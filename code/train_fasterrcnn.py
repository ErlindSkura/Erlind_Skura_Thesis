"""Train and evaluate Faster R-CNN under a given partitioning protocol.

Faster R-CNN detects but does not segment. It is included because the endpoint
this work exists to serve is a *count*, and counting needs only detection: if the
mask branch buys nothing for counting accuracy, then the cheaper model is the
right tool, and that is a result worth reporting rather than assuming.

The comparison against Mask R-CNN is controlled. Both share the ResNet-50 + FPN
backbone, the anchor sizes rescaled to the measured particle size, the detection
budget of 400 objects per image, the optimiser and schedule, the fold definitions
and the rule that the score threshold is chosen on the fold's own training
partition. The mask branch is the only difference, so a gap between them is
attributable to it.

Predictions are stored as boxes with no segmentation field. Consequently AJI,
panoptic quality, merge and split rates and the size distribution are not
reported for this method: they are mask quantities, and filling each box to
manufacture one would score the method on something it never predicted.

    python train_fasterrcnn.py --protocol loso --iters 1500
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import folds as folds_mod
import predio
from config import CROP, PREDICTIONS, SEED, WORK_H, FULL_W, ensure_dirs
from datasets import BeadDataset, collate, load_records, rasterise
from metrics import as_instances, boxes_from_instances, f1_at_iou_boxes
from models import build_fasterrcnn, set_input_size
from runtime import StepTimer, device_report, format_summary


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def infer(model, records, names, device, score_thresh: float):
    """Predict on whole micrographs at native resolution."""
    model.eval()
    set_input_size(model, WORK_H, FULL_W)
    ds = BeadDataset(records, names, train=False)
    out = {}
    for i in range(len(ds)):
        image, target = ds[i]
        with autocast("cuda", enabled=device.type == "cuda"):
            pred = model([image.to(device)])[0]
        keep = pred["scores"] >= score_thresh
        boxes = pred["boxes"][keep].float().cpu().numpy()
        scores = pred["scores"][keep].float().cpu().numpy().tolist()

        # A degenerate box has no area, so it can neither be matched nor scored.
        # Dropping it here keeps boxes and scores aligned everywhere downstream,
        # the same guarantee train_maskrcnn.infer makes for empty masks.
        wh = boxes[:, 2:] - boxes[:, :2] if len(boxes) else np.zeros((0, 2))
        ok = (wh > 0).all(axis=1) if len(boxes) else np.zeros(0, dtype=bool)
        out[target["name"]] = (boxes[ok],
                               [s for s, k in zip(scores, ok) if k])
    return out


@torch.no_grad()
def pick_threshold(model, records, train_names, device) -> float:
    """Choose the score threshold on the training partition only."""
    raw = infer(model, records, train_names, device, score_thresh=0.05)
    gt = {n: boxes_from_instances(
              as_instances(rasterise(records[n].polys, records[n].width,
                                     records[n].height)))
          for n in train_names}
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        scores = []
        for n in train_names:
            boxes, sc = raw[n]
            sel = boxes[[i for i, s in enumerate(sc) if s >= t]] if len(boxes) else boxes
            scores.append(f1_at_iou_boxes(gt[n], sel))
        mean_f1 = float(np.mean(scores))
        if mean_f1 > best_f1:
            best_f1, best_t = mean_f1, float(t)
    return best_t


def train_one_fold(records, fold, *, iters, batch, lr, device, seed):
    torch.manual_seed(seed)
    model = build_fasterrcnn().to(device)
    set_input_size(model, CROP, CROP)

    timer = StepTimer(iters, batch, len(fold["train"]))
    ds = BeadDataset(records, fold["train"], train=True,
                     samples=iters * batch, seed=seed)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2,
                        collate_fn=collate, pin_memory=device.type == "cuda")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    warmup = min(200, iters // 10)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    model.train()
    t0 = time.time()
    for step, (images, targets) in enumerate(loader, start=1):
        factor = (step / max(warmup, 1) if step <= warmup
                  else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(iters - warmup, 1))))
        for g in opt.param_groups:
            g["lr"] = lr * factor

        images = [im.to(device) for im in images]
        # The mask tensors are dropped here: this model has no branch to consume
        # them, and torchvision's detector rejects target keys it cannot use.
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in t.items() if k in ("boxes", "labels")}
                   for t in targets]
        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        with autocast("cuda", enabled=device.type == "cuda"):
            losses = model(images, targets)
            loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt)
        scaler.update()
        timer.tick()

        if step % 100 == 0 or step == 1:
            print(f"    step {step:5d}/{iters}  loss {loss.item():.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.5f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return model, timer.summary()


def run(protocol: str, iters: int, batch: int, lr: float, seed: int = SEED,
        only: list[str] | None = None) -> None:
    ensure_dirs()
    device = _device()
    print(f"device: {device}  protocol: {protocol}  iters: {iters}  batch: {batch}")
    records = load_records()
    all_dets, meta = [], {"protocol": protocol, "method": "fasterrcnn",
                          "iters": iters, "batch": batch, "lr": lr, "seed": seed,
                          "thresholds": {}, "device": str(device),
                          "hardware": device_report(), "runtime": {},
                          "box_only": True}

    selected = folds_mod.load(protocol)
    if only:
        selected = [f for f in selected if f["name"] in only]
        if not selected:
            raise SystemExit(f"no fold named {only}; available: "
                             f"{[f['name'] for f in folds_mod.load(protocol)]}")
        meta["partial"] = True
        meta["folds_run"] = [f["name"] for f in selected]
        print(f"PARTIAL RUN: folds {meta['folds_run']} only. Written to a "
              f"separate file; evaluate.py will not pick it up.")

    for fold in selected:
        print(f"\n[fold {fold['name']}] train={len(fold['train'])} "
              f"test={fold['test']}", flush=True)
        model, timing = train_one_fold(records, fold, iters=iters, batch=batch,
                                       lr=lr, device=device, seed=seed)
        meta["runtime"][fold["name"]] = timing
        print(format_summary(fold["name"], timing), flush=True)
        thr = pick_threshold(model, records, fold["train"], device)
        meta["thresholds"][fold["name"]] = thr
        print(f"    score threshold chosen on training partition: {thr:.2f}")

        preds = infer(model, records, fold["test"], device, score_thresh=thr)
        for name, (boxes, scores) in preds.items():
            all_dets += predio.encode_boxes(boxes, scores, records[name].image_id)
            print(f"    {name}: {len(boxes)} predicted / "
                  f"{len(records[name].polys)} annotated")
        del model
        torch.cuda.empty_cache()

    name = "fasterrcnn" if not only else "fasterrcnn_partial"
    out = PREDICTIONS / protocol / f"{name}.json"
    predio.save(out, all_dets, meta)
    print(f"\nwrote {len(all_dets)} detections to {out}")

    if only:
        for fold in selected:
            for n in fold["test"]:
                gt = boxes_from_instances(
                    as_instances(rasterise(records[n].polys, records[n].width,
                                           records[n].height)))
                got = [d for d in all_dets if d["image_id"] == records[n].image_id]
                pred, _ = predio.to_boxes(got)
                print(f"    {n}: {len(pred):>3} predicted / {len(gt):>3} annotated"
                      f"   F1@0.5 (box) {f1_at_iou_boxes(gt, pred):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="loso", choices=["loso", "random"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", nargs="*", default=None,
                    help="run only these folds (e.g. --folds Z2) as a pilot; "
                         "results go to a separate file and are not evaluated")
    a = ap.parse_args()
    run(a.protocol, a.iters, a.batch, a.lr, a.seed, a.folds)
