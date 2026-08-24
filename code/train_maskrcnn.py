"""Train and evaluate Mask R-CNN under a given partitioning protocol.

One model is trained per fold, from the same initialisation, and predicts only
on that fold's held-out images. Nothing about a fold's test partition -- not the
score threshold, not the number of iterations -- is chosen using the test data:
the score threshold is selected by maximising F1 on the fold's own *training*
images.

    python train_maskrcnn.py --protocol loso --iters 1500
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import checkpoints as ckpt
import folds as folds_mod
from config import CROP, PREDICTIONS, SEED, WORK_H, FULL_W, ensure_dirs
from datasets import BeadDataset, collate, load_records, rasterise
from metrics import as_instances, f1_at_iou
from models import build_maskrcnn, set_input_size
from preprocess import VARIANTS
from runtime import StepTimer, device_report, format_summary
import predio


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def infer(model, records, names, device, score_thresh: float, mask_thresh: float = 0.5,
          preprocess: str = "none", upsample: int = 1):
    """Predict on whole micrographs, always returned at native resolution."""
    model.eval()
    ds = BeadDataset(records, names, train=False, preprocess=preprocess,
                     upsample=upsample)
    out = {}

    if upsample != 1:
        # A magnified frame does not fit through the model in one pass, so it is
        # tiled and the predictions mapped back; see upsample.py. Both the input
        # size and the detection budget become per-tile quantities here, and both
        # are put back afterwards: a model left with min_size at the tile size
        # would silently upscale the next whole frame it was given.
        import upsample as up
        budget = model.roi_heads.detections_per_img
        set_input_size(model, up.TILE, up.TILE)
        model.roi_heads.detections_per_img = up.TILE_DETECTIONS
        try:
            for i in range(len(ds)):
                image, name = ds.frame(i)
                out[name] = up.predict_tiled(
                    model, image, device, native_hw=(WORK_H, FULL_W),
                    factor=upsample, want_masks=True,
                    score_thresh=score_thresh, mask_thresh=mask_thresh)
        finally:
            model.roi_heads.detections_per_img = budget
            set_input_size(model, WORK_H, FULL_W)
        return out

    set_input_size(model, WORK_H, FULL_W)
    for i in range(len(ds)):
        image, name = ds.frame(i)
        with autocast("cuda", enabled=device.type == "cuda"):
            pred = model([image.to(device)])[0]
        keep = pred["scores"] >= score_thresh
        masks = (pred["masks"][keep, 0] > mask_thresh).cpu().numpy().astype(bool)
        scores = pred["scores"][keep].cpu().numpy().tolist()

        # A soft mask can threshold away to nothing even when its box scored
        # well. Such a detection has no area, so it cannot be matched or scored;
        # dropping it here keeps masks and scores aligned everywhere downstream.
        kept = [(m, s) for m, s in zip(masks, scores) if m.any()]
        out[name] = ([m for m, _ in kept], [s for _, s in kept])
    return out


@torch.no_grad()
def pick_threshold(model, records, train_names, device,
                   preprocess: str = "none", upsample: int = 1):
    """Choose the score threshold on the training partition only.

    Returns the threshold *and* the training-partition predictions it implies.
    The predictions are a by-product: choosing the threshold already requires
    inference over every training image, so returning it costs nothing and is
    what makes the training-versus-test comparison of Chapter 5 essentially
    free. Recomputing it later would have doubled the inference bill for a
    number that was already sitting in memory.
    """
    raw = infer(model, records, train_names, device, score_thresh=0.05,
                preprocess=preprocess, upsample=upsample)
    # Both sides converted to sparse instances once, not on every threshold.
    gt = {n: as_instances(rasterise(records[n].polys, records[n].width,
                                    records[n].height))
          for n in train_names}
    cand = {n: (as_instances(raw[n][0]), raw[n][1]) for n in train_names}
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        scores = []
        for n in train_names:
            masks, sc = cand[n]
            sel = [m for m, s in zip(masks, sc) if s >= t]
            scores.append(f1_at_iou(gt[n], sel))
        mean_f1 = float(np.mean(scores))
        if mean_f1 > best_f1:
            best_f1, best_t = mean_f1, float(t)

    at_threshold = {}
    for n in train_names:
        masks, sc = raw[n]
        kept = [(m, s) for m, s in zip(masks, sc) if s >= best_t]
        at_threshold[n] = ([m for m, _ in kept], [s for _, s in kept])
    return best_t, at_threshold


def train_one_fold(records, fold, *, iters, batch, lr, device, seed,
                   preprocess="none", upsample=1):
    torch.manual_seed(seed)
    model = build_maskrcnn().to(device)
    set_input_size(model, CROP, CROP)   # square crops pass through unresampled

    timer = StepTimer(iters, batch, len(fold["train"]))
    # The crop size is deliberately left alone under magnification. A 512 px crop
    # of a 3x frame covers a third of the physical area it covered before, which
    # is the point: the objects in it are three times larger in pixels. Enlarging
    # the crop to hold the same field of view would undo the factor being tested.
    ds = BeadDataset(records, fold["train"], train=True,
                     samples=iters * batch, seed=seed, preprocess=preprocess,
                     upsample=upsample)
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
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in t.items() if k != "name"} for t in targets]
        # Crops that happen to contain no bead carry no detection loss; skipping
        # them avoids a NaN from the empty-target path.
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


def _method_name(preprocess: str, upsample: int) -> str:
    """The name this variant writes under, for predictions and checkpoints alike.

    One function rather than two copies, so that a checkpoint is always findable
    from the prediction file it produced.
    """
    name = "maskrcnn" if preprocess == "none" else f"maskrcnn_{preprocess}"
    return name if upsample == 1 else f"{name}_x{upsample}"


def run(protocol: str, iters: int, batch: int, lr: float, seed: int = SEED,
        only: list[str] | None = None, preprocess: str = "none",
        upsample: int = 1, save_checkpoints: bool = True) -> None:
    ensure_dirs()
    device = _device()
    print(f"device: {device}  protocol: {protocol}  iters: {iters}  batch: {batch}"
          + (f"  upsample: {upsample}x" if upsample != 1 else ""))
    records = load_records()
    all_dets, meta = [], {"protocol": protocol, "method": "maskrcnn",
                          "iters": iters, "batch": batch, "lr": lr, "seed": seed,
                          "thresholds": {}, "device": str(device),
                          "hardware": device_report(), "runtime": {},
                          "preprocess": preprocess, "upsample": upsample}
    # Predictions on each fold's own *training* images, kept apart from the test
    # predictions and tagged with the fold that produced them. A micrograph is in
    # the training partition of three of the four folds and is predicted
    # differently by each, so image_id alone no longer identifies a prediction.
    train_dets: list[dict] = []

    selected = folds_mod.load(protocol)
    if only:
        selected = [f for f in selected if f["name"] in only]
        if not selected:
            raise SystemExit(f"no fold named {only}; available: "
                             f"{[f['name'] for f in folds_mod.load(protocol)]}")
        # A partial run cannot be evaluated as a cross-validation, so it is kept
        # out of the file the evaluator reads.
        meta["partial"] = True
        meta["folds_run"] = [f["name"] for f in selected]
        print(f"PARTIAL RUN: folds {meta['folds_run']} only. Written to a "
              f"separate file; evaluate.py will not pick it up.")

    for fold in selected:
        print(f"\n[fold {fold['name']}] train={len(fold['train'])} "
              f"test={fold['test']}", flush=True)
        model, timing = train_one_fold(records, fold, iters=iters, batch=batch,
                                       lr=lr, device=device, seed=seed,
                                       preprocess=preprocess, upsample=upsample)
        meta["runtime"][fold["name"]] = timing
        print(format_summary(fold["name"], timing), flush=True)
        thr, on_train = pick_threshold(model, records, fold["train"], device,
                                       preprocess, upsample)
        meta["thresholds"][fold["name"]] = thr
        print(f"    score threshold chosen on training partition: {thr:.2f}")

        # Saved with the threshold, and after it has been chosen: the weights on
        # their own reproduce none of the reported numbers.
        ckpt.save(model, _method_name(preprocess, upsample), protocol,
                  fold["name"], threshold=thr, enabled=save_checkpoints,
                  extra={"iters": iters, "batch": batch, "lr": lr, "seed": seed,
                         "preprocess": preprocess, "upsample": upsample})

        for name, (masks, scores) in on_train.items():
            for det in predio.encode(masks, scores, records[name].image_id):
                det["fold"] = fold["name"]
                train_dets.append(det)

        preds = infer(model, records, fold["test"], device, score_thresh=thr,
                      preprocess=preprocess, upsample=upsample)
        for name, (masks, scores) in preds.items():
            all_dets += predio.encode(masks, scores, records[name].image_id)
            print(f"    {name}: {len(masks)} predicted / "
                  f"{len(records[name].polys)} annotated")
        del model
        torch.cuda.empty_cache()

    # Each variant is a separate result, so it gets a separate file. Without this
    # the ablations would overwrite the baseline run.
    name = _method_name(preprocess, upsample)
    if only:
        name += "_partial"
    out = PREDICTIONS / protocol / f"{name}.json"
    predio.save(out, all_dets, meta)
    print(f"\nwrote {len(all_dets)} detections to {out}")

    if train_dets and not only:
        out_train = PREDICTIONS / protocol / f"{name}_train.json"
        predio.save(out_train, train_dets, dict(meta, partition="train"))
        print(f"wrote {len(train_dets)} training-partition detections to "
              f"{out_train}")

    if only:
        # Report the pilot fold's own numbers, so the decision whether to commit
        # to the full run can be made from evidence rather than from hope.
        from data_io import rasterise
        from metrics import aji, as_instances, f1_at_iou
        for fold in selected:
            for n in fold["test"]:
                gt = as_instances(rasterise(records[n].polys, records[n].width,
                                            records[n].height))
                got = [d for d in all_dets if d["image_id"] == records[n].image_id]
                pred = as_instances(predio.to_masks(got)[0])
                print(f"    {n}: {len(pred):>3} predicted / {len(gt):>3} annotated"
                      f"   AJI {aji(gt, pred):.3f}   F1@0.5 {f1_at_iou(gt, pred):.3f}")


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
    ap.add_argument("--preprocess", default="none", choices=VARIANTS,
                    help="preprocessing variant; each writes its own file")
    ap.add_argument("--upsample", type=int, default=1,
                    help="magnify image and label by this factor with bicubic "
                         "interpolation before training and inference; writes "
                         "its own file")
    ap.add_argument("--no-checkpoints", action="store_true",
                    help="do not save the trained weights; saves about 700 MB "
                         "per four-fold run, at the cost of having to retrain "
                         "before any later question about the models")
    a = ap.parse_args()
    run(a.protocol, a.iters, a.batch, a.lr, a.seed, a.folds, a.preprocess,
        a.upsample, save_checkpoints=not a.no_checkpoints)
