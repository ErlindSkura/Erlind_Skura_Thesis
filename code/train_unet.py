"""U-Net semantic baseline: train, then recover instances by connected components.

This baseline exists to show what is lost when the task is treated as semantic
rather than instance segmentation. No boundary-aware post-processing is applied,
so the merge behaviour it is included to demonstrate stays visible instead of
being hidden by a fix that Mask R-CNN does not need.

    python train_unet.py --protocol loso --iters 1500
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
from config import MIN_INSTANCE_PX, PREDICTIONS, SEED, ensure_dirs
from datasets import BeadDataset, collate, load_records, rasterise
from metrics import Instance, as_instances, f1_at_iou
from models import UNetResNet34, dice_bce_loss
from runtime import StepTimer, device_report, format_summary

PROB_GRID = np.arange(0.20, 0.85, 0.05)
MIN_AREA_GRID = (MIN_INSTANCE_PX, 64, 150)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _binary(targets, device):
    out = []
    for t in targets:
        m = t["masks"]
        merged = (m.any(dim=0) if len(m) else
                  torch.zeros(m.shape[-2:], dtype=torch.bool))
        out.append(merged)
    return torch.stack(out).float().unsqueeze(1).to(device)


def train_one_fold(records, fold, *, iters, batch, lr, device, seed):
    torch.manual_seed(seed)
    model = UNetResNet34().to(device)
    timer = StepTimer(iters, batch, len(fold["train"]))
    ds = BeadDataset(records, fold["train"], train=True,
                     samples=iters * batch, seed=seed)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2,
                        collate_fn=collate, pin_memory=device.type == "cuda")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    model.train()
    t0 = time.time()
    for step, (images, targets) in enumerate(loader, start=1):
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / iters))
        x = torch.stack(images).to(device)
        y = _binary(targets, device)
        with autocast("cuda", enabled=device.type == "cuda"):
            loss = dice_bce_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        timer.tick()
        if step % 100 == 0 or step == 1:
            print(f"    step {step:5d}/{iters}  loss {loss.item():.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return model, timer.summary()


@torch.no_grad()
def probability_maps(model, records, names, device):
    model.eval()
    out = {}
    for n in names:
        ds = BeadDataset(records, [n], train=False)
        image, _ = ds[0]
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(image.unsqueeze(0).to(device))
        out[n] = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    return out


def instances_from_probability(prob, thr: float, min_area: int):
    """Connected components of the thresholded map, with mean probability as score."""
    from skimage import measure

    lab = measure.label(prob >= thr, connectivity=1)
    instances, scores = [], []
    for r in measure.regionprops(lab):
        if r.area < min_area:
            continue
        rows, cols = r.slice
        instances.append(Instance(rows.start, rows.stop, cols.start, cols.stop,
                                  r.image, prob.shape))
        scores.append(float(prob[r.slice][r.image].mean()))
    return instances, scores


def pick_params(model, records, train_names, device):
    probs = probability_maps(model, records, train_names, device)
    gt = {n: as_instances(rasterise(records[n].polys, records[n].width,
                                    records[n].height))
          for n in train_names}
    best, best_f1 = (0.5, MIN_INSTANCE_PX), -1.0
    for thr in PROB_GRID:
        for min_area in MIN_AREA_GRID:
            f1 = float(np.mean([
                f1_at_iou(gt[n], instances_from_probability(probs[n], thr, min_area)[0])
                for n in train_names
            ]))
            if f1 > best_f1:
                best_f1, best = f1, (float(thr), int(min_area))
    return best


def run(protocol: str, iters: int, batch: int, lr: float, seed: int = SEED,
        only: list[str] | None = None) -> None:
    ensure_dirs()
    device = _device()
    print(f"device: {device}  protocol: {protocol}  iters: {iters}  batch: {batch}")
    records = load_records()
    all_dets, meta = [], {"protocol": protocol, "method": "unet", "iters": iters,
                          "batch": batch, "lr": lr, "seed": seed, "params": {},
                          "hardware": device_report(), "runtime": {}}

    selected = folds_mod.load(protocol)
    if only:
        selected = [f for f in selected if f["name"] in only]
        if not selected:
            raise SystemExit(f"no fold named {only}")
        meta["partial"] = True
        meta["folds_run"] = [f["name"] for f in selected]
        print(f"PARTIAL RUN: folds {meta['folds_run']} only.")

    for fold in selected:
        print(f"\n[fold {fold['name']}] test={fold['test']}", flush=True)
        model, timing = train_one_fold(records, fold, iters=iters, batch=batch,
                                       lr=lr, device=device, seed=seed)
        meta["runtime"][fold["name"]] = timing
        print(format_summary(fold["name"], timing), flush=True)
        thr, min_area = pick_params(model, records, fold["train"], device)
        meta["params"][fold["name"]] = {"prob_threshold": thr, "min_area_px": min_area}
        print(f"    chosen on training partition: prob>={thr:.2f}, min_area={min_area}px")

        for n, prob in probability_maps(model, records, fold["test"], device).items():
            masks, scores = instances_from_probability(prob, thr, min_area)
            all_dets += predio.encode(masks, scores, records[n].image_id)
            print(f"    {n}: {len(masks)} predicted / {len(records[n].polys)} annotated")
        del model
        torch.cuda.empty_cache()

    out = PREDICTIONS / protocol / ("unet_partial.json" if only else "unet.json")
    predio.save(out, all_dets, meta)
    print(f"\nwrote {len(all_dets)} detections to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="loso", choices=["loso", "random"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", nargs="*", default=None,
                    help="run only these folds as a pilot")
    a = ap.parse_args()
    run(a.protocol, a.iters, a.batch, a.lr, a.seed, a.folds)
