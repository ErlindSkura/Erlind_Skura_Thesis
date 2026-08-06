"""Evaluate every saved prediction file and write one metrics document.

Nothing here trains or predicts. Evaluation is separated from training so that
reporting can be re-run -- and re-checked -- without repeating the expensive
stage, and so that all three methods are measured by literally the same code.

    python evaluate.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json

import numpy as np

import folds as folds_mod
import predio
from config import COCO_GT, MAGNIFICATIONS, PREDICTIONS, RESULTS, ensure_dirs
from data_io import load_records, rasterise
from metrics import (
    aji, areal_density, as_instances, coco_ap, counting_error,
    equivalent_diameters, merge_split, panoptic_quality, summarise,
)

METHODS = ("classical", "unet", "maskrcnn")


def _load_coco():
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(COCO_GT))


def evaluate_method(coco_gt, records, gt_masks, protocol: str, method: str) -> dict | None:
    path = PREDICTIONS / protocol / f"{method}.json"
    if not path.exists():
        return None
    detections, meta = predio.load(path)
    by_image = predio.group_by_image(detections)
    id_to_name = {r.image_id: n for n, r in records.items()}

    per_image, gt_pool, pred_pool = {}, [], []
    for image_id, dets in by_image.items():
        name = id_to_name[image_id]
        rec = records[name]
        # Converted once here; every metric below then reuses the same objects.
        pred_masks = as_instances(predio.to_masks(dets)[0])
        gts = gt_masks[name]

        d_gt = equivalent_diameters(gts, rec.um_per_px)
        d_pred = equivalent_diameters(pred_masks, rec.um_per_px)
        gt_pool.append(d_gt)
        pred_pool.append(d_pred)

        row = {
            "specimen": rec.specimen, "magnification": rec.magnification,
            "n_gt": len(gts), "n_pred": len(pred_masks),
            "counting_error": counting_error(len(gts), len(pred_masks)),
            "aji": aji(gts, pred_masks),
            "density_gt": areal_density(len(gts), rec.width, rec.height, rec.um_per_px),
            "density_pred": areal_density(len(pred_masks), rec.width, rec.height,
                                          rec.um_per_px),
            "median_diam_gt": float(np.median(d_gt)) if len(d_gt) else 0.0,
            "median_diam_pred": float(np.median(d_pred)) if len(d_pred) else 0.0,
        }
        row.update(panoptic_quality(gts, pred_masks))
        row.update(merge_split(gts, pred_masks))
        per_image[name] = row

    per_fold = {}
    for fold in folds_mod.load(protocol):
        names = [n for n in fold["test"] if n in per_image]
        if not names:
            continue
        ids = [records[n].image_id for n in names]
        dets = [d for d in detections if d["image_id"] in ids]
        row = coco_ap(coco_gt, dets, ids)
        for key in ("aji", "pq", "sq", "rq", "merge_rate", "split_rate"):
            row[key] = float(np.mean([per_image[n][key] for n in names]))
        row["counting_error"] = float(np.mean([per_image[n]["counting_error"]
                                               for n in names]))
        row["abs_counting_error"] = float(np.mean([abs(per_image[n]["counting_error"])
                                                   for n in names]))
        row["n_gt"] = int(sum(per_image[n]["n_gt"] for n in names))
        row["n_pred"] = int(sum(per_image[n]["n_pred"] for n in names))
        row["test_images"] = names
        per_fold[fold["name"]] = row

    keys = ("ap", "ap50", "ap75", "aji", "pq", "sq", "rq", "counting_error",
            "abs_counting_error", "merge_rate", "split_rate")
    overall = {k: summarise([f[k] for f in per_fold.values()]) for k in keys}

    by_mag = {}
    for mag in MAGNIFICATIONS:
        names = [n for n, r in per_image.items() if r["magnification"] == mag]
        if not names:
            continue
        ids = [records[n].image_id for n in names]
        row = coco_ap(coco_gt, [d for d in detections if d["image_id"] in ids], ids)
        row["aji"] = float(np.mean([per_image[n]["aji"] for n in names]))
        row["abs_counting_error"] = float(np.mean([abs(per_image[n]["counting_error"])
                                                   for n in names]))
        row["n_gt"] = int(sum(per_image[n]["n_gt"] for n in names))
        row["n_images"] = len(names)
        by_mag[str(mag)] = row

    g = np.concatenate(gt_pool) if gt_pool else np.zeros(0)
    p = np.concatenate(pred_pool) if pred_pool else np.zeros(0)
    size = {"n_gt": int(g.size), "n_pred": int(p.size)}
    if g.size and p.size:
        from scipy import stats
        ks = stats.ks_2samp(g, p)
        size.update({
            "gt_median": float(np.median(g)), "pred_median": float(np.median(p)),
            "gt_iqr": [float(np.percentile(g, 25)), float(np.percentile(g, 75))],
            "pred_iqr": [float(np.percentile(p, 25)), float(np.percentile(p, 75))],
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
        })

    return {"meta": meta, "per_image": per_image, "per_fold": per_fold,
            "overall": overall, "pooled_ap": coco_ap(
                coco_gt, detections, sorted(by_image)),
            "by_magnification": by_mag, "size_agreement": size}


def run(protocols=("loso", "random")) -> dict:
    ensure_dirs()
    coco_gt = _load_coco()
    records = load_records()
    gt_masks = {n: as_instances(rasterise(r.polys, r.width, r.height))
                for n, r in records.items()}

    # The ground-truth size distribution is a property of the annotation, not of
    # any method, so it is computed once over all 11 micrographs. Deriving it
    # from a method's own output would silently drop any image that method
    # failed to produce a single detection for.
    gt_all = np.concatenate([
        equivalent_diameters(gt_masks[n], r.um_per_px) for n, r in records.items()
    ])
    out: dict = {"ground_truth": {
        "n": int(gt_all.size),
        "median": float(np.median(gt_all)),
        "iqr": [float(np.percentile(gt_all, 25)), float(np.percentile(gt_all, 75))],
    }}
    print(f"  ground truth: {gt_all.size} beads, median diameter "
          f"{np.median(gt_all):.2f} um")

    for protocol in protocols:
        out[protocol] = {}
        for method in METHODS:
            res = evaluate_method(coco_gt, records, gt_masks, protocol, method)
            if res is None:
                print(f"  {protocol}/{method}: no predictions, skipped")
                continue
            out[protocol][method] = res
            o = res["overall"]
            print(f"  {protocol}/{method:10s} "
                  f"AP50 {o['ap50']['mean']:.3f}+-{o['ap50']['std']:.3f}  "
                  f"AJI {o['aji']['mean']:.3f}  "
                  f"|count err| {o['abs_counting_error']['mean']:.1f}%  "
                  f"merge {o['merge_rate']['mean']:.1f}%")

    path = RESULTS / "metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="*", default=["loso", "random"])
    run(tuple(ap.parse_args().protocols))
