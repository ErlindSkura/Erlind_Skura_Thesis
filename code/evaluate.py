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
    aji, areal_density, as_instances, boxes_from_instances, coco_ap,
    counting_error, equivalent_diameters, f1_at_iou_boxes, merge_split,
    panoptic_quality, summarise,
)

# The preprocessing variants are separate entries because each is a separate
# trained model, written to its own prediction file. So is the magnified variant,
# for the same reason.
PREPROCESS_VARIANTS = ("maskrcnn_clahe", "maskrcnn_background", "maskrcnn_median")
UPSAMPLE_VARIANTS = ("maskrcnn_x3",)
METHODS = (("classical", "unet", "maskrcnn", "fasterrcnn", "yolov8", "yolov5")
           + PREPROCESS_VARIANTS + UPSAMPLE_VARIANTS)

# Metrics that need a predicted mask. A box-only method leaves these unset rather
# than scoring zero: "did not segment" and "segmented badly" are different claims,
# and a zero here would read as the second while meaning the first.
MASK_ONLY_METRICS = ("aji", "pq", "sq", "rq", "merge_rate", "split_rate")

# The only metrics for which pycocotools can emit its -1 "band not populated"
# sentinel. Every other metric is reported as computed, sign included.
SENTINEL_KEYS = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
                 "ar100", "ar_max", "ar_small", "ar_medium", "ar_large")


def _load_coco():
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(COCO_GT))


def score_image(rec, gts, dets, box_only: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    """Every per-micrograph metric, for one micrograph.

    Split out so that the held-out partition and the training partition are
    scored by literally the same code. A generalisation gap measured with two
    slightly different scorers would be a property of the scorers.
    """
    n_pred = len(dets)
    row = {
        "specimen": rec.specimen, "magnification": rec.magnification,
        "n_gt": len(gts), "n_pred": n_pred,
        "counting_error": counting_error(len(gts), n_pred),
        "density_gt": areal_density(len(gts), rec.width, rec.height, rec.um_per_px),
        "density_pred": areal_density(n_pred, rec.width, rec.height, rec.um_per_px),
    }
    empty = np.zeros(0)

    if box_only:
        # Counting and areal density need only a count, so they are reported.
        # The mask metrics and the size distribution are not, and are marked
        # absent rather than filled with a value this method cannot support.
        gt_boxes = boxes_from_instances(gts)
        pred_boxes, _ = predio.to_boxes(dets)
        row["f1_box"] = f1_at_iou_boxes(gt_boxes, pred_boxes)
        row.update(dict.fromkeys(MASK_ONLY_METRICS, None))
        row.update({"median_diam_gt": None, "median_diam_pred": None,
                    "tp": None, "fp": None, "fn": None})
        return row, empty, empty

    # Converted once here; every metric below then reuses the same objects.
    pred_masks = as_instances(predio.to_masks(dets)[0])
    row["n_pred"] = len(pred_masks)
    row["counting_error"] = counting_error(len(gts), len(pred_masks))
    row["density_pred"] = areal_density(len(pred_masks), rec.width, rec.height,
                                        rec.um_per_px)
    d_gt = equivalent_diameters(gts, rec.um_per_px)
    d_pred = equivalent_diameters(pred_masks, rec.um_per_px)
    row["aji"] = aji(gts, pred_masks)
    row["median_diam_gt"] = float(np.median(d_gt)) if len(d_gt) else 0.0
    row["median_diam_pred"] = float(np.median(d_pred)) if len(d_pred) else 0.0
    row.update(panoptic_quality(gts, pred_masks))
    row.update(merge_split(gts, pred_masks))
    return row, d_gt, d_pred


def _fold_row(coco_gt, per_image, names, detections, ids, iou_type,
              box_only: bool) -> dict:
    """Aggregate one fold's micrographs into a single row."""
    row = coco_ap(coco_gt, detections, ids, iou_type=iou_type)
    for key in MASK_ONLY_METRICS:
        vals = [per_image[n][key] for n in names if per_image[n].get(key) is not None]
        row[key] = float(np.mean(vals)) if vals else None
    if box_only:
        row["f1_box"] = float(np.mean([per_image[n]["f1_box"] for n in names]))
    row["counting_error"] = float(np.mean([per_image[n]["counting_error"]
                                           for n in names]))
    row["abs_counting_error"] = float(np.mean([abs(per_image[n]["counting_error"])
                                               for n in names]))
    row["n_gt"] = int(sum(per_image[n]["n_gt"] for n in names))
    row["n_pred"] = int(sum(per_image[n]["n_pred"] for n in names))
    return row


# Aggregated across folds. pycocotools returns -1 for a size band that the fold's
# ground truth does not populate -- at 3000x, for instance, no particle is
# COCO-"small". That sentinel must not be averaged in as if it were a score of
# -1, so it is dropped; a band absent from every fold stays absent from the
# summary. The sentinel only exists for the AP and AR bands, and the test is
# applied to those alone: signed counting error is legitimately negative whenever
# a method under-counts, and rejecting it as a sentinel would drop exactly the
# folds that carry the bias the mean is meant to expose.
SUMMARY_KEYS = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
                "ar100", "ar_max", "ar_small", "ar_medium", "ar_large",
                "aji", "pq", "sq", "rq", "counting_error",
                "abs_counting_error", "merge_rate", "split_rate", "f1_box")


def _summarise_folds(per_fold: dict) -> dict:
    overall = {}
    for k in SUMMARY_KEYS:
        vals = [f[k] for f in per_fold.values() if f.get(k) is not None]
        if k in SENTINEL_KEYS:
            vals = [v for v in vals if v >= 0]
        if vals:
            overall[k] = summarise(vals)
            overall[k]["n_folds"] = len(vals)
    return overall


def evaluate_train_partition(coco_gt, records, gt_masks, protocol: str,
                             method: str) -> dict | None:
    """Score each fold's model on the images it was trained on.

    The comparison the supervisor asked for -- training accuracy against test
    accuracy -- needs the same model scored on both sides of its own split. The
    predictions come free: choosing the score threshold already required
    inference over every training image.

    There is no pooled figure here, deliberately. A micrograph appears in the
    training partition of three of the four folds, so pooling would score the
    same ground truth three times over, weighting the specimens that happen to be
    in more folds. Only the per-fold rows and their mean are reported.
    """
    path = PREDICTIONS / protocol / f"{method}_train.json"
    if not path.exists():
        return None
    detections, meta = predio.load(path)
    box_only = bool(meta.get("box_only")) or predio.is_box_only(detections)
    iou_type = "bbox" if box_only else "segm"

    per_fold = {}
    for fold in folds_mod.load(protocol):
        mine = [d for d in detections if d.get("fold") == fold["name"]]
        names = [n for n in fold["train"] if n in records]
        if not names:
            continue
        by_image = predio.group_by_image(mine)
        per_image = {}
        for n in names:
            rec = records[n]
            per_image[n], _, _ = score_image(rec, gt_masks[n],
                                             by_image.get(rec.image_id, []),
                                             box_only)
        ids = [records[n].image_id for n in names]
        row = _fold_row(coco_gt, per_image, names, mine, ids, iou_type,
                        box_only)
        row["train_images"] = names
        per_fold[fold["name"]] = row

    if not per_fold:
        return None
    return {"per_fold": per_fold, "overall": _summarise_folds(per_fold),
            "box_only": box_only, "iou_type": iou_type}


def generalisation_gap(test: dict, train: dict) -> dict:
    """Training minus test, for every metric both partitions report.

    Signed and in the metric's own units, not as a ratio: a gap of 0.05 AP50 on
    a method scoring 0.30 and on one scoring 0.70 are different claims, and the
    reader has both means in the same table to divide by if that is what they
    want.
    """
    out = {}
    for k, v in train.items():
        if k in test and isinstance(v, dict) and "mean" in v:
            out[k] = {"train": v["mean"], "test": test[k]["mean"],
                      "gap": v["mean"] - test[k]["mean"]}
    return out


def evaluate_method(coco_gt, records, gt_masks, protocol: str, method: str) -> dict | None:
    path = PREDICTIONS / protocol / f"{method}.json"
    if not path.exists():
        return None
    detections, meta = predio.load(path)
    by_image = predio.group_by_image(detections)

    # A method that predicts boxes and no masks is scored in box space. The flag
    # is taken from the detections themselves rather than from the metadata, so
    # the storage form and the scoring can never disagree.
    box_only = bool(meta.get("box_only")) or predio.is_box_only(detections)
    iou_type = "bbox" if box_only else "segm"

    # Every micrograph is scored, including any for which the method produced no
    # detection at all. Iterating over the predictions instead would drop such an
    # image from the totals, turning a complete failure into a missing row.
    per_image, gt_pool, pred_pool = {}, [], []
    for name, rec in records.items():
        row, d_gt, d_pred = score_image(rec, gt_masks[name],
                                        by_image.get(rec.image_id, []), box_only)
        if not box_only:
            gt_pool.append(d_gt)
            pred_pool.append(d_pred)
        per_image[name] = row

    per_fold = {}
    for fold in folds_mod.load(protocol):
        names = [n for n in fold["test"] if n in per_image]
        if not names:
            continue
        ids = [records[n].image_id for n in names]
        dets = [d for d in detections if d["image_id"] in ids]
        row = _fold_row(coco_gt, per_image, names, dets, ids, iou_type,
                        box_only)
        row["test_images"] = names
        per_fold[fold["name"]] = row

    overall = _summarise_folds(per_fold)

    by_mag = {}
    for mag in MAGNIFICATIONS:
        names = [n for n, r in per_image.items() if r["magnification"] == mag]
        if not names:
            continue
        ids = [records[n].image_id for n in names]
        row = coco_ap(coco_gt, [d for d in detections if d["image_id"] in ids],
                      ids, iou_type=iou_type)
        ajis = [per_image[n]["aji"] for n in names
                if per_image[n].get("aji") is not None]
        row["aji"] = float(np.mean(ajis)) if ajis else None
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

    all_ids = sorted(r.image_id for r in records.values())
    return {"meta": meta, "per_image": per_image, "per_fold": per_fold,
            "overall": overall, "box_only": box_only, "iou_type": iou_type,
            "pooled_ap": coco_ap(coco_gt, detections, all_ids, iou_type=iou_type),
            "by_magnification": by_mag, "size_agreement": size}


def run(protocols=("loso", "random"), force: bool = False) -> dict:
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

            def _m(key, fmt=".3f"):
                """A metric this method does not support prints as '--', not 0."""
                return format(o[key]["mean"], fmt) if key in o else "--"

            print(f"  {protocol}/{method:14s} [{res['iou_type']}] "
                  f"AP50 {_m('ap50')}  AP_s {_m('ap_small')}  "
                  f"AJI {_m('aji')}  "
                  f"|count err| {_m('abs_counting_error', '.1f')}%  "
                  f"merge {_m('merge_rate', '.1f')}%")

            train = evaluate_train_partition(coco_gt, records, gt_masks,
                                             protocol, method)
            if train is not None:
                res["train"] = train
                res["generalisation"] = generalisation_gap(o, train["overall"])
                g = res["generalisation"].get("ap50")
                if g:
                    print(f"  {'':>14}  {'':>7} train AP50 {g['train']:.3f} "
                          f"vs test {g['test']:.3f}  gap {g['gap']:+.3f}")

    path = RESULTS / "metrics.json"
    _guard_overwrite(path, out, force)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return out


def _covered(doc: dict) -> set[tuple[str, str]]:
    """The (protocol, method) pairs a metrics document actually contains."""
    return {(p, m) for p, methods in doc.items()
            if isinstance(methods, dict) and p != "ground_truth"
            for m in methods}


def _guard_overwrite(path, new: dict, force: bool) -> None:
    """Refuse to replace a fuller metrics file with a thinner one.

    Chapter 5 reads this file, and a method missing from it becomes a missing
    row rather than a visible failure. Evaluating a subset -- because a training
    stage has not been re-run yet, or because a prediction file was not copied
    back from Colab -- would otherwise silently discard results that cost hours
    of GPU time. Losing exactly that way is what prompted this check.
    """
    if force or not path.exists():
        return
    try:
        old = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    lost = _covered(old) - _covered(new)
    if lost:
        listing = ", ".join(f"{p}/{m}" for p, m in sorted(lost))
        raise SystemExit(
            f"refusing to overwrite {path}\n"
            f"  it currently holds results this run did not reproduce: {listing}\n"
            f"  re-run the missing stages first, or pass --force to discard them."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="*", default=["loso", "random"])
    ap.add_argument("--force", action="store_true",
                    help="overwrite metrics.json even if this run covers fewer "
                         "methods than the file already holds")
    a = ap.parse_args()
    run(tuple(a.protocols), a.force)
