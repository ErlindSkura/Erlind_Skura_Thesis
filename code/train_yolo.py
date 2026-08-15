"""Train and evaluate a YOLO model under a given partitioning protocol.

Handles both YOLOv8 and YOLOv5 through the Ultralytics package; the family is
taken from the weights filename and each writes its own prediction file.

This method is not a controlled ablation and must not be read as one. Mask R-CNN
and Faster R-CNN differ from each other only in the mask branch, so a gap between
them is attributable to it. A YOLO model shares nothing with either: a different
backbone, an anchor-free head, a different label assignment rule, a different
loss, and its own augmentation and optimiser schedule implemented inside the
Ultralytics package. It answers "does a different family of architecture do
better on this data", not "which component is responsible". Chapter 5 reports it
on that footing.

Two things about the YOLOv5 side need stating, because both invite a wrong
reading:

*It is YOLOv5u, not the YOLOv5 of the literature.* Ultralytics distributes
``yolov5su.pt`` and its siblings, where the "u" marks the original YOLOv5
backbone fitted with YOLOv8's anchor-free, objectness-free split head. Comparing
``yolov5su`` against ``yolov8s`` therefore varies mostly the backbone, and is not
the anchor-based-versus-anchor-free comparison a reader would assume from the
version numbers. Reproducing the published anchor-based YOLOv5 would mean the
separate ``ultralytics/yolov5`` repository and its own training entry point.

*It detects but does not segment.* Ultralytics ships no ``-seg`` variant for
YOLOv5, so it goes through the box-only path: box AP and counting, no AJI,
no panoptic quality and no size distribution. YOLOv8 has ``-seg`` weights and can
be scored either way.

Three settings depart from the Ultralytics defaults, each because of a measured
property of this dataset rather than by search:

``imgsz=1024``      The default of 640 would downscale a 1024x736 micrograph by
                    0.625, taking the median particle at 500x from 14.3 px to
                    8.9 px. Both dimensions are already multiples of 32, so
                    native resolution needs no padding.

``mosaic=0.0``      Mosaic composes four images and rescales, roughly halving
                    apparent object size. With 82.8% of particles already below
                    COCO's small-object threshold, that trades away the signal
                    the model most needs. Left available through ``--mosaic``.

``max_det=400``     A single 500x micrograph holds up to 152 particles, and the
                    default of 300 would truncate the count on dense images --
                    an error in exactly the quantity this work measures.

    python train_yolo.py --protocol loso --iters 1500
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

import folds as folds_mod
import predio
from config import PREDICTIONS, SCRATCH, SEED, ensure_dirs
from data_io import load_records, rasterise
from metrics import as_instances, boxes_from_instances, f1_at_iou, f1_at_iou_boxes
from runtime import device_report

# The exported dataset and the Ultralytics run directory both live here rather
# than under WORK, and the reason is arithmetic. The step budget converts to 750
# epochs over 8 micrographs, and Ultralytics rewrites ``last.pt`` at the end of
# every one of them; on Colab, WORK is Google Drive, so that is 750 sequential
# 23 MB round trips per fold. Neither the exported copies nor the checkpoints are
# results: ``export_fold`` rebuilds the dataset from the records on every run,
# and the predictions go to PREDICTIONS as they do for every other method.
DATASET_ROOT = SCRATCH / "yolo"


def method_name(weights: str) -> str:
    """The key this run is stored and reported under.

    Derived from the weights rather than fixed, because YOLOv5 and YOLOv8 are run
    through the same code path and would otherwise overwrite each other's
    predictions.
    """
    stem = Path(weights).stem.lower()
    for family in ("yolov5", "yolov8", "yolov9", "yolov10", "yolo11", "yolo12"):
        if stem.startswith(family):
            return family
    return "yolo"


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit(
            "ultralytics is not installed. In Colab:  !pip install ultralytics"
        ) from e
    return YOLO


def write_labels(rec, path, segment: bool) -> None:
    """One line per instance, class first, coordinates normalised to [0, 1].

    Segmentation labels carry the polygon; detection labels carry the centre and
    extent of its bounding box. Coordinates are clipped because a vertex may sit
    a fraction of a pixel outside the frame, which Ultralytics rejects.

    Ten decimal places, not the conventional six, which leaves the coordinate
    error at 4e-8 px instead of 5e-4 px. The extra digits cost a few bytes per
    vertex.

    They do not make the round-trip exact, and it is worth recording why, because
    the obvious check for this function fails in a way that looks alarming and is
    not. Many annotation vertices land on exact integers or exact thirds of a
    pixel. Normalising by the frame width and back cannot reproduce such a value
    bit-for-bit, so 73.0 returns as 72.99999997 -- and a vertex sitting precisely
    on a pixel boundary falls to the other side of it when rasterised. On a
    164-pixel particle that showed up as eight flipped boundary pixels and an IoU
    of 0.95 against the source polygon. Raising the precision from six places to
    fourteen does not converge: the error shrinks to 4e-12 px and the raster still
    differs, because the perturbation never becomes exactly zero and the vertex is
    balanced on a knife edge.

    That artefact belongs to the comparison, not to the labels. Ultralytics
    rasterises these polygons with its own code, and 4e-8 px is far below any
    tolerance that matters to it. The contract this function owes is coordinate
    fidelity, which is what ``tests`` checks, rather than pixel-identical
    re-rasterisation, which is not achievable through normalised decimal text and
    would not mean anything if it were.
    """
    lines = []
    for p in rec.polys:
        x = np.clip(p[:, 0] / rec.width, 0.0, 1.0)
        y = np.clip(p[:, 1] / rec.height, 0.0, 1.0)
        if segment:
            coords = " ".join(f"{a:.10f} {b:.10f}" for a, b in zip(x, y))
            lines.append(f"0 {coords}")
        else:
            x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
            lines.append(f"0 {(x0 + x1) / 2:.10f} {(y0 + y1) / 2:.10f} "
                         f"{x1 - x0:.10f} {y1 - y0:.10f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_fold(records, fold, root: Path, segment: bool) -> Path:
    """Lay out one fold as an Ultralytics dataset directory.

    The held-out micrographs are written as the validation split so that training
    logs remain readable, but nothing about them is used to choose anything: the
    score threshold is selected on the training split, exactly as for the other
    methods, and the final weights are the last epoch's rather than the best by
    validation score.
    """
    import yaml

    shutil.rmtree(root, ignore_errors=True)
    for split, names in (("train", fold["train"]), ("val", fold["test"])):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for n in names:
            rec = records[n]
            shutil.copy(rec.path, root / "images" / split / f"{n}.png")
            write_labels(rec, root / "labels" / split / f"{n}.txt", segment)

    cfg = root / "dataset.yaml"
    cfg.write_text(yaml.safe_dump({
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "particle"},
    }), encoding="utf-8")
    return cfg


def predict(model, records, names, conf: float, segment: bool, imgsz: int):
    """Run inference on whole micrographs and return masks or boxes."""
    out = {}
    for n in names:
        rec = records[n]
        res = model.predict(source=str(rec.path), imgsz=imgsz, conf=conf,
                            max_det=400, retina_masks=segment, verbose=False)[0]
        scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
        if segment:
            if res.masks is None:
                out[n] = ([], [])
                continue
            masks = res.masks.data.cpu().numpy() > 0.5
            # A prototype mask can threshold away to nothing while its box still
            # scores; such a detection has no area and cannot be matched, so it is
            # dropped here to keep masks and scores aligned downstream.
            kept = [(m, s) for m, s in zip(masks, scores) if m.any()]
            out[n] = ([m for m, _ in kept], [s for _, s in kept])
        else:
            boxes = (res.boxes.xyxy.cpu().numpy() if res.boxes is not None
                     else np.zeros((0, 4)))
            out[n] = (boxes, scores)
    return out


def pick_threshold(model, records, train_names, segment: bool, imgsz: int) -> float:
    """Choose the confidence threshold on the training partition only."""
    raw = predict(model, records, train_names, 0.05, segment, imgsz)
    best_t, best_f1 = 0.5, -1.0
    gt = {n: as_instances(rasterise(records[n].polys, records[n].width,
                                    records[n].height))
          for n in train_names}
    for t in np.arange(0.05, 0.96, 0.05):
        scores = []
        for n in train_names:
            items, sc = raw[n]
            sel = [m for m, s in zip(items, sc) if s >= t]
            if segment:
                scores.append(f1_at_iou(gt[n], sel))
            else:
                pred = np.array(sel).reshape(-1, 4)
                scores.append(f1_at_iou_boxes(boxes_from_instances(gt[n]), pred))
        mean_f1 = float(np.mean(scores))
        if mean_f1 > best_f1:
            best_f1, best_t = mean_f1, float(t)
    return best_t


def run(protocol: str, iters: int, batch: int, weights: str, imgsz: int,
        mosaic: float, seed: int = SEED, only: list[str] | None = None) -> None:
    YOLO = _require_ultralytics()
    ensure_dirs()
    records = load_records()
    segment = "-seg" in weights

    method = method_name(weights)
    all_dets, meta = [], {
        "protocol": protocol, "method": method, "weights": weights,
        "iters": iters, "batch": batch, "imgsz": imgsz, "mosaic": mosaic,
        "seed": seed, "thresholds": {}, "hardware": device_report(),
        "runtime": {}, "box_only": not segment,
    }

    selected = folds_mod.load(protocol)
    if only:
        selected = [f for f in selected if f["name"] in only]
        if not selected:
            raise SystemExit(f"no fold named {only}")
        meta["partial"] = True
        meta["folds_run"] = [f["name"] for f in selected]
        print(f"PARTIAL RUN: folds {meta['folds_run']} only.")

    for fold in selected:
        n_train = len(fold["train"])
        # Ultralytics counts epochs, the other methods count optimiser steps.
        # Converting here keeps the gradient budget equal across methods, which is
        # the only sense in which this comparison is controlled at all.
        epochs = max(1, round(iters * batch / n_train))
        print(f"\n[fold {fold['name']}] train={n_train} test={fold['test']}")
        print(f"    {iters} steps x batch {batch} = {epochs} epochs "
              f"over {n_train} micrographs", flush=True)

        root = DATASET_ROOT / protocol / fold["name"]
        cfg = export_fold(records, fold, root, segment)

        model = YOLO(weights)
        t0 = time.perf_counter()
        # cache="ram" holds the training split in memory. Eight micrographs at
        # 1024x736x3 is about 18 MB, and it removes 750 re-reads of the same
        # files. It changes what is timed, not what is learnt.
        model.train(data=str(cfg), epochs=epochs, imgsz=imgsz, batch=batch,
                    seed=seed, mosaic=mosaic, max_det=400, val=False,
                    cache="ram", project=str(root / "runs"), name="train",
                    exist_ok=True, plots=False, verbose=False)
        wall = time.perf_counter() - t0
        steps_per_epoch = max(1.0, n_train / batch)
        meta["runtime"][fold["name"]] = {
            "steps": round(epochs * steps_per_epoch), "batch": batch,
            "train_images": n_train, "crops_seen": epochs * n_train,
            "steps_per_epoch": round(steps_per_epoch, 2), "epochs": epochs,
            "s_per_step_median": round(wall / max(epochs * steps_per_epoch, 1), 4),
            "s_per_step_mean": round(wall / max(epochs * steps_per_epoch, 1), 4),
            "s_per_step_p95": 0.0, "s_first_step": 0.0,
            "s_per_epoch": round(wall / epochs, 2),
            "crops_per_s": round(epochs * n_train / wall, 2) if wall else 0.0,
            "train_wall_s": round(wall, 1),
        }
        print(f"    trained in {wall / 60:.1f} min", flush=True)

        thr = pick_threshold(model, records, fold["train"], segment, imgsz)
        meta["thresholds"][fold["name"]] = thr
        print(f"    confidence threshold chosen on training partition: {thr:.2f}")

        for n, (items, scores) in predict(model, records, fold["test"], thr,
                                          segment, imgsz).items():
            if segment:
                all_dets += predio.encode(items, scores, records[n].image_id)
            else:
                all_dets += predio.encode_boxes(items, scores, records[n].image_id)
            print(f"    {n}: {len(items)} predicted / "
                  f"{len(records[n].polys)} annotated")
        del model

    name = method if not only else f"{method}_partial"
    out = PREDICTIONS / protocol / f"{name}.json"
    predio.save(out, all_dets, meta)
    print(f"\nwrote {len(all_dets)} detections to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="loso", choices=["loso", "random"])
    ap.add_argument("--iters", type=int, default=1500,
                    help="optimiser steps, converted to epochs to match the "
                         "gradient budget of the other methods")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--weights", default="yolov8s-seg.pt",
                    help="'-seg' weights predict masks and join the mask "
                         "comparison; plain weights predict boxes only")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--mosaic", type=float, default=0.0,
                    help="Ultralytics default is 1.0; off here because it "
                         "roughly halves apparent object size")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", nargs="*", default=None)
    a = ap.parse_args()
    run(a.protocol, a.iters, a.batch, a.weights, a.imgsz, a.mosaic,
        a.seed, a.folds)
