"""Classical Otsu-plus-watershed baseline, representing current manual practice.

Beads are *darker* than the surrounding fibre mat: measured over the annotated
regions, the mean grey level inside a bead is 98 against 116 outside on Z2-1,
115 against 135 on Z5-2 and 81 against 104 on Z6-4. The pipeline therefore
thresholds for dark objects. That the separation is only about 20 grey levels,
while the foreground fraction ranges from 2% to 20% between micrographs, is the
reason this baseline is expected to struggle -- and is why it belongs in the
comparison rather than being assumed inadequate.

The three free parameters are tuned by grid search on each fold's *training*
partition and then held fixed for that fold's test specimen, so the classical
method is subject to exactly the same protocol as the learned ones.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

import folds as folds_mod
import predio
from config import PREDICTIONS, ensure_dirs
from data_io import load_records, rasterise
from metrics import Instance, aji, as_instances

CLAHE_CLIP = 0.02
OPENING_RADII = (1, 2)
MIN_AREA_UM2 = (2.0, 10.0, 30.0)
MIN_DISTANCE_PX = (5, 10, 20)


def _grey(record) -> np.ndarray:
    return np.asarray(Image.open(record.path).convert("L"), dtype=np.float64) / 255.0


def _equalised(record, cache: dict) -> np.ndarray:
    """CLAHE is the slowest step and does not depend on the tuned parameters."""
    if record.name not in cache:
        cache[record.name] = exposure.equalize_adapthist(_grey(record),
                                                         clip_limit=CLAHE_CLIP)
    return cache[record.name]


def segment(record, cache: dict, opening_radius: int, min_area_um2: float,
            min_distance: int):
    img = _equalised(record, cache)

    # Dark objects: everything below the Otsu level is candidate foreground.
    binary = img < filters.threshold_otsu(img)
    if opening_radius > 0:
        binary = morphology.opening(binary, morphology.disk(opening_radius))
    binary = morphology.closing(binary, morphology.disk(1))
    binary = ndi.binary_fill_holes(binary)

    # Small-object removal by component size, written out rather than taken from
    # skimage, whose parameter for this has changed name across versions.
    min_area_px = max(1, int(round(min_area_um2 / record.um_per_px ** 2)))
    comp = measure.label(binary, connectivity=1)
    counts = np.bincount(comp.ravel())
    big = counts >= min_area_px
    big[0] = False
    binary = big[comp]
    if not binary.any():
        return [], []

    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    markers = np.zeros(distance.shape, dtype=np.int32)
    if len(coords):
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels = segmentation.watershed(-distance, markers, mask=binary)

    background = float(img[~binary].mean()) if (~binary).any() else 1.0
    instances, scores = [], []
    for r in measure.regionprops(labels):
        if r.area < min_area_px:
            continue
        rows, cols = r.slice
        # A monotone confidence proxy: how much darker the object is than the
        # background. Without it every object would carry the same score and
        # average precision would collapse to a single operating point, which
        # would understate this baseline for a reason unrelated to its quality.
        depth = background - float(img[r.slice][r.image].mean())
        instances.append(Instance(rows.start, rows.stop, cols.start, cols.stop,
                                  r.image, img.shape))
        scores.append(float(np.clip(depth / 0.2, 0.01, 1.0)))
    return instances, scores


def tune(records, train_names, cache):
    # Converted to sparse instances once, not on every one of the 144 scorings.
    gt = {n: as_instances(rasterise(records[n].polys, records[n].width,
                                    records[n].height))
          for n in train_names}
    best, best_score = None, -1.0
    for radius, min_area, min_dist in itertools.product(
            OPENING_RADII, MIN_AREA_UM2, MIN_DISTANCE_PX):
        scores = []
        for n in train_names:
            masks, _ = segment(records[n], cache, radius, min_area, min_dist)
            scores.append(aji(gt[n], masks))
        mean_aji = float(np.mean(scores))
        if mean_aji > best_score:
            best_score, best = mean_aji, (radius, min_area, min_dist)
    return best, best_score


def run(protocol: str) -> None:
    ensure_dirs()
    records = load_records()
    cache: dict[str, np.ndarray] = {}
    all_dets, meta = [], {"protocol": protocol, "method": "classical", "params": {}}

    for fold in folds_mod.load(protocol):
        t0 = time.time()
        (radius, min_area, min_dist), train_aji = tune(records, fold["train"], cache)
        meta["params"][fold["name"]] = {"opening_radius": radius,
                                        "min_area_um2": min_area,
                                        "min_distance_px": min_dist}
        print(f"[fold {fold['name']}] tuned on training partition: "
              f"radius={radius} min_area={min_area}um2 min_dist={min_dist} "
              f"(train AJI {train_aji:.3f}, {time.time() - t0:.0f}s)", flush=True)

        for n in fold["test"]:
            masks, scores = segment(records[n], cache, radius, min_area, min_dist)
            all_dets += predio.encode(masks, scores, records[n].image_id)
            print(f"    {n}: {len(masks)} predicted / {len(records[n].polys)} annotated")

    out = PREDICTIONS / protocol / "classical.json"
    predio.save(out, all_dets, meta)
    print(f"\nwrote {len(all_dets)} detections to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="loso", choices=["loso", "random"])
    a = ap.parse_args()
    run(a.protocol)
