"""Checks for the geometry that the pipeline depends on and cannot see go wrong.

These cover the conversions where an error is silent: it produces plausible
numbers rather than an exception, and would reach the thesis as a result. They
run on the real annotations, not on synthetic shapes, because the properties
that break these conversions -- vertices on exact pixel boundaries, overlapping
polygons, objects a dozen pixels across -- are properties of this dataset.

    python tests.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import folds as folds_mod
from data_io import load_records, rasterise
from metrics import (
    as_instances, box_iou_matrix, boxes_from_instances, f1_at_iou_boxes,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --- box geometry ----------------------------------------------------------


def test_box_iou() -> None:
    print("box geometry")
    a = np.array([[0, 0, 10, 10]], float)
    check("identical boxes give IoU 1", box_iou_matrix(a, a)[0, 0] == 1.0)

    # inter 50, union 150
    b = np.array([[5, 0, 15, 10]], float)
    check("half-overlapping boxes give IoU 1/3",
          abs(box_iou_matrix(a, b)[0, 0] - 1 / 3) < 1e-9,
          f"{box_iou_matrix(a, b)[0, 0]:.6f}")

    c = np.array([[20, 20, 30, 30]], float)
    check("disjoint boxes give IoU 0", box_iou_matrix(a, c)[0, 0] == 0.0)

    empty = np.zeros((0, 4))
    check("empty input gives an empty matrix, not an error",
          box_iou_matrix(empty, a).shape == (0, 1))
    check("F1 against no prediction is 0", f1_at_iou_boxes(a, empty) == 0.0)
    check("F1 of a perfect prediction is 1", f1_at_iou_boxes(a, a) == 1.0)


def test_boxes_from_masks() -> None:
    print("box extraction from masks")
    m = np.zeros((20, 20), bool)
    m[3:9, 4:11] = True
    got = boxes_from_instances(as_instances([m]))[0]
    check("bounding box matches the mask extent",
          list(got) == [4, 3, 11, 9], f"got {list(got)}")


# --- YOLO label export -----------------------------------------------------


def test_yolo_labels(records) -> None:
    """The contract is coordinate fidelity.

    Not pixel-identical re-rasterisation: normalising a vertex that sits on an
    exact integer and multiplying back cannot reproduce it bit-for-bit, and such a
    vertex then falls to the other side of the pixel boundary. That is a property
    of comparing through a quantising rasteriser, not of the labels, and it does
    not converge as precision is raised. See ``train_yolo.write_labels``.
    """
    import train_yolo as ty

    print("YOLO label export")
    worst_coord, worst_where = 0.0, ""
    counts_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "fold"
        for fold in folds_mod.load("loso"):
            ty.export_fold(records, fold, root, segment=True)
            for split, names in (("train", fold["train"]), ("val", fold["test"])):
                for n in names:
                    rec = records[n]
                    lines = (root / "labels" / split / f"{n}.txt").read_text().splitlines()
                    if len(lines) != len(rec.polys):
                        counts_ok = False
                    for line, src in zip(lines, rec.polys):
                        v = np.array([float(x) for x in line.split()[1:]]).reshape(-1, 2)
                        back = np.column_stack([v[:, 0] * rec.width,
                                                v[:, 1] * rec.height])
                        if back.shape != src.shape:
                            counts_ok = False
                            continue
                        err = float(np.abs(back - src).max())
                        if err > worst_coord:
                            worst_coord, worst_where = err, n

    check("every polygon becomes exactly one label line", counts_ok)
    check("vertex coordinates survive the round-trip to sub-micropixel",
          worst_coord < 1e-6, f"worst {worst_coord:.2e} px on {worst_where}")


def test_yolo_boxes_match_polygons(records) -> None:
    import train_yolo as ty

    print("YOLO detection labels")
    rec = records["Z6-4"]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "l.txt"
        ty.write_labels(rec, p, segment=False)
        lines = p.read_text().splitlines()
    worst = 0.0
    for line, src in zip(lines, rec.polys):
        _, cx, cy, w, h = (float(v) for v in line.split())
        want_w = (src[:, 0].max() - src[:, 0].min()) / rec.width
        want_h = (src[:, 1].max() - src[:, 1].min()) / rec.height
        want_cx = (src[:, 0].max() + src[:, 0].min()) / 2 / rec.width
        want_cy = (src[:, 1].max() + src[:, 1].min()) / 2 / rec.height
        worst = max(worst, abs(w - want_w), abs(h - want_h),
                    abs(cx - want_cx), abs(cy - want_cy))
    check("box centre and extent match the polygon they came from",
          worst < 1e-9, f"worst {worst:.2e}")


# --- annotation invariants the code relies on ------------------------------


def test_annotation_invariants(records) -> None:
    print("annotation invariants")
    total = sum(len(r.polys) for r in records.values())
    check("606 annotated particles across 11 micrographs", total == 606,
          f"got {total}")

    below = inside = 0
    for r in records.values():
        for p in r.polys:
            inside += 1
            if p[:, 1].max() <= r.height:
                below += 1
    check("no vertex falls below the banner crop", below == inside,
          f"{below}/{inside}")

    # The small-object figure Chapter 2 states, and the AP band table reports
    # against. It must be measured the way pycocotools bands objects -- from the
    # "area" field of the ground-truth file, which prepare_data fills with the
    # shoelace polygon area. Measuring it from rasterised masks instead gives
    # 81.7%, close enough to look right and wrong for the purpose, since it is not
    # the quantity that decides which band an object is scored in.
    import json

    from config import COCO_GT

    areas = np.array([a["area"] for a in
                      json.loads(COCO_GT.read_text())["annotations"]])
    frac = float((areas < 32 * 32).mean())
    check("82.8% of particles are COCO-small, as Chapter 2 states",
          abs(frac - 0.828) < 0.001, f"{100 * frac:.1f}% below 32x32 px")
    check("1.2% are COCO-large",
          abs(float((areas >= 96 * 96).mean()) - 0.012) < 0.001,
          f"{100 * (areas >= 96 * 96).mean():.1f}%")


def test_preprocessing(records) -> None:
    """The invariants a preprocessing variant must not break.

    A variant that silently changed image size or channel count would produce a
    model that trains and scores plausibly while being fed something different
    from what the evaluation assumes.
    """
    import preprocess as pp
    from PIL import Image

    print("preprocessing")
    rec = records["Z6-4"]
    src = Image.open(rec.path).convert("RGB")

    check("'none' returns the image untouched",
          pp.apply(src, "none") is src)

    ok_shape = ok_range = True
    for v in pp.VARIANTS:
        out = pp.apply(src, v)
        if out.size != src.size or out.mode != "RGB":
            ok_shape = False
        a = np.asarray(out)
        if a.dtype != np.uint8 or a.min() < 0 or a.max() > 255:
            ok_range = False
    check("every variant preserves size and RGB mode", ok_shape)
    check("every variant stays in 8-bit range", ok_range)

    a = np.asarray(pp.apply(src, "clahe"))
    b = np.asarray(pp.apply(src, "clahe"))
    check("variants are deterministic", np.array_equal(a, b))

    check("an unknown variant is rejected rather than ignored",
          _raises(lambda: pp.apply(src, "sharpen")))

    # The background estimator must span the largest annotated object (214 px),
    # or a genuine particle is flattened away as if it were illumination.
    check("background footprint exceeds the largest particle",
          pp.BACKGROUND_FOOTPRINT > 214, f"{pp.BACKGROUND_FOOTPRINT} px vs 214 px")


# --- magnification factor --------------------------------------------------


def test_magnification(records) -> None:
    """The 3x factor: image and label must stay in the same coordinate frame.

    The failure this guards against is silent. Scaling the image but not the
    polygons, or scaling them by a rounded factor, produces a perfectly
    well-formed dataset whose labels sit next to the objects instead of on them,
    and training on it yields a plausible, worthless model.
    """
    print("magnification factor")
    import upsample as up
    from PIL import Image

    rec = records["Z2-3"]
    img = Image.open(rec.path).convert("RGB")
    big, polys = up.magnify(img, [p.copy() for p in rec.polys], up.FACTOR)

    check("the image grows by exactly the factor",
          big.size == (img.size[0] * up.FACTOR, img.size[1] * up.FACTOR),
          f"{img.size} -> {big.size}")

    # Checked against the exact consequences of scaling rather than against the
    # native mask upsampled, which is itself lossy: nearest-neighbour growth of
    # an already-quantised boundary disagrees with the true magnified polygon by
    # a pixel all the way round, so an IoU between the two measures the reference
    # and not the code. Area and centroid have exact expected values instead.
    native = rasterise(rec.polys, rec.width, rec.height)
    scaled = rasterise(polys, rec.width * up.FACTOR, rec.height * up.FACTOR)

    # The area tolerance is not a round number chosen to pass. Rasterising at
    # native resolution decides a whole ring of boundary pixels by rounding, and
    # the magnified rasterisation decides the same ring nine times more finely,
    # so the two areas may legitimately differ by about the fraction of the
    # particle that ring occupies. That fraction is measured per particle and
    # used as the bound. A genuine scaling error would miss by a factor, not by
    # a boundary.
    from scipy import ndimage

    worst_ratio, worst_centroid = 0.0, 0.0
    for a, b in zip(native, scaled):
        area_a, area_b = float(a.sum()), float(b.sum())
        ring = float(((a > 0) & ~ndimage.binary_erosion(a > 0)).sum())
        deviation = abs(area_b / (area_a * up.FACTOR ** 2) - 1.0)
        worst_ratio = max(worst_ratio, deviation / (ring / area_a))
        ys, xs = np.nonzero(a)
        yb, xb = np.nonzero(b)
        worst_centroid = max(worst_centroid,
                             np.hypot(xb.mean() - xs.mean() * up.FACTOR,
                                      yb.mean() - ys.mean() * up.FACTOR))

    check("a magnified particle has the factor squared times its area",
          worst_ratio < 1.0,
          f"worst deviation is {worst_ratio:.2f} of its own boundary ring")
    check("a magnified particle sits where the factor puts it",
          worst_centroid < up.FACTOR,
          f"worst centroid shift {worst_centroid:.2f} magnified px "
          f"over {len(native)} particles")

    check("magnification preserves the particle count",
          len(polys) == len(rec.polys), f"{len(rec.polys)} particles")


def test_tile_grid() -> None:
    """Tiling must cover the frame and must not cut an object in every tile."""
    print("tiled inference geometry")
    import upsample as up
    from config import FULL_W, WORK_H

    width, height = FULL_W * up.FACTOR, WORK_H * up.FACTOR
    grid = up.tile_grid(width, height)

    covered = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in grid:
        covered[y0:y1, x0:x1] = True
    check("the tiles cover every pixel of the frame", bool(covered.all()),
          f"{len(grid)} tiles over {width}x{height}")

    check("every tile origin is a multiple of the factor",
          all(x0 % up.FACTOR == 0 and y0 % up.FACTOR == 0
              for x0, y0, _, _ in grid))

    # 214 px is the largest annotated particle; magnified it is 642, and the
    # overlap has to exceed that or an object could straddle every boundary.
    check("the overlap exceeds the largest magnified particle",
          up.OVERLAP > 214 * up.FACTOR,
          f"{up.OVERLAP} px vs {214 * up.FACTOR} px")

    check("an unusable tile geometry is rejected rather than silently used",
          _raises(lambda: up.tile_grid(width, height, tile=1000, overlap=768))
          and _raises(lambda: up.tile_grid(width, height, tile=768, overlap=768)))


def test_tile_ownership() -> None:
    """Each detection is claimed by exactly one tile, and by one that saw it."""
    print("tile ownership")
    import upsample as up

    grid = up.tile_grid(3072, 2208)
    centres = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]
                        for x0, y0, x1, y1 in grid], dtype=np.float64)

    rng = np.random.default_rng(0)
    x0 = rng.uniform(0, 3072 - 120, 600)
    y0 = rng.uniform(0, 2208 - 120, 600)
    size = rng.uniform(20, 120, 600)
    boxes = np.column_stack([x0, y0, x0 + size, y0 + size])

    owners = up._owners(boxes, grid, centres)
    check("every detection gets exactly one owning tile",
          owners.shape == (len(boxes),) and owners.min() >= 0
          and owners.max() < len(grid))

    g = np.asarray(grid, dtype=np.float64)
    whole = 0
    for b, o in zip(boxes, owners):
        t = g[o]
        if t[0] <= b[0] and t[1] <= b[1] and t[2] >= b[2] and t[3] >= b[3]:
            whole += 1
    check("the owning tile contains the whole detection", whole == len(boxes),
          f"{whole}/{len(boxes)} boxes whole inside their owner")

    # An object at the frame edge is contained by no tile in one axis; it must
    # still get a single owner rather than being dropped or double-counted.
    edge = np.array([[3060.0, 2196.0, 3080.0, 2216.0]])
    check("a detection clipped by the frame still gets one owner",
          up._owners(edge, grid, centres).shape == (1,))


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def main() -> int:
    records = load_records()
    test_box_iou()
    test_boxes_from_masks()
    test_yolo_labels(records)
    test_yolo_boxes_match_polygons(records)
    test_annotation_invariants(records)
    test_preprocessing(records)
    test_magnification(records)
    test_tile_grid()
    test_tile_ownership()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
