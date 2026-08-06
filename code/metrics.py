"""Segmentation, counting and physical-quantification metrics.

Detection quality is reported with COCO average precision; instance-level
agreement with the Aggregated Jaccard Index of Kumar et al. and with Panoptic
Quality; and the quantities the laboratory actually uses -- bead count, areal
density and the size distribution in micrometres -- separately, because a method
can score well on AP and still miscount.

Merge and split are reported explicitly because they are the failure modes that
matter here: two touching beads reported as one is a counting error and a size
error at the same time, and 8.8% of annotated bead pixels lie in a region claimed
by more than one polygon.

Instances are held sparsely, as a bounding box plus a mask covering only that
box. A micrograph is 1024x736, and the classical baseline can emit over a
thousand objects for one image; holding each as a full frame made parameter
tuning roughly forty times slower than it needed to be, with almost every byte
touched being a zero outside the object.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np


# --- sparse instance representation ----------------------------------------


class Instance:
    """One object: its bounding box, its mask within that box, and its area."""

    __slots__ = ("y0", "y1", "x0", "x1", "mask", "area", "shape")

    def __init__(self, y0: int, y1: int, x0: int, x1: int,
                 mask: np.ndarray, shape: tuple[int, int]):
        self.y0, self.y1, self.x0, self.x1 = int(y0), int(y1), int(x0), int(x1)
        self.mask = np.ascontiguousarray(mask, dtype=bool)
        self.area = int(self.mask.sum())
        self.shape = shape

    @classmethod
    def from_full(cls, mask: np.ndarray) -> "Instance | None":
        m = np.asarray(mask, dtype=bool)
        rows, cols = m.any(1), m.any(0)
        if not rows.any():
            return None
        y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
        return cls(y0, y1, x0, x1, m[y0:y1, x0:x1], m.shape)

    def full(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=bool)
        out[self.y0:self.y1, self.x0:self.x1] = self.mask
        return out


def as_instances(masks) -> list[Instance]:
    """Accept full-frame masks or ``Instance`` objects; always return the latter."""
    out = []
    for m in masks:
        if isinstance(m, Instance):
            out.append(m)
        else:
            inst = Instance.from_full(m)
            if inst is not None:
                out.append(inst)
    return out


# --- pairwise geometry -----------------------------------------------------


def pairwise_intersection(gts: list[Instance], preds: list[Instance]) -> np.ndarray:
    """Intersection areas, computed only for pairs whose boxes overlap."""
    G, P = len(gts), len(preds)
    inter = np.zeros((G, P), dtype=np.float64)
    if G == 0 or P == 0:
        return inter
    gb = np.array([[i.y0, i.y1, i.x0, i.x1] for i in gts])
    pb = np.array([[i.y0, i.y1, i.x0, i.x1] for i in preds])
    overlap = (
        (gb[:, 0][:, None] < pb[:, 1][None, :]) & (gb[:, 1][:, None] > pb[:, 0][None, :]) &
        (gb[:, 2][:, None] < pb[:, 3][None, :]) & (gb[:, 3][:, None] > pb[:, 2][None, :])
    )
    for i, j in zip(*np.nonzero(overlap)):
        g, p = gts[i], preds[j]
        y0, y1 = max(g.y0, p.y0), min(g.y1, p.y1)
        x0, x1 = max(g.x0, p.x0), min(g.x1, p.x1)
        inter[i, j] = np.logical_and(
            g.mask[y0 - g.y0:y1 - g.y0, x0 - g.x0:x1 - g.x0],
            p.mask[y0 - p.y0:y1 - p.y0, x0 - p.x0:x1 - p.x0],
        ).sum()
    return inter


def iou_matrix(gts, preds):
    gts, preds = as_instances(gts), as_instances(preds)
    inter = pairwise_intersection(gts, preds)
    ga = np.array([i.area for i in gts], dtype=np.float64)
    pa = np.array([i.area for i in preds], dtype=np.float64)
    if len(gts) == 0 or len(preds) == 0:
        return inter, inter.copy(), ga, pa
    union = ga[:, None] + pa[None, :] - inter
    return inter, inter / np.maximum(union, 1.0), ga, pa


# --- instance-level agreement ----------------------------------------------


def aji(gts, preds) -> float:
    """Aggregated Jaccard Index (Kumar et al., 2017)."""
    gts, preds = as_instances(gts), as_instances(preds)
    if len(gts) == 0 and len(preds) == 0:
        return 1.0
    if len(gts) == 0 or len(preds) == 0:
        return 0.0
    inter, iou, ga, pa = iou_matrix(gts, preds)
    union = ga[:, None] + pa[None, :] - inter
    c = u = 0.0
    used: set[int] = set()
    for i in range(len(gts)):
        j = int(iou[i].argmax())
        if iou[i, j] > 0:
            c += inter[i, j]
            u += union[i, j]
            used.add(j)
        else:
            u += ga[i]
    u += sum(pa[j] for j in range(len(preds)) if j not in used)
    return float(c / u) if u > 0 else 0.0


def panoptic_quality(gts, preds, thr: float = 0.5) -> dict:
    """PQ, and its segmentation and recognition factors (Kirillov et al., 2019).

    A ground-truth and a predicted instance can exceed IoU 0.5 with at most one
    partner, so the matching is unambiguous and needs no assignment solver.
    """
    gts, preds = as_instances(gts), as_instances(preds)
    if len(gts) == 0 and len(preds) == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    _, iou, _, _ = iou_matrix(gts, preds)
    tp_ious = [iou[i, j] for i, j in zip(*np.nonzero(iou > thr))] if iou.size else []
    tp = len(tp_ious)
    fp, fn = len(preds) - tp, len(gts) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"pq": float(sum(tp_ious) / denom) if denom else 0.0,
            "sq": float(np.mean(tp_ious)) if tp else 0.0,
            "rq": float(tp / denom) if denom else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def merge_split(gts, preds, thr: float = 0.5) -> dict:
    """Fraction of ground-truth beads lost to a merge or broken by a split.

    A *merge* is a prediction that contains at least half of each of two or more
    ground-truth beads; every ground-truth bead it absorbed counts as merged.
    A *split* is a ground-truth bead containing at least half of each of two or
    more predictions.
    """
    gts, preds = as_instances(gts), as_instances(preds)
    n_gt = len(gts)
    if n_gt == 0 or len(preds) == 0:
        return {"merge_rate": 0.0, "split_rate": 0.0, "n_merged": 0, "n_split": 0}

    inter, _, ga, pa = iou_matrix(gts, preds)
    cov_gt = inter / np.maximum(ga[:, None], 1.0)   # share of each GT inside a pred
    cov_pr = inter / np.maximum(pa[None, :], 1.0)   # share of each pred inside a GT

    merged: set[int] = set()
    for j in range(len(preds)):
        absorbed = np.nonzero(cov_gt[:, j] >= thr)[0]
        if len(absorbed) >= 2:
            merged.update(absorbed.tolist())
    n_split = sum(1 for i in range(n_gt)
                  if np.count_nonzero(cov_pr[i] >= thr) >= 2)

    return {"merge_rate": 100.0 * len(merged) / n_gt,
            "split_rate": 100.0 * n_split / n_gt,
            "n_merged": len(merged), "n_split": n_split}


def f1_at_iou(gts, preds, thr: float = 0.5) -> float:
    gts, preds = as_instances(gts), as_instances(preds)
    if len(gts) == 0 or len(preds) == 0:
        return 0.0
    _, iou, _, _ = iou_matrix(gts, preds)
    tp = int(np.count_nonzero(iou > thr))
    precision, recall = tp / len(preds), tp / len(gts)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# --- counting and physical quantities --------------------------------------


def counting_error(n_gt: int, n_pred: int) -> float:
    """Signed percentage error; positive means the method over-counted."""
    return 0.0 if n_gt == 0 else 100.0 * (n_pred - n_gt) / n_gt


def equivalent_diameters(masks, um_per_px: float) -> np.ndarray:
    """Equivalent circular diameter in micrometres, from mask pixel area.

    Pixel area is used for both prediction and ground truth here so that the two
    distributions are derived identically. The polygon (shoelace) areas reported
    in Chapter 3 are the more accurate figure for the ground truth alone.
    """
    inst = as_instances(masks)
    if not inst:
        return np.zeros(0)
    areas = np.array([i.area for i in inst], dtype=np.float64)
    return 2.0 * np.sqrt(areas * um_per_px ** 2 / np.pi)


def areal_density(n: int, width: int, height: int, um_per_px: float) -> float:
    """Beads per square millimetre of imaged mat."""
    area_mm2 = (width * um_per_px) * (height * um_per_px) / 1e6
    return n / area_mm2 if area_mm2 > 0 else 0.0


# --- COCO average precision ------------------------------------------------


def coco_ap(coco_gt, detections: list[dict], img_ids: list[int],
            max_dets: int = 400) -> dict:
    """AP, AP50 and AP75 on masks, via pycocotools.

    ``max_dets`` is raised well above the COCO default of 100 because a single
    500x micrograph holds up to 152 beads.
    """
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"ap": 0.0, "ap50": 0.0, "ap75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes([dict(d) for d in detections])
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.params.imgIds = list(img_ids)
        ev.params.maxDets = [1, 100, max_dets]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"ap": float(ev.stats[0]), "ap50": float(ev.stats[1]),
            "ap75": float(ev.stats[2])}


def summarise(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
            "n": int(a.size)}
