"""Storage format for predictions.

Every method -- Mask R-CNN, U-Net and the classical pipeline -- writes its
predictions in the same COCO results format, with masks run-length encoded.
One format for all three means the evaluation code cannot accidentally treat
them differently, which is the point of the controlled comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def encode(masks, scores, image_id: int) -> list[dict]:
    """Accepts full-frame masks or sparse ``metrics.Instance`` objects."""
    out = []
    for m, s in zip(masks, scores):
        if hasattr(m, "full"):      # metrics.Instance
            m = m.full()
        rle = mask_utils.encode(np.asfortranarray(np.asarray(m, dtype=np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        out.append({"image_id": int(image_id), "category_id": 1,
                    "segmentation": rle, "score": float(s)})
    return out


def encode_boxes(boxes, scores, image_id: int) -> list[dict]:
    """Box-only detections, in COCO's ``[x, y, width, height]`` convention.

    Written without a ``segmentation`` field rather than with a filled rectangle
    standing in for one. A filled box is not a mask of the object: a roughly
    elliptical particle fills about pi/4 of its bounding box, so a rectangle would
    report an inflated area and would score against the mask metrics as though the
    method had segmented something it never predicted. The absence of the field is
    what lets the evaluator tell a box-only method apart and score it accordingly.
    """
    out = []
    for b, s in zip(boxes, scores):
        x0, y0, x1, y1 = (float(v) for v in b)
        out.append({"image_id": int(image_id), "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
    return out


def is_box_only(detections: list[dict]) -> bool:
    """True when no detection carries a mask, so mask metrics do not apply."""
    return bool(detections) and not any("segmentation" in d for d in detections)


def to_boxes(detections: list[dict]) -> tuple[np.ndarray, list[float]]:
    """``[x0, y0, x1, y1]`` corners and scores, from either storage form."""
    boxes, scores = [], []
    for d in detections:
        if "bbox" in d:
            x, y, w, h = d["bbox"]
            boxes.append([x, y, x + w, y + h])
        else:
            rle = dict(d["segmentation"])
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
            x, y, w, h = mask_utils.toBbox(rle).tolist()
            boxes.append([x, y, x + w, y + h])
        scores.append(float(d.get("score", 1.0)))
    return np.array(boxes, dtype=np.float64).reshape(-1, 4), scores


def save(path: Path, detections: list[dict], meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "detections": detections}))


def load(path: Path) -> tuple[list[dict], dict]:
    blob = json.loads(Path(path).read_text())
    return blob["detections"], blob.get("meta", {})


def to_masks(detections: list[dict]) -> tuple[list[np.ndarray], list[float]]:
    masks, scores = [], []
    for d in detections:
        rle = dict(d["segmentation"])
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        masks.append(mask_utils.decode(rle).astype(bool))
        scores.append(float(d.get("score", 1.0)))
    return masks, scores


def group_by_image(detections: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for d in detections:
        out.setdefault(int(d["image_id"]), []).append(d)
    return out
