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
