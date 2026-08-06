"""Annotation loading, rasterisation and augmentation geometry.

Deliberately free of any deep-learning dependency, so that the classical
baseline and the evaluation code -- neither of which trains anything -- can run
without PyTorch, and so that the geometry below can be tested on its own.

Augmentation is applied to the polygon vertices and to the image together, and
instance masks are rasterised only afterwards. Transforming vertices rather than
pre-rendered masks keeps overlapping beads separable: 8.8% of annotated bead
pixels belong to more than one polygon, and a pipeline that rasterises first
loses that permanently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import COCO_GT, PREPARED_IMAGES


# --- records ---------------------------------------------------------------


@dataclass
class Record:
    name: str
    path: Path
    width: int
    height: int
    specimen: str
    magnification: int
    um_per_px: float
    image_id: int
    polys: list[np.ndarray] = field(default_factory=list)


def load_records(coco_path: Path = COCO_GT) -> dict[str, Record]:
    coco = json.loads(Path(coco_path).read_text())
    recs: dict[str, Record] = {}
    by_id: dict[int, Record] = {}
    for im in coco["images"]:
        r = Record(
            name=im["name"], path=PREPARED_IMAGES / im["file_name"],
            width=im["width"], height=im["height"], specimen=im["specimen"],
            magnification=im["magnification"], um_per_px=im["um_per_px"],
            image_id=im["id"],
        )
        recs[im["name"]] = r
        by_id[im["id"]] = r
    for a in coco["annotations"]:
        flat = a["segmentation"][0]
        by_id[a["image_id"]].polys.append(
            np.asarray(flat, dtype=np.float64).reshape(-1, 2)
        )
    return recs


def rasterise(polys, width: int, height: int) -> np.ndarray:
    """Render polygons to a (N, H, W) uint8 stack, one plane per instance."""
    polys = list(polys)
    if not polys:
        return np.zeros((0, height, width), dtype=np.uint8)
    out = np.zeros((len(polys), height, width), dtype=np.uint8)
    for i, p in enumerate(polys):
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon([tuple(v) for v in p], fill=1)
        out[i] = np.asarray(canvas, dtype=np.uint8)
    return out


# --- geometric transforms on vertices --------------------------------------

# Polygon vertices are continuous coordinates, not pixel indices, so a
# reflection is x -> W - x and not the x -> W - 1 - x that would be correct for
# an integer index. The difference is half a pixel at each edge: negligible for
# a large object, severe here, because the mean bead is 17 px across and a
# one-pixel boundary shift moves roughly 13% of its area. Verified against a
# directly flipped raster mask, which agrees to IoU 0.9995 under this convention
# and only 0.87 under the other.


def scale(img: Image.Image, polys, s: float):
    w, h = img.size
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.BILINEAR)
    fx, fy = nw / w, nh / h
    return img, [p * np.array([fx, fy]) for p in polys]


def flip_h(img: Image.Image, polys):
    w = img.size[0]
    return (img.transpose(Image.FLIP_LEFT_RIGHT),
            [np.column_stack([w - p[:, 0], p[:, 1]]) for p in polys])


def flip_v(img: Image.Image, polys):
    h = img.size[1]
    return (img.transpose(Image.FLIP_TOP_BOTTOM),
            [np.column_stack([p[:, 0], h - p[:, 1]]) for p in polys])


def rot90(img: Image.Image, polys):
    """One 90-degree counter-clockwise rotation; (x, y) -> (y, W - x)."""
    w = img.size[0]
    return (img.transpose(Image.ROTATE_90),
            [np.column_stack([p[:, 1], w - p[:, 0]]) for p in polys])


def crop(img: Image.Image, polys, ox: int, oy: int, size: int):
    out = img.crop((ox, oy, ox + size, oy + size))
    if out.size != (size, size):   # crop ran past the edge; pad with black
        padded = Image.new("RGB", (size, size), (0, 0, 0))
        padded.paste(out, (0, 0))
        out = padded
    return out, [p - np.array([ox, oy]) for p in polys]


def photometric(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Brightness and contrast jitter, modelling detector-gain variation."""
    a = np.asarray(img, dtype=np.float32)
    a = np.clip((a - 128.0) * rng.uniform(0.85, 1.15) + 128.0
                + rng.uniform(-20.0, 20.0), 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def augment(img: Image.Image, polys, rng: np.random.Generator, *,
            crop_size: int, scale_range: tuple[float, float]):
    """Scale, flip, rotate, crop and jitter, keeping vertices in register.

    Elastic and shear deformations are deliberately excluded: they would distort
    bead shape, and bead shape is a measured output of this work rather than an
    incidental property.
    """
    img, polys = scale(img, polys, rng.uniform(*scale_range))
    if rng.random() < 0.5:
        img, polys = flip_h(img, polys)
    if rng.random() < 0.5:
        img, polys = flip_v(img, polys)
    for _ in range(int(rng.integers(4))):
        img, polys = rot90(img, polys)

    w, h = img.size
    ox = int(rng.integers(0, max(1, w - crop_size + 1)))
    oy = int(rng.integers(0, max(1, h - crop_size + 1)))
    img, polys = crop(img, polys, ox, oy, crop_size)
    img = photometric(img, rng)

    keep = [p for p in polys
            if p[:, 0].max() > 0 and p[:, 1].max() > 0
            and p[:, 0].min() < crop_size and p[:, 1].min() < crop_size]
    return img, keep
