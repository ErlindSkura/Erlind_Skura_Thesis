"""Magnification as an experimental factor: the image and the label together.

The supervisor asked for the dataset to be magnified by a factor of three using
bicubic interpolation, and the request has a clear rationale behind it. Under
COCO's own area bands 82.8% of the annotated particles are "small", and 97% are
at 500x; a detector initialised from COCO weights is being asked to work well
below the size regime those weights were learned in. Resampling does not add
information, but it does change the size regime the network operates in, and
whether that alone buys anything is a question the dataset can answer.

Two design points are worth stating because the obvious implementations of each
are wrong.

**The label is magnified with the image.** Annotations are polygons, so they are
scaled as vertices and rasterised afterwards, never rasterised at native
resolution and then resampled. Resampling a mask of a 20 px object and
re-thresholding it moves its boundary by a pixel or more, which on an object
that size is several percent of its area -- the same reason the augmentation
geometry in :mod:`data_io` transforms vertices rather than rasters.

**Predictions are mapped back to native resolution before they are scored.** The
alternative, scoring in the magnified frame against magnified ground truth, is
self-consistent but produces numbers that cannot be put in the same table as the
other five methods: the COCO size bands would move, and a per-micrograph count
would be compared against a different denominator. Mapping back costs one
resampling of each predicted mask and keeps the factor as the only thing that
changed.

Inference cannot be run on the whole magnified micrograph in one pass. At 3x a
frame is 3072x2208, and torchvision pastes every surviving detection into a
full-frame float mask before returning it; with a detection budget of 400 that
is more than ten gigabytes of masks alone, before any activation. The frame is
therefore tiled, with the tiles overlapping by more than the largest annotated
object so that every object is whole in at least one tile, and each detection
assigned to exactly one tile. See :func:`predict_tiled`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Torch is imported inside predict_tiled rather than here, so that the geometry
# above it -- which is what the tests exercise, and what a mistake would be
# silent in -- can be checked on a machine with no deep-learning stack, the same
# convention data_io.py follows.

# The factor the supervisor asked for. Kept as a named constant because it
# appears in the tile geometry as well as in the resampling.
FACTOR = 3

# Tile geometry, in magnified pixels. Both are multiples of FACTOR, which is what
# lets a tile be mapped back to native resolution by integer division: with the
# frame itself a multiple of FACTOR, every tile origin the grid produces is too.
#
# The overlap must exceed the largest object, or an object could be cut by every
# tile that sees it. The largest annotated particle spans 214 px natively, so 642
# magnified; 768 clears that with room to spare.
TILE = 1536
OVERLAP = 768

# Per-tile detection budget. The whole-frame budget is 400, set by the densest
# micrograph (Z6-1, 152 particles). A tile covers 512x512 native pixels, about a
# third of the frame area, so it cannot legitimately hold more than about sixty;
# 200 is generous and, unlike 400, bounds the mask-pasting memory to something a
# 16 GB card can hold.
TILE_DETECTIONS = 200


def magnify(img: Image.Image, polys, factor: int = FACTOR):
    """Bicubic-resample the micrograph and scale the annotation to match.

    Bicubic rather than bilinear or nearest because the request was for a smooth
    image: the particles are separated from the mat by roughly eighteen grey
    levels, and nearest-neighbour magnification would turn that gradient into a
    staircase of blocks whose edges are an artefact of the resampling rather than
    a property of the specimen.
    """
    w, h = img.size
    out = img.resize((w * factor, h * factor), Image.BICUBIC)
    return out, [np.asarray(p, dtype=np.float64) * float(factor) for p in polys]


def tile_grid(width: int, height: int, tile: int = TILE, overlap: int = OVERLAP,
              factor: int = FACTOR) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the frame, as ``(x0, y0, x1, y1)``."""
    if tile % factor or overlap % factor:
        raise ValueError("tile and overlap must be multiples of the factor, so "
                         "that a tile maps back to native resolution exactly")
    if overlap >= tile:
        raise ValueError("overlap must be smaller than the tile")
    xs = _origins(width, tile, tile - overlap, factor)
    ys = _origins(height, tile, tile - overlap, factor)
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in ys for x in xs]


def _origins(extent: int, tile: int, stride: int, factor: int) -> list[int]:
    if extent % factor:
        raise ValueError(f"frame extent {extent} is not a multiple of {factor}")
    if extent <= tile:
        return [0]
    out = list(range(0, extent - tile + 1, stride))
    # The stride rarely divides the frame, so the last tile is placed flush with
    # the far edge. Because the frame and the tile are both multiples of the
    # factor, so is this origin, and no part of the frame is left uncovered.
    if out[-1] != extent - tile:
        out.append(extent - tile)
    return out


def _owners(boxes: np.ndarray, grid, centres: np.ndarray) -> np.ndarray:
    """Assign each detection to exactly one tile.

    A detection is owned by the nearest-centred tile that contains it whole, so
    the tile that owns an object is one that saw all of it. An object clipped by
    the frame edge is contained by no tile; it falls back to the nearest-centred
    tile containing its own centre, which is still a single well-defined owner.
    Ownership by containment, rather than a non-maximum suppression pass over the
    union, keeps the merge rule independent of the scores.
    """
    g = np.asarray(grid, dtype=np.float64)
    bc = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                          (boxes[:, 1] + boxes[:, 3]) / 2.0])
    out = np.empty(len(boxes), dtype=int)
    for i, b in enumerate(boxes):
        cand = np.nonzero((g[:, 0] <= b[0]) & (g[:, 1] <= b[1]) &
                          (g[:, 2] >= b[2]) & (g[:, 3] >= b[3]))[0]
        if not len(cand):
            cand = np.nonzero((g[:, 0] <= bc[i, 0]) & (g[:, 2] > bc[i, 0]) &
                              (g[:, 1] <= bc[i, 1]) & (g[:, 3] > bc[i, 1]))[0]
        if not len(cand):
            cand = np.arange(len(g))
        out[i] = cand[int(np.argmin(((centres[cand] - bc[i]) ** 2).sum(1)))]
    return out


def predict_tiled(model, image, device, *, native_hw,
                  factor: int = FACTOR, tile: int = TILE, overlap: int = OVERLAP,
                  want_masks: bool = True, score_thresh: float = 0.05,
                  mask_thresh: float = 0.5):
    """Run a torchvision detector over a magnified frame, tile by tile.

    Returns predictions already reduced to native resolution: full-frame boolean
    masks when ``want_masks``, otherwise ``[x0, y0, x1, y1]`` boxes, together
    with their scores. Scores are the model's own and are not rescaled, so the
    threshold chosen on the training partition means the same thing here as it
    does for a whole-frame method.
    """
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    _, height, width = image.shape
    nh, nw = native_hw
    grid = tile_grid(width, height, tile, overlap, factor)
    centres = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]
                        for x0, y0, x1, y1 in grid], dtype=np.float64)

    out_items, out_scores = [], []
    with torch.no_grad():
        for index, (x0, y0, x1, y1) in enumerate(grid):
            sub = image[:, y0:y1, x0:x1].to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                pred = model([sub])[0]
            keep = pred["scores"] >= score_thresh
            if not bool(keep.any()):
                continue

            boxes = pred["boxes"][keep].float().cpu().numpy()
            scores = pred["scores"][keep].float().cpu().numpy()
            offset = np.array([x0, y0, x0, y0], dtype=np.float64)
            mine = np.nonzero(_owners(boxes + offset, grid, centres) == index)[0]
            if not len(mine):
                continue

            if not want_masks:
                for j in mine:
                    out_items.append((boxes[j] + offset) / factor)
                    out_scores.append(float(scores[j]))
                continue

            # Downsampled while still a probability map and thresholded
            # afterwards, which averages the soft mask over each native pixel.
            # Thresholding first and downsampling the binary mask would instead
            # let one magnified pixel decide a native one.
            picked = torch.as_tensor(mine, device=pred["masks"].device)
            soft = pred["masks"][keep][picked].float()
            soft = F.interpolate(soft, size=((y1 - y0) // factor,
                                             (x1 - x0) // factor),
                                 mode="bilinear", align_corners=False)
            planes = (soft[:, 0] > mask_thresh).cpu().numpy()
            oy, ox = y0 // factor, x0 // factor
            for plane, j in zip(planes, mine):
                if not plane.any():
                    # A soft mask can threshold away to nothing; such a detection
                    # has no area and cannot be matched, so it is dropped here to
                    # keep masks and scores aligned everywhere downstream.
                    continue
                canvas = np.zeros((nh, nw), dtype=bool)
                canvas[oy:oy + plane.shape[0], ox:ox + plane.shape[1]] = plane
                out_items.append(canvas)
                out_scores.append(float(scores[j]))

    return out_items, out_scores
