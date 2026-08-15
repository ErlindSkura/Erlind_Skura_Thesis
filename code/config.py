"""Shared configuration for the bead segmentation pipeline.

Every path can be overridden through an environment variable so that the same
code runs unchanged on a local Windows machine and on Google Colab:

    BEAD_DATA     the ``Segmentations`` directory holding ``Images`` and ``Labels``
    BEAD_WORK     scratch directory for prepared images, folds and predictions
    BEAD_RESULTS  directory for metric files and generated LaTeX tables
    BEAD_SCRATCH  throwaway directory for data no run ever needs to read back
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- dataset facts ---------------------------------------------------------

# The 11 original micrographs, keyed by specimen and magnification. The suffix
# encodes magnification, not a repeat: -1 is 500x, -2 is 1kx, -3/-4 is 3kx.
IMAGES: dict[str, tuple[str, int]] = {
    "Z2-1": ("Z2", 500), "Z2-2": ("Z2", 1000), "Z2-3": ("Z2", 3000),
    "Z4-2": ("Z4", 1000), "Z4-3": ("Z4", 3000),
    "Z5-1": ("Z5", 500), "Z5-2": ("Z5", 1000), "Z5-3": ("Z5", 3000),
    "Z6-1": ("Z6", 500), "Z6-2": ("Z6", 1000), "Z6-4": ("Z6", 3000),
}

SPECIMENS: tuple[str, ...] = ("Z2", "Z4", "Z5", "Z6")
MAGNIFICATIONS: tuple[int, ...] = (500, 1000, 3000)

# Calibrated from the burned-in scale bars: the field width is 284.4 um at
# 1000x and scales inversely with magnification, over 1024 px.
NM_PER_PX: dict[int, float] = {500: 555.6, 1000: 277.8, 3000: 92.6}

BANNER_H = 32           # burned-in Zeiss info bar at the bottom of every image
FULL_W, FULL_H = 1024, 768
WORK_H = FULL_H - BANNER_H   # 736, the height after the banner is removed

# --- training hyperparameters ----------------------------------------------

CROP = 512                    # random crop size used during training
SCALE_RANGE = (0.5, 2.0)      # deliberately wide: scale robustness is the
                              # central difficulty of this dataset
MIN_INSTANCE_PX = 24          # discard instances left smaller than this by a crop
SEED = 0

# --- paths -----------------------------------------------------------------


def _path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


DATA_ROOT = _path("BEAD_DATA", r"C:/Users/Erlind.Skura/Downloads/Segmentations_VB/Segmentations")
WORK = _path("BEAD_WORK", str(Path(__file__).resolve().parent.parent / "work"))
RESULTS = _path("BEAD_RESULTS", str(Path(__file__).resolve().parent.parent / "results"))

PREPARED_IMAGES = WORK / "images"
COCO_GT = WORK / "coco_gt.json"
PREDICTIONS = WORK / "preds"
CHECKPOINTS = WORK / "checkpoints"

# Intermediate files that are rewritten thousands of times and read back never.
# Deliberately *not* under WORK: on Colab that is a mounted Google Drive, where a
# write costs seconds rather than milliseconds, and a training loop that saves a
# checkpoint every epoch would spend most of its wall clock inside the FUSE
# layer. The local disk is wiped on disconnect, which is the correct trade for
# data that is regenerated at the start of every run anyway.
SCRATCH = _path("BEAD_SCRATCH", str(Path(tempfile.gettempdir()) / "bead_scratch"))


def ensure_dirs() -> None:
    for d in (WORK, RESULTS, PREPARED_IMAGES, PREDICTIONS, CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)
