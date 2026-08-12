"""Preprocessing variants, compared as an experimental factor.

The supervisor asked for a comparison of preprocessing techniques, and prior work
from the same group reports that preprocessing materially changes what a network
can learn from microscopy images suffering from non-uniform illumination and low
contrast (Uka et al., 2020). This dataset has the same character -- a particle is
separated from the surrounding mat by roughly eighteen grey levels -- but differs
in that the background is a dense fibre network rather than a smooth field, so
the variants are chosen for that rather than copied across.

Four variants, each addressing one measured property:

``none``        The baseline. Necessary, not merely conventional: the detectors
                are initialised from COCO and ImageNet weights, which expect
                natural-image statistics, so preprocessing can as easily cost
                accuracy as buy it. A variant that loses to this one is a result.

``clahe``       Contrast-limited adaptive histogram equalisation, against the
                eighteen-grey-level separation. Adaptive rather than global
                because the separation is local: the mat's brightness varies
                across the frame by more than the particle-to-mat difference.

``background``  Subtracts an illumination estimate. The particles are *darker*
                than the mat, so the estimate is a grey closing, which erases
                dark features and keeps the slowly varying field. Its footprint
                must exceed the largest object or that object is treated as
                illumination and flattened away: the largest annotated particle
                spans 214 px, so the footprint is 257.

``median``      A 3x3 median, against SEM shot noise. Small deliberately -- the
                median particle is 20 px across, and a window approaching that
                size removes the object along with the noise.

Every variant is deterministic and depends only on the image, so results are
cached per micrograph rather than recomputed for each of the thousands of crops
drawn during training.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

VARIANTS = ("none", "clahe", "background", "median")

# Must exceed the largest annotated object (214 px) so that a genuine particle is
# never mistaken for illumination. Odd, so the footprint is centred.
BACKGROUND_FOOTPRINT = 257

# skimage's default kernel is one eighth of the image, about 128x92 here, which is
# several times the median 20 px particle and comfortably smaller than the
# illumination scale. Stated explicitly rather than left implicit.
CLAHE_KERNEL = (92, 128)
CLAHE_CLIP = 0.01

MEDIAN_SIZE = 3


def apply(img: Image.Image, variant: str) -> Image.Image:
    """Return a preprocessed copy. ``none`` returns the input unchanged.

    Input and output are both RGB, because the detection backbones expect three
    channels. The transforms act on a single channel and are broadcast back, so
    the three channels stay identical, as they already are in a greyscale
    micrograph saved as RGB.
    """
    if variant == "none":
        return img
    if variant not in VARIANTS:
        raise ValueError(f"unknown preprocessing variant {variant!r}; "
                         f"expected one of {VARIANTS}")

    grey = np.asarray(img.convert("L"), dtype=np.uint8)

    if variant == "clahe":
        from skimage import exposure
        out = exposure.equalize_adapthist(grey, kernel_size=CLAHE_KERNEL,
                                          clip_limit=CLAHE_CLIP)
        out = (out * 255.0).round().astype(np.uint8)

    elif variant == "background":
        from scipy import ndimage
        # Grey closing removes dark features smaller than the footprint, leaving
        # the illumination field. A flat rectangular footprint keeps this
        # separable and therefore fast at this size.
        background = ndimage.grey_closing(
            grey, size=(BACKGROUND_FOOTPRINT, BACKGROUND_FOOTPRINT),
            mode="reflect")
        # Closing is extensive, so the difference is everywhere <= 0 and the
        # image is really a map of how far each pixel sits below its local
        # background. It is rescaled to the full 8-bit range rather than shifted
        # by a constant: shifting by the mean and clipping, the obvious version,
        # drove more than half the frame to pure black here, because next to a
        # bright fibre the local background exceeds a dark void by more than the
        # mean grey level. That would have been read as the transform destroying
        # the image when it was the clipping that destroyed it.
        corrected = grey.astype(np.float64) - background.astype(np.float64)
        lo, hi = corrected.min(), corrected.max()
        scaled = (corrected - lo) / (hi - lo) if hi > lo else np.zeros_like(corrected)
        out = (scaled * 255.0).round().astype(np.uint8)

    elif variant == "median":
        from scipy import ndimage
        out = ndimage.median_filter(grey, size=MEDIAN_SIZE, mode="reflect")

    return Image.fromarray(np.repeat(out[:, :, None], 3, axis=2), mode="RGB")


def separation(img: Image.Image, masks) -> dict:
    """Mean grey level inside particles versus outside, and their separation.

    The quantity preprocessing is meant to improve. Reported in grey levels and
    normalised by the background standard deviation, since a variant that widens
    the gap while also widening the spread has bought nothing: the second figure
    is what a threshold or a convolution filter actually has to work with.
    """
    grey = np.asarray(img.convert("L"), dtype=np.float64)
    inside = np.zeros(grey.shape, dtype=bool)
    for m in masks:
        inside |= m.full() if hasattr(m, "full") else np.asarray(m, dtype=bool)
    if not inside.any() or inside.all():
        return {}
    fg, bg = grey[inside], grey[~inside]
    return {
        "mean_inside": float(fg.mean()),
        "mean_outside": float(bg.mean()),
        "separation": float(bg.mean() - fg.mean()),
        "background_sd": float(bg.std()),
        "normalised": float((bg.mean() - fg.mean()) / bg.std()) if bg.std() else 0.0,
    }
