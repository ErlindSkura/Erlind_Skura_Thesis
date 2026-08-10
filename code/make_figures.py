"""Generate the dataset and analysis figures that are not produced by a training run.

Each figure here exists to carry evidence for a claim the thesis makes and that
prose alone cannot show: that the supplied augmented copies are unusable, that
beads overlap, that bead-to-background contrast is too weak for thresholding, and
that the per-micrograph counting error is far wider than the aggregate suggests.

Figures produced by the pipeline itself -- the qualitative comparison and the
size-distribution agreement -- are written by make_tables.py instead, because
they depend on model predictions.

    python make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

from config import BANNER_H, DATA_ROOT, IMAGES, NM_PER_PX, RESULTS, WORK_H

FIGS = Path(__file__).resolve().parent.parent / "figures"

# Print-safe: the thesis is printed in black ink, so nothing may depend on hue
# alone. Greys carry the data; a single accent is used only for annotation.
GREY = "#4D4D4D"
ACCENT = "#B0413E"
FILL = "#BFBFBF"


def _load_json(name: str) -> dict:
    return json.loads((DATA_ROOT / "Labels" / f"{name}.json").read_text())


def _rgb(name: str, folder: str = "Images", ext: str = "jpg") -> Image.Image:
    return Image.open(DATA_ROOT / folder / f"{name}.{ext}").convert("RGB")


# --- 1. why the supplied augmentations cannot be used ------------------------


def fig_augmentation_problem() -> None:
    """The banner is transformed along with the image, and padding appears.

    This is the visual evidence for discarding the 132 pre-computed copies: they
    are not merely redundant, they carry a mirrored instrument banner and black
    fill that no real micrograph would contain.
    """
    panels = [
        ("Z2-1", "original"),
        ("Z2-1_flip-h", "horizontal flip"),
        ("Z2-1_shift-150-0", "translation"),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(10.5, 4.6),
                            gridspec_kw={"height_ratios": [3, 1]},
                            layout="constrained")
    for col, (name, label) in enumerate(panels):
        im = _rgb(name)
        a = np.asarray(im)
        axes[0, col].imshow(a)
        axes[0, col].set_title(label, fontsize=10)
        axes[0, col].axis("off")
        # Mark the banner region on the full view.
        axes[0, col].add_patch(Rectangle((0, im.height - BANNER_H), im.width,
                                         BANNER_H, fill=False, edgecolor=ACCENT,
                                         linewidth=1.2))
        # And show it enlarged underneath.
        axes[1, col].imshow(a[-BANNER_H:], aspect="auto")
        axes[1, col].axis("off")
    axes[1, 0].set_title("the burned-in instrument banner, enlarged",
                         fontsize=9, loc="left")
    fig.suptitle("The supplied augmented copies transform the instrument banner "
                 "along with the specimen", fontsize=11)
    fig.savefig(FIGS / "fig_augmentation_problem.pdf", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# --- 2. beads overlap --------------------------------------------------------


def _polys(name: str) -> list[np.ndarray]:
    return [np.asarray(s["points"], dtype=float)
            for s in _load_json(name)["shapes"]]


def fig_overlap() -> None:
    """A region where two annotated beads claim the same pixels.

    8.8% of annotated bead pixels lie under more than one polygon. That is the
    reason instances are never flattened into a label map, and it is invisible in
    any summary statistic.
    """
    name = "Z6-1"
    polys = _polys(name)
    W, H = 1024, WORK_H

    acc = np.zeros((H, W), dtype=np.int32)
    for p in polys:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon([tuple(v) for v in p], fill=1)
        acc += np.asarray(m, dtype=np.int32)
    contested = acc > 1

    # Centre the crop on the window holding the most contested pixels. Taking
    # the median coordinate instead lands between clusters and shows nothing.
    half = 130
    box = 2 * half
    integral = np.pad(contested.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, y0, x0 = -1, 0, 0
    for yy in range(0, H - box, 16):
        for xx in range(0, W - box, 16):
            total = (integral[yy + box, xx + box] - integral[yy, xx + box]
                     - integral[yy + box, xx] + integral[yy, xx])
            if total > best:
                best, y0, x0 = total, yy, xx
    y1, x1 = y0 + box, x0 + box

    grey = np.asarray(_rgb(name).convert("L"))[:H]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4), layout="constrained")

    ax1.imshow(grey, cmap="gray")
    ax1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            edgecolor=ACCENT, linewidth=1.5))
    ax1.set_title(f"{name} — {len(polys)} annotated beads", fontsize=10)
    ax1.axis("off")

    ax2.imshow(grey[y0:y1, x0:x1], cmap="gray")
    for p in polys:
        q = p - np.array([x0, y0])
        inside = ((q[:, 0] > -40) & (q[:, 0] < (x1 - x0) + 40)
                  & (q[:, 1] > -40) & (q[:, 1] < (y1 - y0) + 40))
        if inside.any():
            ax2.plot(np.append(q[:, 0], q[0, 0]), np.append(q[:, 1], q[0, 1]),
                     color=ACCENT, linewidth=1.3)
    # The contested region between two touching beads is often only a few pixels
    # wide. Dilating it by two pixels makes it visible at figure scale without
    # misrepresenting where it is; the reported percentage is computed from the
    # undilated mask.
    sub = contested[y0:y1, x0:x1]
    grown = sub.copy()
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            grown |= np.roll(np.roll(sub, dy, axis=0), dx, axis=1)
    ov = np.ma.masked_where(~grown, grown)
    ax2.imshow(ov, cmap=matplotlib.colors.ListedColormap(["#FFD400"]), alpha=0.95)
    ax2.set_xlim(0, x1 - x0)
    ax2.set_ylim(y1 - y0, 0)
    ax2.set_title("annotated outlines, with contested pixels shaded",
                  fontsize=10)
    ax2.axis("off")

    pct = 100 * contested.sum() / max((acc > 0).sum(), 1)
    fig.suptitle(f"Beads touch and overlap: {pct:.1f}% of annotated bead pixels in "
                 f"this micrograph, and 8.8% across the dataset,\nare claimed by "
                 f"more than one bead", fontsize=11)
    fig.savefig(FIGS / "fig_overlap.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 3. why thresholding cannot work ----------------------------------------


def fig_contrast() -> None:
    """Grey-level distributions inside and outside annotated beads.

    The classical baseline reaches AP50 = 0.001. This figure is the reason: the
    two distributions are separated by about 20 grey levels and overlap almost
    completely, so no global threshold can divide them.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained",
                            sharey=True)
    for ax, name in zip(axes, ["Z2-1", "Z5-2", "Z6-4"]):
        grey = np.asarray(_rgb(name).convert("L"), dtype=float)[:WORK_H]
        mask = Image.new("L", (1024, WORK_H), 0)
        d = ImageDraw.Draw(mask)
        for p in _polys(name):
            d.polygon([tuple(v) for v in p], fill=1)
        m = np.asarray(mask, dtype=bool)

        bins = np.arange(0, 257, 4)
        ax.hist(grey[~m], bins=bins, density=True, color=FILL,
                label=f"outside (mean {grey[~m].mean():.0f})")
        ax.hist(grey[m], bins=bins, density=True, histtype="step",
                linewidth=1.8, color=ACCENT,
                label=f"inside (mean {grey[m].mean():.0f})")
        ax.axvline(grey[~m].mean(), color=GREY, linestyle=":", linewidth=1)
        ax.axvline(grey[m].mean(), color=ACCENT, linestyle=":", linewidth=1)
        ax.set_title(f"{name} — {IMAGES[name][1]}$\\times$", fontsize=10)
        ax.set_xlabel("grey level")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density")
    fig.suptitle("Beads are darker than the surrounding mat by only about 20 grey "
                 "levels, and the distributions overlap almost entirely",
                 fontsize=11)
    fig.savefig(FIGS / "fig_contrast.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 4. counting agreement ---------------------------------------------------


def fig_counting() -> None:
    """Predicted against annotated bead count, per micrograph.

    The aggregate shortfall is 10.4%, which flatters the method: the per-image
    errors run from -68% to +95% and largely cancel. A scatter shows that in a
    way a mean cannot.
    """
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    per_image = metrics["loso"]["maskrcnn"]["per_image"]

    markers = {500: "o", 1000: "s", 3000: "^"}
    fig, ax = plt.subplots(figsize=(6.0, 5.4), layout="constrained")

    lim = 5 + max(max(v["n_gt"], v["n_pred"]) for v in per_image.values())
    ax.plot([0, lim], [0, lim], color=GREY, linewidth=1, label="perfect agreement")
    ax.fill_between([0, lim], [0, 0.75 * lim], [0, 1.25 * lim],
                    color=FILL, alpha=0.35, linewidth=0, label="within 25%")

    for mag, mk in markers.items():
        pts = [(v["n_gt"], v["n_pred"]) for v in per_image.values()
               if v["magnification"] == mag]
        if pts:
            ax.scatter(*zip(*pts), marker=mk, s=58, facecolor="white",
                       edgecolor="black", linewidth=1.2, zorder=3,
                       label=f"${mag}\\times$")
    for name, v in per_image.items():
        ax.annotate(name, (v["n_gt"], v["n_pred"]), fontsize=7,
                    xytext=(4, -9), textcoords="offset points", color=GREY)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("annotated beads")
    ax.set_ylabel("beads predicted by Mask R-CNN")
    ax.set_title("Counting agreement per held-out micrograph", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.savefig(FIGS / "fig_counting.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 5. what the model has to find ------------------------------------------


def fig_bead_examples() -> None:
    """The same physical object at each magnification, cropped to one bead.

    The median bead is 14 px across at 500x and 76 at 3000x. The thesis argues
    that object size in pixels, rather than the number of training examples, is
    the binding constraint; this is what that sixfold range looks like.
    """
    picks = [("Z6-1", 500), ("Z6-2", 1000), ("Z6-4", 3000)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")

    for ax, (name, mag) in zip(axes, picks):
        grey = np.asarray(_rgb(name).convert("L"))[:WORK_H]
        polys = _polys(name)
        areas = [abs(sum(p[i][0] * p[(i + 1) % len(p)][1]
                         - p[(i + 1) % len(p)][0] * p[i][1]
                         for i in range(len(p)))) / 2 for p in polys]

        # Take the bead closest to the median area that also sits far enough from
        # every edge for a full square crop, so the three panels are the same
        # shape and can be compared directly.
        order = np.argsort(areas)
        candidates = list(order[len(order) // 2:]) + list(order[:len(order) // 2][::-1])
        for idx in candidates:
            p = polys[idx]
            cx, cy = p[:, 0].mean(), p[:, 1].mean()
            d_px = 2 * math.sqrt(areas[idx] / math.pi)
            half = max(38, int(1.9 * d_px))
            if (cx - half >= 0 and cx + half <= 1024
                    and cy - half >= 0 and cy + half <= WORK_H):
                break
        x0, y0 = int(cx - half), int(cy - half)
        x1, y1 = int(cx + half), int(cy + half)

        ax.imshow(grey[y0:y1, x0:x1], cmap="gray")
        q = p - np.array([x0, y0])
        ax.plot(np.append(q[:, 0], q[0, 0]), np.append(q[:, 1], q[0, 1]),
                color=ACCENT, linewidth=1.6)
        um = d_px * NM_PER_PX[mag] / 1000.0
        ax.set_title(f"{mag}$\\times$ — {d_px:.0f} px across ({um:.1f} µm)",
                     fontsize=10)
        ax.axis("off")

    fig.suptitle("The median bead of specimen Z6 at each magnification, shown at "
                 "the same display size", fontsize=11)
    fig.savefig(FIGS / "fig_bead_examples.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_augmentation_problem()
    fig_overlap()
    fig_contrast()
    fig_bead_examples()
    if (RESULTS / "metrics.json").exists():
        fig_counting()
    else:
        print("no results/metrics.json; skipped the counting figure")
    print(f"figures written to {FIGS}")
