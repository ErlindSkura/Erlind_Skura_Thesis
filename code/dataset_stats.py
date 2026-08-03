"""Descriptive statistics and thesis figures for the electrospun-fibre bead dataset.

Reads the LabelMe annotations shipped in ``Segmentations_VB`` and writes the
dataset figures used in Chapters 3 into ``overleaf/figures``.

Only the 11 original micrographs are analysed. The 132 augmented copies are
deliberately excluded: they carry no additional information and counting them
would inflate every statistic by a factor of 13.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# --- paths -----------------------------------------------------------------

DATA = Path(r"C:/Users/Erlind.Skura/Downloads/Segmentations_VB/Segmentations")
FIGS = Path(__file__).resolve().parent.parent / "overleaf" / "figures"

# The 11 original micrographs, keyed by specimen and magnification. The suffix
# encodes magnification, not a repeat: -1 is 500x, -2 is 1kx, -3/-4 is 3kx.
IMAGES = {
    "Z2-1": ("Z2", 500), "Z2-2": ("Z2", 1000), "Z2-3": ("Z2", 3000),
    "Z4-2": ("Z4", 1000), "Z4-3": ("Z4", 3000),
    "Z5-1": ("Z5", 500), "Z5-2": ("Z5", 1000), "Z5-3": ("Z5", 3000),
    "Z6-1": ("Z6", 500), "Z6-2": ("Z6", 1000), "Z6-4": ("Z6", 3000),
}

# Calibrated from the burned-in scale bars: the field width is 284.4 um at
# 1000x and scales inversely with magnification, over 1024 px.
NM_PER_PX = {500: 555.6, 1000: 277.8, 3000: 92.6}

BANNER_H = 32  # burned-in Zeiss info bar at the bottom of every micrograph


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace area of a closed polygon, in px^2."""
    n = len(points)
    return abs(
        sum(
            points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        )
    ) / 2.0


def load(name: str) -> dict:
    return json.loads((DATA / "Labels" / f"{name}.json").read_text())


def collect() -> dict[str, dict]:
    """Per-image bead areas, equivalent diameters and derived physical sizes."""
    out = {}
    for name, (specimen, mag) in IMAGES.items():
        ann = load(name)
        areas_px = [polygon_area(s["points"]) for s in ann["shapes"]]
        diam_px = [2 * math.sqrt(a / math.pi) for a in areas_px]
        scale = NM_PER_PX[mag] / 1000.0  # um per px
        out[name] = {
            "specimen": specimen,
            "mag": mag,
            "n": len(areas_px),
            "diam_px": np.array(diam_px),
            "diam_um": np.array(diam_px) * scale,
            "area_um2": np.array(areas_px) * scale**2,
            "width": ann["imageWidth"],
            "height": ann["imageHeight"],
        }
    return out


def print_table(stats: dict) -> None:
    print(f"{'image':8} {'spec':5} {'mag':>6} {'beads':>6} {'nm/px':>7} "
          f"{'median d (px)':>14} {'median d (um)':>14}")
    for name, s in stats.items():
        print(f"{name:8} {s['specimen']:5} {s['mag']:>5}x {s['n']:>6} "
              f"{NM_PER_PX[s['mag']]:>7.1f} {np.median(s['diam_px']):>14.1f} "
              f"{np.median(s['diam_um']):>14.2f}")

    total = sum(s["n"] for s in stats.values())
    print(f"\nTotal beads over 11 originals: {total}")

    by_mag = defaultdict(list)
    for s in stats.values():
        by_mag[s["mag"]].extend(s["diam_um"])
    print("\nMedian bead diameter by magnification (um) -- a consistency check "
          "on the scale calibration:")
    for mag in sorted(by_mag):
        v = np.array(by_mag[mag])
        print(f"  {mag:>5}x  n={len(v):>4}  median={np.median(v):.2f}  "
              f"IQR=[{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]")


def fig_dataset_overview(stats: dict) -> None:
    """One specimen shown at all three magnifications, with annotations."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, name in zip(axes, ["Z2-1", "Z2-2", "Z2-3"]):
        im = Image.open(DATA / "Images" / f"{name}.jpg").convert("RGB")
        ann = load(name)
        draw = ImageDraw.Draw(im)
        for s in ann["shapes"]:
            draw.line([tuple(p) for p in s["points"]] + [tuple(s["points"][0])],
                      fill=(255, 40, 40), width=3)
        ax.imshow(im)
        ax.set_title(f"{name} — {stats[name]['mag']}× — {stats[name]['n']} beads",
                     fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_dataset_overview.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_bead_statistics(stats: dict) -> None:
    """Bead count per micrograph and size distribution across magnifications."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    names = list(stats)
    colors = {500: "#4C72B0", 1000: "#DD8452", 3000: "#55A868"}
    ax1.bar(names, [stats[n]["n"] for n in names],
            color=[colors[stats[n]["mag"]] for n in names])
    ax1.set_ylabel("annotated beads")
    ax1.set_title("Beads per micrograph", fontsize=10)
    ax1.tick_params(axis="x", rotation=60, labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax1.legend(handles, [f"{m}×" for m in colors], fontsize=8, title="magnification",
               title_fontsize=8)

    by_mag = defaultdict(list)
    for s in stats.values():
        by_mag[s["mag"]].extend(s["diam_um"])
    ax2.boxplot([by_mag[m] for m in sorted(by_mag)],
                tick_labels=[f"{m}×" for m in sorted(by_mag)], showfliers=False)
    ax2.set_ylabel("equivalent bead diameter (µm)")
    ax2.set_title("Physical bead size is stable across magnifications", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_bead_statistics.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_split_protocol() -> None:
    """Illustrates why the split must be by specimen rather than by image."""
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    specimens = ["Z2", "Z4", "Z5", "Z6"]
    mags = [500, 1000, 3000]
    for i, sp in enumerate(specimens):
        for j, mg in enumerate(mags):
            present = any(s["specimen"] == sp and s["mag"] == mg
                          for s in collect().values())
            ax.add_patch(plt.Rectangle((i, j), 0.92, 0.92,
                                       facecolor="#4C72B0" if present else "#EEEEEE",
                                       edgecolor="grey"))
            if not present:
                ax.text(i + 0.46, j + 0.46, "—", ha="center", va="center",
                        color="grey")
    ax.set_xticks([i + 0.46 for i in range(4)])
    ax.set_xticklabels(specimens)
    ax.set_yticks([j + 0.46 for j in range(3)])
    ax.set_yticklabels([f"{m}×" for m in mags])
    ax.set_xlim(0, 4); ax.set_ylim(0, 3)
    ax.set_xlabel("specimen (the unit that must not be split)")
    ax.set_title("Dataset layout: 4 specimens × 3 magnifications, one cell missing",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_split_protocol.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    stats = collect()
    print_table(stats)
    fig_dataset_overview(stats)
    fig_bead_statistics(stats)
    fig_split_protocol()
    print(f"\nFigures written to {FIGS}")
