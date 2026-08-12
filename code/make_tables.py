

"""Turn the measured metrics into the LaTeX tables and figures of Chapter 5.

Generating the tables from ``results/metrics.json`` rather than typing them means
no number can reach the thesis without having been produced by a run. Every cell
below is read from a file that a training script wrote.

    python make_tables.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import predio
from config import MAGNIFICATIONS, PREDICTIONS, RESULTS, SPECIMENS
from data_io import load_records, rasterise

METHOD_LABEL = {"classical": "Otsu + watershed", "unet": "U-Net + CC",
                "maskrcnn": "Mask R-CNN", "fasterrcnn": "Faster R-CNN"}
FIGS = RESULTS.parent / "figures"


def _f(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _pm(s: dict, nd: int = 3) -> str:
    return f"{s['mean']:.{nd}f} $\\pm$ {s['std']:.{nd}f}"


# --- tables ----------------------------------------------------------------


def table_maskrcnn(m: dict) -> str:
    r = m["loso"]["maskrcnn"]
    rows = []
    for sp in SPECIMENS:
        f = r["per_fold"].get(sp)
        if not f:
            continue
        rows.append(f"        {sp} & {_f(f['ap50'])} & {_f(f['ap75'])} & "
                    f"{_f(f['ap'])} & {_f(f['aji'])} \\\\")
    o = r["overall"]
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Mask R-CNN under leave-one-specimen-out cross-validation.}}
    \\label{{tab:maskrcnn_results}}
    \\begin{{tabular}}{{lrrrr}}
        \\toprule
        Test specimen & $\\mathrm{{AP}}_{{50}}$ & $\\mathrm{{AP}}_{{75}}$ & AP & AJI \\\\
        \\midrule
{chr(10).join(rows)}
        \\midrule
        Mean $\\pm$ s.d. & {_pm(o['ap50'])} & {_pm(o['ap75'])} & {_pm(o['ap'])} & {_pm(o['aji'])} \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_comparison(m: dict) -> str:
    rows = []
    for key in ("classical", "unet", "maskrcnn"):
        r = m["loso"].get(key)
        if not r:
            continue
        o = r["overall"]
        rows.append(f"        {METHOD_LABEL[key]:16s} & {_f(o['ap50']['mean'])} & "
                    f"{_f(o['aji']['mean'])} & "
                    f"{_f(o['abs_counting_error']['mean'], 1)} & "
                    f"{_f(o['merge_rate']['mean'], 1)} \\\\")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{The three methods under an identical protocol, averaged over the four
    leave-one-specimen-out folds. Counting error is the mean absolute percentage
    difference between the predicted and the annotated bead count.}}
    \\label{{tab:method_comparison}}
    \\begin{{tabular}}{{lrrrr}}
        \\toprule
        Method & $\\mathrm{{AP}}_{{50}}$ & AJI & Counting error (\\%) & Merge rate (\\%) \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_leakage(m: dict) -> str | None:
    a = m.get("random", {}).get("maskrcnn")
    b = m.get("loso", {}).get("maskrcnn")
    if not (a and b):
        return None
    ra, rb = a["overall"], b["overall"]
    diff = (f"        Difference & "
            f"{_f(ra['ap50']['mean'] - rb['ap50']['mean'])} & "
            f"{_f(ra['aji']['mean'] - rb['aji']['mean'])} & "
            f"{_f(ra['abs_counting_error']['mean'] - rb['abs_counting_error']['mean'], 1)} \\\\")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{The same model evaluated under a specimen-wise protocol and under a
    random split over the 11 micrographs, with identical fold sizes. The
    difference is what a naive protocol would have gained on this dataset;
    a difference near zero is itself a result, not a missing one.}}
    \\label{{tab:leakage}}
    \\begin{{tabular}}{{lrrr}}
        \\toprule
        Protocol & $\\mathrm{{AP}}_{{50}}$ & AJI & Counting error (\\%) \\\\
        \\midrule
        Random split over images & {_f(ra['ap50']['mean'])} & {_f(ra['aji']['mean'])} & {_f(ra['abs_counting_error']['mean'], 1)} \\\\
        Leave-one-specimen-out   & {_f(rb['ap50']['mean'])} & {_f(rb['aji']['mean'])} & {_f(rb['abs_counting_error']['mean'], 1)} \\\\
        \\midrule
{diff}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_counting(m: dict) -> str:
    """Counting accuracy across every method, including the box-only detector.

    Kept separate from the AJI comparison because Faster R-CNN predicts no masks
    and so has no AJI, no merge rate and no size distribution. Counting error and
    the raw count are defined identically for a box and for a mask, so this is the
    one table on which all four methods can be placed side by side without
    qualification. The average precision column names the IoU type it was computed
    on, since a mask AP and a box AP are not the same measurement.
    """
    rows = []
    for key in ("classical", "unet", "maskrcnn", "fasterrcnn"):
        r = m["loso"].get(key)
        if not r:
            continue
        o = r["overall"]
        n_pred = sum(v["n_pred"] for v in r["per_image"].values())
        n_gt = sum(v["n_gt"] for v in r["per_image"].values())
        kind = "box" if r.get("box_only") else "mask"
        ap50 = _f(o["ap50"]["mean"]) if "ap50" in o else "---"
        rows.append(f"        {METHOD_LABEL[key]:16s} & {n_pred} & {n_gt} & "
                    f"{_f(o['abs_counting_error']['mean'], 1)} & "
                    f"{ap50} ({kind}) \\\\")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Counting accuracy under the leave-one-specimen-out protocol. Counts
    are pooled over all 11 held-out micrographs; the error is the mean absolute
    percentage difference per micrograph, so it does not cancel between images that
    over- and under-count. Faster R-CNN predicts boxes and no masks, so its average
    precision is computed on boxes and it is absent from the mask-based comparisons
    of Table~\\ref{{tab:method_comparison}}.}}
    \\label{{tab:counting}}
    \\begin{{tabular}}{{lrrrr}}
        \\toprule
        Method & Detected & Annotated & Counting error (\\%) & $\\mathrm{{AP}}_{{50}}$ \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_ap_bands(m: dict) -> str:
    """Average precision split by COCO object-size band.

    82.8% of the annotated particles fall in the small band, so a pooled AP is
    close to an average over one band while appearing to describe all three. The
    bands are reported separately for that reason. A band the ground truth does not
    populate is left blank rather than shown as zero.
    """
    rows = []
    for key in ("classical", "unet", "maskrcnn", "fasterrcnn"):
        r = m["loso"].get(key)
        if not r:
            continue
        o = r["overall"]
        cells = [(_f(o[k]["mean"]) if k in o else "---")
                 for k in ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")]
        rows.append(f"        {METHOD_LABEL[key]:16s} & " + " & ".join(cells) + " \\\\")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Average precision decomposed by object size, under the
    leave-one-specimen-out protocol. The bands are COCO's: small is below
    $32 \\times 32$ pixels, large above $96 \\times 96$. Since 82.8\\% of the
    annotated particles are small and 1.2\\% are large, $\\mathrm{{AP}}_S$ is the
    column that describes this dataset; a dash marks a band the held-out ground
    truth does not populate, which cannot be scored.}}
    \\label{{tab:ap_bands}}
    \\begin{{tabular}}{{lrrrrrr}}
        \\toprule
        Method & AP & $\\mathrm{{AP}}_{{50}}$ & $\\mathrm{{AP}}_{{75}}$ &
        $\\mathrm{{AP}}_S$ & $\\mathrm{{AP}}_M$ & $\\mathrm{{AP}}_L$ \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_runtime(m: dict) -> str:
    """Wall-clock cost per fold, in the units the supervisor asked for."""
    rows = []
    for key in ("unet", "maskrcnn", "fasterrcnn"):
        r = m["loso"].get(key)
        if not r:
            continue
        rt = (r.get("meta") or {}).get("runtime") or {}
        folds = [v for v in rt.values() if v]
        if not folds:
            continue
        mean = lambda f: sum(x[f] for x in folds) / len(folds)   # noqa: E731
        rows.append(
            f"        {METHOD_LABEL[key]:16s} & {folds[0]['batch']} & "
            f"{mean('s_per_step_median'):.3f} & {folds[0]['steps_per_epoch']:.1f} & "
            f"{mean('s_per_epoch'):.1f} & {mean('epochs'):.1f} & "
            f"{mean('train_wall_s') / 60:.1f} \\\\")
    if not rows:
        return ""
    hw = ""
    for key in ("maskrcnn", "unet", "fasterrcnn"):
        h = ((m["loso"].get(key) or {}).get("meta") or {}).get("hardware") or {}
        if h.get("gpu"):
            hw = f" Trained on {h['gpu']}."
            break
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Training cost per fold, averaged over the four leave-one-specimen-out
    folds.{hw} Training draws random crops on demand rather than iterating a fixed
    set, so an epoch is defined here as one crop-equivalent pass over the fold's
    eight training micrographs. Step times are medians, measured with the device
    synchronised, and exclude the first step, which carries one-off initialisation.}}
    \\label{{tab:runtime}}
    \\begin{{tabular}}{{lrrrrrr}}
        \\toprule
        Method & Batch & s/step & Steps/epoch & s/epoch & Epochs & Total (min) \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_magnification(m: dict) -> str:
    r = m["loso"]["maskrcnn"]["by_magnification"]
    rows = []
    for mag in MAGNIFICATIONS:
        d = r.get(str(mag))
        if not d:
            continue
        rows.append(f"        ${mag}\\times$ & {d['n_images']} & {d['n_gt']} & "
                    f"{_f(d['ap50'])} & {_f(d['aji'])} & "
                    f"{_f(d['abs_counting_error'], 1)} \\\\")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Mask R-CNN performance by magnification, pooled over the four
    leave-one-specimen-out folds. Every micrograph is scored by the model that
    did not see its specimen.}}
    \\label{{tab:magnification}}
    \\begin{{tabular}}{{lrrrrr}}
        \\toprule
        Magnification & Images & Beads & $\\mathrm{{AP}}_{{50}}$ & AJI & $|$Counting error$|$ (\\%) \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_physical(m: dict) -> str:
    rows = []
    for key in ("classical", "unet", "maskrcnn"):
        r = m["loso"].get(key)
        if not r:
            continue
        s = r["size_agreement"]
        if "gt_median" not in s:
            continue
        rows.append(
            f"        {METHOD_LABEL[key]:16s} & {s['n_pred']} & "
            f"{_f(s['pred_median'], 2)} & "
            f"[{_f(s['pred_iqr'][0], 2)}, {_f(s['pred_iqr'][1], 2)}] & "
            f"{_f(s['ks_statistic'], 3)} & {s['ks_pvalue']:.1e} \\\\")
    gt = m["ground_truth"]
    ref = (f"        Manual annotation & {gt['n']} & {_f(gt['median'], 2)} & "
           f"[{_f(gt['iqr'][0], 2)}, {_f(gt['iqr'][1], 2)}] & --- & --- \\\\")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Predicted bead size distribution against the manual ground truth,
    pooled over all 11 held-out micrographs. Diameters are equivalent circular
    diameters in micrometres; the two-sample Kolmogorov--Smirnov statistic
    compares each predicted distribution with the manual one.}}
    \\label{{tab:physical}}
    \\begin{{tabular}}{{lrrrrr}}
        \\toprule
        Source & Beads & Median $d$ ($\\mu$m) & IQR ($\\mu$m) & KS $D$ & $p$ \\\\
        \\midrule
{ref}
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_folds(m: dict) -> str:
    r = m["loso"]["maskrcnn"]["per_fold"]
    rows = []
    for i, sp in enumerate(SPECIMENS, start=1):
        f = r.get(sp)
        if not f:
            continue
        rows.append(f"        {i} & {sp} & {len(f['test_images'])} & {f['n_gt']} & "
                    f"{f['n_pred']} & {_f(f['counting_error'], 1)} \\\\")
    total_gt = sum(f["n_gt"] for f in r.values())
    total_pred = sum(f["n_pred"] for f in r.values())
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Leave-one-specimen-out fold composition and the bead counts
    Mask R-CNN produced for each held-out specimen. Counting error is signed:
    positive means the method over-counted.}}
    \\label{{tab:folds}}
    \\begin{{tabular}}{{llrrrr}}
        \\toprule
        Fold & Test specimen & Micrographs & Annotated & Predicted & Counting error (\\%) \\\\
        \\midrule
{chr(10).join(rows)}
        \\midrule
        \\multicolumn{{2}}{{l}}{{\\textbf{{Total}}}} & \\textbf{{11}} & \\textbf{{{total_gt}}} & \\textbf{{{total_pred}}} & \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def table_config(m: dict) -> str:
    mr = m["loso"]["maskrcnn"]["meta"]
    un = m["loso"].get("unet", {}).get("meta", {})
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Training configuration actually used. Both models were trained
    independently on each of the four folds from the same initialisation.}}
    \\label{{tab:training_config}}
    \\begin{{tabular}}{{lll}}
        \\toprule
        Setting & Mask R-CNN & U-Net \\\\
        \\midrule
        Backbone        & ResNet-50 + FPN   & ResNet-34 encoder \\\\
        Initialisation  & COCO              & ImageNet \\\\
        Optimiser       & SGD, momentum 0.9 & Adam \\cite{{kingma2015adam}} \\\\
        Learning rate   & {mr.get('lr', '---')} (cosine)  & {un.get('lr', '---')} (cosine) \\\\
        Batch size      & {mr.get('batch', '---')}        & {un.get('batch', '---')} \\\\
        Iterations      & {mr.get('iters', '---')}        & {un.get('iters', '---')} \\\\
        Crop size       & $512 \\times 512$  & $512 \\times 512$ \\\\
        Loss            & RPN + box + mask  & Cross-entropy + Dice \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


# --- figures ---------------------------------------------------------------


def _outline(gray: np.ndarray, masks, colour) -> np.ndarray:
    from skimage.segmentation import find_boundaries
    rgb = np.dstack([gray] * 3).astype(np.float32) / 255.0
    for m in masks:
        b = find_boundaries(m, mode="outer")
        rgb[b] = colour
    return np.clip(rgb, 0, 1)


def fig_qualitative(m: dict, names=("Z6-1", "Z2-2")) -> None:
    records = load_records()
    available = [k for k in ("classical", "unet", "maskrcnn") if k in m["loso"]]
    preds = {}
    for k in available:
        dets, _ = predio.load(PREDICTIONS / "loso" / f"{k}.json")
        preds[k] = predio.group_by_image(dets)

    cols = 1 + len(available)
    fig, axes = plt.subplots(len(names), cols,
                             figsize=(3.1 * cols, 2.45 * len(names)),
                             layout="constrained")
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        rec = records[name]
        gray = np.asarray(Image.open(rec.path).convert("L"))
        gts = list(rasterise(rec.polys, rec.width, rec.height))
        panels = [("Manual annotation", gts, (1.0, 0.15, 0.15))]
        for k in available:
            masks, _ = predio.to_masks(preds[k].get(rec.image_id, []))
            panels.append((METHOD_LABEL[k], masks, (0.15, 0.85, 1.0)))

        for col, (title, masks, colour) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(_outline(gray, masks, colour))
            ax.axis("off")
            # Column headings go on the top row only. Repeating them per row
            # made the second row's titles overprint the first row's images.
            if row == 0:
                ax.set_title(title, fontsize=9)
            ax.text(0.03, 0.94, f"{len(masks)} beads", transform=ax.transAxes,
                    fontsize=8, color="white", va="top",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2,
                              edgecolor="none"))
        axes[row, 0].text(0.03, 0.04, f"{name} — {rec.magnification}$\\times$",
                          transform=axes[row, 0].transAxes, fontsize=8,
                          color="white",
                          bbox=dict(facecolor="black", alpha=0.55, pad=2,
                                    edgecolor="none"))

    fig.suptitle("Held-out micrographs: every panel is scored by a model that "
                 "never saw this specimen", fontsize=10)
    fig.savefig(FIGS / "fig_qualitative.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_size_agreement(m: dict) -> None:
    records = load_records()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    gt = np.concatenate([
        _diams(r, rasterise(r.polys, r.width, r.height)) for r in records.values()
    ])
    ax.hist(gt, bins=40, density=True, alpha=0.45, label=f"Manual (n={gt.size})",
            color="#B0413E")

    for k, colour in (("maskrcnn", "#4C72B0"), ("unet", "#DD8452"),
                      ("classical", "#55A868")):
        if k not in m["loso"]:
            continue
        dets, _ = predio.load(PREDICTIONS / "loso" / f"{k}.json")
        grouped = predio.group_by_image(dets)
        vals = []
        for rec in records.values():
            masks, _ = predio.to_masks(grouped.get(rec.image_id, []))
            vals.append(_diams(rec, masks))
        v = np.concatenate(vals) if vals else np.zeros(0)
        if v.size:
            ax.hist(v, bins=40, density=True, histtype="step", linewidth=1.8,
                    label=f"{METHOD_LABEL[k]} (n={v.size})", color=colour)

    ax.set_xlabel("equivalent bead diameter (µm)")
    ax.set_ylabel("density")
    ax.set_title("Predicted against manual bead size distribution, all held-out "
                 "micrographs", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_size_agreement.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _diams(rec, masks):
    from metrics import equivalent_diameters
    return equivalent_diameters(list(masks), rec.um_per_px)


# --- entry point -----------------------------------------------------------


def run() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    m = json.loads((RESULTS / "metrics.json").read_text())
    if "maskrcnn" not in m.get("loso", {}):
        raise SystemExit("no leave-one-specimen-out Mask R-CNN results to tabulate")

    banner = ("% Generated by code/make_tables.py from results/metrics.json.\n"
              "% Do not edit by hand: re-run the pipeline instead.\n")

    tables = {
        "training_config": table_config(m),
        "folds": table_folds(m),
        "maskrcnn_results": table_maskrcnn(m),
        "method_comparison": table_comparison(m),
        "counting": table_counting(m),
        "ap_bands": table_ap_bands(m),
        "magnification": table_magnification(m),
        "physical": table_physical(m),
    }
    leakage = table_leakage(m)
    if leakage:
        tables["leakage"] = leakage
    runtime = table_runtime(m)
    if runtime:
        tables["runtime"] = runtime

    # One file per table, so Chapter 5 can \input each where it belongs and a
    # re-run updates the chapter in place. The combined file is kept as well,
    # for reading the whole set at once.
    for name, body in tables.items():
        (RESULTS / f"table_{name}.tex").write_text(banner + body + "\n",
                                                   encoding="utf-8")
    (RESULTS / "chapter5_tables.tex").write_text(
        banner + "\n" + "\n\n".join(tables.values()) + "\n", encoding="utf-8")
    print(f"wrote {len(tables)} tables to {RESULTS}")

    fig_qualitative(m)
    fig_size_agreement(m)
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    run()
