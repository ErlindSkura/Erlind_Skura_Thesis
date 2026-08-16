"""Generate Thesis_Colab.ipynb.

The notebook is generated rather than hand-edited: a 28-cell .ipynb is painful to
diff and easy to corrupt by hand. Edit this file and re-run it instead.

    python notebooks/_generate_notebook.py
"""
import json
from pathlib import Path

REPO = "https://github.com/ErlindSkura/Erlind_Skura_Thesis.git"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text, meta=None):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": meta or {}, "outputs": [], "source": text.strip("\n")})


md(r"""
# Bead segmentation in SEM micrographs — training and evaluation

Runs the full pipeline for the MSc thesis *Deep Learning for Instance Segmentation and
Quantification of Bead Defects in SEM Images of Electrospun Nanofibre Mats*.

**Before you start:** `Runtime → Change runtime type → T4 GPU`.

Each stage writes its output to disk, so if Colab disconnects you can re-run only the
stage that was interrupted. Run the cells in order.

| Stage | Roughly |
|---|---|
| Setup and data preparation | 2 min |
| Classical baseline (no GPU needed) | 5 min |
| U-Net, 4 folds | 25 min |
| Mask R-CNN, 4 folds | 40 min |
| Mask R-CNN naive-split control | 40 min |
| Evaluation and tables | 3 min |
""")

md("## 1 · Check the GPU")
code(r"""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
      or "NO GPU - set Runtime > Change runtime type > T4 GPU, then rerun this cell.")
""")

md(r"""
## 2 · Mount Drive and locate the data

Upload `bead_data.zip` to the top level of your Google Drive first (My Drive).

Mounting Drive also means the results survive a disconnect: everything is written to
`MyDrive/thesis_work`, so a re-run picks up where it stopped.
""")
code(r"""
import os, zipfile
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/thesis_work")
except Exception as e:
    print(f"Drive not mounted ({e}); results will be lost on disconnect.")
    ROOT = Path("/content/thesis_work")

ROOT.mkdir(parents=True, exist_ok=True)
DATA = Path("/content/data")

zips = list(Path("/content/drive/MyDrive").glob("bead_data.zip")) if Path("/content/drive").exists() else []
if not zips:
    from google.colab import files
    print("bead_data.zip not found in My Drive — upload it now.")
    up = files.upload()
    zips = [Path(next(iter(up)))]

with zipfile.ZipFile(zips[0]) as z:
    z.extractall(DATA)
print("data:", sorted(p.name for p in (DATA / "Segmentations").iterdir()))
print("images:", len(list((DATA / "Segmentations" / "Images").glob("*.jpg"))))
""")

md("## 3 · Get the code")
code(rf"""
import os, subprocess
from pathlib import Path

CODE = Path("/content/thesis")
if CODE.exists():
    subprocess.run(["git", "-C", str(CODE), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}", str(CODE)], check=True)

os.environ["BEAD_DATA"] = str(DATA / "Segmentations")
os.environ["BEAD_WORK"] = str(ROOT / "work")
os.environ["BEAD_RESULTS"] = str(ROOT / "results")
os.chdir(CODE / "code")
print("cwd:", Path.cwd())
print("work:", os.environ["BEAD_WORK"])
""")

md("## 4 · Verify the environment")
code(r"""
import importlib, subprocess, sys

# Import name -> pip name, which differ for two of these.
NEEDED = {"torch": "torch", "torchvision": "torchvision",
          "pycocotools": "pycocotools", "skimage": "scikit-image",
          "scipy": "scipy"}

for mod_name, pip_name in NEEDED.items():
    try:
        mod = importlib.import_module(mod_name)
        print(f"{mod_name:14} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"{mod_name:14} missing - installing {pip_name}")
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", pip_name],
                       check=True)

import torch
print("\ncuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
""")

md(r"""
## 5 · Prepare the data

Crops the instrument banner, converts the LabelMe polygons to COCO, and writes the
fold manifests. Expect **11 images and 606 beads** — if you see anything else, stop
and check the upload.
""")
code(r"""
!python prepare_data.py
!python folds.py
""")

md(r"""
### 5b · Check the geometry

Every conversion the pipeline depends on, checked against the real annotations:
box extraction, the YOLO label export, and the dataset invariants the thesis
quotes. These are the errors that would produce plausible numbers rather than a
crash, so they have to be checked rather than assumed. All should pass.
""")
code(r"""
!python tests.py
""")

md(r"""
## 6 · Smoke test

Eight training iterations per fold. The numbers this produces are meaningless — the
point is to find out in two minutes, rather than after two hours, whether anything
crashes.
""")
code(r"""
!python run_all.py --smoke
""")

md(r"""
## 7 · The real run

From here the stages are separate cells. Each saves its predictions to Drive, so a
disconnect costs you one stage and not the whole run.

### 7a · Classical baseline (Otsu + watershed)
""")
code(r"""
!python classical.py --protocol loso
""")

md("### 7b · U-Net semantic baseline")
code(r"""
!python train_unet.py --protocol loso --iters 1500 --batch 8
""")

md("### 7c · Mask R-CNN, leave-one-specimen-out")
code(r"""
!python train_maskrcnn.py --protocol loso --iters 1500 --batch 4
""")

md(r"""
### 7d · Faster R-CNN, detection only

The same backbone, anchors and schedule as 7c, without the mask branch. The
laboratory endpoint is a count, and counting needs only detection, so this
measures what the mask branch is actually worth for the quantity of interest.
It reports box AP rather than mask AP, and no AJI — it predicts no masks.
""")
code(r"""
!python train_fasterrcnn.py --protocol loso --iters 1500 --batch 4
""")

md(r"""
### 7e · YOLOv8

A different family of architecture, not a controlled ablation: YOLOv8 shares no
backbone, head, loss or augmentation with the R-CNN models, so it answers "does
another family do better here", not "which component is responsible".

Three settings differ from the Ultralytics defaults, each for a measured reason.
`imgsz=1024` keeps native resolution — the default 640 would shrink the median
500× particle from 14.3 px to 8.9 px. `mosaic=0.0` is off because mosaic roughly
halves apparent object size, and 82.8% of these particles are already below
COCO's small-object threshold. `max_det=400` prevents truncating dense images.

`-seg` weights predict masks and join the mask comparison; swap to `yolov8s.pt`
for detection only.
""")
code(r"""
!pip install -q ultralytics
!python train_yolo.py --protocol loso --iters 1500 --batch 4 --weights yolov8s-seg.pt
""")

md(r"""
### 7f · YOLOv5

Two things to be clear about when you present this.

It is **YOLOv5u**, not the YOLOv5 of the 2020 papers: Ultralytics ships the
YOLOv5 backbone fitted with YOLOv8's anchor-free head. So `yolov5su` against
`yolov8s` varies mostly the *backbone* — it is not the anchor-based versus
anchor-free comparison the version numbers suggest.

Ultralytics has no `-seg` variant for YOLOv5, so it detects only: box AP and
counting, no AJI and no size distribution. It writes to its own prediction file,
so it does not overwrite 7e.
""")
code(r"""
!python train_yolo.py --protocol loso --iters 1500 --batch 4 --weights yolov5su.pt
""")

md(r"""
### 7g · Preprocessing ablation

The same architecture, schedule, folds and threshold rule — only the input
transform changes, so a difference is attributable to preprocessing alone.

Contrast measured on the annotations *before* training — separation between
particle and mat (Δ), divided by background spread (σ), which is what a filter
actually has to work with:

| Variant | Δ (grey levels) | σ | Δ/σ | vs baseline |
|---|---|---|---|---|
| none | 18.0 | 33.2 | 0.550 | — |
| median | 18.3 | 30.5 | 0.612 | +11% |
| background | 20.3 | 34.9 | 0.593 | +8% |
| clahe | 23.1 | 62.5 | 0.372 | −32% |

CLAHE is the case to understand: it gives the **widest** raw gap of the four,
23.1 levels against 18.0, and on that number alone would be the obvious choice.
It also nearly doubles the background spread, because it amplifies the fibre
texture as much as the particles. Usable contrast falls by a third.

This measures contrast, not accuracy. A CNN is not a threshold — it can use
texture and shape the statistic ignores — so treat it as a prior on where to
look, not a prediction. Each variant is a full 4-fold run, about an hour apiece.
""")
code(r"""
!python train_maskrcnn.py --protocol loso --iters 1500 --batch 4 --preprocess median
!python train_maskrcnn.py --protocol loso --iters 1500 --batch 4 --preprocess clahe
!python train_maskrcnn.py --protocol loso --iters 1500 --batch 4 --preprocess background
""")

md(r"""
### 7h · Mask R-CNN under the naive random split

This is the control for the data-leakage contribution: the same model and the same
fold sizes, but the split ignores specimen identity. The gap between this and 7c is
the amount a naive protocol would have overstated performance.
""")
code(r"""
!python train_maskrcnn.py --protocol random --iters 1500 --batch 4
""")

md("## 8 · Evaluate and build the Chapter 5 tables")
code(r"""
!python evaluate.py
!python make_tables.py
""")

md("## 9 · Look at the results")
code(r"""
import json, os
from pathlib import Path

RES = Path(os.environ["BEAD_RESULTS"])
m = json.loads((RES / "metrics.json").read_text())

g = m["ground_truth"]
print(f"ground truth: {g['n']} beads, median diameter {g['median']:.2f} um, "
      f"IQR [{g['iqr'][0]:.2f}, {g['iqr'][1]:.2f}]")

for protocol, methods in m.items():
    if protocol == "ground_truth":
        continue
    print(f"\n=== {protocol} ===")
    for name, r in methods.items():
        o = r["overall"]
        print(f"{name:11s} AP50 {o['ap50']['mean']:.3f}+-{o['ap50']['std']:.3f}   "
              f"AP {o['ap']['mean']:.3f}   AJI {o['aji']['mean']:.3f}   "
              f"PQ {o['pq']['mean']:.3f}   "
              f"|count err| {o['abs_counting_error']['mean']:5.1f}%   "
              f"merge {o['merge_rate']['mean']:4.1f}%   "
              f"split {o['split_rate']['mean']:4.1f}%")
""")

code(r"""
import os
from pathlib import Path

print((Path(os.environ["BEAD_RESULTS"]) / "chapter5_tables.tex").read_text())
""")

md("""
## 10 · Download everything

`thesis_results.zip` contains `metrics.json`, the generated LaTeX tables, the
figures, and — importantly — the raw per-image predictions.

The predictions are included because every metric in the thesis is a *function*
of them. Shipping them back means a new metric (a different IoU threshold, a
size-stratified AP, a counting statistic nobody has thought of yet) can be
computed on a laptop in seconds instead of costing another GPU session. They
cost well under a megabyte per method. An earlier version of this notebook left
them behind on the Colab VM, and when the session was reset the numbers could
only be recovered by retraining.
""")
code(r"""
import os, shutil
from pathlib import Path
from google.colab import files

RES = Path(os.environ["BEAD_RESULTS"])
out = Path("/content/thesis_results")
shutil.rmtree(out, ignore_errors=True)
out.mkdir()
shutil.copytree(RES, out / "results", dirs_exist_ok=True)
figs = RES.parent / "figures"
if figs.exists():
    shutil.copytree(figs, out / "figures", dirs_exist_ok=True)

# The predictions and the ground-truth COCO file: together these make every
# metric recomputable offline, with no GPU and no retraining.
work = Path(os.environ["BEAD_WORK"])
if (work / "preds").exists():
    shutil.copytree(work / "preds", out / "preds", dirs_exist_ok=True)
for extra in ("coco_gt.json", "folds_loso.json", "folds_random.json"):
    if (work / extra).exists():
        shutil.copy(work / extra, out / extra)

shutil.make_archive("/content/thesis_results", "zip", out)
print("bundled:", sorted(p.name for p in out.rglob("*") if p.is_file())[:20])
files.download("/content/thesis_results.zip")
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [],
}
for c in cells:
    src = c["source"].split("\n")
    c["source"] = [l + "\n" for l in src[:-1]] + [src[-1]]
    nb["cells"].append(c)

dest = Path(__file__).resolve().parent / "Thesis_Colab.ipynb"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", dest, len(nb["cells"]), "cells")
