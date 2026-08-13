# Deep Learning for Instance Segmentation and Quantification of Bead Defects in SEM Images of Electrospun Nanofibre Mats

MSc thesis, Epoka University, Department of Computer Engineering.
Supervisor: Assoc. Prof. Dr. Arban Uka. Defence: September 2026.

## What this is

Bead defects in electrospun nanofibre mats are currently quantified by hand: an
operator outlines each bead in ImageJ. This work automates that measurement and
reports the quantities the laboratory actually uses — bead count, areal density and
the size distribution in micrometres — rather than detection scores alone.

## The dataset

11 SEM micrographs of 4 specimens (Z2, Z4, Z5, Z6) at 500×, 1000× and 3000×,
carrying **606 manually delineated beads**. Z4 has no 500× micrograph.

| Magnification | nm/px | Field width | Micrographs | Beads |
|---|---|---|---|---|
| 500× | 555.6 | 568.9 µm | 3 | 325 |
| 1000× | 277.8 | 284.4 µm | 4 | 229 |
| 3000× | 92.6 | 94.8 µm | 4 | 52 |

The images themselves are not in this repository.

### Two decisions that shape everything downstream

**The 143 supplied pairs are 11 originals and 132 pre-computed augmented copies.**
The copies are correctly transformed, but they are not independent samples. Letting
one cross a partition boundary would put a transformed version of a test image into
training. They are discarded, and augmentation is applied on the fly after
partitioning instead.

**The 11 micrographs come from only 4 specimens.** The specimen, not the image, is
the unit of partitioning, so evaluation uses leave-one-specimen-out
cross-validation. A random split over images is also run as a deliberate control,
with identical fold sizes, to measure how much a naive protocol would overstate
performance.

## Layout

```
thesis.tex, metadata.tex, epoka.cls   LaTeX entry point (Overleaf-compatible, flat)
chapters/                             chapters 1-6 and appendix A
figures/                              generated figures
references.bib                        42 references
code/                                 the pipeline (see below)
notebooks/Thesis_Colab.ipynb          runs the training stages on Colab
MANUAL.md                             execution order, and what each run puts in the thesis
```

## The pipeline

| File | Role |
|---|---|
| `config.py` | dataset facts, calibration constants, paths (all overridable by env var) |
| `prepare_data.py` | banner cropping, LabelMe → COCO |
| `folds.py` | fold manifests for both protocols |
| `data_io.py` | annotation loading, rasterisation, augmentation geometry — **no torch** |
| `datasets.py`, `models.py` | PyTorch dataset; Mask R-CNN, Faster R-CNN and U-Net |
| `preprocess.py` | the four input variants compared as an experimental factor |
| `runtime.py` | step, epoch and wall-clock timing, device-synchronised |
| `classical.py` | Otsu + watershed baseline |
| `train_unet.py` | U-Net encoder–decoder, semantic, plus connected components |
| `train_maskrcnn.py` | Mask R-CNN, and the preprocessing ablation via `--preprocess` |
| `train_fasterrcnn.py` | Faster R-CNN, detection only, no mask branch |
| `train_yolo.py` | both YOLO families, selected by `--weights` |
| `predio.py` | shared prediction format (COCO RLE) for every method |
| `metrics.py`, `evaluate.py` | AP panel, AJI, PQ, counting, merge/split, physical units |
| `make_tables.py` | generates the Chapter 5 tables and two figures from the metrics |
| `make_figures.py` | the figures that do not depend on a run, plus the counting figure |
| `run_all.py` | end to end, with `--smoke` for a fast correctness check |
| `tests.py` | 20 checks on geometry, annotations and preprocessing invariants |
| `dataset_stats.py` | the descriptive figures of Chapter 3 |
| `make_colab_bundle.py` | packages the 11 originals for upload |

The execution order, and which run is needed for each table in the thesis, is in
[`MANUAL.md`](MANUAL.md).

### Running it

No GPU is available locally, so the training stages run on Google Colab:

```bash
python code/make_colab_bundle.py     # -> bead_data.zip, upload to Drive
```

then open `notebooks/Thesis_Colab.ipynb` in Colab, select a T4 GPU, and run the
cells in order. Each stage writes its predictions to Drive and can be resumed on
its own, so a disconnect costs one stage rather than the whole run.

Locally, without a GPU, the parts that do not train still run:

```bash
cd code
python prepare_data.py && python folds.py
python classical.py --protocol loso
python evaluate.py && python make_tables.py
```

## Two things worth knowing about the implementation

**Overlap is preserved.** 8.8% of annotated bead pixels are claimed by more than
one polygon — beads touch. Instances are therefore never flattened into a label
map, which would discard the overlap exactly where merge and split errors are
decided.

**Augmentation transforms vertices, not rasters.** Reflection of a continuous
coordinate is `x → W - x`, not the `x → W - 1 - x` that is correct for a pixel
index. Half a pixel is nothing on a large object; on a 17-pixel bead it is 13% of
its area. The two conventions are distinguished by a check that compares the
rasterised transformed polygon against the directly transformed raster mask:
IoU 0.9995 against 0.87.

## Results

Chapter 5 is generated by `make_tables.py` from `results/metrics.json`. No number
enters the thesis without a run having produced it; the tables ship empty until
then.
