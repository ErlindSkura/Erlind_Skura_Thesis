# Bead-Defect Segmentation in SEM Images of Electrospun Nanofibre Mats

MSc thesis, Department of Computer Engineering, Epoka University.

**Author:** Erlind Skura
**Supervisor:** Assoc. Prof. Dr. Arban Uka
**Defence:** September 2026

Instance segmentation and quantification of bead defects in scanning electron
microscopy micrographs of electrospun nanofibre mats. The aim is to replace the
manual ImageJ workflow currently used to count and measure beads with an automated
pipeline that reports bead count, areal density and size distribution in physical
units.

## Layout

```
chapters/        thesis chapters (chapter1-6, appendixA)
figures/         figures generated from the dataset
code/            analysis code
epoka.cls        Epoka University thesis class
metadata.tex     title, abstracts, committee, abbreviations
references.bib   bibliography
thesis.tex       main document
```

The repository root is a valid Overleaf project: zip it and upload, or compile
locally.

## Building

```bash
pdflatex thesis && bibtex thesis && pdflatex thesis && pdflatex thesis
```

Produces a 54-page PDF. Build artefacts are gitignored.

## Dataset

Not included in this repository — the micrographs are held separately.

| | |
|---|---|
| Micrographs | 11 originals (the 143 supplied files are 11 × 13 augmented variants) |
| Specimens | 4 (Z2, Z4, Z5, Z6) × 3 magnifications (500×, 1000×, 3000×); Z4-1 absent |
| Annotations | 606 LabelMe polygons, single class `bead` |
| Image size | 1024 × 768, 8-bit greyscale; bottom 32 px are a burned-in instrument banner |
| Pixel size | 555.6 / 277.8 / 92.6 nm per pixel at 500× / 1000× / 3000× |

Run `python code/dataset_stats.py` to regenerate the statistics and the dataset
figures.

## Notes on method

Two decisions follow from the structure of the data rather than from convention:

**The specimen is the unit of partitioning.** Eleven micrographs derive from four
specimens, so evaluation uses leave-one-specimen-out cross-validation. A split over
images would place different magnifications of the same specimen on both sides.

**The supplied augmentations are not used.** They were generated before
partitioning, so a random split over the 143 files would put transformed copies of
one micrograph in both training and test. Augmentation is applied on the fly after
partitioning instead.

## Status

Chapters 1–3 are complete. Chapters 4–6 are scaffolds: the result tables are
deliberately empty and are to be filled only from executed runs.

Open questions for the supervisor are tracked in `pyetje_per_supervizorin.md`.
