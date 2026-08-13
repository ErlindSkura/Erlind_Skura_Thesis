# Manuali i ekzekutimit

Si ekzekutohen skedarët e `code/`, në ç'rend, dhe cili prodhon çfarë në tezë.

**Parimi:** asnjë numër nuk hyn në tezë me dorë. Tabelat e Kapitullit 5
gjenerohen nga `results/metrics.json`, dhe ai skedar prodhohet vetëm nga
ekzekutime reale. Nëse një tabelë është bosh ose me viza, do të thotë se
ekzekutimi përkatës nuk është bërë — jo se numri mungon.

---

## 1 · Ku ekzekutohet çfarë

| Vendi | Çfarë |
|---|---|
| **Lokalisht** (pa GPU) | përgatitja e të dhënave, baza klasike, vlerësimi, tabelat, figurat, kompilimi i LaTeX-it |
| **Colab** (T4) | çdo gjë që trajnon: U-Net, Mask R-CNN, Faster R-CNN, YOLO, ablacioni i preprocessing-ut |

Të tri variablat e mjedisit i vendos notebook-u vetë. Lokalisht përdoren
parazgjedhjet nga `config.py`:

```
BEAD_DATA     …/Segmentations   (Images/ dhe Labels/)
BEAD_WORK     thesis/work       (imazhet e përgatitura, folds, parashikimet)
BEAD_RESULTS  thesis/results    (metrics.json dhe tabelat .tex)
```

---

## 2 · Rendi

### Faza 0 — një herë, lokalisht

```bash
python code/make_colab_bundle.py     # -> bead_data.zip
```

Ngarko `bead_data.zip` në rrënjën e My Drive.

### Faza 1 — përgatitja (e detyrueshme para çdo trajnimi)

```bash
cd code
python prepare_data.py    # pret banderolën, LabelMe -> COCO
python folds.py           # manifestet e folds për të dy protokollet
python tests.py           # 20 kontrolle; duhen 20/20
```

**Rendi këtu është i detyruar:** `folds.py` lexon `work/coco_gt.json`, të cilin e
shkruan `prepare_data.py`. `tests.py` nuk prodhon asgjë për tezën, por ndalon një
ekzekutim disaorësh mbi kod të prishur.

### Faza 2 — metodat

Këto janë **të pavarura mes tyre**. Secila shkruan skedarin e vet të
parashikimeve në `work/preds/<protokolli>/<metoda>.json`, ndaj rendi nuk ka
rëndësi dhe një ndërprerje kushton vetëm një fazë.

```bash
python classical.py        --protocol loso
python train_unet.py       --protocol loso --iters 1500 --batch 8
python train_maskrcnn.py   --protocol loso --iters 1500 --batch 4
python train_fasterrcnn.py --protocol loso --iters 1500 --batch 4
python train_yolo.py       --protocol loso --iters 1500 --batch 4 --weights yolov8s-seg.pt
python train_yolo.py       --protocol loso --iters 1500 --batch 4 --weights yolov5su.pt
```

Ablacioni i preprocessing-ut — e njëjta arkitekturë, vetëm hyrja ndryshon:

```bash
python train_maskrcnn.py --protocol loso --iters 1500 --batch 4 --preprocess median
python train_maskrcnn.py --protocol loso --iters 1500 --batch 4 --preprocess clahe
python train_maskrcnn.py --protocol loso --iters 1500 --batch 4 --preprocess background
```

Kontrolli i rrjedhjes së të dhënave — ndarje e rastësishme në vend të LOSO:

```bash
python train_maskrcnn.py --protocol random --iters 1500 --batch 4
```

### Faza 3 — mbledhja

```bash
python evaluate.py        # -> results/metrics.json
python make_tables.py     # -> results/table_*.tex + 2 figura
```

`evaluate.py` lexon parashikimet dhe llogarit çdo metrikë. `make_tables.py`
lexon `metrics.json` dhe shkruan tabelat.

### Faza 4 — lokalisht, pas shkarkimit të `thesis_results.zip`

```bash
python code/make_figures.py     # rigjeneron fig_counting nga metrics.json i ri
pdflatex thesis && bibtex thesis && pdflatex thesis && pdflatex thesis
```

`dataset_stats.py` ekzekutohet vetëm nëse ndryshon dataseti — figurat e tij
përshkruajnë të dhënat, jo rezultatet.

---

## 3 · Çfarë duhet ekzekutuar që një gjë të hyjë në tezë

| Do në tezë | Ekzekuto | Prodhon |
|---|---|---|
| Rreshtat e metodave në krahasim | `train_*.py` → `evaluate.py` → `make_tables.py` | `table_method_comparison.tex` |
| Brezat AP_S / AP_M / AP_L | `evaluate.py` → `make_tables.py` **(pa GPU nëse `preds/` ekziston)** | `table_ap_bands.tex` |
| Gabimi i numërimit | `evaluate.py` → `make_tables.py` | `table_counting.tex` |
| Kohët e ekzekutimit | **ritrajnim** i çdo metode → `evaluate.py` → `make_tables.py` | `table_runtime.tex` |
| Ablacioni i preprocessing-ut | `train_maskrcnn.py --preprocess` × 3 | `table_preprocessing.tex` |
| Kostoja e ndarjes naive | `train_maskrcnn.py --protocol random` | `table_leakage.tex` |
| Sjellja sipas zmadhimit | `evaluate.py` → `make_tables.py` | `table_magnification.tex` |
| Madhësitë në mikrometra | `evaluate.py` → `make_tables.py` | `table_physical.tex` |
| Shembujt cilësorë | `make_tables.py` | `fig_qualitative.pdf` |
| Pajtimi i shpërndarjes së madhësive | `make_tables.py` | `fig_size_agreement.pdf` |
| Figura e numërimit | `make_figures.py` **(vetëm lokalisht)** | `fig_counting.pdf` |

Tre rreshta meritojnë vëmendje:

**Brezat e AP-së nuk kërkojnë GPU.** Çdo metrikë është funksion i parashikimeve.
Kudo ku ekziston `work/preds/`, mjafton `evaluate.py` → `make_tables.py`.
Kujdes: në këtë makinë `work/preds/loso/` mban vetëm `classical.json` — të tjerat
kanë mbetur në Drive. Pra lokalisht kjo shkurtore funksionon **vetëm pasi të
shpaketosh `preds/` nga `thesis_results.zip`**.

**Kohët kërkojnë ritrajnim.** Maten gjatë trajnimit dhe ruhen në `meta.runtime`.
Rezultatet ekzistuese janë nga para se të shtohej `runtime.py`, ndaj nuk i kanë.

**`fig_counting.pdf` nuk rigjenerohet në Colab.** Notebook-u nuk e thërret
`make_figures.py`. Pa hapin lokal, figura mbetet me tri metodat e vjetra ndërsa
tabelat kanë nëntë.

---

## 4 · Skedarët që nuk thirren kurrë direkt

Këta janë biblioteka. Kodi i tyre ekzekutohet, por përmes importimit:

| Skedari | Përdoret nga |
|---|---|
| `config.py` | të gjithë — faktet e datasetit, kalibrimi, shtigjet |
| `data_io.py` | ngarkimi i anotimeve, rasterizimi, gjeometria e augmentimit |
| `datasets.py` | çdo trajnim — këtu aplikohet `--preprocess` |
| `models.py` | Mask R-CNN, Faster R-CNN, U-Net |
| `predio.py` | formati i përbashkët i parashikimeve (COCO RLE) |
| `metrics.py` | `evaluate.py` — AP, AJI, PQ, numërimi, merge/split |
| `preprocess.py` | `datasets.py` dhe `make_figures.py` |
| `runtime.py` | çdo trajnim — matja e kohës |

---

## 5 · Kurthe

**`refusing to overwrite`** nga `evaluate.py` nuk është defekt. Është roja që
ndalon një ekzekutim të pjesshëm të fshijë rezultate që nuk i riprodhoi.
Ekzekuto fazat që mungojnë, ose `--force` nëse vërtet do t'i hedhësh.

**`no leave-one-specimen-out Mask R-CNN results to tabulate`** nga
`make_tables.py` do të thotë se `metrics.json` s'ka ende Mask R-CNN me LOSO — kjo
metodë është referenca ndaj së cilës ndërtohen tabelat.

**`make_tables.py` ka nevojë edhe për parashikimet, jo vetëm për `metrics.json`.**
Tabelat i shkruan që në fillim, por pastaj vizaton `fig_qualitative` duke lexuar
`work/preds/<protokolli>/<metoda>.json`. Nëse ndonjë mungon, tabelat dalin dhe
skripti bie me `FileNotFoundError` te figura. Ekzekutoje aty ku janë
parashikimet — në Colab, ose lokalisht pasi të shpaketosh `preds/` nga zip-i.

**YOLOv5 dhe YOLOv8** shkruajnë në skedarë të veçantë sipas familjes, ndaj
ekzekutimi i njërës nuk e prish tjetrën.

**`--folds Z2`** ekzekuton vetëm një fold si pilot; rezultatet shkojnë në një
skedar të veçantë dhe **nuk vlerësohen**. E dobishme për të matur kohën para se
të nisësh të katërt.

**Tabelat e kushtëzuara.** `table_preprocessing`, `table_runtime` dhe
`table_leakage` gjenerohen vetëm kur ekzistojnë të dhënat përkatëse, dhe
Kapitulli 5 i merr me `\IfFileExists`. Teza kompilon edhe pa to.
