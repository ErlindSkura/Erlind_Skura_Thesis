"""Partition manifests for the two evaluation protocols.

``loso``   leave-one-specimen-out: all micrographs of a specimen move together,
           so no specimen is ever in both partitions. Four folds.
``random`` the deliberately flawed control: the 11 micrographs are split at
           random, ignoring specimen identity, so the same specimen can appear
           at one magnification in training and at another in testing.

The random control uses the same four test-set sizes as the specimen-wise
protocol (3, 2, 3, 3), so the two differ only in what the split respects and
not in how much data each fold sees. The manifests are written to disk rather
than recomputed at training time, so that every method provably sees identical
partitions.
"""

from __future__ import annotations

import json
import random

from config import IMAGES, SEED, SPECIMENS, WORK, ensure_dirs


def loso_folds() -> list[dict]:
    folds = []
    for sp in SPECIMENS:
        test = [n for n, (s, _) in IMAGES.items() if s == sp]
        train = [n for n in IMAGES if n not in test]
        folds.append({"name": sp, "held_out_specimen": sp,
                      "test": sorted(test), "train": sorted(train)})
    return folds


def random_folds(seed: int = SEED) -> list[dict]:
    names = sorted(IMAGES)
    rng = random.Random(seed)
    rng.shuffle(names)
    sizes = [len(f["test"]) for f in loso_folds()]  # 3, 2, 3, 3
    folds, cut = [], 0
    for i, k in enumerate(sizes, start=1):
        test = sorted(names[cut:cut + k])
        cut += k
        folds.append({"name": f"R{i}", "held_out_specimen": None,
                      "test": test, "train": sorted(n for n in IMAGES if n not in test)})
    return folds


def build() -> dict[str, list[dict]]:
    ensure_dirs()
    out = {"loso": loso_folds(), "random": random_folds()}
    for protocol, folds in out.items():
        (WORK / f"folds_{protocol}.json").write_text(json.dumps(folds, indent=2))
    return out


def load(protocol: str) -> list[dict]:
    path = WORK / f"folds_{protocol}.json"
    if not path.exists():
        build()
    return json.loads(path.read_text())


if __name__ == "__main__":
    for protocol, folds in build().items():
        print(f"\n{protocol}:")
        for f in folds:
            leaks = {IMAGES[n][0] for n in f["test"]} & {IMAGES[n][0] for n in f["train"]}
            note = f"  <-- specimen(s) {sorted(leaks)} on both sides" if leaks else ""
            print(f"  {f['name']:4} test={f['test']}{note}")
