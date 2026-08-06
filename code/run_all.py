"""Run the whole pipeline end to end.

    python run_all.py --smoke          # a few iterations, to check nothing crashes
    python run_all.py                  # the real run

The smoke run exists because the expensive mistake is discovering a bug after
two hours of GPU time. Its numbers are meaningless and must never be reported.
"""

from __future__ import annotations

import argparse
import time

import classical
import evaluate
import folds as folds_mod
import make_tables
import prepare_data
import train_maskrcnn
import train_unet
from config import RESULTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify the pipeline; results are not usable")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--maskrcnn-batch", type=int, default=4)
    ap.add_argument("--unet-batch", type=int, default=8)
    ap.add_argument("--skip-random", action="store_true",
                    help="skip the naive-split control")
    a = ap.parse_args()

    iters = 8 if a.smoke else a.iters
    t0 = time.time()

    print("=" * 70, "\n1/6  preparing data", flush=True)
    coco = prepare_data.build()
    print(f"     {len(coco['images'])} images, {len(coco['annotations'])} beads")

    print("=" * 70, "\n2/6  building fold manifests", flush=True)
    for protocol, f in folds_mod.build().items():
        print(f"     {protocol}: {[x['name'] for x in f]}")

    print("=" * 70, "\n3/6  classical baseline (leave-one-specimen-out)", flush=True)
    classical.run("loso")

    print("=" * 70, "\n4/6  U-Net baseline (leave-one-specimen-out)", flush=True)
    train_unet.run("loso", iters=iters, batch=a.unet_batch, lr=3e-4)

    print("=" * 70, "\n5/6  Mask R-CNN (leave-one-specimen-out)", flush=True)
    train_maskrcnn.run("loso", iters=iters, batch=a.maskrcnn_batch, lr=0.005)

    protocols = ["loso"]
    if not a.skip_random:
        print("=" * 70, "\n5b/6 Mask R-CNN under the naive random split (control)",
              flush=True)
        train_maskrcnn.run("random", iters=iters, batch=a.maskrcnn_batch, lr=0.005)
        protocols.append("random")

    print("=" * 70, "\n6/6  evaluating and tabulating", flush=True)
    evaluate.run(tuple(protocols))
    make_tables.run()

    mins = (time.time() - t0) / 60
    print("=" * 70)
    print(f"done in {mins:.1f} min")
    print(f"metrics : {RESULTS / 'metrics.json'}")
    print(f"tables  : {RESULTS / 'chapter5_tables.tex'}")
    if a.smoke:
        print("\nTHIS WAS A SMOKE RUN. The numbers are meaningless -- "
              "re-run without --smoke.")


if __name__ == "__main__":
    main()
