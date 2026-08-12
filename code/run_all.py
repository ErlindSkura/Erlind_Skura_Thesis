"""Run the whole pipeline end to end.

    python run_all.py --smoke          # a few iterations, to check nothing crashes
    python run_all.py                  # the real run

The smoke run exists because the expensive mistake is discovering a bug after two
hours of GPU time.

A smoke run writes to a *separate* work and results directory. It has to: its
predictions come from eight training iterations and are meaningless, and an
earlier version of this script wrote them to the same place as the real ones.
Re-running the smoke cell after a real run therefore destroyed four hours of
results in about ninety seconds, silently, because both runs produce a
perfectly well-formed predictions file. Nothing warns you; the numbers just
become wrong. Keeping the two apart is the only reliable fix.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run in a separate directory; results are not usable")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--maskrcnn-batch", type=int, default=4)
    ap.add_argument("--skip-fasterrcnn", action="store_true",
                    help="omit the detection-only comparison")
    ap.add_argument("--unet-batch", type=int, default=8)
    ap.add_argument("--skip-random", action="store_true",
                    help="skip the naive-split control")
    return ap.parse_args()


def _redirect_smoke_paths() -> tuple[Path, Path]:
    """Point BEAD_WORK and BEAD_RESULTS at smoke-only directories.

    Done before the pipeline modules are imported, because config.py resolves
    every path from the environment at import time.
    """
    repo = Path(__file__).resolve().parent.parent
    work = Path(os.environ.get("BEAD_WORK", repo / "work"))
    results = Path(os.environ.get("BEAD_RESULTS", repo / "results"))

    smoke_work = work.with_name(work.name + "_smoke")
    smoke_results = results.with_name(results.name + "_smoke")
    os.environ["BEAD_WORK"] = str(smoke_work)
    os.environ["BEAD_RESULTS"] = str(smoke_results)
    return smoke_work, smoke_results


def main() -> None:
    a = _parse()

    if a.smoke:
        work, results = _redirect_smoke_paths()
        print("=" * 70)
        print("SMOKE RUN. Eight iterations per fold; the numbers are meaningless.")
        print(f"Writing to {work}")
        print(f"        and {results}")
        print("Any real run's predictions are left untouched.")
        print("=" * 70, flush=True)

    # Imported only now: config.py reads the environment when it is imported, so
    # a smoke run must have redirected the paths before this point.
    import classical
    import evaluate
    import folds as folds_mod
    import make_tables
    import prepare_data
    import train_fasterrcnn
    import train_maskrcnn
    import train_unet
    from config import RESULTS, WORK

    iters = 8 if a.smoke else a.iters
    t0 = time.time()

    print("=" * 70, "\n1/6  preparing data", flush=True)
    coco = prepare_data.build()
    print(f"     {len(coco['images'])} images, {len(coco['annotations'])} beads")

    print("=" * 70, "\n2/6  building fold manifests", flush=True)
    for protocol, f in folds_mod.build().items():
        print(f"     {protocol}: {[x['name'] for x in f]}")

    print("=" * 70, "\n3/6  classical baseline (leave-one-specimen-out)", flush=True)
    classical.run("loso", quick=a.smoke)

    print("=" * 70, "\n4/6  U-Net baseline (leave-one-specimen-out)", flush=True)
    train_unet.run("loso", iters=iters, batch=a.unet_batch, lr=3e-4)

    print("=" * 70, "\n5/7  Mask R-CNN (leave-one-specimen-out)", flush=True)
    train_maskrcnn.run("loso", iters=iters, batch=a.maskrcnn_batch, lr=0.005)

    if not a.skip_fasterrcnn:
        print("=" * 70, "\n6/7  Faster R-CNN, detection only (leave-one-specimen-out)",
              flush=True)
        train_fasterrcnn.run("loso", iters=iters, batch=a.maskrcnn_batch, lr=0.005)

    protocols = ["loso"]
    if not a.skip_random:
        print("=" * 70, "\n6b/7 Mask R-CNN under the naive random split (control)",
              flush=True)
        train_maskrcnn.run("random", iters=iters, batch=a.maskrcnn_batch, lr=0.005)
        protocols.append("random")

    print("=" * 70, "\n7/7  evaluating and tabulating", flush=True)
    evaluate.run(tuple(protocols))
    make_tables.run()

    print("=" * 70)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    print(f"work    : {WORK}")
    print(f"metrics : {RESULTS / 'metrics.json'}")
    print(f"tables  : {RESULTS}")
    if a.smoke:
        print("\nTHIS WAS A SMOKE RUN in a separate directory. The numbers are "
              "meaningless.\nRe-run without --smoke for results you can report.")


if __name__ == "__main__":
    main()
