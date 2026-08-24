"""Save one trained model per fold, so a later question does not cost a re-run.

The absence of this module cost two and a half hours. When the supervisor asked
for training accuracy against test accuracy, every model that could have answered
it had already been freed: the pipeline wrote predictions and threw the weights
away. Both region-based detectors had to be retrained from scratch, and because
no cuDNN determinism is set, the re-run moved numbers the results chapter had
already quoted -- so a question about the models turned into an edit across forty
sentences.

A checkpoint holds the weights *and* the operating point. A state dict on its own
cannot reproduce a single number in this thesis, because every reported figure
comes from predictions filtered at a score threshold that was itself selected on
that fold's training partition. Saving the weights without the threshold would
store something that looks like a model and answers nothing.

Written under ``WORK`` rather than ``SCRATCH``, deliberately and against the rule
that governs the YOLO export. The rule exists because Ultralytics rewrites
``last.pt`` after every epoch, and 750 epochs of 23 MB writes onto a mounted
Drive is pathological. This writes once per fold. The whole point is that it
survives the runtime being recycled, which is exactly what ``SCRATCH`` does not
do.

Size is the reason this is opt-out rather than mandatory: a Mask R-CNN
ResNet-50-FPN state dict is about 170 MB, so one four-fold run is roughly 700 MB
on the user's Drive. Every script therefore takes ``--no-checkpoints``.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from config import CHECKPOINTS


def name(method: str, protocol: str, fold: str) -> str:
    """The stem identifying one fold's model.

    Kept in one place because ``load`` has to be able to find what ``save``
    wrote, and a naming convention duplicated across four training scripts is a
    naming convention that will drift.
    """
    return f"{method}_{protocol}_{fold}"


def path(method: str, protocol: str, fold: str) -> Path:
    return CHECKPOINTS / f"{name(method, protocol, fold)}.pt"


def save(model, method: str, protocol: str, fold: str, *,
         threshold: float | None = None, extra: dict | None = None,
         enabled: bool = True) -> Path | None:
    """Write one fold's weights and the operating point that goes with them.

    ``threshold`` is the score cut selected on this fold's training partition.
    ``extra`` carries whatever else is needed to reconstruct the input pipeline
    -- the preprocessing variant, the magnification factor -- since a model
    trained on 3x bicubic input is not interchangeable with one trained on
    native resolution even though their state dicts have identical shapes.

    Returns the path written, or None when checkpointing is off.
    """
    if not enabled:
        return None

    import torch

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    p = path(method, protocol, fold)
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "method": method,
        "protocol": protocol,
        "fold": fold,
        "threshold": threshold,
        "torch_version": torch.__version__,
    }
    payload.update(extra or {})
    torch.save(payload, p)

    # A sidecar in JSON so the run can be identified without loading 170 MB of
    # weights, and without torch installed at all.
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    meta["bytes"] = p.stat().st_size
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"    checkpoint: {p.name}  ({p.stat().st_size / 1e6:.0f} MB)")
    return p


def save_file(src, method: str, protocol: str, fold: str, *,
              threshold: float | None = None, extra: dict | None = None,
              enabled: bool = True) -> Path | None:
    """Copy a checkpoint that another framework wrote, into the same place.

    Ultralytics serialises its own self-contained ``.pt`` and reloading it
    through ``YOLO(path)`` needs that file rather than a bare state dict, so it
    is copied verbatim instead of being unpacked and rewritten. It lands under
    ``WORK`` for the reason the module docstring gives: the directory it was
    written to is temporary and does not survive the runtime.
    """
    if not enabled:
        return None

    import shutil

    src = Path(src)
    if not src.exists():
        print(f"    no weights at {src}; checkpoint skipped")
        return None

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    dst = path(method, protocol, fold)
    shutil.copy2(src, dst)

    meta = {"method": method, "protocol": protocol, "fold": fold,
            "threshold": threshold, "source": str(src),
            "bytes": dst.stat().st_size}
    meta.update(extra or {})
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"    checkpoint: {dst.name}  ({dst.stat().st_size / 1e6:.0f} MB)")
    return dst


def load(model, method: str, protocol: str, fold: str, *, device=None) -> dict:
    """Restore one fold's weights into ``model`` and return everything else.

    The model passed in must have been built by the same factory that built the
    one saved -- this restores weights, it does not reconstruct an architecture.
    The returned dict carries the threshold and the input-pipeline settings, and
    the caller needs all of them to reproduce the fold's reported numbers.
    """
    import torch

    p = path(method, protocol, fold)
    if not p.exists():
        raise FileNotFoundError(
            f"no checkpoint at {p}. Either the run predates checkpointing or it "
            f"was made with --no-checkpoints; the model has to be retrained.")
    payload = torch.load(p, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return {k: v for k, v in payload.items() if k != "state_dict"}


def available() -> list[dict]:
    """Every checkpoint on disk, read from the sidecars rather than the weights.

    An unreadable sidecar is skipped rather than failing the listing: the point
    of this function is to answer "what do I already have", and one corrupt file
    should not hide the rest.
    """
    out = []
    for j in sorted(CHECKPOINTS.glob("*.json")):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            out.append(json.loads(j.read_text()))
    return out
