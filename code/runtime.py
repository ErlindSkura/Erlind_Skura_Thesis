"""Wall-clock instrumentation for training runs.

The supervisor asked for time per epoch, steps per epoch and time per step. Two
things about this pipeline make those quantities need defining before they can
be reported honestly.

First, training does not iterate over a fixed dataset. ``BeadDataset`` draws
random crops on demand and is asked for exactly ``iters * batch`` of them, so
there is no natural epoch boundary. An epoch is therefore defined here as one
crop-equivalent pass over the fold's training micrographs: with ``n_images``
training micrographs, each contributing ``CROPS_PER_IMAGE`` crops of area
``CROP^2`` to tile a 1024x736 frame, one epoch is ``n_images * CROPS_PER_IMAGE``
crops. This is a stated convention, not a property of the code, and it is
reported as such.

Second, CUDA kernels launch asynchronously, so timing a step without
synchronising measures how long it took to *queue* the work, not to do it. Every
step is therefore synchronised before the clock is read. The cost of that is
negligible against a step that already runs to completion, and without it the
reported numbers would be fiction.

The first step is always excluded from the averages and reported separately:
cuDNN autotuning, lazy CUDA context creation and dataloader worker startup all
land on it, and on a T4 it typically runs several times longer than a settled
step.
"""

from __future__ import annotations

import time

CROP_AREA = 512 * 512
FRAME_AREA = 1024 * 736
CROPS_PER_IMAGE = FRAME_AREA / CROP_AREA   # 2.875, a tiling equivalent


class StepTimer:
    """Collects per-step wall-clock times and reports them in the shapes asked for."""

    def __init__(self, iters: int, batch: int, n_images: int):
        self.iters = iters
        self.batch = batch
        self.n_images = n_images
        self.times: list[float] = []
        self._t0 = time.perf_counter()
        self._start = self._t0

    def tick(self) -> None:
        """Record one completed step. Call after the optimiser step."""
        _sync()
        now = time.perf_counter()
        self.times.append(now - self._t0)
        self._t0 = now

    @property
    def steps_per_epoch(self) -> float:
        return max(self.n_images * CROPS_PER_IMAGE / self.batch, 1.0)

    def summary(self) -> dict:
        """Runtime facts for this fold, in the units the supervisor asked for."""
        if not self.times:
            return {}
        total = time.perf_counter() - self._start
        first, rest = self.times[0], self.times[1:] or self.times
        rest = sorted(rest)
        n = len(rest)
        mean = sum(rest) / n
        median = rest[n // 2] if n % 2 else 0.5 * (rest[n // 2 - 1] + rest[n // 2])
        spe = self.steps_per_epoch
        return {
            "steps": len(self.times),
            "batch": self.batch,
            "train_images": self.n_images,
            "crops_seen": len(self.times) * self.batch,
            "steps_per_epoch": round(spe, 2),
            "epochs": round(len(self.times) / spe, 2),
            "s_per_step_mean": round(mean, 4),
            "s_per_step_median": round(median, 4),
            "s_per_step_p95": round(rest[min(int(0.95 * n), n - 1)], 4),
            "s_first_step": round(first, 3),
            "s_per_epoch": round(mean * spe, 2),
            "crops_per_s": round(self.batch / mean, 2),
            "train_wall_s": round(total, 1),
        }


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def device_report() -> dict:
    """What hardware actually ran this, so a reported time can be interpreted."""
    out: dict = {}
    try:
        import torch
        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["gpu"] = p.name
            out["gpu_total_mem_gb"] = round(p.total_memory / 1024 ** 3, 1)
            out["cuda"] = torch.version.cuda
        else:
            out["gpu"] = "cpu"
    except Exception as e:                                   # pragma: no cover
        out["gpu"] = f"unknown ({e})"
    return out


def format_summary(name: str, s: dict) -> str:
    if not s:
        return f"  [{name}] no timing recorded"
    return (
        f"  [{name}] {s['steps']} steps x batch {s['batch']} "
        f"= {s['crops_seen']} crops = {s['epochs']} epochs\n"
        f"      {s['s_per_step_median']:.3f} s/step (median), "
        f"{s['s_per_step_mean']:.3f} mean, {s['s_per_step_p95']:.3f} p95, "
        f"first step {s['s_first_step']:.1f} s\n"
        f"      {s['steps_per_epoch']:.2f} steps/epoch, "
        f"{s['s_per_epoch']:.1f} s/epoch, "
        f"{s['crops_per_s']:.2f} crops/s, "
        f"total {s['train_wall_s']:.0f} s"
    )
