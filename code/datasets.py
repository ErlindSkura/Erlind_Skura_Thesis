"""PyTorch dataset over the bead micrographs.

The geometry lives in :mod:`data_io`; this module only turns it into tensors.
In training mode ``__len__`` is the number of random crops drawn per epoch and
has no relation to the number of source micrographs, of which a
leave-one-specimen-out training partition holds only eight.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CROP, MIN_INSTANCE_PX, SCALE_RANGE, SEED
from data_io import Record, augment, load_records, rasterise

__all__ = ["BeadDataset", "collate", "load_records", "rasterise", "Record"]


class BeadDataset(Dataset):
    """Random augmented crops when ``train``; whole micrographs otherwise."""

    def __init__(self, records: dict[str, Record], names: list[str], *,
                 train: bool, samples: int = 400, crop: int = CROP,
                 scale_range: tuple[float, float] = SCALE_RANGE, seed: int = SEED,
                 preprocess: str = "none"):
        self.records = [records[n] for n in names]
        self.train = train
        self.samples = samples
        self.crop = crop
        self.scale_range = scale_range
        self.seed = seed
        self.preprocess = preprocess
        self._cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return self.samples if self.train else len(self.records)

    def _image(self, rec: Record) -> Image.Image:
        """The micrograph, preprocessed once and cached.

        Preprocessing is applied here, to the whole micrograph before cropping and
        augmentation, for two reasons. It is where training and inference share a
        code path, so the two cannot diverge -- a model trained on equalised images
        and evaluated on raw ones would produce a plausible, wrong result. And the
        variants are deterministic functions of the image, so computing them per
        crop would repeat identical work thousands of times per fold.
        """
        if rec.name not in self._cache:
            img = Image.open(rec.path).convert("RGB")
            if self.preprocess != "none":
                import preprocess as pp
                img = pp.apply(img, self.preprocess)
            self._cache[rec.name] = img
        return self._cache[rec.name]

    def __getitem__(self, idx: int):
        if self.train:
            # Seeded per item so a run is reproducible regardless of how many
            # worker processes the DataLoader uses.
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            rec = self.records[int(rng.integers(len(self.records)))]
            img, polys = augment(self._image(rec), [p.copy() for p in rec.polys],
                                 rng, crop_size=self.crop,
                                 scale_range=self.scale_range)
        else:
            rec = self.records[idx]
            img, polys = self._image(rec), rec.polys

        w, h = img.size
        masks = rasterise(polys, w, h)
        if len(masks):
            areas = masks.reshape(len(masks), -1).sum(1)
            masks = masks[areas >= MIN_INSTANCE_PX]

        boxes, keep = [], []
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            boxes.append([x0, y0, x1, y1])
            keep.append(i)
        masks = masks[keep] if keep else np.zeros((0, h, w), np.uint8)

        image = torch.from_numpy(
            np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes_t,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": torch.from_numpy(masks.astype(np.uint8)),
            "image_id": torch.tensor(rec.image_id),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "name": rec.name,
        }
        return image, target


def collate(batch):
    return tuple(zip(*batch))
