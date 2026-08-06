"""Package just the 11 original micrographs for upload to Colab.

The supplied dataset is 143 image--annotation pairs, but 132 of them are
pre-computed augmented copies that this work does not use. Uploading only the
originals keeps the bundle small and makes it impossible to accidentally train
on a copy.

    python make_colab_bundle.py [output.zip]
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from config import DATA_ROOT, IMAGES

DEFAULT_OUT = Path.home() / "Desktop" / "bead_data.zip"


def build(out: Path = DEFAULT_OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in IMAGES:
            z.write(DATA_ROOT / "Images" / f"{name}.jpg",
                    f"Segmentations/Images/{name}.jpg")
            z.write(DATA_ROOT / "Labels" / f"{name}.json",
                    f"Segmentations/Labels/{name}.json")
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    path = build(dest)
    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path}  ({size_mb:.1f} MB, {2 * len(IMAGES)} files)")
    print("Upload this to Google Drive, then run the Colab notebook.")
