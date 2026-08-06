"""Convert the LabelMe annotations into a COCO dataset with the banner removed.

Only the 11 original micrographs are used. The 132 pre-computed augmented
copies are ignored: they are correctly transformed but they are not independent
samples, and allowing them to cross a partition boundary would leak information
from a training specimen into its own test partition.

Polygons are kept as polygons rather than being flattened into a single label
map, because 8.8% of annotated bead pixels are claimed by more than one polygon.
Beads in these micrographs touch and overlap, and a label map would silently
discard that overlap -- which is exactly the region where merge and split errors
are decided.

Output
------
``$BEAD_WORK/images/<name>.png``  banner-cropped 1024x736 micrograph
``$BEAD_WORK/coco_gt.json``       COCO annotations, with specimen, magnification
                                  and pixel size recorded on each image entry
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from config import (
    BANNER_H, COCO_GT, DATA_ROOT, FULL_H, FULL_W, IMAGES, NM_PER_PX,
    PREPARED_IMAGES, WORK_H, ensure_dirs,
)


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace area of a closed polygon, in px^2.

    Computed from the vertices rather than from a rasterised mask: at 500x the
    median bead is only about 14 px across, so a one-pixel rasterisation
    boundary error would change its area by roughly 15%.
    """
    n = len(points)
    return abs(
        sum(
            points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        )
    ) / 2.0


def _bbox(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def build() -> dict:
    ensure_dirs()
    images, annotations = [], []
    ann_id = 1

    for img_id, (name, (specimen, mag)) in enumerate(IMAGES.items(), start=1):
        ann = json.loads((DATA_ROOT / "Labels" / f"{name}.json").read_text())
        if (ann["imageWidth"], ann["imageHeight"]) != (FULL_W, FULL_H):
            raise ValueError(f"{name}: unexpected size "
                             f"{ann['imageWidth']}x{ann['imageHeight']}")

        # Crop the instrument banner from the bottom. The crop is taken from the
        # bottom edge, so polygon coordinates need no adjustment; the assertion
        # below is what makes that safe rather than assumed.
        src = Image.open(DATA_ROOT / "Images" / f"{name}.jpg").convert("RGB")
        src.crop((0, 0, FULL_W, WORK_H)).save(PREPARED_IMAGES / f"{name}.png")

        for shape in ann["shapes"]:
            if shape["shape_type"] != "polygon":
                raise ValueError(f"{name}: unsupported shape {shape['shape_type']}")
            pts = [[float(x), float(y)] for x, y in shape["points"]]
            lowest = max(p[1] for p in pts)
            if lowest > WORK_H:
                raise ValueError(f"{name}: annotation extends {lowest:.1f} px "
                                 f"into the banner region (limit {WORK_H})")
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [[c for p in pts for c in p]],
                "area": polygon_area(pts),
                "bbox": _bbox(pts),
                "iscrowd": 0,
                "label": shape["label"],
            })
            ann_id += 1

        images.append({
            "id": img_id,
            "file_name": f"{name}.png",
            "name": name,
            "width": FULL_W,
            "height": WORK_H,
            "specimen": specimen,
            "magnification": mag,
            "nm_per_px": NM_PER_PX[mag],
            "um_per_px": NM_PER_PX[mag] / 1000.0,
        })

    coco = {
        "info": {"description": "Bead defects in SEM micrographs of electrospun "
                                "nanofibre mats; 11 originals, banner removed."},
        "licenses": [],
        "categories": [{"id": 1, "name": "bead", "supercategory": "defect"}],
        "images": images,
        "annotations": annotations,
    }
    COCO_GT.write_text(json.dumps(coco))
    return coco


if __name__ == "__main__":
    coco = build()
    labels = {a["label"] for a in coco["annotations"]}
    print(f"images      : {len(coco['images'])}")
    print(f"annotations : {len(coco['annotations'])}")
    print(f"labels      : {sorted(labels)}")
    print(f"written to  : {COCO_GT}")
    print(f"images in   : {PREPARED_IMAGES}")
