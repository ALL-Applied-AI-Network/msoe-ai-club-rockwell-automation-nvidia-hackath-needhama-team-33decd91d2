#!/usr/bin/env python3
"""Build a one-class real-only proposal dataset with missing-cap context tiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

from PIL import Image


EXPECTED_CLASSES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")
MISSING_CLASS_ID = 2
CONTEXT_SCALES = (6.0, 10.0, 14.0)


def read_labels(path: Path, width: int, height: int):
    labels = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five fields")
        class_id = int(fields[0])
        cx, cy, bw, bh = (float(value) for value in fields[1:])
        if not 0 <= class_id < len(EXPECTED_CLASSES):
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        labels.append((class_id, ((cx - bw / 2) * width, (cy - bh / 2) * height,
                                  (cx + bw / 2) * width, (cy + bh / 2) * height)))
    return labels


def yolo_line(box, tile):
    left, top, right, bottom = tile
    x1, y1, x2, y2 = box
    x1, x2 = max(left, x1), min(right, x2)
    y1, y2 = max(top, y1), min(bottom, y2)
    width, height = right - left, bottom - top
    return (
        f"0 {((x1 + x2) / 2 - left) / width:.8f} {((y1 + y2) / 2 - top) / height:.8f} "
        f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
    )


def context_tile(target_box, image_size, scale):
    x1, y1, x2, y2 = target_box
    object_size = max(x2 - x1, y2 - y1)
    side = min(float(min(image_size)), max(160.0, object_size * scale))
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2 + side * 0.12
    left = min(max(0.0, center_x - side / 2), image_size[0] - side)
    top = min(max(0.0, center_y - side / 2), image_size[1] - side)
    return (left, top, left + side, top + side)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {args.output}")
    classes = tuple(line.strip() for line in (args.export / "classes.txt").read_text().splitlines() if line.strip())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES}, received {classes}")
    image_out = args.output / "images" / "train"
    label_out = args.output / "labels" / "train"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)

    full_images = 0
    context_tiles = 0
    missing_instances = 0
    total_labels = 0
    for image_path in sorted((args.export / "images").glob("*.png")):
        label_path = args.export / "labels" / f"{image_path.stem}.txt"
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        labels = read_labels(label_path, *image.size)
        shutil.copy2(image_path, image_out / image_path.name)
        (label_out / label_path.name).write_text(
            "\n".join(yolo_line(box, (0.0, 0.0, float(image.width), float(image.height))) for _, box in labels) + "\n"
        )
        full_images += 1
        total_labels += len(labels)

        for missing_index, (class_id, target_box) in enumerate(labels):
            if class_id != MISSING_CLASS_ID:
                continue
            missing_instances += 1
            for scale in CONTEXT_SCALES:
                tile = context_tile(target_box, image.size, scale)
                left, top, right, bottom = tile
                raster = (math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom))
                tile_image = image.crop(raster)
                label_tile = tuple(float(value) for value in raster)
                tile_labels = []
                for _, box in labels:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if left <= cx <= right and top <= cy <= bottom:
                        tile_labels.append(yolo_line(box, label_tile))
                stem = f"{image_path.stem}__missing{missing_index:02d}__context{int(scale):02d}x"
                tile_image.save(image_out / f"{stem}.png")
                (label_out / f"{stem}.txt").write_text("\n".join(tile_labels) + "\n")
                context_tiles += 1
                total_labels += len(tile_labels)

    if missing_instances != 4:
        raise ValueError(f"Expected four missing-cap examples, found {missing_instances}")
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/train\n"
        "names:\n  0: cap_or_vial_top\n"
    )
    provenance = {
        "schema_version": 1,
        "purpose": "One-class real-only proposal detector with missing-cap context oversampling",
        "source_export": str(args.export.resolve()),
        "class_mapping": {name: "cap_or_vial_top" for name in EXPECTED_CLASSES},
        "full_images": full_images,
        "missing_cap_source_instances": missing_instances,
        "context_scales": list(CONTEXT_SCALES),
        "missing_context_tiles": context_tiles,
        "training_images_total": full_images + context_tiles,
        "training_box_instances_total": total_labels,
        "synthetic_pixels_used": False,
        "validation_warning": "Training images are reused for diagnostic validation.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
