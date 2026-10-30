#!/usr/bin/env python3
"""Freeze a Label Studio YOLO export as a real-only seed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


EXPECTED_CLASSES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {args.output}")
    classes = tuple(line.strip() for line in (args.export / "classes.txt").read_text().splitlines() if line.strip())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Expected class order {EXPECTED_CLASSES}, received {classes}")
    images = sorted((args.export / "images").glob("*.png"))
    if len(images) != 2:
        raise ValueError(f"Expected exactly two seed images, found {len(images)}")

    image_out = args.output / "images" / "train"
    label_out = args.output / "labels" / "train"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    counts = {name: 0 for name in classes}
    for image in images:
        label = args.export / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        shutil.copy2(image, image_out / image.name)
        shutil.copy2(label, label_out / label.name)
        for line_number, line in enumerate(label.read_text().splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label}:{line_number}: expected five fields")
            class_id = int(fields[0])
            if not 0 <= class_id < len(classes):
                raise ValueError(f"{label}:{line_number}: invalid class {class_id}")
            if any(not 0.0 <= float(value) <= 1.0 for value in fields[1:]):
                raise ValueError(f"{label}:{line_number}: coordinates outside [0,1]")
            counts[classes[class_id]] += 1

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(classes))
    )
    provenance = {
        "schema_version": 1,
        "purpose": "Real-only two-image YOLO26x seed experiment",
        "source_export": str(args.export.resolve()),
        "images": [image.name for image in images],
        "classes": list(classes),
        "object_counts": counts,
        "validation_warning": "Training images are reused for diagnostic validation; metrics do not estimate generalization.",
        "synthetic_domain_images_used": False,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
