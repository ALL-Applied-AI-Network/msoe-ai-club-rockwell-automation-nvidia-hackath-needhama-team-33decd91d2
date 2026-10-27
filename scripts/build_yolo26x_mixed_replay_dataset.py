#!/usr/bin/env python3
"""Build a balanced synthetic/physical replay dataset for YOLO26x.

The five reviewed physical frames are repeated with distinct symlink names so
Ultralytics sees an equal number of synthetic and physical training samples.
The validation and test splits continue to point at the untouched synthetic
splits. This is replay training, not an independent physical evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-total", type=int, default=800)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Mixed replay dataset already exists: {args.out}")
    if args.real_total <= 0:
        parser.error("--real-total must be positive")

    synthetic_images = sorted(
        path for path in (args.synthetic / "images" / "train").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    real_images = sorted(
        path for path in (args.real / "images" / "all").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not synthetic_images or not real_images:
        raise RuntimeError("Both source datasets must contain training images")
    if args.real_total % len(real_images):
        raise ValueError("--real-total must be divisible by the number of physical images")

    image_out = args.out / "images" / "train"
    label_out = args.out / "labels" / "train"
    for index, image in enumerate(synthetic_images):
        label = args.synthetic / "labels" / "train" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        stem = f"synthetic_{index:04d}_{image.stem}"
        relative_symlink(image, image_out / f"{stem}{image.suffix.lower()}")
        relative_symlink(label, label_out / f"{stem}.txt")

    repetitions = args.real_total // len(real_images)
    for image_index, image in enumerate(real_images):
        label = args.real / "labels" / "all" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        for repetition in range(repetitions):
            stem = f"physical_{image_index:02d}_{repetition:03d}_{image.stem}"
            relative_symlink(image, image_out / f"{stem}{image.suffix.lower()}")
            relative_symlink(label, label_out / f"{stem}.txt")

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": str((args.synthetic / "images" / "val").resolve()),
        "test": str((args.synthetic / "images" / "test").resolve()),
        "names": {
            0: "good_cap",
            1: "crooked_cap",
            2: "missing_cap",
            3: "table_cap",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (args.out / "provenance.json").write_text(json.dumps({
        "purpose": "balanced synthetic/physical replay after physical-only adaptation",
        "synthetic_source": str(args.synthetic.resolve()),
        "physical_source": str(args.real.resolve()),
        "synthetic_unique_training_images": len(synthetic_images),
        "physical_unique_training_images": len(real_images),
        "physical_repetitions_per_image": repetitions,
        "physical_training_entries": args.real_total,
        "total_training_entries": len(synthetic_images) + args.real_total,
        "validation_and_test": "untouched synthetic validation/test splits",
        "physical_evaluation_warning": "the same five physical images are present in training",
    }, indent=2) + "\n")
    print(args.out / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
