#!/usr/bin/env python3
"""Build balanced synthetic/corrected-real replay using lightweight symlinks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-repeats", type=int, default=160)
    args = parser.parse_args()

    train_images = args.output / "images" / "train"
    train_labels = args.output / "labels" / "train"
    train_images.mkdir(parents=True, exist_ok=False)
    train_labels.mkdir(parents=True, exist_ok=False)

    synthetic_images = sorted((args.synthetic / "images" / "train").glob("*"))
    synthetic_count = 0
    for index, image in enumerate(synthetic_images):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        label = args.synthetic / "labels" / "train" / f"{image.stem}.txt"
        name = f"synthetic_{index:04d}_{image.name}"
        symlink(image, train_images / name)
        symlink(label, train_labels / f"{Path(name).stem}.txt")
        synthetic_count += 1

    real_images = sorted((args.real / "images" / "all").glob("*"))
    real_unique = 0
    real_entries = 0
    for image_index, image in enumerate(real_images):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        real_unique += 1
        label = args.real / "labels" / "all" / f"{image.stem}.txt"
        for repeat in range(args.real_repeats):
            name = f"physical_{image_index:02d}_{repeat:03d}_{image.name}"
            symlink(image, train_images / name)
            symlink(label, train_labels / f"{Path(name).stem}.txt")
            real_entries += 1

    # Validation and test remain untouched synthetic holdouts.
    for split in ("val", "test"):
        symlink(args.synthetic / "images" / split, args.output / "images" / split)
        symlink(args.synthetic / "labels" / split, args.output / "labels" / split)

    data_yaml = (
        f"path: {args.output.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: good_cap\n  1: crooked_cap\n  2: missing_cap\n  3: table_cap\n"
    )
    (args.output / "data.yaml").write_text(data_yaml)
    provenance = {
        "purpose": "YOLO26x mixed replay with reviewed classes and VLM-refined good/crooked real boxes",
        "synthetic_source": str(args.synthetic.resolve()),
        "corrected_real_source": str(args.real.resolve()),
        "synthetic_training_entries": synthetic_count,
        "real_unique_images": real_unique,
        "real_repetitions_per_image": args.real_repeats,
        "real_training_entries": real_entries,
        "total_training_entries": synthetic_count + real_entries,
        "validation_and_test": "untouched synthetic holdout splits",
        "warning": "all five real images are training examples; real-image scores are memorization checks",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
