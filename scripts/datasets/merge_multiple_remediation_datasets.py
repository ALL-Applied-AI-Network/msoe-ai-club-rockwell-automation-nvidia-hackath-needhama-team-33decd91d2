#!/usr/bin/env python3
"""Merge any number of compatible remediation YOLO training datasets."""

import argparse
import json
import shutil
from pathlib import Path


NAMES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")


def copy_split(source: Path, output: Path, split: str, prefix: str) -> int:
    count = 0
    image_dir = source / "images" / split
    for image in sorted(image_dir.glob("*")):
        if not image.is_file():
            continue
        label = source / "labels" / split / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image}: {label}")
        destination_image = output / "images" / split / f"{prefix}_{image.name}"
        destination_label = output / "labels" / split / f"{prefix}_{image.stem}.txt"
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination_image)
        shutil.copy2(label, destination_label)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True,
                        help="prefix=/path/to/dataset; repeat for each source")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already contains files: {output}")
    report = {}
    for specification in args.dataset:
        prefix, raw_path = specification.split("=", 1)
        source = Path(raw_path).resolve()
        if not (source / "data.yaml").is_file():
            raise FileNotFoundError(f"Not a YOLO dataset: {source}")
        report[prefix] = {
            split: copy_split(source, output, split, prefix)
            for split in ("train", "val", "test")
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(json.dumps({
        "path": str(output), "train": "images/train", "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(NAMES)),
    }, indent=2) + "\n")
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
