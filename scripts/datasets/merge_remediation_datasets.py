#!/usr/bin/env python3
"""Merge compatible remediation YOLO datasets without filename collisions."""

import argparse
import json
import shutil
from pathlib import Path

NAMES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")


def copy_split(src: Path, dst: Path, split: str, prefix: str) -> int:
    count = 0
    for image in sorted((src / "images" / split).glob("*")):
        if not image.is_file():
            continue
        label = src / "labels" / split / f"{image.stem}.txt"
        out_image = dst / "images" / split / f"{prefix}_{image.name}"
        out_label = dst / "labels" / split / f"{prefix}_{image.stem}.txt"
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, out_image)
        shutil.copy2(label, out_label)
        count += 1
    return count


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--extra", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already contains files: {out}")
    report = {}
    report["base_train"] = copy_split(args.base, out, "train", "base")
    report["base_val"] = copy_split(args.base, out, "val", "base")
    report["extra_train"] = copy_split(args.extra, out, "train", "paper")
    report["extra_val"] = copy_split(args.extra, out, "val", "paper")
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        json.dumps({"path": str(out), "train": "images/train", "val": "images/val",
                    "names": dict(enumerate(NAMES))}, indent=2) + "\n"
    )
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
