#!/usr/bin/env python3
"""Build reviewed real table-cap labels and a balanced YOLO replay dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Tight flange bounds reviewed against the 1920x1080 processed images. The
# yellow connected component was padded by two pixels to include its dark rim.
TABLE_CAP_BOXES = {
    "20261028_002417.png": [438, 406, 548, 510],
    "20261028_002430.png": [1168, 625, 1265, 716],
}


def symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def yolo_line(box: list[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return f"3 {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}\n"


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--prior-real", type=Path, required=True)
    parser.add_argument("--table-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-real-repeats", type=int, default=80)
    parser.add_argument("--table-repeats", type=int, default=400)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    reviewed = args.output / "reviewed_table_caps"
    dataset = args.output / "mixed_replay_dataset"
    (reviewed / "images" / "all").mkdir(parents=True)
    (reviewed / "labels" / "all").mkdir(parents=True)
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)

    review_cells = []
    records = []
    for name, box in TABLE_CAP_BOXES.items():
        source = args.table_images / name
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        symlink(source, reviewed / "images" / "all" / name)
        label = reviewed / "labels" / "all" / f"{Path(name).stem}.txt"
        label.write_text(yolo_line(box, width, height))
        rendered = image.copy()
        draw = ImageDraw.Draw(rendered)
        draw.rectangle(tuple(box), outline=(180, 40, 220), width=5)
        draw.text((box[0], max(0, box[1] - 34)), "table_cap", fill=(180, 40, 220), font=font(28))
        rendered.thumbnail((960, 540), Image.Resampling.LANCZOS)
        review_cells.append(rendered)
        records.append({"image": name, "size": [width, height], "class_id": 3,
                        "class_name": "table_cap", "box_xyxy": box})

    sheet = Image.new("RGB", (960, 540 * len(review_cells)), "white")
    for index, cell in enumerate(review_cells):
        sheet.paste(cell, (0, index * 540))
    sheet.save(reviewed / "reviewed_table_cap_boxes.png")
    (reviewed / "labels.json").write_text(json.dumps(records, indent=2) + "\n")

    synthetic_count = 0
    for index, image in enumerate(sorted((args.synthetic / "images" / "train").iterdir())):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        label = args.synthetic / "labels" / "train" / f"{image.stem}.txt"
        name = f"synthetic_{index:04d}_{image.name}"
        symlink(image, dataset / "images" / "train" / name)
        symlink(label, dataset / "labels" / "train" / f"{Path(name).stem}.txt")
        synthetic_count += 1

    prior_entries = 0
    prior_images = sorted((args.prior_real / "images" / "all").iterdir())
    for image_index, image in enumerate(prior_images):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        label = args.prior_real / "labels" / "all" / f"{image.stem}.txt"
        for repeat in range(args.prior_real_repeats):
            name = f"prior_real_{image_index:02d}_{repeat:03d}_{image.name}"
            symlink(image, dataset / "images" / "train" / name)
            symlink(label, dataset / "labels" / "train" / f"{Path(name).stem}.txt")
            prior_entries += 1

    table_entries = 0
    for image_index, image in enumerate(sorted((reviewed / "images" / "all").iterdir())):
        label = reviewed / "labels" / "all" / f"{image.stem}.txt"
        for repeat in range(args.table_repeats):
            name = f"table_real_{image_index:02d}_{repeat:03d}_{image.name}"
            symlink(image, dataset / "images" / "train" / name)
            symlink(label, dataset / "labels" / "train" / f"{Path(name).stem}.txt")
            table_entries += 1

    for split in ("val", "test"):
        symlink(args.synthetic / "images" / split, dataset / "images" / split)
        symlink(args.synthetic / "labels" / split, dataset / "labels" / split)
    (dataset / "data.yaml").write_text(
        f"path: {dataset.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: good_cap\n  1: crooked_cap\n  2: missing_cap\n  3: table_cap\n"
    )
    provenance = {
        "purpose": "YOLO26x replay fine-tune with two reviewed physical table-cap examples",
        "synthetic_source": str(args.synthetic.resolve()),
        "prior_real_source": str(args.prior_real.resolve()),
        "table_real_source": str(reviewed.resolve()),
        "synthetic_entries": synthetic_count,
        "prior_real_unique": len(prior_images),
        "prior_real_repeats": args.prior_real_repeats,
        "prior_real_entries": prior_entries,
        "table_real_unique": len(TABLE_CAP_BOXES),
        "table_real_repeats": args.table_repeats,
        "table_real_entries": table_entries,
        "total_train_entries": synthetic_count + prior_entries + table_entries,
        "validation_and_test": "untouched synthetic holdouts",
    }
    (dataset / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
