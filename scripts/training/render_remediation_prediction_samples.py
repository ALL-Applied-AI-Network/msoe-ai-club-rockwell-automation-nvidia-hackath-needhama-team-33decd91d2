#!/usr/bin/env python3
"""Render detector predictions and ground-truth overlays for audit samples."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Running a script by path puts its directory, rather than the repository
# root, first on sys.path. Add the root so the evaluator helper is importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_remediation_comparison import (
    classify_real_slots,
    cropped_aperture_centers,
    load_real_records,
    model_names,
)


COLORS = {
    "good_cap": (40, 180, 70),
    "crooked_cap": (255, 150, 20),
    "missing_cap": (220, 50, 50),
    "table_cap": (160, 70, 210),
    "no_vial": (80, 80, 80),
}


def font(size: int = 12):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def short_label(label: str) -> str:
    return {
        "good_cap": "good",
        "crooked_cap": "crook",
        "missing_cap": "miss",
        "table_cap": "table",
        "no_vial": "none",
    }.get(label, label)


def prediction_image(model: YOLO, image_path: Path, imgsz: int = 640):
    result = model.predict(
        source=[image_path], device=0, imgsz=imgsz, conf=0.25, iou=0.7,
        max_det=200, verbose=False,
    )[0]
    rendered = Image.fromarray(result.plot(labels=True, boxes=True, conf=True)[..., ::-1]).convert("RGB")
    return result, rendered


def draw_gt_boxes(image: Image.Image, label_path: Path, names: dict[int, str]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    label_font = font(max(10, min(width, height) // 55))
    if not label_path.is_file():
        return image
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_index, cx, cy, box_w, box_h = int(fields[0]), *(float(value) for value in fields[1:])
        x1 = int((cx - box_w / 2) * width)
        y1 = int((cy - box_h / 2) * height)
        x2 = int((cx + box_w / 2) * width)
        y2 = int((cy + box_h / 2) * height)
        label = names.get(class_index, str(class_index))
        color = COLORS.get(label, (20, 180, 20))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, width // 320))
        draw.text((x1 + 2, max(0, y1 - 14)), f"GT {short_label(label)}", fill=color, font=label_font)
    return image


def draw_real_slots(image: Image.Image, detail: dict[str, Any], centers: dict[str, tuple[float, float]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    label_font = font(max(8, min(image.size) // 34))
    for slot, values in detail["slots"].items():
        x, y = centers[slot]
        truth = values["truth"]
        prediction = values["prediction"]
        correct = truth == prediction
        color = (40, 180, 70) if correct else (220, 40, 40)
        radius = max(3, min(image.size) // 55)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        text = f"{slot} {short_label(truth)}>{short_label(prediction)}"
        draw.text((x + radius + 1, y - radius - 1), text, fill=color, font=label_font)
    return image


def titled_cell(image: Image.Image, title: str, width: int = 720, height: int = 520) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((width - 12, height - 34), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(cell)
    draw.text((8, 5), title, fill="black", font=font(14))
    cell.paste(image, ((width - image.width) // 2, 30))
    return cell


def contact_sheet(cells: list[Image.Image], output: Path, columns: int = 4) -> None:
    if not cells:
        return
    rows = math.ceil(len(cells) / columns)
    width = cells[0].width * columns
    height = cells[0].height * rows
    sheet = Image.new("RGB", (width, height), "#d9d9d9")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * cell.width, (index // columns) * cell.height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def read_synthetic_samples(data_path: Path, split: str, count: int) -> tuple[list[Path], Path, dict[int, str]]:
    config = yaml.safe_load(data_path.read_text())
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    image_dir = root / str(config[split])
    names_raw = config.get("names", {})
    names = {int(key): str(value) for key, value in names_raw.items()} if isinstance(names_raw, dict) else dict(enumerate(names_raw))
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if len(images) <= count:
        selected = images
    else:
        indices = sorted(set(round(index * (len(images) - 1) / (count - 1)) for index in range(count)))
        selected = [images[index] for index in indices]
    return selected, root, names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--synthetic-data", type=Path, required=True)
    parser.add_argument("--synthetic-split", default="val")
    parser.add_argument("--synthetic-count", type=int, default=3)
    parser.add_argument("--hand-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--apertures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model_specs = []
    for value in args.models:
        name, checkpoint = value.split("=", 1)
        model_specs.append((name, Path(checkpoint)))
    records = load_real_records(args.hand_dir, args.image_dir)
    centers = cropped_aperture_centers(args.apertures)
    synthetic_images, synthetic_root, synthetic_names = read_synthetic_samples(args.synthetic_data, args.synthetic_split, args.synthetic_count)

    real_cells = []
    synthetic_cells = []
    manifest = {"real_images": [record["filename"] for record in records], "synthetic_images": [str(path) for path in synthetic_images], "models": {}}

    for model_name, checkpoint in model_specs:
        model = YOLO(str(checkpoint))
        names = model_names(model)
        _, _, real_details = classify_real_slots(
            model, records, centers, confidence=0.25,
            max_slot_distance=18.0, imgsz=args.imgsz
        )
        details_by_filename = {item["filename"]: item for item in real_details}
        manifest["models"][model_name] = {"checkpoint": str(checkpoint), "real": {}, "synthetic": {}}

        for record in records:
            result, image = prediction_image(model, record["image"], args.imgsz)
            image = draw_real_slots(image, details_by_filename[record["filename"]], centers)
            target = args.out / "real" / model_name / record["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, quality=95)
            real_cells.append(titled_cell(image, f"{model_name} | {record['filename']}"))
            manifest["models"][model_name]["real"][record["filename"]] = str(target)

        for image_path in synthetic_images:
            result, image = prediction_image(model, image_path, args.imgsz)
            label_path = synthetic_root / "labels" / args.synthetic_split / f"{image_path.stem}.txt"
            image = draw_gt_boxes(image, label_path, synthetic_names)
            target = args.out / "synthetic" / model_name / image_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, quality=95)
            synthetic_cells.append(titled_cell(image, f"{model_name} | {image_path.name}"))
            manifest["models"][model_name]["synthetic"][image_path.name] = str(target)

    contact_sheet(real_cells, args.out / "real_prediction_samples.png")
    contact_sheet(synthetic_cells, args.out / "synthetic_prediction_samples.png")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "real_images": len(records), "synthetic_images": len(synthetic_images), "models": [name for name, _ in model_specs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
