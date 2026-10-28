#!/usr/bin/env python3
"""Crop phone table-cap samples to TableCam aspect and compare YOLO26x models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.refine_yolo_boxes_with_vlm import (  # noqa: E402
    CLASS_NAMES,
    draw_detections,
    get_font,
    panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--operational-confidence", type=float, default=0.40)
    parser.add_argument("--display-confidence", type=float, default=0.05)
    parser.add_argument("--crop-top-fraction", type=float, default=0.23)
    return parser.parse_args()


def landscape_crop(image: Image.Image, top_fraction: float) -> tuple[Image.Image, list[int]]:
    """Use full phone width and a 16:9 crop positioned over the paper sheets."""
    width, height = image.size
    crop_height = round(width * 9 / 16)
    top = min(max(0, round(height * top_fraction)), height - crop_height)
    box = [0, top, width, top + crop_height]
    return image.crop(tuple(box)), box


def detections_from_result(result: Any) -> list[dict[str, Any]]:
    output = []
    if result.boxes is None:
        return output
    for box, class_id, confidence in zip(
        result.boxes.xyxy.detach().cpu().tolist(),
        result.boxes.cls.detach().cpu().tolist(),
        result.boxes.conf.detach().cpu().tolist(),
    ):
        class_id = int(class_id)
        output.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(confidence),
            "box_xyxy": [float(value) for value in box],
            "box": [float(value) for value in box],
            "center": [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)],
        })
    return output


def status_banner(image: Image.Image, model_name: str, detections: list[dict[str, Any]], threshold: float) -> Image.Image:
    operational = [item for item in detections if item["confidence"] >= threshold]
    table_caps = [item for item in operational if item["class_name"] == "table_cap"]
    if len(table_caps) == 1 and len(operational) == 1:
        status = f"PASS: one table_cap at {table_caps[0]['confidence']:.3f}"
        color = (20, 125, 45)
    else:
        status = f"CHECK: operational={len(operational)}, table_cap={len(table_caps)}"
        color = (180, 45, 35)
    canvas = Image.new("RGB", (image.width, image.height + 42), "white")
    canvas.paste(image, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{model_name} — {status}", fill=color, font=get_font(20))
    return canvas


def contact_sheet(rows: list[Image.Image], output: Path) -> None:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    sheet = Image.new("RGB", (width, height), "#c8c8c8")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output, quality=95)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    image_paths = sorted(args.input.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images under {args.input}")

    models = {}
    for spec in args.models:
        name, checkpoint = spec.split("=", 1)
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        models[name] = YOLO(str(checkpoint_path))

    payload: dict[str, Any] = {
        "preprocessing": {
            "crop": "full source width, 16:9 height, top at 23% of source height",
            "resize": [1920, 1080],
            "yolo_imgsz": args.imgsz,
        },
        "operational_confidence": args.operational_confidence,
        "display_confidence": args.display_confidence,
        "models": {name: str(Path(spec.split("=", 1)[1]).resolve()) for name, spec in zip(models, args.models)},
        "images": [],
    }
    rows = []
    for image_path in image_paths:
        source = Image.open(image_path).convert("RGB")
        cropped, crop_box = landscape_crop(source, args.crop_top_fraction)
        processed = cropped.resize((1920, 1080), Image.Resampling.LANCZOS)
        processed_path = args.output / "processed" / f"{image_path.stem}.png"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(processed_path)

        row_cells = [panel(processed, f"Processed input — {image_path.name}", (720, 450))]
        record: dict[str, Any] = {
            "source": str(image_path.resolve()),
            "source_size": list(source.size),
            "crop_box_xyxy": crop_box,
            "processed": str(processed_path.resolve()),
            "processed_size": [1920, 1080],
            "models": {},
        }
        for model_name, model in models.items():
            result = model.predict(
                source=str(processed_path), imgsz=args.imgsz, device=0,
                conf=0.001, iou=0.70, max_det=100, verbose=False,
            )[0]
            detected = detections_from_result(result)
            visible = [item for item in detected if item["confidence"] >= args.display_confidence]
            operational = [item for item in detected if item["confidence"] >= args.operational_confidence]
            rendered = draw_detections(processed, visible, True)
            rendered = status_banner(rendered, model_name, detected, args.operational_confidence)
            prediction_path = args.output / "predictions" / model_name / f"{image_path.stem}.png"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(prediction_path)
            row_cells.append(panel(rendered, model_name, (720, 450)))
            record["models"][model_name] = {
                "all_detections": [{key: value for key, value in item.items() if key != "box"} for item in detected],
                "operational_detections": [{key: value for key, value in item.items() if key != "box"} for item in operational],
                "table_cap_detected": sum(item["class_name"] == "table_cap" for item in operational),
                "prediction_image": str(prediction_path.resolve()),
            }

        row = Image.new("RGB", (720 * len(row_cells), 450), "#c8c8c8")
        for index, cell in enumerate(row_cells):
            row.paste(cell, (720 * index, 0))
        rows.append(row)
        payload["images"].append(record)

    contact_sheet(rows, args.output / "table_cap_model_comparison.png")
    (args.output / "predictions.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
