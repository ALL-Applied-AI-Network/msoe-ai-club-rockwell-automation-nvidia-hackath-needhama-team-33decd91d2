#!/usr/bin/env python3
"""Render saved Qwen grounding responses separately and as a color-coded overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inspection.defect_training.qwen_grounding import parse_qwen_grounding_boxes  # noqa: E402


COLORS = ((40, 190, 90), (255, 190, 30), (235, 70, 75), (70, 145, 245))
SHORT_NAMES = ("vials", "mounted_or_nearby_caps", "uncapped_rims", "loose_table_caps")


def prompt_name(prompt: str, index: int) -> str:
    for class_name in ("good_cap", "crooked_cap", "missing_cap", "table_cap"):
        if prompt.strip().lower().startswith(class_name):
            return class_name
    return SHORT_NAMES[index] if index < len(SHORT_NAMES) else f"prompt_{index + 1}"


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def draw_boxes(image: Image.Image, groups, *, legend: bool) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    line_width = max(3, round(min(image.size) / 350))
    label_font = font(max(15, round(min(image.size) / 60)))
    for name, color, boxes in groups:
        for box in boxes:
            coordinates = tuple(round(value) for value in box.as_list())
            draw.rectangle(coordinates, outline=color, width=line_width)
    if legend:
        x, y, gap = 20, 20, max(28, round(min(image.size) / 32))
        for name, color, boxes in groups:
            text = f"{name}: {len(boxes)}"
            bounds = draw.textbbox((x + 28, y), text, font=label_font)
            draw.rectangle((x - 6, y - 5, bounds[2] + 7, bounds[3] + 5), fill=(0, 0, 0))
            draw.rectangle((x, y + 3, x + 18, y + 21), fill=color)
            draw.text((x + 28, y), text, fill=color, font=label_font)
            y += gap
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for record_path in sorted(args.records.glob("*.json")):
        record = json.loads(record_path.read_text())
        image_path = Path(record["image"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        groups = []
        counts = {}
        for index, prompt in enumerate(record["prompts"]):
            name = prompt_name(prompt, index)
            response = record["responses"][prompt]["raw_response"]
            boxes, _ = parse_qwen_grounding_boxes(response, image.size, args.nms_iou)
            color = COLORS[index % len(COLORS)]
            groups.append((name, color, boxes))
            counts[name] = len(boxes)
            prompt_dir = args.output / "by_prompt" / name
            prompt_dir.mkdir(parents=True, exist_ok=True)
            draw_boxes(image, [(name, color, boxes)], legend=True).save(prompt_dir / image_path.name)
        combined_dir = args.output / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        draw_boxes(image, groups, legend=True).save(combined_dir / image_path.name)
        summary.append({"image": image_path.name, "counts": counts})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
