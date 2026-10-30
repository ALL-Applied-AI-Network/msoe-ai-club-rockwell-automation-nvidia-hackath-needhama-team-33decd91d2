#!/usr/bin/env python3
"""Combine vial and cap proposal manifests and visualize inferred missing caps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inspection.defect_training.boxes import Box  # noqa: E402
from src.inspection.defect_training.schemas import read_object_records  # noqa: E402
from src.inspection.defect_training.vial_cap_association import (  # noqa: E402
    AssociationConfig,
    associate_vials_and_caps,
)


COLORS = {
    "vial": (40, 140, 240),
    "matched_cap": (245, 190, 35),
    "missing": (235, 55, 65),
    "unmatched_cap": (180, 70, 210),
}


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box: Box, color: tuple[int, int, int], label: str, width: int, label_font) -> None:
    xy = tuple(round(value) for value in box.as_list())
    draw.rectangle(xy, outline=color, width=width)
    text_box = draw.textbbox((xy[0], xy[1]), label, font=label_font)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.text((xy[0], xy[1]), label, fill=color, font=label_font)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vial-manifest", type=Path, required=True)
    parser.add_argument("--cap-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upper-fraction", type=float, default=0.45)
    parser.add_argument("--horizontal-margin", type=float, default=0.30)
    parser.add_argument("--maximum-distance", type=float, default=1.25)
    parser.add_argument("--top-width-scale", type=float, default=1.10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    vials_by_image = {}
    caps_by_image = {}
    for record in read_object_records(args.vial_manifest):
        vials_by_image.setdefault(record.image, []).append(record.source_box)
    for record in read_object_records(args.cap_manifest):
        caps_by_image.setdefault(record.image, []).append(record.source_box)
    missing_caps = sorted(set(vials_by_image) - set(caps_by_image))
    if missing_caps:
        raise ValueError(f"Cap manifest has no records for images: {missing_caps}")
    config = AssociationConfig(
        upper_fraction=args.upper_fraction,
        horizontal_margin_fraction=args.horizontal_margin,
        maximum_normalized_center_distance=args.maximum_distance,
        inferred_top_width_scale=args.top_width_scale,
    )
    images = []
    summary = {"images": 0, "vials": 0, "caps": 0, "matches": 0, "inferred_missing": 0, "unmatched_caps": 0}
    for image_path_text in sorted(vials_by_image):
        image_path = Path(image_path_text)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        vials = vials_by_image[image_path_text]
        caps = caps_by_image[image_path_text]
        result = associate_vials_and_caps(vials, caps, image.size, config)
        rendered = image.copy()
        draw = ImageDraw.Draw(rendered)
        line_width = max(2, round(min(image.size) / 400))
        label_font = font(max(14, round(min(image.size) / 60)))
        matched_vials = {item["vial_index"] for item in result["matches"]}
        for index, vial in enumerate(vials):
            draw_box(draw, vial, COLORS["vial"], f"vial {index + 1}", line_width, label_font)
        for item in result["matches"]:
            draw_box(draw, caps[item["cap_index"]], COLORS["matched_cap"], "matched cap", line_width, label_font)
        for item in result["inferred_missing"]:
            draw_box(draw, item["box"], COLORS["missing"], "inferred missing", line_width + 1, label_font)
        for index in result["unmatched_cap_indices"]:
            draw_box(draw, caps[index], COLORS["unmatched_cap"], "unmatched cap", line_width, label_font)
        overlay_path = args.output / "overlays" / image_path.name
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(overlay_path)
        record = {
            "image": image_path_text,
            "image_size": list(image.size),
            "vial_boxes_xyxy": [box.as_list() for box in vials],
            "cap_boxes_xyxy": [box.as_list() for box in caps],
            "matches": result["matches"],
            "inferred_missing": [
                {"vial_index": item["vial_index"], "box_xyxy": item["box"].as_list()}
                for item in result["inferred_missing"]
            ],
            "unmatched_cap_indices": result["unmatched_cap_indices"],
            "overlay": str(overlay_path),
        }
        (args.output / "records").mkdir(exist_ok=True)
        (args.output / "records" / f"{image_path.stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        images.append(record)
        summary["images"] += 1
        summary["vials"] += len(vials)
        summary["caps"] += len(caps)
        summary["matches"] += len(result["matches"])
        summary["inferred_missing"] += len(result["inferred_missing"])
        summary["unmatched_caps"] += len(result["unmatched_cap_indices"])
    payload = {"schema_version": 1, "config": vars(config), "summary": summary, "images": images}
    (args.output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
