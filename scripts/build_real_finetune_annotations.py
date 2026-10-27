#!/usr/bin/env python3
"""Build provisional annotations and inspection graphics for real fine-tuning frames."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {0: "good_cap", 1: "crooked_cap", 2: "missing_cap"}
CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
SHORT_NAMES = {"good_cap": "G", "crooked_cap": "C", "missing_cap": "M"}
COLORS = {
    "good_cap": (25, 180, 80),
    "crooked_cap": (245, 135, 15),
    "missing_cap": (220, 35, 45),
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def grid_boxes(
    centers: list[list[tuple[int, int]]],
    states: list[list[str]],
    row_sizes: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for row_index, (center_row, state_row) in enumerate(zip(centers, states), start=1):
        width, height = row_sizes[row_index - 1]
        for column_index, ((center_x, center_y), short_state) in enumerate(
            zip(center_row, state_row), start=1
        ):
            class_name = {"G": "good_cap", "C": "crooked_cap", "M": "missing_cap"}[short_state]
            x1 = round(center_x - width / 2)
            y1 = round(center_y - height / 2)
            x2 = round(center_x + width / 2)
            y2 = round(center_y + height / 2)
            boxes.append({
                "slot": f"R{row_index}C{column_index}",
                "class_id": CLASS_IDS[class_name],
                "class_name": class_name,
                "xyxy": [x1, y1, x2, y2],
                "confidence": "operator_reviewed",
            })
    return boxes


FRAMES: list[dict[str, Any]] = [
    {
        "filename": "run_161143_0027.png",
        "source": "shared/runs/2026-10-26_16-11-43_physical_policy/tablecam_raw/0027.png",
        "description": "6x5 tray; mostly crooked caps with some seated caps",
        "crop": [620, 270, 1320, 830],
        "centers": [
            [(870, 374), (941, 377), (1021, 389), (1097, 389), (1168, 390)],
            [(861, 429), (940, 438), (1020, 439), (1098, 439), (1178, 439)],
            [(853, 485), (935, 487), (1020, 491), (1100, 491), (1183, 491)],
            [(850, 541), (936, 544), (1019, 547), (1100, 548), (1187, 549)],
            [(840, 603), (925, 606), (1017, 606), (1104, 607), (1190, 609)],
            [(837, 666), (925, 668), (1015, 669), (1105, 670), (1197, 675)],
        ],
        "states": [
            list("CCGGG"),
            list("CGGGG"),
            list("CCGGG"),
            list("GCMGG"),
            list("CCCCG"),
            list("CMCCG"),
        ],
        "row_sizes": [(50, 46), (52, 48), (54, 50), (56, 52), (58, 54), (60, 56)],
    },
    {
        "filename": "run_155442_0030.png",
        "source": "shared/runs/2026-10-26_15-54-42_physical_policy/tablecam_raw/0030.png",
        "description": "3x5 tray; every present cap is crooked and three vials are uncapped",
        "crop": [600, 430, 1250, 860],
        "centers": [
            [(742, 579), (833, 574), (929, 571), (1008, 581), (1098, 585)],
            [(731, 637), (822, 638), (923, 643), (1011, 644), (1100, 644)],
            [(733, 709), (815, 725), (917, 713), (1019, 710), (1114, 714)],
        ],
        "states": [
            list("CCCCC"),
            list("MCCMC"),
            list("CMCCC"),
        ],
        "row_sizes": [(54, 54), (56, 56), (58, 58)],
    },
    {
        "filename": "run_155442_0225.png",
        "source": "shared/runs/2026-10-26_15-54-42_physical_policy/tablecam_raw/0225.png",
        "description": "6x5 tray; mixed seated/crooked caps and two visually uncapped vials",
        "crop": [650, 280, 1350, 820],
        "centers": [
            [(864, 360), (934, 362), (1015, 372), (1088, 374), (1163, 376)],
            [(859, 414), (934, 417), (1015, 418), (1090, 419), (1170, 420)],
            [(855, 470), (932, 470), (1013, 474), (1094, 477), (1177, 477)],
            [(850, 525), (930, 527), (1015, 531), (1097, 532), (1182, 533)],
            [(842, 583), (925, 585), (1014, 588), (1102, 588), (1189, 590)],
            [(838, 643), (925, 647), (1015, 650), (1105, 651), (1197, 654)],
        ],
        "states": [
            list("CCGGG"),
            list("CGGGC"),
            list("CCGGG"),
            list("GCMGC"),
            list("CCCCC"),
            list("CMCCC"),
        ],
        "row_sizes": [(50, 46), (52, 48), (54, 50), (56, 52), (58, 54), (60, 56)],
    },
]


def annotate_frame(frame: dict[str, Any], boxes: list[dict[str, Any]]) -> Image.Image:
    image = Image.open(frame["source"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    box_font = font(14)
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        class_name = box["class_name"]
        color = COLORS[class_name]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        tag = f"{box['slot']} {SHORT_NAMES[class_name]}"
        text_box = draw.textbbox((0, 0), tag, font=box_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        tag_y = max(0, y1 - text_height - 5)
        draw.rectangle((x1, tag_y, x1 + text_width + 5, y1), fill=color)
        draw.text((x1 + 2, tag_y - text_box[1] + 1), tag, fill="white", font=box_font)
    return image


def make_panel(frame: dict[str, Any], annotated: Image.Image, boxes: list[dict[str, Any]]) -> Image.Image:
    x1, y1, x2, y2 = frame["crop"]
    crop = annotated.crop((x1, y1, x2, y2))
    target_width = 1400
    target_height = round(crop.height * target_width / crop.width)
    crop = crop.resize((target_width, target_height), Image.Resampling.LANCZOS)

    counts = Counter(box["class_name"] for box in boxes)
    header_height = 82
    panel = Image.new("RGB", (target_width, target_height + header_height), "white")
    panel.paste(crop, (0, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), frame["filename"], fill="black", font=font(20))
    summary = (
        f"G good={counts['good_cap']}   C crooked={counts['crooked_cap']}   "
        f"M missing={counts['missing_cap']}   |   loose table caps intentionally omitted"
    )
    draw.text((12, 38), summary, fill="#202020", font=font(16))
    return panel


def main() -> int:
    output = Path("images/Real_Finetune")
    inspection = output / "inspection"
    inspection.mkdir(parents=True, exist_ok=True)

    annotation_frames: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for frame in FRAMES:
        boxes = grid_boxes(frame["centers"], frame["states"], frame["row_sizes"])
        annotated = annotate_frame(frame, boxes)
        overlay_path = inspection / f"{Path(frame['filename']).stem}_labeled.png"
        annotated.save(overlay_path)
        panels.append(make_panel(frame, annotated, boxes))
        annotation_frames.append({
            "filename": frame["filename"],
            "source": frame["source"],
            "description": frame["description"],
            "grid": {"rows": len(frame["states"]), "columns": 5},
            "loose_table_caps_annotated": False,
            "boxes": boxes,
        })

    sheet_width = max(panel.width for panel in panels)
    sheet_height = sum(panel.height for panel in panels) + 20 * (len(panels) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#d8dee9")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height + 20
    sheet.save(output / "real_finetune_annotation_inspection.png")

    payload = {
        "schema_version": 1,
        "status": "operator_reviewed",
        "image_size": [1920, 1080],
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "slot_order": "left-to-right, then top-to-bottom; R1C1 is the back-left tray slot",
        "labeling_policy": {
            "good_cap": "cap appears centered and seated squarely on the vial mouth",
            "crooked_cap": "cap is visibly angled, shifted, or incompletely seated",
            "missing_cap": "a transparent vial occupies the slot but has no cap",
            "table_cap": "not annotated in these frames because deployment will use pink paper",
        },
        "evaluation_warning": (
            "These frames will be used for both fine-tuning and evaluation due to the tiny real dataset; "
            "reported real metrics are training-set fit, not an unbiased estimate of generalization."
        ),
        "images": annotation_frames,
    }
    (output / "annotations.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "annotations": str(output / "annotations.json"),
        "inspection_sheet": str(output / "real_finetune_annotation_inspection.png"),
        "individual_overlays": [str(path) for path in sorted(inspection.glob("*_labeled.png"))],
        "counts": {
            item["filename"]: dict(Counter(box["class_name"] for box in item["boxes"]))
            for item in annotation_frames
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
