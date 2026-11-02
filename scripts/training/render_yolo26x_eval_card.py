#!/usr/bin/env python3
"""Render a compact YOLO26x evaluation card with synthetic, real, and table-cap samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.evaluate_yolo_vlm_box_refinement import match_operational  # noqa: E402
from scripts.training.refine_yolo_boxes_with_vlm import (  # noqa: E402
    CLASS_NAMES,
    COLORS,
    draw_detections,
    get_font,
    load_reviewed_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--synthetic-image", type=Path, required=True)
    parser.add_argument("--synthetic-label", type=Path, required=True)
    parser.add_argument("--real-image", type=Path, required=True)
    parser.add_argument("--real-label", type=Path, required=True)
    parser.add_argument("--table-images", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--crop-top-fraction", type=float, default=0.23)
    parser.add_argument("--heldout-map", type=float, default=0.9724)
    return parser.parse_args()


def landscape_crop(image: Image.Image, top_fraction: float) -> tuple[Image.Image, list[int]]:
    width, height = image.size
    crop_height = round(width * 9 / 16)
    top = min(max(0, round(height * top_fraction)), height - crop_height)
    crop = [0, top, width, top + crop_height]
    return image.crop(tuple(crop)), crop


def predictions(model: YOLO, image_path: Path, imgsz: int, confidence: float) -> list[dict]:
    result = model.predict(
        source=str(image_path), imgsz=imgsz, device=0, conf=0.001,
        iou=0.70, max_det=300, verbose=False,
    )[0]
    output = []
    if result.boxes is None:
        return output
    for box, class_id, score in zip(
        result.boxes.xyxy.detach().cpu().tolist(),
        result.boxes.cls.detach().cpu().tolist(),
        result.boxes.conf.detach().cpu().tolist(),
    ):
        if score < confidence:
            continue
        class_id = int(class_id)
        output.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(score),
            "box": [float(value) for value in box],
        })
    return output


def counts(items: list[dict]) -> str:
    found = []
    for class_id, name in enumerate(CLASS_NAMES):
        count = sum(item["class_id"] == class_id for item in items)
        if count:
            found.append(f"{name}={count}")
    return ", ".join(found) if found else "none"


def labeled_metrics(image: Image.Image, truth: list[dict], detected: list[dict], confidence: float) -> dict:
    record = {
        "size": image.size,
        "ground_truth": truth,
        "predictions": detected,
    }
    return match_operational([record], "predictions", None, confidence)


def fit_view(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    view = image.copy()
    view.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#101720")
    canvas.paste(view, ((size[0] - view.width) // 2, (size[1] - view.height) // 2))
    return canvas


def sample_panel(
    title: str,
    subtitle: str,
    left: Image.Image,
    right: Image.Image,
    left_name: str,
    right_name: str,
    status: str,
    pass_state: bool,
) -> Image.Image:
    width, height = 1080, 650
    canvas = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((3, 3, width - 4, height - 4), radius=16, outline="#829ab1", width=3)
    draw.text((24, 16), title, fill="#102a43", font=get_font(30))
    draw.text((24, 55), subtitle, fill="#486581", font=get_font(18))
    view_size = (505, 465)
    left_view, right_view = fit_view(left, view_size), fit_view(right, view_size)
    canvas.paste(left_view, (24, 110))
    canvas.paste(right_view, (551, 110))
    draw.text((24, 86), left_name, fill="#334e68", font=get_font(18))
    draw.text((551, 86), right_name, fill="#334e68", font=get_font(18))
    color = "#147d3f" if pass_state else "#c53030"
    draw.rounded_rectangle((24, 590, width - 24, 632), radius=8, fill="#e7f7ed" if pass_state else "#fff0f0")
    draw.text((40, 599), status, fill=color, font=get_font(20))
    return canvas


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(args.model.resolve()))
    records = []
    panels = []

    for kind, image_path, label_path in (
        ("Held-out synthetic", args.synthetic_image, args.synthetic_label),
        ("Reviewed real tray", args.real_image, args.real_label),
    ):
        image = Image.open(image_path).convert("RGB")
        truth = load_reviewed_labels(label_path, *image.size)
        detected = predictions(model, image_path, args.imgsz, args.confidence)
        metric = labeled_metrics(image, truth, detected, args.confidence)
        passed = metric["false_positive"] == 0 and metric["false_negative"] == 0
        status = (
            f'{"PASS" if passed else "CHECK"} — TP {metric["true_positive"]}, '
            f'FP {metric["false_positive"]}, FN {metric["false_negative"]}, '
            f'mean IoU {metric["mean_iou_matched"]:.3f}'
        )
        panels.append(sample_panel(
            kind,
            f"{image_path.name} | GT: {counts(truth)}",
            draw_detections(image, truth),
            draw_detections(image, detected, True),
            "Ground truth",
            f"YOLO prediction @ {args.confidence:.2f}",
            status,
            passed,
        ))
        records.append({
            "kind": kind, "image": str(image_path.resolve()),
            "ground_truth": truth, "predictions": detected, "metrics": metric,
        })

    for index, image_path in enumerate(args.table_images, 1):
        source = Image.open(image_path).convert("RGB")
        processed, crop = landscape_crop(source, args.crop_top_fraction)
        processed = processed.resize((1920, 1080), Image.Resampling.LANCZOS)
        processed_path = args.output / "processed" / f"{image_path.stem}.png"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(processed_path)
        detected = predictions(model, processed_path, args.imgsz, args.confidence)
        table = [item for item in detected if item["class_id"] == 3]
        passed = len(table) == 1 and len(detected) == 1
        confidence = table[0]["confidence"] if len(table) == 1 else None
        status = (
            f'{"PASS" if passed else "CHECK"} — expected one table_cap; '
            f'detected {len(table)} table_cap / {len(detected)} total'
            + (f", confidence {confidence:.3f}" if confidence is not None else "")
        )
        panels.append(sample_panel(
            f"Unseen table-cap photo {index}",
            f"{image_path.name} | expected: table_cap=1",
            processed,
            draw_detections(processed, detected, True),
            "Processed 16:9 input",
            f"YOLO prediction @ {args.confidence:.2f}",
            status,
            passed,
        ))
        records.append({
            "kind": f"Unseen table-cap photo {index}",
            "image": str(image_path.resolve()), "crop_box_xyxy": crop,
            "processed": str(processed_path.resolve()),
            "expected": {"table_cap": 1, "total": 1},
            "predictions": detected, "pass": passed,
        })

    margin, gap, header, footer = 28, 22, 142, 100
    card_width = margin * 2 + panels[0].width * 2 + gap
    card_height = header + panels[0].height * 2 + gap + footer
    card = Image.new("RGB", (card_width, card_height), "#d9e2ec")
    draw = ImageDraw.Draw(card)
    draw.text((margin, 20), "YOLO26x table-cap fine-tune — evaluation card", fill="#102a43", font=get_font(42))
    draw.text(
        (margin, 76),
        f"confidence {args.confidence:.2f} | imgsz {args.imgsz} | held-out synthetic mAP50–95 {args.heldout_map:.4f}",
        fill="#334e68", font=get_font(23),
    )
    for index, panel_image in enumerate(panels):
        x = margin + (index % 2) * (panel_image.width + gap)
        y = header + (index // 2) * (panel_image.height + gap)
        card.paste(panel_image, (x, y))
    footer_y = card_height - footer + 16
    draw.text((margin, footer_y), "Classes:", fill="#102a43", font=get_font(21))
    x = margin + 104
    for name, color in zip(CLASS_NAMES, COLORS):
        draw.rounded_rectangle((x, footer_y + 2, x + 24, footer_y + 26), radius=4, fill=color)
        draw.text((x + 32, footer_y), name, fill="#243b53", font=get_font(20))
        x += 250
    draw.text(
        (margin, footer_y + 38),
        "Synthetic is held out; the reviewed real tray image was used during fine-tuning; both table photographs are unseen.",
        fill="#486581", font=get_font(19),
    )
    card_path = args.output / "yolo26x_new_table_photo_eval_card.png"
    card.save(card_path)
    payload = {
        "checkpoint": str(args.model.resolve()), "confidence": args.confidence,
        "imgsz": args.imgsz, "heldout_synthetic_map50_95": args.heldout_map,
        "records": records, "card": str(card_path.resolve()),
    }
    (args.output / "eval_card_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "card": str(card_path),
        "results": [
            {"kind": record["kind"], "metrics": record.get("metrics"), "pass": record.get("pass")}
            for record in records
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
