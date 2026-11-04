#!/usr/bin/env python3
"""Evaluate NVIDIA LocateAnything-3B on the five reviewed physical frames.

The script runs each semantic class as a separate referring-expression query so
every returned box has an unambiguous class. Coordinates are emitted by the
model in [0, 1000] and converted back to pixels for metrics and rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import AutoModel, AutoProcessor, AutoTokenizer


CLASS_NAMES = ["good_cap", "crooked_cap", "missing_cap", "table_cap"]
CLASS_SHORT = ["PASS", "CROOKED", "MISSING", "TABLE"]
COLORS = [(40, 205, 80), (255, 145, 20), (235, 60, 65), (45, 145, 255)]
SCORED_CLASSES = {0, 1, 2}  # table caps were intentionally omitted from GT
PROMPTS = {
    # LocateAnything is trained on short referring expressions. Keep each query
    # to one category phrase so the decoder emits one unambiguous <ref> section.
    0: "yellow cap seated flat and centered on top of a clear plastic vial in the metal tray",
    1: "yellow cap tilted, diagonal, or off-center on top of a clear plastic vial in the metal tray",
    2: "open top of an uncapped clear plastic vial in the metal tray",
    3: "loose yellow cap lying on the white table outside the metal tray",
}
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-root", type=Path,
        help="Fine-tune checkpoint containing llm_lora_adapter/ and vision_projector.pt",
    )
    parser.add_argument("--views", default="raw,clahe")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def make_view(image: Image.Image, view: str) -> Image.Image:
    if view == "raw":
        return image
    rgb = np.asarray(image.convert("RGB"))
    if view == "clahe":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # A restrained unsharp mask makes faint vial rims more visible without
        # introducing a second geometry or changing any coordinates.
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
        enhanced = cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)
        return Image.fromarray(enhanced)
    raise ValueError(f"Unknown view: {view}")


def load_ground_truth(label_path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()[:5]
        cls_i = int(cls)
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        boxes.append(
            {
                "class_id": cls_i,
                "class_name": CLASS_NAMES[cls_i],
                "box": [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ],
            }
        )
    return boxes


def parse_boxes(answer: str, width: int, height: int, class_id: int) -> list[dict]:
    parsed = []
    for match in BOX_RE.finditer(answer):
        x1, y1, x2, y2 = (int(value) for value in match.groups())
        coords = [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]
        coords[0], coords[2] = sorted((max(0.0, coords[0]), min(float(width), coords[2])))
        coords[1], coords[3] = sorted((max(0.0, coords[1]), min(float(height), coords[3])))
        if coords[2] - coords[0] >= 2 and coords[3] - coords[1] >= 2:
            parsed.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "box": coords})
    return nms(parsed, 0.75)


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def nms(items: list[dict], threshold: float) -> list[dict]:
    # No model confidence is available, so prefer the tighter box when exact
    # duplicates appear in a response.
    ordered = sorted(items, key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]))
    kept = []
    for item in ordered:
        if all(iou(item["box"], other["box"]) < threshold for other in kept):
            kept.append(item)
    return kept


def match_at_iou(gt: list[dict], pred: list[dict], threshold: float) -> tuple[int, int, int, list[float]]:
    candidates = []
    for gi, truth in enumerate(gt):
        for pi, prediction in enumerate(pred):
            if truth["class_id"] == prediction["class_id"]:
                candidates.append((iou(truth["box"], prediction["box"]), gi, pi))
    used_gt, used_pred, overlaps = set(), set(), []
    for overlap, gi, pi in sorted(candidates, reverse=True):
        if overlap < threshold:
            break
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            overlaps.append(overlap)
    return len(overlaps), len(pred) - len(used_pred), len(gt) - len(used_gt), overlaps


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def center_pairing(gt: list[dict], pred: list[dict], diagonal: float) -> list[float]:
    """Greedy nearest same-class center distances, normalized by image diagonal."""
    pairs = []
    for gi, truth in enumerate(gt):
        gx, gy = center(truth["box"])
        for pi, prediction in enumerate(pred):
            if truth["class_id"] == prediction["class_id"]:
                px, py = center(prediction["box"])
                pairs.append((math.hypot(gx - px, gy - py), gi, pi))
    used_gt, used_pred, distances = set(), set(), []
    for distance, gi, pi in sorted(pairs):
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            distances.append(distance / diagonal)
    return distances


def summarize(records: list[dict], threshold: float) -> dict:
    result = {"iou_threshold": threshold, "overall": {}, "per_class": {}}
    all_stats = defaultdict(float)
    by_class = {class_id: defaultdict(float) for class_id in SCORED_CLASSES}
    center_distances = defaultdict(list)
    for record in records:
        width, height = record["size"]
        diagonal = math.hypot(width, height)
        for class_id in SCORED_CLASSES:
            gt = [item for item in record["ground_truth"] if item["class_id"] == class_id]
            pred = [item for item in record["predictions"] if item["class_id"] == class_id]
            tp, fp, fn, overlaps = match_at_iou(gt, pred, threshold)
            stats = by_class[class_id]
            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn
            stats["gt"] += len(gt)
            stats["pred"] += len(pred)
            stats["iou_sum"] += sum(overlaps)
            stats["iou_count"] += len(overlaps)
            center_distances[class_id].extend(center_pairing(gt, pred, diagonal))
            for key in ("tp", "fp", "fn", "gt", "pred", "iou_sum", "iou_count"):
                all_stats[key] += stats[key] - all_stats.get(f"seen_{class_id}_{key}", 0)
                all_stats[f"seen_{class_id}_{key}"] = stats[key]

    def finalize(stats: dict, distances: list[float]) -> dict:
        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "ground_truth_count": int(stats["gt"]),
            "prediction_count": int(stats["pred"]),
            "true_positive": int(stats["tp"]),
            "false_positive": int(stats["fp"]),
            "false_negative": int(stats["fn"]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou_matched": stats["iou_sum"] / stats["iou_count"] if stats["iou_count"] else None,
            "mean_center_error_percent_diagonal": 100 * float(np.mean(distances)) if distances else None,
        }

    all_distances = [distance for values in center_distances.values() for distance in values]
    result["overall"] = finalize(all_stats, all_distances)
    for class_id in sorted(SCORED_CLASSES):
        result["per_class"][CLASS_NAMES[class_id]] = finalize(by_class[class_id], center_distances[class_id])
    result["per_class"]["table_cap"] = {
        "ground_truth_count": 0,
        "prediction_count": sum(
            1 for record in records for item in record["predictions"] if item["class_id"] == 3
        ),
        "scored": False,
        "reason": "The reviewed five-image ground truth intentionally excludes loose table caps.",
    }
    return result


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_boxes(image: Image.Image, items: list[dict], prefix: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    label_font = font(max(13, round(image.width / 120)))
    for index, item in enumerate(items, 1):
        class_id = item["class_id"]
        color = COLORS[class_id]
        box = tuple(round(value) for value in item["box"])
        draw.rectangle(box, outline=color, width=max(2, image.width // 640))
        cx, cy = center(item["box"])
        radius = max(3, image.width // 400)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(0, 0, 0))
        text = f"{prefix}{index}:{CLASS_SHORT[class_id]}"
        text_box = draw.textbbox((box[0], box[1]), text, font=label_font, stroke_width=2)
        tx = min(box[0], image.width - (text_box[2] - text_box[0]) - 2)
        ty = max(0, box[1] - (text_box[3] - text_box[1]) - 3)
        draw.text((tx, ty), text, fill=color, font=label_font, stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def render_comparison(
    image: Image.Image,
    gt: list[dict],
    pred: list[dict],
    title: str,
    path: Path,
    backend_name: str = "LocateAnything",
) -> None:
    left = draw_boxes(image, gt, "G")
    right = draw_boxes(image, pred, "P")
    header_h = 72
    sheet = Image.new("RGB", (image.width * 2, image.height + header_h), "white")
    sheet.paste(left, (0, header_h))
    sheet.paste(right, (image.width, header_h))
    draw = ImageDraw.Draw(sheet)
    title_font = font(24)
    draw.text((20, 8), title, fill=(0, 0, 0), font=title_font)
    draw.text((20, 39), "Reviewed ground truth", fill=(0, 0, 0), font=font(18))
    draw.text((image.width + 20, 39), f"{backend_name} prediction (box center marked)", fill=(0, 0, 0), font=font(18))
    sheet.thumbnail((2400, 1400), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)


def metric_text(value: float | int | None) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def render_report(
    output: Path,
    summaries: dict,
    sample_paths: dict[str, list[Path]],
    title: str = "LocateAnything-3B — five reviewed physical frames",
    filename: str = "locateanything_real5_report.png",
) -> None:
    views = list(summaries)
    width, row_h = 1800, 56
    table_rows = 2 + len(views) * 4
    thumbnails = []
    for view in views:
        for path in sample_paths[view]:
            image = Image.open(path).convert("RGB")
            image.thumbnail((850, 480), Image.Resampling.LANCZOS)
            thumbnails.append((view, path.stem, image.copy()))
    height = 145 + table_rows * row_h + math.ceil(len(thumbnails) / 2) * 525 + 90
    report = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(report)
    draw.text((45, 28), title, fill=(20, 25, 32), font=font(34))
    draw.text((45, 78), "Boxes and centers; IoU@0.50. Table-cap predictions shown but not scored.", fill=(60, 65, 75), font=font(20))
    columns = [(45, "View / class"), (460, "GT"), (555, "Pred"), (670, "Precision"), (850, "Recall"), (1020, "F1"), (1140, "Mean IoU"), (1340, "Center error (% diagonal)")]
    y = 135
    draw.rectangle((30, y, width - 30, y + row_h), fill=(31, 45, 61))
    for x, label in columns:
        draw.text((x, y + 15), label, fill="white", font=font(18))
    y += row_h
    for view in views:
        summary = summaries[view]["iou_0.50"]
        rows = [("OVERALL", summary["overall"])] + list(summary["per_class"].items())[:3]
        for row_index, (name, values) in enumerate(rows):
            fill = (224, 230, 238) if row_index == 0 else (255, 255, 255)
            draw.rectangle((30, y, width - 30, y + row_h), fill=fill)
            label = f"{view} / {name}" if row_index == 0 else f"    {name}"
            draw.text((45, y + 15), label, fill=(20, 25, 32), font=font(18))
            cells = [
                values.get("ground_truth_count"), values.get("prediction_count"), values.get("precision"),
                values.get("recall"), values.get("f1"), values.get("mean_iou_matched"),
                values.get("mean_center_error_percent_diagonal"),
            ]
            for (x, _), value in zip(columns[1:], cells):
                draw.text((x, y + 15), metric_text(value), fill=(20, 25, 32), font=font(18))
            y += row_h
    y += 25
    for index, (view, name, image) in enumerate(thumbnails):
        col = index % 2
        if col == 0 and index:
            y += 525
        x = 35 + col * 880
        draw.text((x, y), f"{view}: {name}", fill=(20, 25, 32), font=font(18))
        report.paste(image, (x, y + 32))
    report.save(output / filename)


class Worker:
    def __init__(self, model_path: str, adapter_root: Path | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
        )
        if adapter_root is not None:
            adapter = adapter_root / "llm_lora_adapter"
            projector = adapter_root / "vision_projector.pt"
            if not adapter.is_dir() or not projector.is_file():
                raise FileNotFoundError(f"Incomplete LocateAnything checkpoint: {adapter_root}")
            self.model.language_model = PeftModel.from_pretrained(self.model.language_model, adapter)
            self.model.mlp1.load_state_dict(torch.load(projector, map_location="cpu", weights_only=True))
        self.model = self.model.to("cuda").eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image, phrase: str, max_new_tokens: int) -> str:
        question = f"Locate all the instances that match the following description: {phrase}."
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="hybrid",
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return answer if isinstance(answer, str) else str(answer)


def main() -> None:
    args = parse_args()
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    image_dir = args.dataset / "images" / "test"
    label_dir = args.dataset / "labels" / "test"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(images) != 5:
        raise RuntimeError(f"Expected exactly five reviewed images, found {len(images)} in {image_dir}")

    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)}", flush=True)
    worker = Worker(args.model, args.adapter_root)
    records_by_view = {view: [] for view in views}
    sample_paths = {view: [] for view in views}
    started = time.time()
    for image_index, image_path in enumerate(images, 1):
        original = Image.open(image_path).convert("RGB")
        width, height = original.size
        gt = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        for view in views:
            inference_image = make_view(original, view)
            view_path = args.output / "inference_views" / view / f"{image_path.stem}.png"
            view_path.parent.mkdir(parents=True, exist_ok=True)
            inference_image.save(view_path)
            predictions, responses = [], {}
            for class_id in range(len(CLASS_NAMES)):
                class_start = time.time()
                answer = worker.predict(inference_image, PROMPTS[class_id], args.max_new_tokens)
                boxes = parse_boxes(answer, width, height, class_id)
                predictions.extend(boxes)
                responses[CLASS_NAMES[class_id]] = {"prompt": PROMPTS[class_id], "answer": answer, "box_count": len(boxes)}
                print(
                    f"[{image_index}/{len(images)}] {image_path.name} {view} {CLASS_NAMES[class_id]}: "
                    f"{len(boxes)} boxes in {time.time() - class_start:.1f}s",
                    flush=True,
                )
            record = {
                "image": str(image_path), "view": view, "size": [width, height],
                "ground_truth": gt, "predictions": predictions, "responses": responses,
            }
            records_by_view[view].append(record)
            sample = args.output / "prediction_examples" / view / f"{image_path.stem}_gt_vs_prediction.png"
            render_comparison(original, gt, predictions, f"{image_path.name} — {view}", sample)
            sample_paths[view].append(sample)

    summaries = {}
    for view, records in records_by_view.items():
        summaries[view] = {
            "iou_0.25": summarize(records, 0.25),
            "iou_0.50": summarize(records, 0.50),
        }
    metrics = {
        "schema_version": 1,
        "model": args.model,
        "adapter_root": str(args.adapter_root) if args.adapter_root else None,
        "runtime": "official Transformers BF16, generation_mode=hybrid",
        "dataset": str(args.dataset),
        "image_count": len(images),
        "views": views,
        "prompts": {CLASS_NAMES[key]: value for key, value in PROMPTS.items()},
        "notes": [
            "The model does not emit confidence scores, so AP/mAP is not defined.",
            "The report includes precision, recall, F1, matched IoU, and box-center error.",
            "Loose table caps are rendered but not scored because the reviewed annotations intentionally omit them.",
            "These five frames are the same reviewed physical frames used by prior YOLO experiments.",
        ],
        "elapsed_seconds": time.time() - started,
        "results": summaries,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    serializable_records = {view: records for view, records in records_by_view.items()}
    (args.output / "predictions_and_raw_responses.json").write_text(json.dumps(serializable_records, indent=2) + "\n")
    render_report(args.output, summaries, sample_paths)
    print(f"Finished in {metrics['elapsed_seconds'] / 60:.1f} minutes", flush=True)
    print(f"Report: {args.output / 'locateanything_real5_report.png'}", flush=True)


if __name__ == "__main__":
    main()
