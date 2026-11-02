#!/usr/bin/env python3
"""Evaluate YOLO geometry before/after class-agnostic VLM box refinement."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.refine_yolo_boxes_with_vlm import (  # noqa: E402
    CLASS_NAMES,
    cluster_vlm_boxes,
    draw_detections,
    get_font,
    match_proposals,
    panel,
    write_yolo_labels,
    box_iou,
    center_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--minimum-inference-confidence", type=float, default=0.001)
    parser.add_argument("--cluster-iou", type=float, default=0.65)
    parser.add_argument("--minimum-match-score", type=float, default=0.35)
    return parser.parse_args()


def refine_high_confidence(detections: list[dict], proposals: list[dict], confidence: float, score: float) -> list[dict]:
    high_indices = [index for index, item in enumerate(detections) if item["confidence"] >= confidence]
    high = [detections[index] for index in high_indices]
    refined_high = match_proposals(high, proposals, score)
    refined_by_index = dict(zip(high_indices, refined_high))
    output = []
    for index, detection in enumerate(detections):
        if index in refined_by_index:
            output.append(refined_by_index[index])
        else:
            item = dict(detection)
            item.update({"yolo_box": item["box"], "refined": False, "match": None})
            output.append(item)
    return output


def match_operational(records: list[dict], prediction_key: str, class_id: int | None, confidence: float, threshold: float = 0.5) -> dict:
    true_positive = false_positive = false_negative = 0
    matched_ious, center_errors = [], []
    prediction_count = ground_truth_count = 0
    for record in records:
        ground_truth = [item for item in record["ground_truth"] if class_id is None or item["class_id"] == class_id]
        predictions = [
            item for item in record[prediction_key]
            if item["confidence"] >= confidence and (class_id is None or item["class_id"] == class_id)
        ]
        # Overall matching must still require equal classes.
        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        used = set()
        prediction_count += len(predictions)
        ground_truth_count += len(ground_truth)
        for prediction in predictions:
            candidates = [
                (box_iou(prediction["box"], truth["box"]), index, truth)
                for index, truth in enumerate(ground_truth)
                if index not in used and truth["class_id"] == prediction["class_id"]
            ]
            best = max(candidates, default=(0.0, -1, None), key=lambda item: item[0])
            if best[0] >= threshold:
                true_positive += 1
                used.add(best[1])
                matched_ious.append(best[0])
                center_errors.append(center_distance(prediction["box"], best[2]["box"]) / math.hypot(*record["size"]) * 100)
            else:
                false_positive += 1
        false_negative += len(ground_truth) - len(used)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_iou_matched": float(np.mean(matched_ious)) if matched_ious else None,
        "mean_center_error_percent_diagonal": float(np.mean(center_errors)) if center_errors else None,
    }


def average_precision(records: list[dict], prediction_key: str, class_id: int, threshold: float) -> float:
    ground_truth = {}
    total_ground_truth = 0
    predictions = []
    for image_index, record in enumerate(records):
        boxes = [item["box"] for item in record["ground_truth"] if item["class_id"] == class_id]
        ground_truth[image_index] = {"boxes": boxes, "used": set()}
        total_ground_truth += len(boxes)
        for item in record[prediction_key]:
            if item["class_id"] == class_id:
                predictions.append((item["confidence"], image_index, item["box"]))
    if total_ground_truth == 0:
        return float("nan")
    predictions.sort(reverse=True, key=lambda item: item[0])
    tp, fp = [], []
    for _, image_index, prediction_box in predictions:
        image_truth = ground_truth[image_index]
        candidates = [
            (box_iou(prediction_box, box), index)
            for index, box in enumerate(image_truth["boxes"])
            if index not in image_truth["used"]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1), key=lambda item: item[0])
        if best_iou >= threshold:
            image_truth["used"].add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not predictions:
        return 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recalls = cumulative_tp / total_ground_truth
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return float(np.mean([np.max(precisions[recalls >= level], initial=0.0) for level in np.linspace(0, 1, 101)]))


def summarize(records: list[dict], prediction_key: str, confidence: float) -> dict:
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    per_class = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        ap_values = [average_precision(records, prediction_key, class_id, threshold) for threshold in thresholds]
        per_class[class_name] = {
            **match_operational(records, prediction_key, class_id, confidence),
            "ap50": ap_values[0],
            "map50_95": float(np.nanmean(ap_values)),
        }
    overall = match_operational(records, prediction_key, None, confidence)
    overall["ap50"] = float(np.mean([item["ap50"] for item in per_class.values()]))
    overall["map50_95"] = float(np.mean([item["map50_95"] for item in per_class.values()]))
    return {"overall": overall, "per_class": per_class}


def metric(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def render_report(results: dict, sample_rows: list[Image.Image], output: Path) -> None:
    report_width = 2400
    row_height = 52
    top_height = 170 + row_height * 12
    report_height = top_height + len(sample_rows) * 470 + 40
    report = Image.new("RGB", (report_width, report_height), "white")
    draw = ImageDraw.Draw(report)
    draw.text((40, 28), "YOLO26x + LocateAnything geometry — held-out synthetic test", fill=(15, 20, 28), font=get_font(38))
    draw.text(
        (40, 82),
        "100 images; YOLO retains count/class/confidence; VLM only redraws matched boxes; operational IoU@0.50, confidence 0.40",
        fill=(55, 60, 70), font=get_font(22),
    )
    columns = [(40, 420), (490, 180), (690, 180), (890, 180), (1090, 180), (1290, 200), (1510, 210), (1740, 220), (1980, 350)]
    headers = ["Method / class", "Precision", "Recall", "F1", "AP50", "mAP50–95", "Mean IoU", "Center err %", "Pred / GT"]
    y = 135
    draw.rectangle((25, y, report_width - 25, y + row_height), fill=(35, 45, 58))
    for (x, _), header in zip(columns, headers):
        draw.text((x, y + 13), header, fill="white", font=get_font(20))
    y += row_height
    for method_key, method_name in (("yolo", "YOLO original"), ("hybrid", "YOLO + VLM geometry")):
        summary = results[method_key]
        rows = [(method_name, summary["overall"], True)] + [
            (f"  {class_name}", summary["per_class"][class_name], False) for class_name in CLASS_NAMES
        ]
        for label, values, overall in rows:
            draw.rectangle((25, y, report_width - 25, y + row_height), fill=(220, 228, 237) if overall else (248, 249, 251))
            cells = [
                label, metric(values["precision"]), metric(values["recall"]), metric(values["f1"]),
                metric(values["ap50"]), metric(values["map50_95"]), metric(values["mean_iou_matched"]),
                metric(values["mean_center_error_percent_diagonal"]),
                f'{values["prediction_count"]} / {values["ground_truth_count"]}',
            ]
            for (x, _), value in zip(columns, cells):
                draw.text((x, y + 13), value, fill=(20, 25, 32), font=get_font(20))
            y += row_height
        y += 8
    sample_y = top_height
    for row in sample_rows:
        row.thumbnail((report_width - 40, 450), Image.Resampling.LANCZOS)
        report.paste(row, ((report_width - row.width) // 2, sample_y))
        sample_y += 470
    report.save(output)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.vlm_predictions.read_text())
    vlm_by_name = {Path(record["image"]).name: record for record in payload["raw"]}
    image_dir = args.dataset / "images" / args.split
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    model = YOLO(str(args.model))
    records = []
    total_refined = 0

    for image_index, image_path in enumerate(image_paths, 1):
        vlm_record = vlm_by_name[image_path.name]
        result = model.predict(
            source=str(image_path), device=0, imgsz=args.imgsz,
            conf=args.minimum_inference_confidence, iou=0.70, max_det=300, verbose=False,
        )[0]
        yolo = []
        for box, class_id, confidence in zip(
            result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.cls.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
        ):
            class_id = int(class_id)
            yolo.append({
                "class_id": class_id, "class_name": CLASS_NAMES[class_id],
                "confidence": float(confidence), "box": [float(value) for value in box],
            })
        proposals = cluster_vlm_boxes(vlm_record["predictions"], args.cluster_iou)
        hybrid = refine_high_confidence(yolo, proposals, args.confidence, args.minimum_match_score)
        total_refined += sum(item["refined"] for item in hybrid)
        record = {
            "image": str(image_path), "size": vlm_record["size"],
            "ground_truth": vlm_record["ground_truth"], "yolo": yolo, "hybrid": hybrid,
            "vlm_box_count": len(vlm_record["predictions"]), "vlm_proposal_count": len(proposals),
        }
        records.append(record)
        write_yolo_labels(args.output / "labels" / "hybrid" / f"{image_path.stem}.txt", hybrid, *vlm_record["size"])
        print(f"[{image_index}/{len(image_paths)}] {image_path.name}: yolo={len(yolo)} proposals={len(proposals)}", flush=True)

    summaries = {
        "yolo": summarize(records, "yolo", args.confidence),
        "hybrid": summarize(records, "hybrid", args.confidence),
    }
    metrics = {
        "schema_version": 1,
        "model": str(args.model), "adapter_root": payload.get("adapter_root"),
        "dataset": str(args.dataset), "split": args.split, "image_count": len(records),
        "operational_confidence": args.confidence, "imgsz": args.imgsz,
        "refined_detection_count": total_refined,
        "method": "YOLO count/class/confidence + class-agnostic clustered LocateAnything geometry",
        "results": summaries,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    sample_indices = sorted(set(round(index * (len(records) - 1) / 5) for index in range(6)))
    sample_rows = []
    for index in sample_indices:
        record = records[index]
        image = Image.open(record["image"]).convert("RGB")
        operational_yolo = [item for item in record["yolo"] if item["confidence"] >= args.confidence]
        operational_hybrid = [item for item in record["hybrid"] if item["confidence"] >= args.confidence]
        cells = [
            panel(draw_detections(image, record["ground_truth"]), f"Synthetic ground truth — {Path(record['image']).name}", (790, 470)),
            panel(draw_detections(image, operational_yolo, True), "YOLO original", (790, 470)),
            panel(draw_detections(image, operational_hybrid, True), "YOLO classes + VLM geometry", (790, 470)),
        ]
        row = Image.new("RGB", (2370, 470), "#d0d0d0")
        for cell_index, cell in enumerate(cells):
            row.paste(cell, (790 * cell_index, 0))
        sample_rows.append(row)
    render_report(summaries, sample_rows, args.output / "hybrid_synthetic_evaluation.png")
    (args.output / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(metrics["results"], indent=2), flush=True)
    print(f"Report: {args.output / 'hybrid_synthetic_evaluation.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
