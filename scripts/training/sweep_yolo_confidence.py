#!/usr/bin/env python3
"""Sweep YOLO confidence thresholds against fixed labeled datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def load_truth(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    output = []
    for line in path.read_text().splitlines():
        class_id, cx, cy, box_width, box_height = line.split()
        class_id = int(class_id)
        cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
        output.append({
            "class_id": class_id,
            "box": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width, (cy + box_height / 2) * height],
        })
    return output


def score(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    totals = defaultdict(int)
    class_totals = {name: defaultdict(int) for name in CLASS_NAMES}
    matched_ious = []
    for record in records:
        truths = record["truth"]
        predictions = sorted(
            (item for item in record["predictions"] if item["confidence"] >= threshold),
            key=lambda item: item["confidence"], reverse=True,
        )
        used = set()
        for prediction in predictions:
            same_class = [
                (box_iou(prediction["box"], truth["box"]), index)
                for index, truth in enumerate(truths)
                if index not in used and truth["class_id"] == prediction["class_id"]
            ]
            best_iou, best_index = max(same_class, default=(0.0, -1), key=lambda item: item[0])
            class_name = CLASS_NAMES[prediction["class_id"]]
            if best_iou >= 0.50:
                used.add(best_index)
                totals["tp"] += 1
                class_totals[class_name]["tp"] += 1
                matched_ious.append(best_iou)
                continue
            wrong_class_overlap = max(
                (box_iou(prediction["box"], truth["box"])
                 for index, truth in enumerate(truths)
                 if index not in used and truth["class_id"] != prediction["class_id"]),
                default=0.0,
            )
            totals["fp"] += 1
            class_totals[class_name]["fp"] += 1
            if wrong_class_overlap >= 0.50:
                totals["class_errors"] += 1
                class_totals[class_name]["class_errors"] += 1
            else:
                totals["localization_or_duplicate_errors"] += 1
                class_totals[class_name]["localization_or_duplicate_errors"] += 1
        for index, truth in enumerate(truths):
            class_name = CLASS_NAMES[truth["class_id"]]
            class_totals[class_name]["ground_truth"] += 1
            totals["ground_truth"] += 1
            if index not in used:
                totals["fn"] += 1
                class_totals[class_name]["fn"] += 1
        totals["predictions"] += len(predictions)

    # Materialize zero-valued counters so JSON/CSV consumers receive a stable
    # schema even when the model makes no errors of a particular type.
    for key in ("ground_truth", "predictions", "tp", "fp", "fn", "class_errors",
                "localization_or_duplicate_errors"):
        totals[key] += 0
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        **dict(totals),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        "zero_error": totals["fp"] == 0 and totals["fn"] == 0,
        "per_class": {name: dict(values) for name, values in class_totals.items()},
    }


def render_report(payload: dict[str, Any], output: Path) -> None:
    datasets = list(payload["results"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1.5, 1]})
    fig.suptitle("YOLO26x confidence sweep", fontsize=21, fontweight="bold")
    for dataset_name in datasets:
        rows = payload["results"][dataset_name]
        thresholds = [row["threshold"] for row in rows]
        axes[0].plot(thresholds, [row["precision"] for row in rows], marker="o", label=f"{dataset_name} precision")
        axes[0].plot(thresholds, [row["recall"] for row in rows], marker="x", linestyle="--", label=f"{dataset_name} recall")
    axes[0].axvline(0.40, color="black", linestyle=":", label="current 0.40")
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("Confidence threshold (log scale)")
    axes[0].set_ylabel("Precision / recall at IoU≥0.50")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)

    table_rows = []
    for name in datasets:
        rows = payload["results"][name]
        zero = [row["threshold"] for row in rows if row["zero_error"]]
        at_point_four = min(rows, key=lambda row: abs(row["threshold"] - 0.40))
        table_rows.append([
            name, str(rows[0]["ground_truth"]),
            f"{min(zero):.3f}" if zero else "none",
            str(at_point_four["fp"]), str(at_point_four["fn"]),
            f'{at_point_four["precision"]:.4f}', f'{at_point_four["recall"]:.4f}',
        ])
    axes[1].axis("off")
    axes[1].set_title("Lowest tested zero-error threshold and current-threshold behavior", loc="left", fontweight="bold")
    table = axes[1].table(
        cellText=table_rows,
        colLabels=["Dataset", "GT", "Lowest zero-error", "FP @ .40", "FN @ .40", "P @ .40", "R @ .40"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#243b53")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf2f7")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="name=image_dir,label_dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--thresholds", default="0.001,0.002,0.003,0.004,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    model = YOLO(str(args.model.resolve()))

    dataset_records = {}
    for dataset_spec in args.datasets:
        name, paths = dataset_spec.split("=", 1)
        image_dir_text, label_dir_text = paths.split(",", 1)
        image_dir, label_dir = Path(image_dir_text), Path(label_dir_text)
        image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        records = []
        # A Python list is treated as one in-memory batch by this Ultralytics
        # release even when batch=1 is requested. Iterate explicitly so a
        # 1280px YOLO26x sweep fits a 16 GB T4 exactly like deployment.
        for image_path in image_paths:
            result = model.predict(
                source=str(image_path), imgsz=args.imgsz, device=0,
                conf=min(thresholds), iou=0.70, max_det=300, verbose=False,
            )[0]
            with Image.open(image_path) as image:
                width, height = image.size
            truth = load_truth(label_dir / f"{image_path.stem}.txt", width, height)
            predictions = []
            if result.boxes is not None:
                for box, class_id, confidence in zip(
                    result.boxes.xyxy.detach().cpu().tolist(),
                    result.boxes.cls.detach().cpu().tolist(),
                    result.boxes.conf.detach().cpu().tolist(),
                ):
                    predictions.append({"class_id": int(class_id), "confidence": float(confidence),
                                        "box": [float(value) for value in box]})
            records.append({"image": str(image_path), "truth": truth, "predictions": predictions})
        dataset_records[name] = records

    dataset_records["combined"] = [record for name, records in dataset_records.items() if name != "combined" for record in records]
    payload = {
        "model": str(args.model.resolve()),
        "imgsz": args.imgsz,
        "iou_match_threshold": 0.50,
        "thresholds": thresholds,
        "results": {name: [score(records, threshold) for threshold in thresholds]
                    for name, records in dataset_records.items()},
    }
    (args.output / "confidence_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "confidence_sweep.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "threshold", "gt", "predictions", "tp", "fp", "fn",
                         "class_errors", "precision", "recall", "f1", "zero_error"])
        for name, rows in payload["results"].items():
            for row in rows:
                writer.writerow([name, row["threshold"], row["ground_truth"], row["predictions"],
                                 row["tp"], row["fp"], row["fn"], row["class_errors"],
                                 row["precision"], row["recall"], row["f1"], row["zero_error"]])
    render_report(payload, args.output / "confidence_sweep.png")
    summary = {}
    for name, rows in payload["results"].items():
        zero = [row["threshold"] for row in rows if row["zero_error"]]
        summary[name] = {"lowest_zero_error_threshold": min(zero) if zero else None,
                         "ground_truth_count": rows[0]["ground_truth"]}
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
