#!/usr/bin/env python3
"""Evaluate remediation checkpoints on synthetic and hand-labeled real data.

The synthetic evaluation uses Ultralytics' normal detection metrics.  The real
frames are evaluated as 30 fixed fixture slots per image: a detector prediction
is assigned to the nearest aperture when it contains the aperture center or is
within the configured slot-distance threshold.  An unassigned slot is scored
as ``no_vial``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ultralytics import YOLO


MODEL_CLASSES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")
REAL_CLASSES = MODEL_CLASSES + ("no_vial",)
REAL_LABEL_MAP = {
    "pass": "good_cap",
    "crooked_cap": "crooked_cap",
    "missing_cap": "missing_cap",
    "no_vial": "no_vial",
}


def model_names(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def evaluate_synthetic(model_path: Path, data_path: Path, split: str, work_dir: Path,
                       imgsz: int = 640) -> dict[str, Any]:
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=split,
        device=0,
        imgsz=imgsz,
        batch=8,
        workers=2,
        plots=False,
        verbose=False,
        project=str(work_dir / "ultralytics"),
        name=model_path.stem,
        exist_ok=True,
    )
    names = model_names(model)
    per_class: dict[str, dict[str, float]] = {}
    for index, name in names.items():
        if index >= len(metrics.box.p):
            continue
        per_class[name] = {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "mAP50": float(metrics.box.ap50[index]),
            "mAP50_95": float(metrics.box.ap[index]),
        }
    return {
        "data": str(data_path),
        "split": split,
        "metrics": {
            "precision": float(metrics.results_dict["metrics/precision(B)"]),
            "recall": float(metrics.results_dict["metrics/recall(B)"]),
            "mAP50": float(metrics.results_dict["metrics/mAP50(B)"]),
            "mAP50_95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
        },
        "per_class": per_class,
    }


def cropped_aperture_centers(apertures_path: Path) -> dict[str, tuple[float, float]]:
    source = json.loads(apertures_path.read_text())
    # The real frames are 50% center crops of 640x480 frames, so the crop is
    # x=[160,480], y=[120,360], resized to 320x240.
    return {
        slot: ((float(value["x"]) - 160.0) / 320.0 * 320.0,
               (float(value["y"]) - 120.0) / 240.0 * 240.0)
        for slot, value in source.items()
    }


def load_real_records(hand_dir: Path, image_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(hand_dir.glob("*.jpg.json")):
        record = json.loads(path.read_text())
        image = image_dir / record["filename"]
        if not image.is_file():
            raise FileNotFoundError(f"Missing real test image for {path}: {image}")
        truth = {}
        for slot, value in record["apertures"].items():
            label = str(value.get("label", "")).lower()
            if label not in REAL_LABEL_MAP:
                raise ValueError(f"Unsupported real label {label!r} in {path} at {slot}")
            truth[slot] = REAL_LABEL_MAP[label]
        records.append({"json": str(path), "image": image, "filename": record["filename"], "truth": truth})
    if not records:
        raise FileNotFoundError(f"No hand-labeled test JSON files found in {hand_dir}")
    return records


def classify_real_slots(
    model: YOLO,
    records: list[dict[str, Any]],
    centers: dict[str, tuple[float, float]],
    confidence: float,
    max_slot_distance: float,
    imgsz: int = 640,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    details: list[dict[str, Any]] = []
    names = model_names(model)
    results = model.predict(
        source=[record["image"] for record in records],
        device=0,
        imgsz=imgsz,
        conf=confidence,
        iou=0.7,
        max_det=200,
        verbose=False,
    )
    for record, result in zip(records, results):
        boxes = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            for box, class_index, score in zip(xyxy, classes, scores):
                x1, y1, x2, y2 = box
                boxes.append({
                    "class": names.get(int(class_index), str(int(class_index))),
                    "confidence": float(score),
                    "box": [float(value) for value in box],
                    "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                })
        image_predictions = {}
        for slot in sorted(record["truth"]):
            point = centers[slot]
            candidates = []
            for candidate in boxes:
                cx, cy = candidate["center"]
                distance = ((cx - point[0]) ** 2 + (cy - point[1]) ** 2) ** 0.5
                x1, y1, x2, y2 = candidate["box"]
                contains = x1 <= point[0] <= x2 and y1 <= point[1] <= y2
                if contains or distance <= max_slot_distance:
                    candidates.append((not contains, distance, -candidate["confidence"], candidate))
            if candidates:
                _, _, _, selected = min(candidates, key=lambda item: item[:3])
                prediction = selected["class"]
                if prediction not in REAL_CLASSES:
                    prediction = "no_vial"
                selected_output = dict(selected)
            else:
                prediction = "no_vial"
                selected_output = None
            truth = record["truth"][slot]
            y_true.append(truth)
            y_pred.append(prediction)
            image_predictions[slot] = {"truth": truth, "prediction": prediction, "detection": selected_output}
        details.append({"filename": record["filename"], "slots": image_predictions})
    return y_true, y_pred, details


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def real_metrics(y_true: list[str], y_pred: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_class = {}
    supported_precisions = []
    supported_recalls = []
    supported_f1s = []
    for class_name in REAL_CLASSES:
        tp = sum(true == class_name and pred == class_name for true, pred in zip(y_true, y_pred))
        fp = sum(true != class_name and pred == class_name for true, pred in zip(y_true, y_pred))
        fn = sum(true == class_name and pred != class_name for true, pred in zip(y_true, y_pred))
        support = sum(true == class_name for true in y_true)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1, "support": support,
                                 "true_positive": tp, "false_positive": fp, "false_negative": fn}
        if support:
            supported_precisions.append(precision or 0.0)
            supported_recalls.append(recall or 0.0)
            supported_f1s.append(f1 or 0.0)
    overall = {
        "accuracy": safe_div(sum(true == pred for true, pred in zip(y_true, y_pred)), len(y_true)),
        "macro_precision": sum(supported_precisions) / len(supported_precisions),
        "macro_recall": sum(supported_recalls) / len(supported_recalls),
        "macro_f1": sum(supported_f1s) / len(supported_f1s),
        "slots": len(y_true),
        "confusion": {true: dict(Counter(pred for actual, pred in zip(y_true, y_pred) if actual == true)) for true in REAL_CLASSES},
    }
    return overall, per_class


def value_text(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def make_sheet(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    model_order = list(result["models"])
    synthetic_rows = []
    synthetic_class_rows = []
    real_rows = []
    real_class_rows = []
    for model_name in model_order:
        item = result["models"][model_name]
        syn = item["synthetic"]["metrics"]
        synthetic_rows.append([model_name, *(value_text(syn[key]) for key in ("precision", "recall", "mAP50", "mAP50_95"))])
        for class_name in MODEL_CLASSES:
            values = item["synthetic"]["per_class"].get(class_name, {})
            synthetic_class_rows.append([model_name, class_name, *(value_text(values.get(key)) for key in ("precision", "recall", "mAP50", "mAP50_95"))])
        real = item["real"]["metrics"]
        real_rows.append([model_name, *(value_text(real[key]) for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")), str(real["slots"])])
        for class_name in REAL_CLASSES:
            values = item["real"]["per_class"][class_name]
            real_class_rows.append([model_name, class_name, *(value_text(values.get(key)) for key in ("precision", "recall", "f1")), str(values["support"])])

    fig, axes = plt.subplots(2, 2, figsize=(24, 17), gridspec_kw={"height_ratios": [1, 2.5]})
    fig.suptitle("Remediation detector comparison", fontsize=22, fontweight="bold", y=0.985)
    fig.text(0.5, 0.962, "Synthetic: YOLO detection metrics. Real: 2 hand-labeled frames / 60 fixture slots; unmatched detections score as no_vial.", ha="center", fontsize=11)

    def add_table(ax, title, columns, rows, font_size=9):
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=10)
        table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.55)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#244062")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#edf2f7")
        return table

    add_table(axes[0, 0], "Synthetic overall", ["Model", "Precision", "Recall", "mAP50", "mAP50-95"], synthetic_rows)
    add_table(axes[0, 1], "Real overall", ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Slots"], real_rows)
    add_table(axes[1, 0], "Synthetic per class", ["Model", "Class", "Precision", "Recall", "mAP50", "mAP50-95"], synthetic_class_rows, 8)
    add_table(axes[1, 1], "Real per class", ["Model", "Class", "Precision", "Recall", "F1", "Support"], real_class_rows, 8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.03, hspace=0.12, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--synthetic-data", type=Path, required=True)
    parser.add_argument("--synthetic-split", default="test")
    parser.add_argument("--hand-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--apertures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-confidence", type=float, default=0.25)
    parser.add_argument("--max-slot-distance", type=float, default=18.0)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    models = {}
    for item in args.models:
        name, checkpoint = item.split("=", 1)
        models[name] = Path(checkpoint)
        if not models[name].is_file():
            raise FileNotFoundError(models[name])
    records = load_real_records(args.hand_dir, args.image_dir)
    centers = cropped_aperture_centers(args.apertures)
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "synthetic_data": str(args.synthetic_data),
        "synthetic_split": args.synthetic_split,
        "real_hand_dir": str(args.hand_dir),
        "real_images": [record["filename"] for record in records],
        "real_confidence": args.real_confidence,
        "max_slot_distance": args.max_slot_distance,
        "imgsz": args.imgsz,
        "models": {},
    }
    for name, checkpoint in models.items():
        print(f"[synthetic] {name}", flush=True)
        synthetic = evaluate_synthetic(
            checkpoint, args.synthetic_data, args.synthetic_split, args.out, args.imgsz
        )
        print(f"[real] {name}", flush=True)
        model = YOLO(str(checkpoint))
        y_true, y_pred, details = classify_real_slots(
            model, records, centers, args.real_confidence,
            args.max_slot_distance, args.imgsz
        )
        overall, per_class = real_metrics(y_true, y_pred)
        result["models"][name] = {
            "checkpoint": str(checkpoint),
            "synthetic": synthetic,
            "real": {"metrics": overall, "per_class": per_class, "details": details},
        }
        (args.out / f"{name}.json").write_text(json.dumps(result["models"][name], indent=2) + "\n")
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    make_sheet(result, args.out / "remediation_model_comparison.png")
    print(json.dumps({"output": str(args.out), "models": list(models), "real_slots": len(records) * 30}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
