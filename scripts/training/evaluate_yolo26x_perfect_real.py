#!/usr/bin/env python3
"""Compare the preserved YOLO26x baseline with the corrected-box fine-tune."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.evaluate_yolo_vlm_box_refinement import match_operational  # noqa: E402
from scripts.training.refine_yolo_boxes_with_vlm import (  # noqa: E402
    CLASS_NAMES,
    draw_detections,
    load_reviewed_labels,
    panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint")
    parser.add_argument("--synthetic-data", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--synthetic-samples", type=int, default=4)
    return parser.parse_args()


def metric_payload(metrics: Any) -> dict[str, Any]:
    names_raw = metrics.names
    names = {int(key): str(value) for key, value in names_raw.items()} if isinstance(names_raw, dict) else dict(enumerate(names_raw))
    class_indices = [int(value) for value in metrics.box.ap_class_index]
    per_class = {}
    for metric_index, class_index in enumerate(class_indices):
        per_class[names[class_index]] = {
            "precision": float(metrics.box.p[metric_index]),
            "recall": float(metrics.box.r[metric_index]),
            "ap50": float(metrics.box.ap50[metric_index]),
            "map50_95": float(metrics.box.ap[metric_index]),
        }
    return {
        "overall": {
            "precision": float(metrics.results_dict["metrics/precision(B)"]),
            "recall": float(metrics.results_dict["metrics/recall(B)"]),
            "ap50": float(metrics.results_dict["metrics/mAP50(B)"]),
            "map50_95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
        },
        "per_class": per_class,
    }


def evaluate_standard(model: YOLO, data: Path, split: str, output: Path, name: str, imgsz: int) -> dict[str, Any]:
    metrics = model.val(
        data=str(data), split=split, imgsz=imgsz, batch=4, device=0, workers=4,
        conf=0.001, iou=0.70, max_det=300, plots=True, verbose=False,
        project=str(output / "ultralytics"), name=name, exist_ok=True,
    )
    return metric_payload(metrics)


def predictions(model: YOLO, image_paths: list[Path], imgsz: int) -> list[list[dict[str, Any]]]:
    results = model.predict(
        source=[str(path) for path in image_paths], imgsz=imgsz, device=0,
        conf=0.001, iou=0.70, max_det=300, verbose=False,
    )
    output = []
    for result in results:
        detections = []
        if result.boxes is not None:
            for box, class_id, confidence in zip(
                result.boxes.xyxy.detach().cpu().tolist(),
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
            ):
                class_id = int(class_id)
                detections.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": float(confidence),
                    "box": [float(value) for value in box],
                })
        output.append(detections)
    return output


def real_records(real_root: Path, model_predictions: list[list[dict[str, Any]]], image_paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for image_path, detected in zip(image_paths, model_predictions):
        with Image.open(image_path) as image:
            size = image.size
        truth = load_reviewed_labels(real_root / "labels" / "all" / f"{image_path.stem}.txt", *size)
        records.append({"image": str(image_path), "size": size, "ground_truth": truth, "predictions": detected})
    return records


def operational_summary(records: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    overall = match_operational(records, "predictions", None, confidence)
    per_class = {
        class_name: match_operational(records, "predictions", class_id, confidence)
        for class_id, class_name in enumerate(CLASS_NAMES)
        if any(item["class_id"] == class_id for record in records for item in record["ground_truth"])
    }
    return {"overall": overall, "per_class": per_class}


def make_contact_sheet(rows: list[Image.Image], output: Path) -> None:
    if not rows:
        return
    sheet = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "#c8c8c8")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * row.height))
    sheet.save(output, quality=95)


def render_rows(
    image_paths: list[Path], truths: list[list[dict[str, Any]]],
    model_order: list[str], all_predictions: dict[str, list[list[dict[str, Any]]]],
    confidence: float, output: Path,
) -> None:
    rows = []
    for image_index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        cells = [panel(draw_detections(image, truths[image_index]), f"Ground truth — {image_path.name}", (640, 420))]
        for model_name in model_order:
            visible = [item for item in all_predictions[model_name][image_index] if item["confidence"] >= confidence]
            cells.append(panel(draw_detections(image, visible, True), model_name, (640, 420)))
        row = Image.new("RGB", (640 * len(cells), 420), "#c8c8c8")
        for cell_index, cell in enumerate(cells):
            row.paste(cell, (640 * cell_index, 0))
        rows.append(row)
    make_contact_sheet(rows, output)


def sample_paths(data: Path, count: int) -> tuple[list[Path], Path]:
    import yaml

    config = yaml.safe_load(data.read_text())
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    image_dir = root / config["test"]
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) > count:
        indices = sorted(set(round(index * (len(paths) - 1) / (count - 1)) for index in range(count)))
        paths = [paths[index] for index in indices]
    return paths, root


def fmt(value: Any) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.3f}"


def render_metric_report(payload: dict[str, Any], output: Path) -> None:
    synthetic_rows, real_rows, localization_rows = [], [], []
    for model_name, values in payload["models"].items():
        syn = values["synthetic"]["overall"]
        real = values["real"]["standard"]["overall"]
        loc = values["real"]["operational"]["overall"]
        synthetic_rows.append([model_name, *(fmt(syn[key]) for key in ("precision", "recall", "ap50", "map50_95"))])
        real_rows.append([model_name, *(fmt(real[key]) for key in ("precision", "recall", "ap50", "map50_95"))])
        localization_rows.append([
            model_name, fmt(loc["precision"]), fmt(loc["recall"]), fmt(loc["f1"]),
            fmt(loc["mean_iou_matched"]), fmt(loc["mean_center_error_percent_diagonal"]),
            f'{loc["prediction_count"]}/{loc["ground_truth_count"]}',
        ])

    class_rows = []
    for model_name, values in payload["models"].items():
        syn_classes = values["synthetic"]["per_class"]
        real_classes = values["real"]["standard"]["per_class"]
        real_loc = values["real"]["operational"]["per_class"]
        for class_name in CLASS_NAMES:
            syn = syn_classes.get(class_name, {})
            real = real_classes.get(class_name, {})
            loc = real_loc.get(class_name, {})
            class_rows.append([
                model_name, class_name,
                fmt(syn.get("map50_95")), fmt(real.get("map50_95")),
                fmt(loc.get("mean_iou_matched")), fmt(loc.get("mean_center_error_percent_diagonal")),
            ])

    fig, axes = plt.subplots(4, 1, figsize=(18, 18), gridspec_kw={"height_ratios": [1, 1, 1, 2.4]})
    fig.suptitle("YOLO26x corrected real-box fine-tune", fontsize=22, fontweight="bold", y=0.985)
    fig.text(0.5, 0.963, "Synthetic metrics are held-out. Real metrics use all five training images and are memorization/localization checks.", ha="center", fontsize=11)

    def table(ax: Any, title: str, columns: list[str], rows: list[list[str]], size: int = 10) -> None:
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=8)
        item = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
        item.auto_set_font_size(False)
        item.set_fontsize(size)
        item.scale(1, 1.45)
        for (row, _), cell in item.get_celld().items():
            if row == 0:
                cell.set_facecolor("#243b53")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#edf2f7")

    columns = ["Model", "Precision", "Recall", "AP50", "mAP50–95"]
    table(axes[0], "Held-out synthetic test", columns, synthetic_rows)
    table(axes[1], "Five corrected real images", columns, real_rows)
    table(
        axes[2],
        f'Real operational localization at confidence {payload["operational_confidence"]:.2f}',
        ["Model", "Precision", "Recall", "F1", "Mean IoU", "Center err % diag", "Pred/GT"],
        localization_rows,
    )
    table(axes[3], "Per-class localization summary", ["Model", "Class", "Synthetic mAP", "Real mAP", "Real mean IoU", "Real center err %"], class_rows, 9)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.025, hspace=0.18)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    models = {}
    for spec in args.models:
        name, checkpoint = spec.split("=", 1)
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        models[name] = path

    real_images = sorted(path for path in (args.real_root / "images" / "all").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    real_data = args.output / "real_data.yaml"
    real_data.write_text(
        f"path: {args.real_root.resolve()}\ntrain: images/all\nval: images/all\ntest: images/all\n"
        "names:\n  0: good_cap\n  1: crooked_cap\n  2: missing_cap\n  3: table_cap\n"
    )
    synthetic_images, synthetic_root = sample_paths(args.synthetic_data, args.synthetic_samples)
    real_truth = []
    for path in real_images:
        with Image.open(path) as image:
            real_truth.append(load_reviewed_labels(args.real_root / "labels" / "all" / f"{path.stem}.txt", *image.size))
    synthetic_truth = []
    for path in synthetic_images:
        with Image.open(path) as image:
            synthetic_truth.append(load_reviewed_labels(synthetic_root / "labels" / "test" / f"{path.stem}.txt", *image.size))

    payload: dict[str, Any] = {
        "warning": "The five real images were used for training; only synthetic test metrics are held out.",
        "imgsz": args.imgsz,
        "operational_confidence": args.confidence,
        "models": {},
    }
    real_predictions, synthetic_predictions = {}, {}
    for model_name, checkpoint in models.items():
        print(f"[evaluate] {model_name}: {checkpoint}", flush=True)
        model = YOLO(str(checkpoint))
        synthetic = evaluate_standard(model, args.synthetic_data, "test", args.output, f"{model_name}_synthetic", args.imgsz)
        real_standard = evaluate_standard(model, real_data, "test", args.output, f"{model_name}_real", args.imgsz)
        real_predictions[model_name] = predictions(model, real_images, args.imgsz)
        synthetic_predictions[model_name] = predictions(model, synthetic_images, args.imgsz)
        records = real_records(args.real_root, real_predictions[model_name], real_images)
        payload["models"][model_name] = {
            "checkpoint": str(checkpoint),
            "synthetic": synthetic,
            "real": {"standard": real_standard, "operational": operational_summary(records, args.confidence)},
        }

    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    render_metric_report(payload, args.output / "yolo26x_perfect_real_finetune_report.png")
    render_rows(real_images, real_truth, list(models), real_predictions, args.confidence, args.output / "real_all5_predictions.png")
    render_rows(synthetic_images, synthetic_truth, list(models), synthetic_predictions, args.confidence, args.output / "synthetic_prediction_samples.png")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
