#!/usr/bin/env python3
"""Evaluate a YOLO checkpoint on synthetic and manually labeled real data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


CLASS_NAMES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")
COLORS = {
    0: (30, 190, 80),
    1: (255, 145, 20),
    2: (225, 45, 45),
    3: (160, 70, 210),
}
SHORT_LABELS = {
    0: "G",
    1: "C",
    2: "M",
    3: "T",
}
DEFAULT_LIMITATION = (
    "PARTIAL CHECKPOINT: training was canceled after epoch 22 (23/50), and the model "
    "was trained on the now-known WRONG synthetic tray scene. Results are diagnostic only."
)


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def prepare_real_dataset(annotation_path: Path, repo_root: Path, output: Path) -> tuple[Path, dict[str, Any]]:
    annotations = json.loads(annotation_path.read_text())
    width, height = annotations["image_size"]
    image_dir = output / "images" / "test"
    label_dir = output / "labels" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    provenance = []
    for item in annotations["images"]:
        source = repo_root / item["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = image_dir / item["filename"]
        shutil.copy2(source, destination)
        with Image.open(destination) as image:
            if image.size != (width, height):
                raise ValueError(f"Unexpected image size for {destination}: {image.size}")
        lines = []
        for box in item["boxes"]:
            x1, y1, x2, y2 = (float(value) for value in box["xyxy"])
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid box in {item['filename']}: {box}")
            cx = (x1 + x2) / (2 * width)
            cy = (y1 + y2) / (2 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(f"{int(box['class_id'])} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}")
        label_path = label_dir / f"{destination.stem}.txt"
        label_path.write_text("\n".join(lines) + "\n")
        provenance.append({
            "filename": item["filename"],
            "source": item["source"],
            "selection_reason": item.get("selection_reason", item.get("description", "operator-selected frame")),
            "box_count": len(item["boxes"]),
            "uncertain_boxes": [box for box in item["boxes"] if box.get("confidence") != "high" and "confidence" in box],
        })

    data_yaml = output / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(output.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }, sort_keys=False))
    (output / "provenance.json").write_text(json.dumps({
        "labeling_policy": annotations["labeling_policy"],
        "images": provenance,
    }, indent=2) + "\n")
    return data_yaml, annotations


def resolve_dataset(data_yaml: Path, split: str) -> tuple[Path, Path]:
    config = yaml.safe_load(data_yaml.read_text())
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    image_dir = root / str(config[split])
    label_dir = root / "labels" / split
    return image_dir, label_dir


def label_counts(label_dir: Path) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for path in sorted(label_dir.glob("*.txt")):
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields:
                counts[int(fields[0])] += 1
    return {name: counts[index] for index, name in enumerate(CLASS_NAMES)}


def metric_value(results: dict[str, Any], key: str) -> float:
    return float(results[key])


def extract_metrics(metrics: Any, counts: dict[str, int]) -> dict[str, Any]:
    result_dict = metrics.results_dict
    class_indices = getattr(metrics.box, "ap_class_index", np.arange(len(metrics.box.p)))
    class_indices = [int(value) for value in np.asarray(class_indices).tolist()]
    positions = {class_id: position for position, class_id in enumerate(class_indices)}
    per_class: dict[str, Any] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        support = counts[name]
        position = positions.get(class_id)
        if support == 0 or position is None:
            values = {"precision": None, "recall": None, "mAP50": None, "mAP50_95": None}
        else:
            values = {
                "precision": float(metrics.box.p[position]),
                "recall": float(metrics.box.r[position]),
                "mAP50": float(metrics.box.ap50[position]),
                "mAP50_95": float(metrics.box.ap[position]),
            }
        per_class[name] = {"ground_truth_count": support, **values}
    return {
        "overall": {
            "precision": metric_value(result_dict, "metrics/precision(B)"),
            "recall": metric_value(result_dict, "metrics/recall(B)"),
            "mAP50": metric_value(result_dict, "metrics/mAP50(B)"),
            "mAP50_95": metric_value(result_dict, "metrics/mAP50-95(B)"),
            "ground_truth_count": sum(counts.values()),
        },
        "per_class": per_class,
    }


def evaluate_domain(model: YOLO, data_yaml: Path, split: str, output: Path, name: str, imgsz: int) -> dict[str, Any]:
    _, label_dir = resolve_dataset(data_yaml, split)
    counts = label_counts(label_dir)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        device=0,
        imgsz=imgsz,
        batch=2,
        workers=4,
        plots=True,
        verbose=False,
        project=str((output / "ultralytics").resolve()),
        name=name,
        exist_ok=True,
    )
    return {"data": str(data_yaml), "split": split, "ground_truth_counts": counts, **extract_metrics(metrics, counts)}


def yolo_boxes(label_path: Path, size: tuple[int, int]) -> list[tuple[int, tuple[int, int, int, int]]]:
    width, height = size
    boxes = []
    for line in label_path.read_text().splitlines():
        class_id, cx, cy, box_width, box_height = line.split()
        cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
        boxes.append((int(class_id), (
            round((cx - box_width / 2) * width),
            round((cy - box_height / 2) * height),
            round((cx + box_width / 2) * width),
            round((cy + box_height / 2) * height),
        )))
    return boxes


def draw_detections(
    image: Image.Image,
    detections: list[tuple[int, tuple[int, int, int, int]]],
) -> Image.Image:
    """Draw compact tags inside boxes so dense tray detections stay legible."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = get_font(max(11, image.width // 160))
    line_width = max(2, image.width // 500)
    tag_padding = max(2, image.width // 900)

    for class_id, raw_box in detections:
        x1, y1, x2, y2 = raw_box
        x1 = max(0, min(image.width - 1, x1))
        y1 = max(0, min(image.height - 1, y1))
        x2 = max(x1 + 1, min(image.width - 1, x2))
        y2 = max(y1 + 1, min(image.height - 1, y2))
        color = COLORS[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        # A one-letter tag conveys the class without covering adjacent vials.
        tag = SHORT_LABELS[class_id]
        text_box = draw.textbbox((0, 0), tag, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        tag_x = x1 + line_width
        tag_y = y1 + line_width
        tag_right = min(x2, tag_x + text_width + 2 * tag_padding)
        tag_bottom = min(y2, tag_y + text_height + 2 * tag_padding)
        draw.rectangle((tag_x, tag_y, tag_right, tag_bottom), fill=color)
        draw.text(
            (tag_x + tag_padding, tag_y + tag_padding - text_box[1]),
            tag,
            fill="white",
            font=font,
        )
    return image


def draw_ground_truth(image_path: Path, label_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return draw_detections(image, yolo_boxes(label_path, image.size))


def draw_prediction(prediction: Any) -> Image.Image:
    image = Image.fromarray(prediction.orig_img[..., ::-1]).convert("RGB")
    detections: list[tuple[int, tuple[int, int, int, int]]] = []
    if prediction.boxes is not None:
        xyxy = prediction.boxes.xyxy.detach().cpu().numpy()
        classes = prediction.boxes.cls.detach().cpu().numpy()
        for coordinates, class_id in zip(xyxy, classes):
            detections.append((
                int(class_id),
                tuple(round(float(value)) for value in coordinates),
            ))
    return draw_detections(image, detections)


def titled_panel(image: Image.Image, title: str, width: int = 900, height: int = 590) -> Image.Image:
    image = image.copy()
    header_height = 72
    image.thumbnail((width - 10, height - header_height - 5), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((10, 7), title, fill="black", font=get_font(18))

    legend_font = get_font(12)
    legend_x = 10
    legend_y = 38
    for class_id, name in enumerate(CLASS_NAMES):
        color = COLORS[class_id]
        tag = SHORT_LABELS[class_id]
        draw.rectangle((legend_x, legend_y, legend_x + 16, legend_y + 16), fill=color)
        draw.text((legend_x + 4, legend_y + 1), tag, fill="white", font=legend_font)
        legend_text = name.replace("_", " ")
        draw.text((legend_x + 21, legend_y), legend_text, fill="black", font=legend_font)
        legend_x += 21 + draw.textlength(legend_text, font=legend_font) + 22

    panel.paste(image, ((width - image.width) // 2, header_height))
    return panel


def render_pair(
    model: YOLO,
    image_path: Path,
    label_path: Path,
    output: Path,
    imgsz: int,
    annotation_title: str,
) -> Image.Image:
    ground_truth = draw_ground_truth(image_path, label_path)
    prediction = model.predict(
        source=[image_path], device=0, imgsz=imgsz, conf=0.25, iou=0.7,
        max_det=200, verbose=False,
    )[0]
    predicted = draw_prediction(prediction)
    left = titled_panel(ground_truth, annotation_title)
    right = titled_panel(predicted, "Model predictions (confidence >= 0.25)")
    pair = Image.new("RGB", (left.width + right.width, left.height), "#d9d9d9")
    pair.paste(left, (0, 0))
    pair.paste(right, (left.width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    pair.save(output)
    return pair


def select_synthetic_samples(data_yaml: Path, count: int = 4) -> list[tuple[Path, Path]]:
    image_dir, label_dir = resolve_dataset(data_yaml, "test")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise FileNotFoundError(image_dir)
    indices = sorted(set(round(index * (len(images) - 1) / max(1, count - 1)) for index in range(count)))
    return [(images[index], label_dir / f"{images[index].stem}.txt") for index in indices]


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def add_table(axis: Any, title: str, columns: list[str], rows: list[list[str]], font_size: int = 9) -> None:
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=8)
    table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.52)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#243b53")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf2f7")


def make_report(result: dict[str, Any], sample_pairs: dict[str, list[tuple[str, Path]]], output: Path) -> None:
    sample_rows: list[tuple[str, list[tuple[str, Path]]]] = [
        ("synthetic", sample_pairs["synthetic"][:2]),
    ]
    for start in range(0, len(sample_pairs["real"]), 2):
        sample_rows.append(("real", sample_pairs["real"][start:start + 2]))

    figure = plt.figure(figsize=(23, 14 + 5 * len(sample_rows)), facecolor="white")
    grid = figure.add_gridspec(
        3 + len(sample_rows), 2,
        height_ratios=[0.8, 1.4, 2.4, *([4.2] * len(sample_rows))],
        hspace=0.22, wspace=0.08,
    )
    figure.suptitle(
        result["title"].upper(),
        color="#b91c1c", fontsize=22, fontweight="bold", y=0.992,
    )

    metadata = figure.add_subplot(grid[0, :])
    metadata.axis("off")
    metadata.text(0.0, 0.85, result["limitation"], fontsize=13, fontweight="bold",
                  color="#991b1b", va="top", wrap=True)
    metadata.text(
        0.0, 0.45,
        f"Checkpoint: {result['checkpoint']}\n"
        f"Synthetic: 100 held-out qualified-scene images at {result['imgsz']} px. "
        f"Real: {result['real']['image_count']} labeled frames / {result['real']['overall']['ground_truth_count']} annotated objects.\n"
        "Interpret the real metrics under the limitation stated above. "
        "Box tags: G=good, C=crooked, M=missing, T=table. N/A means the class has no annotation support.",
        fontsize=11, va="top", wrap=True,
    )

    overall_rows = []
    for domain in ("synthetic", "real"):
        values = result[domain]["overall"]
        overall_rows.append([domain.title(), str(values["ground_truth_count"]), *(fmt(values[key]) for key in ("precision", "recall", "mAP50", "mAP50_95"))])
    overall_axis = figure.add_subplot(grid[1, :])
    add_table(overall_axis, "Aggregate detection metrics", ["Domain", "GT boxes", "Precision", "Recall", "mAP50", "mAP50-95"], overall_rows, 11)

    for column, domain in enumerate(("synthetic", "real")):
        rows = []
        for name in CLASS_NAMES:
            values = result[domain]["per_class"][name]
            rows.append([name, str(values["ground_truth_count"]), *(fmt(values[key]) for key in ("precision", "recall", "mAP50", "mAP50_95"))])
        axis = figure.add_subplot(grid[2, column])
        add_table(axis, f"{domain.title()} per-class metrics", ["Class", "GT", "Precision", "Recall", "mAP50", "mAP50-95"], rows, 9)

    for row_index, (domain, samples) in enumerate(sample_rows, start=3):
        for column, (name, path) in enumerate(samples):
            axis = figure.add_subplot(grid[row_index, column])
            axis.axis("off")
            axis.set_title(f"{domain.title()} example: {name}", fontsize=12, fontweight="bold")
            axis.imshow(Image.open(path))

    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_csv(result: dict[str, Any], output: Path) -> None:
    fields = ["domain", "scope", "class", "ground_truth_count", "precision", "recall", "mAP50", "mAP50_95"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for domain in ("synthetic", "real"):
            writer.writerow({"domain": domain, "scope": "overall", "class": "all", **result[domain]["overall"]})
            for name in CLASS_NAMES:
                writer.writerow({"domain": domain, "scope": "class", "class": name, **result[domain]["per_class"][name]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic-data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--title", default="YOLO remediation detector evaluation")
    parser.add_argument("--limitation", default="Results are diagnostic until the real set is expanded and human-reviewed.")
    parser.add_argument("--training-state", default="{}",
                        help="JSON object describing checkpoint training provenance")
    parser.add_argument("--report-name", default="yolo_remediation_eval_report.png")
    args = parser.parse_args()
    training_state = json.loads(args.training_state)
    if not isinstance(training_state, dict):
        parser.error("--training-state must decode to a JSON object")

    repo_root = Path.cwd().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for required in (args.checkpoint, args.synthetic_data, args.annotations):
        if not required.is_file():
            raise FileNotFoundError(required)

    real_data, annotations = prepare_real_dataset(args.annotations, repo_root, args.out / "real_dataset")
    model = YOLO(str(args.checkpoint))
    print("[evaluate] synthetic held-out test", flush=True)
    synthetic = evaluate_domain(model, args.synthetic_data, "test", args.out, "synthetic", args.imgsz)
    print("[evaluate] manually labeled real test", flush=True)
    real = evaluate_domain(model, real_data, "test", args.out, "real", args.imgsz)
    real["image_count"] = len(annotations["images"])

    result = {
        "schema_version": 1,
        "title": args.title,
        "limitation": args.limitation,
        "checkpoint": str(args.checkpoint),
        "training_state": training_state,
        "imgsz": args.imgsz,
        "synthetic": synthetic,
        "real": real,
        "real_annotation_file": str(args.annotations),
    }

    sample_pairs: dict[str, list[tuple[str, Path]]] = {"synthetic": [], "real": []}
    synthetic_samples = select_synthetic_samples(args.synthetic_data, 4)
    for image_path, label_path in synthetic_samples:
        target = args.out / "prediction_examples" / "synthetic" / f"{image_path.stem}_gt_vs_prediction.png"
        render_pair(model, image_path, label_path, target, args.imgsz, "Generated synthetic annotations")
        sample_pairs["synthetic"].append((image_path.name, target))
    real_image_dir, real_label_dir = resolve_dataset(real_data, "test")
    for image_path in sorted(real_image_dir.glob("*.png")):
        label_path = real_label_dir / f"{image_path.stem}.txt"
        target = args.out / "prediction_examples" / "real" / f"{image_path.stem}_gt_vs_prediction.png"
        render_pair(model, image_path, label_path, target, args.imgsz, "Reference real annotations")
        sample_pairs["real"].append((image_path.name, target))

    result["prediction_examples"] = {domain: [str(path) for _name, path in values] for domain, values in sample_pairs.items()}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "synthetic_metrics.json").write_text(json.dumps(synthetic, indent=2) + "\n")
    (args.out / "real_metrics.json").write_text(json.dumps(real, indent=2) + "\n")
    write_csv(result, args.out / "metrics.csv")
    report_path = args.out / args.report_name
    make_report(result, sample_pairs, report_path)
    print(json.dumps({
        "report": str(report_path),
        "metrics": str(args.out / "metrics.json"),
        "real_images": real["image_count"],
        "real_ground_truth": real["ground_truth_counts"],
        "synthetic_ground_truth": synthetic["ground_truth_counts"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
