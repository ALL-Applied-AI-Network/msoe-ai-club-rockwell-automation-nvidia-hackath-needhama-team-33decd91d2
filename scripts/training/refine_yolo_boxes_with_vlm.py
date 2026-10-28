#!/usr/bin/env python3
"""Refine YOLO box geometry with class-agnostic LocateAnything proposals.

YOLO remains authoritative for object count, class, and confidence. VLM class
names are discarded: overlapping VLM outputs are clustered into geometry-only
proposals, then matched one-to-one to YOLO detections.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


CLASS_NAMES = ["good_cap", "crooked_cap", "missing_cap", "table_cap"]
SHORT_NAMES = ["GOOD", "CROOK", "MISS", "TABLE"]
COLORS = [(40, 190, 75), (255, 145, 25), (225, 55, 60), (45, 135, 240)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--vlm-view", default="raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--yolo-iou", type=float, default=0.70)
    parser.add_argument("--cluster-iou", type=float, default=0.65)
    parser.add_argument("--minimum-match-score", type=float, default=0.35)
    return parser.parse_args()


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def diagonal(box: list[float]) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def center_distance(a: list[float], b: list[float]) -> float:
    ac, bc = center(a), center(b)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


def cluster_vlm_boxes(predictions: list[dict], threshold: float) -> list[dict]:
    """Union near-identical cross-query boxes and retain median geometry."""
    count = len(predictions)
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    for left in range(count):
        for right in range(left + 1, count):
            a, b = predictions[left]["box"], predictions[right]["box"]
            norm_distance = center_distance(a, b) / max(diagonal(a), diagonal(b), 1.0)
            # Repeated generative queries can shift a box slightly. Require strong
            # overlap, or very close centers with comparable box dimensions.
            area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
            area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            area_ratio = max(area_a, area_b) / min(area_a, area_b)
            if box_iou(a, b) >= threshold or (norm_distance <= 0.20 and area_ratio <= 1.8):
                union(left, right)

    groups: dict[int, list[dict]] = {}
    for index, prediction in enumerate(predictions):
        groups.setdefault(find(index), []).append(prediction)

    proposals = []
    for members in groups.values():
        geometry = np.median(np.asarray([item["box"] for item in members], dtype=np.float64), axis=0).tolist()
        proposals.append({
            "box": geometry,
            "support": len(members),
            "source_classes": sorted({int(item["class_id"]) for item in members}),
            "members": members,
        })
    return proposals


def match_score(yolo_box: list[float], proposal_box: list[float]) -> tuple[float, float, float]:
    overlap = box_iou(yolo_box, proposal_box)
    norm_distance = center_distance(yolo_box, proposal_box) / max(
        diagonal(yolo_box), diagonal(proposal_box), 1.0
    )
    center_score = max(0.0, 1.0 - norm_distance)
    return 0.75 * overlap + 0.25 * center_score, overlap, norm_distance


def match_proposals(detections: list[dict], proposals: list[dict], minimum_score: float) -> list[dict]:
    candidates = []
    for detection_index, detection in enumerate(detections):
        for proposal_index, proposal in enumerate(proposals):
            score, overlap, norm_distance = match_score(detection["box"], proposal["box"])
            if (overlap >= 0.15 or norm_distance <= 0.50) and score >= minimum_score:
                candidates.append((score, overlap, -norm_distance, detection_index, proposal_index))
    candidates.sort(reverse=True)

    used_detections, used_proposals = set(), set()
    matches = {}
    for score, overlap, neg_distance, detection_index, proposal_index in candidates:
        if detection_index in used_detections or proposal_index in used_proposals:
            continue
        used_detections.add(detection_index)
        used_proposals.add(proposal_index)
        matches[detection_index] = {
            "proposal_index": proposal_index,
            "score": score,
            "iou": overlap,
            "normalized_center_distance": -neg_distance,
        }

    refined = []
    for index, detection in enumerate(detections):
        item = dict(detection)
        item["yolo_box"] = detection["box"]
        match = matches.get(index)
        if match is None:
            item.update({"refined": False, "match": None})
        else:
            proposal = proposals[match["proposal_index"]]
            item["box"] = proposal["box"]
            item["refined"] = True
            item["match"] = {
                **match,
                "vlm_support": proposal["support"],
                "vlm_source_classes": proposal["source_classes"],
                "center_shift_pixels": center_distance(detection["box"], proposal["box"]),
            }
        refined.append(item)
    return refined


def load_reviewed_labels(path: Path, width: int, height: int) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id = int(fields[0])
        cx, cy, box_width, box_height = map(float, fields[1:])
        records.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "box": [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ],
        })
    return records


def write_yolo_labels(path: Path, detections: list[dict], width: int, height: int) -> None:
    lines = []
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        box_width, box_height = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f'{item["class_id"]} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def get_font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list[dict], show_confidence: bool = False) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 600)
    label_font = get_font(max(14, image.width // 115))
    for item in detections:
        class_id = int(item["class_id"])
        box = tuple(round(value) for value in item["box"])
        color = COLORS[class_id]
        draw.rectangle(box, outline=color, width=width)
        label = SHORT_NAMES[class_id]
        if show_confidence and "confidence" in item:
            label += f' {item["confidence"]:.2f}'
        text_box = draw.textbbox((box[0], box[1]), label, font=label_font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((box[0], box[1]), label, fill=color, font=label_font)
        cx, cy = center(item["box"])
        radius = max(2, width)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    return image


def panel(image: Image.Image, title: str, target_size: tuple[int, int] = (960, 590)) -> Image.Image:
    canvas = Image.new("RGB", target_size, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="black", font=get_font(22))
    view = image.copy()
    view.thumbnail((target_size[0] - 16, target_size[1] - 48), Image.Resampling.LANCZOS)
    canvas.paste(view, ((target_size[0] - view.width) // 2, 42))
    return canvas


def save_comparison(image: Image.Image, reviewed: list[dict], yolo: list[dict], refined: list[dict], path: Path) -> None:
    panels = [
        panel(draw_detections(image, reviewed), "Reviewed human labels"),
        panel(draw_detections(image, yolo, True), "YOLO classes and original boxes"),
        panel(draw_detections(image, refined, True), "YOLO classes + VLM-refined boxes"),
    ]
    sheet = Image.new("RGB", (panels[0].width * 3, panels[0].height), "#d0d0d0")
    for index, item in enumerate(panels):
        sheet.paste(item, (index * item.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vlm_payload = json.loads(args.vlm_predictions.read_text())
    if args.vlm_view not in vlm_payload:
        raise KeyError(f"VLM view {args.vlm_view!r} not found; choices: {list(vlm_payload)}")
    vlm_by_name = {Path(record["image"]).name: record for record in vlm_payload[args.vlm_view]}

    model = YOLO(str(args.model))
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    manifest = {
        "model": str(args.model),
        "vlm_predictions": str(args.vlm_predictions),
        "vlm_view": args.vlm_view,
        "principle": "YOLO count/class/confidence retained; VLM class discarded; matched geometry only replaced",
        "confidence": args.confidence,
        "imgsz": args.imgsz,
        "images": [],
    }
    comparison_paths = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = args.reviewed_labels / f"{image_path.stem}.txt"
        vlm_record = vlm_by_name.get(image_path.name)
        if vlm_record is None:
            raise KeyError(f"No VLM record for {image_path.name}")

        result = model.predict(
            source=str(image_path), device=0, imgsz=args.imgsz, conf=args.confidence,
            iou=args.yolo_iou, max_det=200, verbose=False,
        )[0]
        yolo = []
        for box, class_id, confidence in zip(
            result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.cls.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
        ):
            class_id = int(class_id)
            yolo.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(confidence),
                "box": [float(value) for value in box],
            })

        proposals = cluster_vlm_boxes(vlm_record["predictions"], args.cluster_iou)
        refined = match_proposals(yolo, proposals, args.minimum_match_score)
        reviewed = load_reviewed_labels(label_path, width, height)
        write_yolo_labels(args.output / "labels" / "yolo" / f"{image_path.stem}.txt", yolo, width, height)
        write_yolo_labels(args.output / "labels" / "vlm_refined" / f"{image_path.stem}.txt", refined, width, height)
        comparison_path = args.output / "comparisons" / f"{image_path.stem}_comparison.png"
        save_comparison(image, reviewed, yolo, refined, comparison_path)
        comparison_paths.append(comparison_path)

        changed = [item for item in refined if item["refined"]]
        record = {
            "image": str(image_path),
            "reviewed_label": str(label_path),
            "yolo_detection_count": len(yolo),
            "vlm_raw_box_count": len(vlm_record["predictions"]),
            "vlm_geometry_proposal_count": len(proposals),
            "refined_count": len(changed),
            "retained_yolo_count": len(refined) - len(changed),
            "mean_center_shift_pixels": float(np.mean([item["match"]["center_shift_pixels"] for item in changed])) if changed else 0.0,
            "detections": refined,
        }
        manifest["images"].append(record)
        (args.output / "metadata" / f"{image_path.stem}.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output / "metadata" / f"{image_path.stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"{image_path.name}: YOLO={len(yolo)} VLM proposals={len(proposals)} "
            f"refined={len(changed)} retained={len(refined) - len(changed)}",
            flush=True,
        )

    if comparison_paths:
        thumbnails = []
        for path in comparison_paths:
            image = Image.open(path).convert("RGB")
            image.thumbnail((1440, 295), Image.Resampling.LANCZOS)
            thumbnails.append(image.copy())
        contact = Image.new("RGB", (1440, 295 * len(thumbnails)), "white")
        for index, image in enumerate(thumbnails):
            contact.paste(image, ((1440 - image.width) // 2, index * 295))
        contact.save(args.output / "refinement_contact_sheet.png")

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote review artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
