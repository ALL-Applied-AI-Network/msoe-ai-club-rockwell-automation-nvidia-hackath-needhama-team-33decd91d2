#!/usr/bin/env python3
"""Combine reviewed real classes with class-agnostic VLM cap geometry."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.refine_yolo_boxes_with_vlm import (  # noqa: E402
    cluster_vlm_boxes,
    draw_detections,
    load_reviewed_labels,
    panel,
    write_yolo_labels,
    box_iou,
    center_distance,
    diagonal,
)


def score_match(reviewed_box: list[float], proposal_box: list[float]) -> tuple[float, float, float]:
    overlap = box_iou(reviewed_box, proposal_box)
    normalized_distance = center_distance(reviewed_box, proposal_box) / max(
        diagonal(reviewed_box), diagonal(proposal_box), 1.0
    )
    return 0.75 * overlap + 0.25 * max(0.0, 1.0 - normalized_distance), overlap, normalized_distance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster-iou", type=float, default=0.65)
    args = parser.parse_args()

    image_output = args.output / "images" / "all"
    label_output = args.output / "labels" / "all"
    comparison_output = args.output / "comparisons"
    for path in (image_output, label_output, comparison_output):
        path.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.vlm_predictions.read_text())
    records = payload["raw"] if isinstance(payload, dict) else payload
    vlm_by_name = {Path(record["image"]).name: record for record in records}
    manifest = {
        "policy": "reviewed class/count; full VLM geometry for good/crooked caps; reviewed geometry for missing caps",
        "images": [],
    }
    comparison_rows = []
    for image_path in sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        reviewed = load_reviewed_labels(args.reviewed_labels / f"{image_path.stem}.txt", width, height)
        proposals = cluster_vlm_boxes(vlm_by_name[image_path.name]["predictions"], args.cluster_iou)

        candidates = []
        for reviewed_index, item in enumerate(reviewed):
            if item["class_id"] not in {0, 1}:
                continue
            for proposal_index, proposal in enumerate(proposals):
                score, overlap, distance = score_match(item["box"], proposal["box"])
                if score >= 0.35 and (overlap >= 0.15 or distance <= 0.50):
                    candidates.append((score, overlap, -distance, reviewed_index, proposal_index))
        candidates.sort(reverse=True)
        used_reviewed, used_proposals, matches = set(), set(), {}
        for score, overlap, negative_distance, reviewed_index, proposal_index in candidates:
            if reviewed_index in used_reviewed or proposal_index in used_proposals:
                continue
            used_reviewed.add(reviewed_index)
            used_proposals.add(proposal_index)
            matches[reviewed_index] = {
                "proposal_index": proposal_index,
                "score": score,
                "iou_with_reviewed": overlap,
                "normalized_center_distance": -negative_distance,
            }

        corrected, details = [], []
        for index, item in enumerate(reviewed):
            corrected_item = dict(item)
            match = matches.get(index)
            if item["class_id"] in {0, 1} and match is not None:
                proposal = proposals[match["proposal_index"]]
                corrected_item["box"] = proposal["box"]
                corrected_item["geometry_source"] = "vlm"
                details.append({
                    "class_id": item["class_id"], "class_name": item["class_name"],
                    "reviewed_box": item["box"], "corrected_box": proposal["box"],
                    "center_shift_pixels": center_distance(item["box"], proposal["box"]),
                    "match": match, "vlm_support": proposal["support"],
                })
            else:
                corrected_item["geometry_source"] = "reviewed"
                details.append({
                    "class_id": item["class_id"], "class_name": item["class_name"],
                    "reviewed_box": item["box"], "corrected_box": item["box"],
                    "center_shift_pixels": 0.0, "match": None,
                })
            corrected.append(corrected_item)

        shutil.copy2(image_path, image_output / image_path.name)
        write_yolo_labels(label_output / f"{image_path.stem}.txt", corrected, width, height)
        cells = [
            panel(draw_detections(image, reviewed), f"Reviewed labels — {image_path.name}", (960, 590)),
            panel(draw_detections(image, corrected), "Reviewed classes + VLM cap boxes", (960, 590)),
        ]
        row = Image.new("RGB", (1920, 590), "#d0d0d0")
        row.paste(cells[0], (0, 0))
        row.paste(cells[1], (960, 0))
        comparison_output_path = comparison_output / f"{image_path.stem}.png"
        row.save(comparison_output_path)
        comparison_rows.append(row)
        manifest["images"].append({
            "image": str(image_output / image_path.name),
            "reviewed_count": len(reviewed),
            "vlm_refined_count": sum(item["geometry_source"] == "vlm" for item in corrected),
            "reviewed_geometry_count": sum(item["geometry_source"] == "reviewed" for item in corrected),
            "details": details,
        })

    sheet = Image.new("RGB", (1920, 590 * len(comparison_rows)), "white")
    for index, row in enumerate(comparison_rows):
        sheet.paste(row, (0, index * 590))
    sheet.save(args.output / "reviewed_vs_vlm_boxes.png")
    shifts = [
        item["center_shift_pixels"] for image in manifest["images"] for item in image["details"]
        if item["center_shift_pixels"] > 0
    ]
    manifest["summary"] = {
        "image_count": len(manifest["images"]),
        "object_count": sum(image["reviewed_count"] for image in manifest["images"]),
        "vlm_refined_count": sum(image["vlm_refined_count"] for image in manifest["images"]),
        "reviewed_geometry_count": sum(image["reviewed_geometry_count"] for image in manifest["images"]),
        "mean_vlm_center_shift_pixels": sum(shifts) / len(shifts) if shifts else 0.0,
        "max_vlm_center_shift_pixels": max(shifts, default=0.0),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
