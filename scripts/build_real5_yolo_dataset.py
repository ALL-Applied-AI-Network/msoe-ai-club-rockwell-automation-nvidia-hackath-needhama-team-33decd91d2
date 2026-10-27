#!/usr/bin/env python3
"""Combine the three approved training frames and two reviewed audit frames."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


CLASS_NAMES = {0: "good_cap", 1: "crooked_cap", 2: "missing_cap", 3: "table_cap"}
REVIEWED_FILENAMES = {"run_154105_0000.png", "run_155442_0000.png"}


def yolo_line(box: dict, width: int, height: int) -> str:
    x1, y1, x2, y2 = (float(value) for value in box["xyxy"])
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{int(box['class_id'])} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"


def main() -> int:
    repo = Path.cwd().resolve()
    provisional_path = repo / "images/Real_Finetune/annotations.json"
    reviewed_path = repo / "runs/remediation_scene_finetune/eval_partial_yolo26l_289047/real_annotations.json"
    output = repo / "runs/remediation_scene_finetune/real_finetune_5"
    combined_path = repo / "images/Real_Finetune/five_image_annotations.json"

    provisional = json.loads(provisional_path.read_text())
    reviewed = json.loads(reviewed_path.read_text())
    width, height = provisional["image_size"]
    if reviewed["image_size"] != [width, height]:
        raise ValueError("Annotation image-size mismatch")

    images: list[dict] = []
    for item in reviewed["images"]:
        if item["filename"] not in REVIEWED_FILENAMES:
            continue
        copied = dict(item)
        copied["boxes"] = [dict(box) for box in item["boxes"] if int(box["class_id"]) != 3]
        copied["review_status"] = "operator_reviewed"
        copied["loose_table_caps_annotated"] = False
        images.append(copied)

    for item in provisional["images"]:
        copied = dict(item)
        copied["boxes"] = [dict(box) for box in item["boxes"] if int(box["class_id"]) != 3]
        copied["review_status"] = "operator_reviewed"
        images.append(copied)

    if len(images) != 5:
        raise RuntimeError(f"Expected exactly five reviewed frames, found {len(images)}")
    if len({item['filename'] for item in images}) != 5:
        raise RuntimeError("Duplicate destination filename in five-image dataset")

    image_dir = output / "images" / "all"
    label_dir = output / "labels" / "all"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[int] = Counter()
    for item in images:
        source = repo / item["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = image_dir / item["filename"]
        shutil.copy2(source, destination)
        with Image.open(destination) as image:
            if image.size != (width, height):
                raise ValueError(f"Unexpected image size for {destination}: {image.size}")
        lines = []
        for box in item["boxes"]:
            class_id = int(box["class_id"])
            counts[class_id] += 1
            lines.append(yolo_line(box, width, height))
        (label_dir / f"{destination.stem}.txt").write_text("\n".join(lines) + "\n")

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output}\n"
        "train: images/all\n"
        "val: images/all\n"
        "test: images/all\n"
        "names:\n"
        + "".join(f"  {class_id}: {name}\n" for class_id, name in CLASS_NAMES.items())
    )

    combined = {
        "schema_version": 1,
        "status": "operator_reviewed",
        "image_size": [width, height],
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "labeling_policy": {
            "good_cap": "cap is centered and seated squarely on the vial mouth",
            "crooked_cap": "cap is visibly angled, shifted, or incompletely seated",
            "missing_cap": "a transparent vial occupies the slot but has no cap",
            "table_cap": "excluded from all five physical frames because deployment table caps will use pink paper",
        },
        "evaluation_warning": (
            "All five frames are used for both fine-tuning and evaluation. Real metrics measure training-set fit "
            "and are not an unbiased estimate of physical generalization."
        ),
        "images": images,
    }
    combined_path.write_text(json.dumps(combined, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "annotations": str(combined_path.relative_to(repo)),
        "frame_count": len(images),
        "class_counts": {CLASS_NAMES[index]: counts[index] for index in CLASS_NAMES},
        "train_equals_val_equals_test": True,
        "table_caps_excluded": True,
    }, indent=2) + "\n")
    print(json.dumps({
        "dataset": str(output),
        "annotations": str(combined_path),
        "frames": [item["filename"] for item in images],
        "class_counts": {CLASS_NAMES[index]: counts[index] for index in CLASS_NAMES},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
