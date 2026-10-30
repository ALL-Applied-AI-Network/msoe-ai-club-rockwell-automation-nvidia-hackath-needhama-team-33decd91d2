"""Class-agnostic LocateAnything proposal generation with resumable records."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from PIL import Image, ImageDraw, ImageFont

from .boxes import Box
from .config import DefectProjectConfig
from .schemas import ObjectRecord, stable_object_id, write_object_records
from .seed_import import COLORS, IMAGE_SUFFIXES


BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def box_iou(left: Box, right: Box) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(boxes: list[Box], threshold: float) -> list[Box]:
    """Remove repeated generative boxes, preferring tighter geometry."""

    ordered = sorted(boxes, key=lambda box: (box.width * box.height, box.as_list()))
    kept: list[Box] = []
    for box in ordered:
        if all(box_iou(box, other) < threshold for other in kept):
            kept.append(box)
    return sorted(kept, key=lambda box: (box.y1, box.x1, box.y2, box.x2))


def merge_prompt_proposals(
    proposals: list[tuple[Box, str]], threshold: float
) -> list[tuple[Box, tuple[str, ...]]]:
    """Merge duplicate boxes across prompts while retaining prompt provenance."""

    ordered = sorted(proposals, key=lambda item: (item[0].width * item[0].height, item[0].as_list(), item[1]))
    kept: list[dict[str, Any]] = []
    for box, prompt in ordered:
        duplicate = next((item for item in kept if box_iou(box, item["box"]) >= threshold), None)
        if duplicate is None:
            kept.append({"box": box, "prompts": {prompt}})
        else:
            duplicate["prompts"].add(prompt)
    return [
        (item["box"], tuple(sorted(item["prompts"])))
        for item in sorted(kept, key=lambda item: (item["box"].y1, item["box"].x1))
    ]


def parse_locate_anything_boxes(
    response: str, image_size: tuple[int, int], nms_iou: float
) -> tuple[list[Box], int]:
    """Parse LocateAnything's 0–1000 coordinates and return boxes plus raw count."""

    width, height = image_size
    parsed = []
    for match in BOX_PATTERN.finditer(response):
        x1, y1, x2, y2 = (int(value) for value in match.groups())
        left, right = sorted((max(0.0, x1 / 1000.0 * width), min(float(width), x2 / 1000.0 * width)))
        top, bottom = sorted((max(0.0, y1 / 1000.0 * height), min(float(height), y2 / 1000.0 * height)))
        if right - left >= 2 and bottom - top >= 2:
            parsed.append(Box(left, top, right, bottom))
    return non_maximum_suppression(parsed, nms_iou), len(parsed)


def _fingerprint(config: DefectProjectConfig, model: str, adapter: str | None) -> str:
    payload = {
        "schema_version": 1,
        "model": model,
        "adapter": adapter,
        "prompts": list(config.localization.prompts),
        "nms_iou": config.localization.nms_iou,
        "maximum_objects_per_image": config.localization.maximum_objects_per_image,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def _overlay(image: Image.Image, boxes: list[Box], destination: Path) -> None:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    width = max(2, round(min(image.size) / 400))
    font = _font(max(14, round(min(image.size) / 55)))
    for index, box in enumerate(boxes, 1):
        coordinates = tuple(round(value) for value in box.as_list())
        draw.rectangle(coordinates, outline=COLORS[3], width=width)
        label = f"proposal {index}"
        text_box = draw.textbbox((coordinates[0], coordinates[1]), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((coordinates[0], coordinates[1]), label, fill=COLORS[3], font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(destination)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def generate_proposals(
    config: DefectProjectConfig,
    images_dir: Path,
    split_assignments: Mapping[str, Mapping[str, str]],
    output: Path,
    predictor: Callable[[Image.Image, str], str],
    *,
    model: str,
    adapter: str | None = None,
    include_splits: set[str] | None = None,
    response_parser: Callable[[str, tuple[int, int], float], tuple[list[Box], int]] = parse_locate_anything_boxes,
    localization_source: str = "locateanything",
) -> dict[str, Any]:
    """Generate or resume class-agnostic proposals for a complete image set."""

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {images_dir}")
    image_names = {path.name for path in image_paths}
    if image_names != set(split_assignments):
        raise ValueError(
            f"Split manifest mismatch; missing={sorted(image_names - set(split_assignments))}, "
            f"extra={sorted(set(split_assignments) - image_names)}"
        )
    if include_splits is not None:
        invalid_splits = include_splits - {"train", "val", "test"}
        if invalid_splits:
            raise ValueError(f"Invalid included splits: {sorted(invalid_splits)}")
        image_paths = [
            path for path in image_paths if split_assignments[path.name]["split"] in include_splits
        ]
        if not image_paths:
            raise ValueError(f"No images remain after filtering to splits {sorted(include_splits)}")
    fingerprint = _fingerprint(config, model, adapter)
    record_dir = output / "records"
    records: list[ObjectRecord] = []
    image_summaries = []
    resumed = 0
    for image_path in image_paths:
        record_path = record_dir / f"{image_path.stem}.json"
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        assignment = split_assignments[image_path.name]
        if record_path.is_file():
            payload = json.loads(record_path.read_text())
            if payload.get("fingerprint") != fingerprint:
                raise ValueError(f"Resume fingerprint mismatch: {record_path}")
            if Path(payload["image"]).resolve() != image_path.resolve():
                raise ValueError(f"Resume image mismatch: {record_path}")
            if "proposals" in payload:
                proposals = [
                    (Box.from_sequence(item["box_xyxy"]), tuple(item["source_prompts"]))
                    for item in payload["proposals"]
                ]
            else:
                proposals = [
                    (Box.from_sequence(values), (payload["prompt"],))
                    for values in payload["boxes_xyxy"]
                ]
            raw_count = int(payload["raw_box_count"])
            resumed += 1
        else:
            responses = {}
            prompt_candidates = []
            raw_count = 0
            for prompt in config.localization.prompts:
                response = predictor(image, prompt)
                prompt_boxes, prompt_raw_count = response_parser(
                    response, image.size, config.localization.nms_iou
                )
                responses[prompt] = {
                    "raw_response": response,
                    "raw_box_count": prompt_raw_count,
                    "parsed_box_count": len(prompt_boxes),
                }
                raw_count += prompt_raw_count
                prompt_candidates.extend((box, prompt) for box in prompt_boxes)
            proposals = merge_prompt_proposals(prompt_candidates, config.localization.nms_iou)
            payload = {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "image": str(image_path.resolve()),
                "image_size": list(image.size),
                "split": assignment["split"],
                "split_group": assignment["group"],
                "model": model,
                "adapter": adapter,
                "prompts": list(config.localization.prompts),
                "responses": responses,
                "raw_box_count": raw_count,
                "proposals": [
                    {"box_xyxy": box.as_list(), "source_prompts": list(source_prompts)}
                    for box, source_prompts in proposals
                ],
            }
            _write_json_atomic(record_path, payload)
        boxes = [box for box, _ in proposals]
        _overlay(image, boxes, output / "overlays" / assignment["split"] / image_path.name)
        for box, source_prompts in proposals:
            records.append(ObjectRecord(
                object_id=stable_object_id(image_path.resolve(), box, localization_source),
                image=str(image_path.resolve()),
                image_size=image.size,
                split=assignment["split"],
                split_group=assignment["group"],
                source_box=box,
                crop_box=None,
                localization_source=localization_source,
                localization_prompt=" | ".join(source_prompts),
                status="unlabeled",
                metadata={
                    "model": model,
                    "adapter": adapter,
                    "record": str(record_path),
                    "source_prompts": list(source_prompts),
                },
            ))
        image_summaries.append({
            "image": image_path.name,
            "split": assignment["split"],
            "split_group": assignment["group"],
            "raw_box_count": raw_count,
            "proposal_count": len(boxes),
            "zero_proposals": not boxes,
            "exceeds_maximum_objects": (
                config.localization.maximum_objects_per_image is not None
                and len(boxes) > config.localization.maximum_objects_per_image
            ),
        })
    write_object_records(output / "manifest.jsonl", records)
    counts = Counter(item.split for item in records)
    proposal_counts = [item["proposal_count"] for item in image_summaries]
    report = {
        "schema_version": 1,
        "project_name": config.project_name,
        "fingerprint": fingerprint,
        "model": model,
        "adapter": adapter,
        "included_splits": sorted(include_splits) if include_splits is not None else None,
        "prompts": list(config.localization.prompts),
        "images": len(image_paths),
        "objects": len(records),
        "objects_by_split": dict(sorted(counts.items())),
        "zero_proposal_images": sum(item["zero_proposals"] for item in image_summaries),
        "maximum_objects_per_image": config.localization.maximum_objects_per_image,
        "over_limit_images": [
            item["image"] for item in image_summaries if item["exceeds_maximum_objects"]
        ],
        "minimum_proposals_per_image": min(proposal_counts),
        "maximum_proposals_per_image": max(proposal_counts),
        "resumed_images": resumed,
        "image_summaries": image_summaries,
    }
    _write_json_atomic(output / "report.json", report)
    return report
