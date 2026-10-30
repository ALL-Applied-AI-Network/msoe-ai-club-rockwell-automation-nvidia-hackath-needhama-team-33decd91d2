"""Import reviewed YOLO annotations as VLM reference examples."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

from .boxes import Box, expand_box, rasterize_box, yolo_box_to_pixel
from .config import DefectProjectConfig
from .schemas import ObjectRecord, stable_object_id, write_object_records


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
COLORS = (
    (50, 190, 80),
    (245, 145, 35),
    (225, 65, 70),
    (65, 140, 235),
    (180, 70, 210),
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_split_manifest(path: Path | str) -> dict[str, dict[str, str]]:
    """Load ``{image name: {split, group}}`` assignments and validate uniqueness."""

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping) and "images" in payload:
        payload = payload["images"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Split manifest must contain a non-empty image mapping")
    result = {}
    for image_name, assignment in payload.items():
        if not isinstance(image_name, str) or not image_name:
            raise ValueError("Split manifest image names must be non-empty strings")
        if not isinstance(assignment, Mapping):
            raise ValueError(f"Split assignment for {image_name!r} must be a mapping")
        split = assignment.get("split")
        group = assignment.get("group")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {image_name!r}: {split!r}")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"Invalid group for {image_name!r}: {group!r}")
        result[image_name] = {"split": split, "group": group.strip()}
    groups: dict[str, set[str]] = defaultdict(set)
    for assignment in result.values():
        groups[assignment["group"]].add(assignment["split"])
    leaked = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Split groups cross dataset splits: {leaked}")
    return result


def read_yolo_labels(path: Path, image_size: tuple[int, int], class_count: int) -> list[tuple[int, Box]]:
    labels = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        try:
            class_id = int(fields[0])
            geometry = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: class ID {class_id} is not configured")
        labels.append((class_id, yolo_box_to_pixel(geometry, image_size)))
    return labels


def _draw_overlay(image: Image.Image, items: list[tuple[str, Box]], class_names: tuple[str, ...]) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(image.size) / 400))
    font = _font(max(14, round(min(image.size) / 55)))
    for class_name, box in items:
        class_id = class_names.index(class_name)
        color = COLORS[class_id % len(COLORS)]
        coordinates = tuple(round(value) for value in box.as_list())
        draw.rectangle(coordinates, outline=color, width=line_width)
        text_box = draw.textbbox((coordinates[0], coordinates[1]), class_name, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((coordinates[0], coordinates[1]), class_name, fill=color, font=font)
    return rendered


def _contact_sheet(paths: list[Path], title: str, destination: Path) -> None:
    if not paths:
        return
    tile_width, tile_height, header = 256, 256, 38
    columns = min(6, len(paths))
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, header + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{title} — {len(paths)} references", fill="black", font=_font(20))
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            crop = opened.convert("RGB")
        crop.thumbnail((tile_width - 12, tile_height - 28), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = header + (index // columns) * tile_height
        sheet.paste(crop, (x + (tile_width - crop.width) // 2, y + 2))
        label = path.stem[:34]
        draw.text((x + 6, y + tile_height - 23), label, fill="black", font=_font(12))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def import_seed_dataset(
    config: DefectProjectConfig,
    images_dir: Path,
    labels_dir: Path,
    split_assignments: Mapping[str, Mapping[str, str]],
    output: Path,
) -> dict[str, Any]:
    """Import reviewed YOLO labels and create deterministic reference artifacts."""

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite seed output: {output}")
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {images_dir}")
    image_names = {path.name for path in image_paths}
    missing_assignments = sorted(image_names - split_assignments.keys())
    extra_assignments = sorted(split_assignments.keys() - image_names)
    if missing_assignments or extra_assignments:
        raise ValueError(
            f"Split manifest mismatch; missing={missing_assignments}, extra={extra_assignments}"
        )

    records: list[ObjectRecord] = []
    crops_by_class: dict[str, list[Path]] = defaultdict(list)
    object_counts_by_split: Counter[str] = Counter()
    class_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts: Counter[str] = Counter()
    output.mkdir(parents=True)
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        image_size = image.size
        assignment = split_assignments[image_path.name]
        labels = read_yolo_labels(label_path, image_size, len(config.classes))
        overlay_items = []
        for class_id, source_box in labels:
            class_name = config.classes[class_id].name
            requested_crop, clipped = expand_box(
                source_box, image_size, config.crop.scale, square=config.crop.square
            )
            raster_crop = rasterize_box(requested_crop, image_size)
            object_id = stable_object_id(image_path.resolve(), source_box, "human")
            record = ObjectRecord(
                object_id=object_id,
                image=str(image_path.resolve()),
                image_size=image_size,
                split=assignment["split"],
                split_group=assignment["group"],
                source_box=source_box,
                crop_box=raster_crop,
                localization_source="human",
                appearance_label=class_name,
                label_source="human",
                confidence=1.0,
                status="accepted",
                crop_clipped=clipped,
                metadata={"requested_crop_box_xyxy": requested_crop.as_list()},
            )
            records.append(record)
            overlay_items.append((class_name, source_box))
            object_counts_by_split[assignment["split"]] += 1
            class_counts_by_split[assignment["split"]][class_name] += 1
            class_counts[class_name] += 1
            if assignment["split"] == "train":
                crop_path = output / "crops" / class_name / f"{object_id}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop(tuple(int(value) for value in raster_crop.as_list())).save(crop_path)
                crops_by_class[class_name].append(crop_path)
        overlay = _draw_overlay(image, overlay_items, config.class_names)
        overlay_path = output / "overlays" / assignment["split"] / image_path.name
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)

    write_object_records(output / "manifest.jsonl", records)
    for class_name in config.class_names:
        _contact_sheet(
            crops_by_class[class_name], class_name, output / "contact_sheets" / f"{class_name}.png"
        )
    report = {
        "schema_version": 1,
        "project_name": config.project_name,
        "images": len(image_paths),
        "objects": len(records),
        "objects_by_class": dict(sorted(class_counts.items())),
        "objects_by_split": dict(sorted(object_counts_by_split.items())),
        "classes_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(class_counts_by_split.items())
        },
        "reference_crops_by_class": {
            name: len(crops_by_class[name]) for name in config.class_names
        },
        "crop_scale": config.crop.scale,
        "class_names": list(config.class_names),
        "source_images": str(images_dir.resolve()),
        "source_labels": str(labels_dir.resolve()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    source_data = labels_dir.parent / "data.yaml"
    if source_data.is_file():
        shutil.copy2(source_data, output / "source_data.yaml")
    return report
