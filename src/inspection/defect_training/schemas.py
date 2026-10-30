"""Canonical, versioned records for the defect-training data flow."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable, Mapping

from .boxes import Box, validate_image_size


VALID_SPLITS = frozenset({"train", "val", "test"})
VALID_STATUSES = frozenset({"unlabeled", "accepted", "review", "rejected", "human_corrected"})


def normalize_image_reference(image: str | Path) -> str:
    value = str(image).replace("\\", "/")
    normalized = str(PurePosixPath(value))
    if normalized in {"", "."}:
        raise ValueError("Image reference must not be empty")
    return normalized


def stable_object_id(
    image: str | Path,
    source_box: Box,
    localization_source: str,
    *,
    precision: int = 6,
) -> str:
    """Create an order-independent ID from immutable proposal identity fields."""

    image_ref = normalize_image_reference(image)
    if not localization_source.strip():
        raise ValueError("localization_source must not be empty")
    canonical = json.dumps(
        {
            "image": image_ref,
            "source_box_xyxy": [round(value, precision) for value in source_box.as_list()],
            "localization_source": localization_source.strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    stem = Path(image_ref).stem
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return f"{safe_stem}__{digest}"


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    image: str
    image_size: tuple[int, int]
    split: str
    split_group: str
    source_box: Box
    crop_box: Box | None
    localization_source: str
    localization_prompt: str | None = None
    appearance_label: str | None = None
    label_source: str | None = None
    confidence: float | None = None
    status: str = "unlabeled"
    reference_ids: tuple[str, ...] = ()
    crop_clipped: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported object schema version: {self.schema_version}")
        if not self.object_id.strip():
            raise ValueError("object_id must not be empty")
        object.__setattr__(self, "image", normalize_image_reference(self.image))
        validate_image_size(self.image_size)
        self.source_box.validate_within(self.image_size)
        if self.crop_box is not None:
            self.crop_box.validate_within(self.image_size)
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {self.split!r}")
        if not self.split_group.strip():
            raise ValueError("split_group must not be empty")
        if not self.localization_source.strip():
            raise ValueError("localization_source must not be empty")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {self.status!r}")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.status in {"accepted", "human_corrected"} and not self.appearance_label:
            raise ValueError(f"status {self.status!r} requires appearance_label")
        if self.appearance_label and not self.label_source:
            raise ValueError("appearance_label requires label_source")
        if len(set(self.reference_ids)) != len(self.reference_ids):
            raise ValueError("reference_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "object_id": self.object_id,
            "image": self.image,
            "image_size": list(self.image_size),
            "split": self.split,
            "split_group": self.split_group,
            "source_box_xyxy": self.source_box.as_list(),
            "crop_box_xyxy": self.crop_box.as_list() if self.crop_box is not None else None,
            "localization_source": self.localization_source,
            "localization_prompt": self.localization_prompt,
            "appearance_label": self.appearance_label,
            "label_source": self.label_source,
            "confidence": self.confidence,
            "status": self.status,
            "reference_ids": list(self.reference_ids),
            "crop_clipped": self.crop_clipped,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectRecord":
        required = {
            "object_id", "image", "image_size", "split", "split_group",
            "source_box_xyxy", "localization_source",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Object record is missing required fields: {missing}")
        return cls(
            schema_version=payload.get("schema_version", 1),
            object_id=str(payload["object_id"]),
            image=str(payload["image"]),
            image_size=tuple(payload["image_size"]),
            split=str(payload["split"]),
            split_group=str(payload["split_group"]),
            source_box=Box.from_sequence(payload["source_box_xyxy"]),
            crop_box=(
                Box.from_sequence(payload["crop_box_xyxy"])
                if payload.get("crop_box_xyxy") is not None else None
            ),
            localization_source=str(payload["localization_source"]),
            localization_prompt=payload.get("localization_prompt"),
            appearance_label=payload.get("appearance_label"),
            label_source=payload.get("label_source"),
            confidence=payload.get("confidence"),
            status=payload.get("status", "unlabeled"),
            reference_ids=tuple(payload.get("reference_ids", ())),
            crop_clipped=payload.get("crop_clipped", False),
            metadata=payload.get("metadata", {}),
        )


def write_object_records(path: Path | str, records: Iterable[ObjectRecord]) -> None:
    """Atomically replace a JSONL manifest after checking duplicate IDs."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    items = list(records)
    identifiers = [item.object_id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Object record manifest contains duplicate object IDs")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_object_records(path: Path | str) -> list[ObjectRecord]:
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = ObjectRecord.from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid object record on line {line_number}: {exc}") from exc
        if record.object_id in seen:
            raise ValueError(f"Duplicate object_id {record.object_id!r} on line {line_number}")
        seen.add(record.object_id)
        records.append(record)
    return records
