"""Associate visible caps with vial proposals and infer unmatched vial tops."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .boxes import Box


@dataclass(frozen=True)
class AssociationConfig:
    upper_fraction: float = 0.45
    horizontal_margin_fraction: float = 0.30
    maximum_normalized_center_distance: float = 1.25
    inferred_top_width_scale: float = 1.10

    def __post_init__(self) -> None:
        if not 0 < self.upper_fraction <= 1:
            raise ValueError("upper_fraction must be in (0, 1]")
        if self.horizontal_margin_fraction < 0:
            raise ValueError("horizontal_margin_fraction must be non-negative")
        if self.maximum_normalized_center_distance <= 0:
            raise ValueError("maximum_normalized_center_distance must be positive")
        if self.inferred_top_width_scale <= 0:
            raise ValueError("inferred_top_width_scale must be positive")


def inferred_top_box(vial: Box, image_size: tuple[int, int], width_scale: float = 1.10) -> Box:
    """Create a square top proposal centered on the upper edge of a vial box."""

    image_width, image_height = image_size
    side = vial.width * width_scale
    cx = vial.center[0]
    cy = vial.y1 + side * 0.35
    return Box(
        max(0.0, cx - side / 2),
        max(0.0, cy - side / 2),
        min(float(image_width), cx + side / 2),
        min(float(image_height), cy + side / 2),
    )


def _candidate_score(vial: Box, cap: Box, config: AssociationConfig) -> tuple[float, float] | None:
    cap_x, cap_y = cap.center
    margin = vial.width * config.horizontal_margin_fraction
    upper_bottom = vial.y1 + vial.height * config.upper_fraction
    if not vial.x1 - margin <= cap_x <= vial.x2 + margin:
        return None
    if not vial.y1 - margin <= cap_y <= upper_bottom:
        return None
    target_x = vial.center[0]
    target_y = vial.y1 + min(cap.height, vial.width) * 0.25
    distance = math.hypot(cap_x - target_x, cap_y - target_y) / max(vial.width, 1.0)
    if distance > config.maximum_normalized_center_distance:
        return None
    horizontal_overlap = max(0.0, min(vial.x2 + margin, cap.x2) - max(vial.x1 - margin, cap.x1))
    overlap_ratio = horizontal_overlap / max(cap.width, 1.0)
    score = distance + 0.25 * (1.0 - min(1.0, overlap_ratio))
    return score, distance


def associate_vials_and_caps(
    vials: list[Box],
    caps: list[Box],
    image_size: tuple[int, int],
    config: AssociationConfig = AssociationConfig(),
) -> dict[str, Any]:
    """Greedily form one-to-one vial/cap matches and infer unmatched tops."""

    candidates = []
    for vial_index, vial in enumerate(vials):
        for cap_index, cap in enumerate(caps):
            result = _candidate_score(vial, cap, config)
            if result is not None:
                score, distance = result
                candidates.append((score, distance, vial_index, cap_index))
    used_vials: set[int] = set()
    used_caps: set[int] = set()
    matches = []
    for score, distance, vial_index, cap_index in sorted(candidates):
        if vial_index in used_vials or cap_index in used_caps:
            continue
        used_vials.add(vial_index)
        used_caps.add(cap_index)
        matches.append({
            "vial_index": vial_index,
            "cap_index": cap_index,
            "score": score,
            "normalized_center_distance": distance,
        })
    unmatched_vials = [index for index in range(len(vials)) if index not in used_vials]
    unmatched_caps = [index for index in range(len(caps)) if index not in used_caps]
    inferred = [
        {
            "vial_index": index,
            "box": inferred_top_box(vials[index], image_size, config.inferred_top_width_scale),
        }
        for index in unmatched_vials
    ]
    return {
        "matches": matches,
        "unmatched_vial_indices": unmatched_vials,
        "unmatched_cap_indices": unmatched_caps,
        "inferred_missing": inferred,
    }
