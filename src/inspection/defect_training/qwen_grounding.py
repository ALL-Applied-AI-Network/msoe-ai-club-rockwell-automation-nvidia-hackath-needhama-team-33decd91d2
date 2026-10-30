"""Parsing helpers for Qwen3-VL JSON grounding responses."""

from __future__ import annotations

import json
import re
from typing import Any

from .boxes import Box
from .locate_anything import non_maximum_suppression


FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
BBOX_FRAGMENT = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
    r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def _json_payload(response: str) -> Any:
    fenced = FENCED_JSON.search(response)
    candidate = fenced.group(1) if fenced else response.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        starts = [index for index in (candidate.find("["), candidate.find("{")) if index >= 0]
        if not starts:
            return []
        start = min(starts)
        closing = "]" if candidate[start] == "[" else "}"
        end = candidate.rfind(closing)
        if end < start:
            return []
        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return []


def parse_qwen_grounding_boxes(
    response: str, image_size: tuple[int, int], nms_iou: float
) -> tuple[list[Box], int]:
    """Parse normalized 0–1000 ``bbox_2d`` values from Qwen3-VL output."""

    payload = _json_payload(response)
    if isinstance(payload, dict):
        payload = payload.get("objects", payload.get("detections", [payload]))
    if not isinstance(payload, list) or not payload:
        payload = [
            {"bbox_2d": [float(value) for value in match.groups()]}
            for match in BBOX_FRAGMENT.finditer(response)
        ]
    width, height = image_size
    parsed: list[Box] = []
    for item in payload:
        values = item.get("bbox_2d") if isinstance(item, dict) else None
        if not isinstance(values, list) or len(values) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in values)
        except (TypeError, ValueError):
            continue
        left, right = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        top, bottom = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        box = Box(left / 1000.0 * width, top / 1000.0 * height,
                  right / 1000.0 * width, bottom / 1000.0 * height)
        if box.width >= 2 and box.height >= 2:
            parsed.append(box)
    return non_maximum_suppression(parsed, nms_iou), len(parsed)
