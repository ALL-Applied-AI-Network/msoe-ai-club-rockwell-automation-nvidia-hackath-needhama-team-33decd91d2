"""Reusable building blocks for weakly supervised defect-detector training."""

from .boxes import Box, expand_box, pixel_box_to_yolo, rasterize_box, yolo_box_to_pixel
from .config import DefectProjectConfig, load_project_config
from .schemas import ObjectRecord, read_object_records, stable_object_id, write_object_records

__all__ = [
    "Box",
    "DefectProjectConfig",
    "ObjectRecord",
    "expand_box",
    "load_project_config",
    "pixel_box_to_yolo",
    "rasterize_box",
    "read_object_records",
    "stable_object_id",
    "write_object_records",
    "yolo_box_to_pixel",
]
