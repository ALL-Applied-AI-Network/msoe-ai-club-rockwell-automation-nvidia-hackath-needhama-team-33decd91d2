"""Box geometry shared by seed import, crop generation, and YOLO export."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Box:
    """Pixel-space ``[x1, y1, x2, y2]`` box with an exclusive lower-right edge."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in (self.x1, self.y1, self.x2, self.y2))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Box coordinates must be finite: {values}")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"Box must have positive area: {values}")
        for name, value in zip(("x1", "y1", "x2", "y2"), values):
            object.__setattr__(self, name, value)

    @classmethod
    def from_sequence(cls, values: Iterable[float]) -> "Box":
        coordinates = tuple(float(value) for value in values)
        if len(coordinates) != 4:
            raise ValueError(f"Expected four box coordinates, got {len(coordinates)}")
        return cls(*coordinates)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def validate_within(self, image_size: tuple[int, int], *, tolerance: float = 1e-9) -> None:
        width, height = validate_image_size(image_size)
        if self.x1 < -tolerance or self.y1 < -tolerance:
            raise ValueError(f"Box starts outside image {image_size}: {self.as_list()}")
        if self.x2 > width + tolerance or self.y2 > height + tolerance:
            raise ValueError(f"Box ends outside image {image_size}: {self.as_list()}")


def validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("Image size must contain width and height")
    width, height = image_size
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError(f"Image dimensions must be positive integers: {image_size}")
    if int(width) != width or int(height) != height or width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive integers: {image_size}")
    return int(width), int(height)


def expand_box(
    box: Box,
    image_size: tuple[int, int],
    scale: float = 1.25,
    *,
    square: bool = False,
) -> tuple[Box, bool]:
    """Expand a box about its center and clamp it to the image.

    Returns the expanded box and a flag indicating whether the ideal expanded
    box was clipped by an image boundary. Clipping does not shift the opposite
    edge to preserve size; pixels outside the source image do not exist.
    """

    width, height = validate_image_size(image_size)
    box.validate_within((width, height))
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(f"Crop scale must be finite and at least 1.0, got {scale}")
    cx, cy = box.center
    expanded_width = box.width * scale
    expanded_height = box.height * scale
    if square:
        expanded_width = expanded_height = max(expanded_width, expanded_height)
    ideal = Box(
        cx - expanded_width / 2.0,
        cy - expanded_height / 2.0,
        cx + expanded_width / 2.0,
        cy + expanded_height / 2.0,
    )
    clipped = ideal.x1 < 0 or ideal.y1 < 0 or ideal.x2 > width or ideal.y2 > height
    result = Box(
        max(0.0, ideal.x1),
        max(0.0, ideal.y1),
        min(float(width), ideal.x2),
        min(float(height), ideal.y2),
    )
    return result, clipped


def yolo_box_to_pixel(
    values: Iterable[float], image_size: tuple[int, int], *, clamp: bool = False
) -> Box:
    """Convert normalized YOLO ``cx cy width height`` geometry to pixels."""

    width, height = validate_image_size(image_size)
    coordinates = tuple(float(value) for value in values)
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        raise ValueError("YOLO box must contain four finite values")
    cx, cy, box_width, box_height = coordinates
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"YOLO box must have positive area: {coordinates}")
    box = Box(
        (cx - box_width / 2.0) * width,
        (cy - box_height / 2.0) * height,
        (cx + box_width / 2.0) * width,
        (cy + box_height / 2.0) * height,
    )
    if clamp:
        return Box(
            max(0.0, box.x1),
            max(0.0, box.y1),
            min(float(width), box.x2),
            min(float(height), box.y2),
        )
    box.validate_within((width, height))
    return box


def pixel_box_to_yolo(box: Box, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Convert a valid pixel box to normalized YOLO geometry."""

    width, height = validate_image_size(image_size)
    box.validate_within((width, height))
    cx, cy = box.center
    return cx / width, cy / height, box.width / width, box.height / height


def rasterize_box(box: Box, image_size: tuple[int, int]) -> Box:
    """Round a valid pixel box outward to the integer window used for image crops."""

    width, height = validate_image_size(image_size)
    box.validate_within((width, height))
    return Box(
        max(0, math.floor(box.x1)),
        max(0, math.floor(box.y1)),
        min(width, math.ceil(box.x2)),
        min(height, math.ceil(box.y2)),
    )
