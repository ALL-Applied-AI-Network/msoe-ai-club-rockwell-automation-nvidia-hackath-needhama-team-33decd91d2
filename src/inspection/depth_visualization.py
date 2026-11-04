"""Readable visualizations for raw millimetre depth frames."""

from __future__ import annotations

import cv2
import numpy as np


def render_depth_heatmap(depth_mm, *, edge_threshold_mm=8.0):
    """Return a BGR turbo heatmap with white depth-discontinuity outlines.

    Zero and non-finite pixels are invalid and remain black. Valid values are
    scaled between the 2nd and 98th percentiles so distant outliers do not turn
    the useful tray/vial range into a nearly binary image. Near surfaces are
    warm and far surfaces are cool.
    """
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("depth image must be a two-dimensional array")
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    heatmap = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return heatmap

    values = depth[valid]
    low, high = np.percentile(values, [2.0, 98.0])
    if high - low < 1.0:
        low, high = float(values.min()), float(values.max())
    if high - low < 1.0:
        high = low + 1.0

    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip((depth[valid] - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    near_is_hot = 255 - scaled
    smoothed = cv2.medianBlur(near_is_hot, 3)
    heatmap = cv2.applyColorMap(smoothed, cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0

    # Outline real metric jumps as well as contrast edges in the normalized
    # visualization. Mark both pixels beside a jump so thin vial walls remain
    # visible after resizing in the run browser.
    edges = cv2.Canny(smoothed, 40, 100) > 0
    threshold = float(edge_threshold_mm)
    horizontal = valid[:, 1:] & valid[:, :-1] & (np.abs(depth[:, 1:] - depth[:, :-1]) >= threshold)
    vertical = valid[1:, :] & valid[:-1, :] & (np.abs(depth[1:, :] - depth[:-1, :]) >= threshold)
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edges[1:, :] |= vertical
    edges[:-1, :] |= vertical
    edges &= cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1) > 0
    heatmap[edges] = (255, 255, 255)
    heatmap[~valid] = 0
    return heatmap
