"""Camera-only feature extraction shared by physical and simulated policy adapters."""
from __future__ import annotations

import numpy as np


def estimate_crooked_axis(rgb, depth_mm, target_base, T_base_flange):
    """Estimate a crooked cap's upward axis from the calibrated wrist RGB-D streams."""
    from src.remediation.wrist_camera_model import (T_FLANGE_COLOUR, colour_K,
                                                     distort_px, register_depth,
                                                     undistort_px)
    from src.remediation.wrist_servo import (choose_normal, circle_normals,
                                             depth_rise_direction, find_cap_opening)
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("wrist RGB must be HxWx3")
    T = np.asarray(T_base_flange, dtype=float) @ T_FLANGE_COLOUR
    point_cam = T[:3, :3].T @ (np.asarray(target_base, dtype=float) - T[:3, 3])
    if point_cam[2] <= 0:
        raise ValueError("crooked target is behind the wrist camera")
    K = colour_K()
    expected = distort_px((K @ point_cam)[:2] / point_cam[2])[0]
    ellipse, ring = find_cap_opening(rgb, expected, to_pinhole=undistort_px)
    if ellipse is None:
        raise ValueError("crooked cap opening not found in wrist image")
    candidates = circle_normals(ellipse, K)
    registered = register_depth(np.asarray(depth_mm)) if depth_mm is not None else None
    rise = depth_rise_direction(registered, ring) if registered is not None else None
    normal_cam = choose_normal(candidates, rise)
    if normal_cam is None:
        base_candidates = [T[:3, :3] @ candidate for candidate in candidates]
        axis = max(base_candidates, key=lambda candidate: candidate[2])
    else:
        axis = T[:3, :3] @ normal_cam
    axis = np.asarray(axis, dtype=float)
    if axis[2] < 0:
        axis = -axis
    axis /= np.linalg.norm(axis)
    return axis, {"ellipse": [list(ellipse[0]), list(ellipse[1]), float(ellipse[2])],
                  "axis_base": axis.tolist(), "source": "wrist_rgb_depth"}


def find_free_table_spot(points_base, anchors, exclusions=(), surface_z=-0.0051):
    """Find a camera-observed clear table point near a detected loose-cap anchor."""
    points = np.asarray(points_base, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    anchors = [np.asarray(value, dtype=float)[:2] for value in anchors]
    excluded = [np.asarray(value, dtype=float)[:2] for value in exclusions]
    offsets = [(dx, dy) for radius in (0.06, 0.08, 0.10)
               for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius))]
    candidates = [anchor + offset for anchor in anchors for offset in offsets]
    if not candidates:
        # Crooked-only scenes may have no loose cap. Derive candidate XY cells
        # directly from observed work-surface points; the bounds are the robot's
        # calibrated reachable table region, not fixture/aperture coordinates.
        support_cloud = points[(np.abs(points[:, 2] - surface_z) < 0.004)
                               & (points[:, 0] >= 0.18) & (points[:, 0] <= 0.42)
                               & (points[:, 1] >= -0.30) & (points[:, 1] <= 0.25)]
        cells = np.unique(np.round(support_cloud[:, :2] / 0.02).astype(int), axis=0)
        candidates = [cell.astype(float) * 0.02 for cell in cells]
        candidates.sort(key=lambda xy: (xy[1], xy[0]))
    for xy in candidates:
        if not (0.18 <= xy[0] <= 0.42 and -0.30 <= xy[1] <= 0.25):
            continue
        if any(np.linalg.norm(xy - other) < 0.045 for other in anchors + excluded):
            continue
        radial = np.linalg.norm(points[:, :2] - xy, axis=1)
        obstacle = points[(radial < 0.025) & (points[:, 2] > surface_z + 0.004)]
        support = points[(radial < 0.020) & (np.abs(points[:, 2] - surface_z) < 0.004)]
        if not len(obstacle) and len(support) >= 10:
            # UF0's calibrated lower TCP limit corresponds to base Z=-4 mm.
            # Release 1 mm above that boundary so a zero provisional place
            # offset remains valid and the cap can settle the remaining 2.1 mm
            # under gravity.  XY and support still come exclusively from depth.
            reachable_release_z = max(float(surface_z), -0.003)
            return np.array([xy[0], xy[1], reachable_release_z], dtype=float)
    raise ValueError("no camera-observed clear table regrasp spot found")


def upright_cap_near(detections, expected_xy, max_distance_m=0.025):
    """Return a camera-localized upright loose cap near the expected regrasp spot."""
    candidates = [item for item in detections
                  if item.get("class_name") == "table_cap"
                  and item.get("depth_available") is True]
    if not candidates:
        return None
    expected = np.asarray(expected_xy, dtype=float)[:2]
    nearest = min(candidates, key=lambda item: np.linalg.norm(
        np.asarray(item["base_pos_m"], dtype=float)[:2] - expected))
    position = np.asarray(nearest["base_pos_m"], dtype=float)
    return position if np.linalg.norm(position[:2] - expected) <= max_distance_m else None
