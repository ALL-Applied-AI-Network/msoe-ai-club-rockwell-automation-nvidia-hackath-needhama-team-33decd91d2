"""Robust parallel work-surface fitting for physical remediation.

The physical TableCam depth stream is sparse around clear vials and reflective
metal, but it sees a large amount of the opaque work surface.  Fit only the
surface's base-frame Z offset; robot-base/table parallelism remains the trusted
calibration.  The vial-rim plane is then derived from measured fixture geometry.
"""
from __future__ import annotations

import numpy as np


def fit_parallel_work_surface(points_base_m, nominal_z_m, *, min_points=500,
                              candidate_band_m=0.020,
                              inlier_band_m=0.004,
                              max_rms_m=0.0035,
                              min_xy_span_m=0.150):
    """Fit a constant-Z work-surface plane from calibrated depth points.

    Points outside a narrow band around the saved calibration are excluded so
    the tray top, vials, robot and distant background cannot pull the estimate.
    Median-centred clipping makes the result insensitive to paper, scratches and
    isolated stereo mismatches.  The returned diagnostics are JSON-safe.
    """
    points = np.asarray(points_base_m, dtype=float).reshape(-1, 3)
    nominal = float(nominal_z_m)
    if not np.isfinite(nominal):
        raise ValueError("nominal work-surface Z must be finite")
    finite = np.isfinite(points).all(axis=1)
    candidates = points[finite & (np.abs(points[:, 2] - nominal) <=
                                  float(candidate_band_m))]
    if len(candidates) < int(min_points):
        raise ValueError(
            f"only {len(candidates)} depth points near the work-surface plane")

    kept = candidates
    for _ in range(2):
        centre = float(np.median(kept[:, 2]))
        inliers = np.abs(kept[:, 2] - centre) <= float(inlier_band_m)
        if int(inliers.sum()) < int(min_points):
            raise ValueError(
                f"only {int(inliers.sum())} work-surface depth inliers")
        kept = kept[inliers]

    fitted = float(np.median(kept[:, 2]))
    residual = kept[:, 2] - fitted
    rms = float(np.sqrt(np.mean(residual * residual)))
    xy_low = np.percentile(kept[:, :2], 5, axis=0)
    xy_high = np.percentile(kept[:, :2], 95, axis=0)
    xy_span = xy_high - xy_low
    if rms > float(max_rms_m):
        raise ValueError(f"work-surface depth RMS is {rms * 1000:.1f} mm")
    if float(np.max(xy_span)) < float(min_xy_span_m):
        raise ValueError(
            f"work-surface depth spans only {float(np.max(xy_span)) * 1000:.1f} mm")
    return {
        "fitted_z_base_m": fitted,
        "nominal_z_base_m": nominal,
        "fit_correction_m": fitted - nominal,
        "candidate_count": int(len(candidates)),
        "inlier_count": int(len(kept)),
        "rms_m": rms,
        "xy_span_m": xy_span.tolist(),
        "constraint": "parallel to calibrated robot-base XY plane",
        "source": "TableCam depth work-surface fit",
    }
