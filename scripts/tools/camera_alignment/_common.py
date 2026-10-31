"""Shared helpers for the TableCam alignment tests (see CLAUDE_REPORT_CAMERA.md).

cameras.json is treated as the source of truth for intrinsics and distortion.
Everything here is numpy/OpenCV only, so the offline tests run in .conda_env
without Isaac or YOLO.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CLIENT = ROOT / 'PhysicalAI-Hackathon-2026'
for _path in (ROOT, CLIENT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from calibration.geometry import (depth_to_base, flange_pose, load as load_cameras,  # noqa: E402
                                  pixel_to_base, project)

UF0_Z_MM = 330.0
UTOOL1_FROM_J6_M = 0.293     # FRAMES.md: 223 mm past a flange face 70 mm past the wrist centre
CLASS_PLANE_NOTE = {
    'table_cap': 'work surface + plug depth + flange thickness',
    'missing_cap': 'vial rim',
    'crooked_cap': 'vial rim + flange thickness',
    'good_cap': 'vial rim + flange thickness',
}


def out_dir(task, override=None):
    """shared/runs/YYYY-MM-DD_camera_alignment/<task>/ unless overridden."""
    path = Path(override) if override else (
        ROOT / 'shared/runs' / f'{_dt.date.today():%Y-%m-%d}_camera_alignment' / task)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_path(run):
    """Accept a run name (shared/runs/<name>) or a path."""
    path = Path(run)
    return path if path.is_dir() else ROOT / 'shared/runs' / run


def trace(run):
    return [json.loads(line) for line in open(run_path(run) / 'trace.jsonl')]


def frame_record(run, frame):
    return next(r for r in trace(run) if r.get('frame') == frame)


def uf0_mm_to_base(xyz_mm):
    p = np.asarray(xyz_mm, dtype=float)[..., :3].copy()
    p[..., 2] += UF0_Z_MM
    return p / 1000.0


def base_to_uf0_mm(xyz_m):
    p = np.asarray(xyz_m, dtype=float).copy() * 1000.0
    p[..., 2] -= UF0_Z_MM
    return p


def wpr_to_rotation(wpr_deg):
    """FANUC W/P/R, applied as Rz(R) Ry(P) Rx(W); columns are tool x, y, z in base."""
    w, p, r = np.radians(np.asarray(wpr_deg, dtype=float))
    Rx = np.array([[1, 0, 0], [0, np.cos(w), -np.sin(w)], [0, np.sin(w), np.cos(w)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def vertical_vanishing_pixel(cam):
    """Image point where vertical base-frame lines meet.

    A height error moves a point along the line through this pixel, so its
    apparent left/right direction flips across the vanishing column.
    """
    d = cam['T'][:3, :3].T @ np.array([0.0, 0.0, -1.0])
    w, h = cam['size']
    return np.array([w / 2 + cam['fx'] * d[0] / d[2], h / 2 + cam['fx'] * d[1] / d[2]])


def plane_error_xy(cam, point_m, dz_m):
    """Base-frame XY error of a ray/plane target when the plane is dz_m too low."""
    eye = cam['T'][:3, 3]
    p = np.asarray(point_m, dtype=float)
    return (p - eye)[:2] / (eye[2] - p[2]) * dz_m


def triangulate(cams, observations):
    """Least-squares 3-D point from {camera_name: (u, v)} pixels of fixed cameras.

    Undistorts the cameras.json single-k1 model (fixed principal point at the
    image centre), then intersects the rays in the base frame.
    """
    A, b = np.zeros((3, 3)), np.zeros(3)
    for name, uv in observations.items():
        cam = cams[name]
        w, h = cam['size']
        xd, yd = (np.asarray(uv, dtype=float) - [w / 2, h / 2]) / cam['fx']
        x, y = xd, yd
        for _ in range(10):
            d = 1 + cam.get('k1', 0.0) * (x * x + y * y)
            x, y = xd / d, yd / d
        ray = cam['T'][:3, :3] @ np.array([x, y, 1.0])
        ray /= np.linalg.norm(ray)
        P = np.eye(3) - np.outer(ray, ray)
        A += P
        b += P @ cam['T'][:3, 3]
    return np.linalg.solve(A, b)


def ray_point_on_plane(cam, uv, z):
    return pixel_to_base(cam, uv, z)


__all__ = [
    'ROOT', 'CLIENT', 'UF0_Z_MM', 'UTOOL1_FROM_J6_M', 'CLASS_PLANE_NOTE', 'out_dir', 'run_path', 'trace',
    'frame_record', 'uf0_mm_to_base', 'base_to_uf0_mm', 'wpr_to_rotation', 'vertical_vanishing_pixel',
    'plane_error_xy', 'triangulate', 'ray_point_on_plane', 'load_cameras', 'project', 'pixel_to_base',
    'depth_to_base', 'flange_pose',
]
