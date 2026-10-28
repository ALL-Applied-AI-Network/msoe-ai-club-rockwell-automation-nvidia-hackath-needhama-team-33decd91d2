"""Four-class YOLO detection and calibrated TableCam localization.

Images enter as RGB. Bounding boxes returned by Ultralytics are already in the
original image coordinates; only the camera-resolution mapping is applied here.
A coarse target is not a contact-qualified grasp or press location.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.remediation.geometry import CLASS_NAMES, SPEC, base_to_uf0, rim_base_z

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = ROOT / 'PhysicalAI-Hackathon-2026/calibration'


def _cluster_1d(values, count):
    """Small deterministic 1-D k-means used to initialize the tray lattice."""
    values = np.asarray(values, dtype=float)
    centres = np.linspace(float(values.min()), float(values.max()), int(count))
    labels = np.zeros(len(values), dtype=int)
    for _ in range(30):
        new_labels = np.argmin(abs(values[:, None] - centres[None, :]), axis=1)
        new_centres = centres.copy()
        for index in range(len(centres)):
            members = values[new_labels == index]
            if len(members):
                new_centres[index] = float(np.median(members))
        if np.array_equal(labels, new_labels) and np.allclose(centres, new_centres):
            break
        labels, centres = new_labels, new_centres
    order = np.argsort(centres)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return centres[order], inverse[labels]


def self_consistent_cap_height(surface_top):
    """Reject obvious robot/table false positives from the tray-grid fit."""
    return rim_base_z() - 0.020 <= surface_top <= rim_base_z() + 0.055


def refine_defect_centres_from_grid(targets, grid_shape=(6, 5), min_visible=6):
    """Snap missing/crooked contact XY to a camera-derived tray lattice.

    A crooked cap's RGB-D/box centre follows the leaning flange, not the vial
    axis. Likewise, a mouth box can be biased by reflections. Fit an affine
    6x5 lattice from all visible tray detections, then replace only defect XY
    with the nearest fitted aperture. The fit has no stored tray origin,
    rotation, spacing, or slot coordinates.
    """
    candidates = []
    candidate_indices = []
    for index, target in enumerate(targets):
        if target.get('class_name') not in {'good_cap', 'crooked_cap', 'missing_cap'}:
            continue
        position = np.asarray(target.get('base_pos_m', ()), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            continue
        surface = target.get('surface_top_base_z_m')
        if (target.get('class_name') == 'crooked_cap' and surface is not None
                and np.isfinite(surface) and not self_consistent_cap_height(float(surface))):
            continue
        candidates.append(position[:2])
        candidate_indices.append(index)
    points = np.asarray(candidates, dtype=float)
    if len(points) < int(min_visible):
        return targets

    # Reject isolated detections on the robot/table before estimating lattice
    # direction. This uses only relative point density, not a tray envelope.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[distances == 0.0] = np.inf
    nearest = distances.min(axis=1)
    typical = float(np.median(nearest[np.isfinite(nearest)]))
    keep = nearest <= max(typical * 1.8, typical + 0.004)
    points = points[keep]
    candidate_indices = [index for index, accepted in zip(candidate_indices, keep) if accepted]
    if len(points) < int(min_visible):
        return targets

    nearest_vectors = []
    for row, point in enumerate(points):
        delta = points - point
        norms = np.linalg.norm(delta, axis=1)
        norms[row] = np.inf
        nearest_vectors.append(delta[int(np.argmin(norms))])
    angles = np.arctan2(np.asarray(nearest_vectors)[:, 1],
                        np.asarray(nearest_vectors)[:, 0])
    theta = 0.5 * np.arctan2(np.mean(np.sin(2.0 * angles)),
                             np.mean(np.cos(2.0 * angles)))
    first_axis = np.array([np.cos(theta), np.sin(theta)])
    second_axis = np.array([-first_axis[1], first_axis[0]])
    projections = np.column_stack((points @ first_axis, points @ second_axis))

    fits = []
    for first_count, second_count in (grid_shape, grid_shape[::-1]):
        if len(points) < max(first_count, second_count):
            continue
        _, first_labels = _cluster_1d(projections[:, 0], first_count)
        _, second_labels = _cluster_1d(projections[:, 1], second_count)
        if len(np.unique(first_labels)) != first_count or len(np.unique(second_labels)) != second_count:
            continue
        design = np.column_stack((np.ones(len(points)), first_labels, second_labels))
        coefficients, _, _, _ = np.linalg.lstsq(design, points, rcond=None)
        residuals = np.linalg.norm(design @ coefficients - points, axis=1)
        robust_limit = max(0.004, 3.0 * float(np.median(residuals)))
        inliers = residuals <= robust_limit
        if inliers.sum() < int(min_visible):
            continue
        coefficients, _, _, _ = np.linalg.lstsq(design[inliers], points[inliers], rcond=None)
        residuals = np.linalg.norm(design[inliers] @ coefficients - points[inliers], axis=1)
        basis_lengths = np.linalg.norm(coefficients[1:], axis=1)
        if np.min(basis_lengths) <= 1e-6:
            continue
        score = float(np.sqrt(np.mean(residuals ** 2)) / np.min(basis_lengths))
        fits.append((score, first_count, second_count, coefficients))
    if not fits:
        return targets
    score, first_count, second_count, coefficients = min(fits, key=lambda item: item[0])
    if score > 0.20:
        return targets

    lattice = np.asarray([
        coefficients[0] + i * coefficients[1] + j * coefficients[2]
        for i in range(first_count) for j in range(second_count)
    ])
    spacing = min(np.linalg.norm(coefficients[1]), np.linalg.norm(coefficients[2]))
    for target in targets:
        if target.get('class_name') not in {'crooked_cap', 'missing_cap'}:
            continue
        position = np.asarray(target.get('base_pos_m', ()), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            continue
        feature_position = position.copy()
        nearest_slot = lattice[int(np.argmin(np.linalg.norm(lattice - position[:2], axis=1)))]
        if np.linalg.norm(nearest_slot - position[:2]) > 0.45 * spacing:
            continue
        position[:2] = nearest_slot
        # A missing-cap depth crop sees the inside wall/bore below the rim.
        # That depth is useful for confirming a vial exists, but it is not the
        # cap seating TCP height. Keep the calibrated ray/plane rim height.
        if target.get('class_name') == 'missing_cap':
            position[2] = float(target.get('plane_base_z_m', rim_base_z()))
        else:
            # Preserve the measured leaning-flange centre separately from the
            # aperture centre. The cone uses it as the initial touch point,
            # then guides the cap inward to the fitted vial axis.
            target['feature_base_pos_m'] = feature_position.tolist()
        target['base_pos_m'] = position.tolist()
        target['tcp_pos_mm'] = base_to_uf0(position).tolist()
        target['localization_note'] = (
            'Contact XY refined by affine 6x5 tray-grid fit from visible TableCam detections; '
            'defect Z uses its calibrated contact plane and cap surface remains an aligned RGB-D observation')
    return targets


def cap_axis_from_rgbd_points(points, *, min_points=12, min_surface_z=None):
    """Estimate an upward cap axis from a TableCam RGB-D box point cloud.

    A crooked cap's flange is locally planar, so its least-variance PCA axis is
    the cap axis.  This is camera evidence only and is shared by simulation and
    the physical adapter.
    """
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < min_points:
        return None
    # Suppress the vial/cavity below the cap.  Live remediation has a calibrated
    # rim plane; the percentile fallback keeps the standalone estimator useful.
    threshold = (float(min_surface_z) if min_surface_z is not None else
                 float(np.percentile(points[:, 2], 40)))
    points = points[points[:, 2] >= threshold]
    if len(points) < min_points:
        return None
    # Fit z = ax + by + c directly.  PCA on a cropped hollow flange is biased
    # by its asymmetric point footprint; robust vertical residual rejection
    # preserves the actual high-side gradient used for contact.
    A = np.c_[points[:, :2], np.ones(len(points))]
    q = points
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        residual = q[:, 2] - A @ coefficients
        keep = np.abs(residual) < max(0.0015, 2.0 * float(residual.std()))
        A, q = A[keep], q[keep]
        if len(q) < min_points:
            return None
    coefficients, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
    axis = np.array([-coefficients[0], -coefficients[1], 1.0])
    axis /= np.linalg.norm(axis)
    # Reject side walls and ill-conditioned fits rather than inventing a pose.
    if axis[2] < 0.50:
        return None
    return axis


def load_cameras(calib_dir=CALIBRATION_DIR):
    raw = json.loads((Path(calib_dir) / 'cameras.json').read_text())
    return {name: {'T': np.array(c.get('T_base_cam', c.get('T_flange_cam'))),
                   'fx': c['fx'], 'k1': c.get('k1', 0.), 'size': tuple(c['size']),
                   'frame': 'base' if 'T_base_cam' in c else 'flange'}
            for name, c in raw.items() if isinstance(c, dict)}


def pixel_to_base(cam, uv, z=SPEC.surface_base_z, image_size=None):
    if cam.get('frame', 'base') != 'base':
        raise ValueError('Fixed camera in robot-base frame required')
    w, h = cam['size']
    pixel = np.asarray(uv, dtype=float)
    if image_size is not None:
        iw, ih = image_size
        if min(iw, ih) <= 0 or abs(iw / ih - w / h) > 1e-6:
            raise ValueError('Image aspect ratio changed: supply a calibrated crop transform')
        pixel = pixel * np.array([w / iw, h / ih])
    xd, yd = (pixel - np.array([w / 2, h / 2])) / cam['fx']
    x, y = xd, yd
    for _ in range(10):
        d = 1 + cam.get('k1', 0.) * (x*x + y*y)
        x, y = xd/d, yd/d
    ray = cam['T'][:3, :3] @ np.array([x, y, 1.])
    origin = cam['T'][:3, 3]
    if abs(ray[2]) < 1e-9:
        raise ValueError('Ray parallel to support plane')
    t = (z-origin[2])/ray[2]
    if t <= 0 or not np.isfinite(t):
        raise ValueError('Support plane behind camera')
    return origin + ray*t


def camera_from_usd_capture(metadata):
    """Convert a captured USD camera to the fixed-camera convention, in base metres."""
    m = np.asarray(metadata['camera_to_world'], dtype=float)
    T = np.eye(4)
    T[:3, :3] = m[:3, :3].T @ np.diag([1., -1., -1.])
    T[:3, 3] = m[3, :3] - [0, 0, SPEC.world_base_z]
    w, h = metadata['resolution']
    return {'T': T, 'fx': w*metadata['focal_length_mm']/metadata['horizontal_aperture_mm'],
            'k1': 0., 'size': (w, h), 'frame': 'base'}


class RemediationVisionDetector:
    def __init__(self, checkpoint, calib_dir=CALIBRATION_DIR, *, camera=None, model=None,
                 confidence=.6, device='cpu', rim_z=None, imgsz=1280):
        if model is None:
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError('A validated remediation detector checkpoint is required')
            from ultralytics import YOLO
            model = YOLO(str(checkpoint))
        self.model = model
        names = model.names
        actual = tuple(names[i] for i in range(len(names))) if isinstance(names, dict) else tuple(names)
        if actual == ('pass', 'crooked_cap', 'missing_cap', 'damaged_vial'):
            self.class_map = {0: 'good_cap', 1: 'crooked_cap', 2: 'missing_cap', 3: 'table_cap'}
        elif actual == CLASS_NAMES:
            self.class_map = {i: name for i, name in enumerate(CLASS_NAMES)}
        else:
            self.class_map = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        self.table_cam = camera if camera is not None else load_cameras(calib_dir)['table']
        self.confidence, self.device, self.imgsz = float(confidence), device, int(imgsz)
        if not 0.0 < self.confidence < 1.0:
            raise ValueError('confidence must be between 0 and 1')
        if self.imgsz <= 0:
            raise ValueError('imgsz must be positive')
        self.rim_z = rim_base_z() if rim_z is None else float(rim_z)

    def detect(self, rgb, timestamp=None, depth_m=None):
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected an RGB image with three channels')
        h, w = rgb.shape[:2]
        if depth_m is not None:
            depth_m=np.asarray(depth_m,dtype=float)
            if depth_m.shape[:2] != (h,w):
                raise ValueError('Depth and RGB must share the calibrated native pixel grid')
        results = self.model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=self.confidence,
                                     imgsz=self.imgsz, device=self.device, verbose=False)
        targets = []
        for result in results:
            for xyxy, confidence, cls_id in zip(result.boxes.xyxy.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy(),
                                                result.boxes.cls.cpu().numpy()):
                name = self.class_map.get(int(cls_id), CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else 'unknown')
                center = (xyxy[:2]+xyxy[2:])/2
                # Object origins differ: spare caps rest on the plug, seated caps on the rim.
                plane = (SPEC.surface_base_z + SPEC.plug_depth + SPEC.flange_thickness
                         if name == 'table_cap' else self.rim_z + (0 if name == 'missing_cap' else SPEC.flange_thickness))
                position = pixel_to_base(self.table_cam, center, plane, (w, h))
                depth_value=None
                surface_top=None
                cap_axis=None
                cap_high_point=None
                if depth_m is not None:
                    x0,y0,x1,y1=np.round(xyxy).astype(int)
                    x0,x1=max(x0,0),min(x1,w); y0,y1=max(y0,0),min(y1,h)
                    mx,my=int(.15*(x1-x0)),int(.15*(y1-y0))
                    xa,xb,ya,yb=x0+mx,x1-mx,y0+my,y1-my
                    crop=depth_m[ya:yb,xa:xb]
                    vv,uu=np.mgrid[ya:yb,xa:xb]
                    valid=np.isfinite(crop) & (crop>0)
                    if valid.sum() >= 8:
                        z=crop[valid]
                        native_w,native_h=self.table_cam['size']
                        su,sv=native_w/w,native_h/h
                        pc=np.column_stack([((uu[valid]*su-native_w/2)/self.table_cam['fx'])*z,
                                            ((vv[valid]*sv-native_h/2)/self.table_cam['fx'])*z,z])
                        points=pc@self.table_cam['T'][:3,:3].T+self.table_cam['T'][:3,3]
                        position=np.median(points,axis=0)
                        depth_value=float(np.median(z))
                        surface_top=float(np.percentile(points[:,2],90))
                        if name == 'crooked_cap':
                            cap_axis = cap_axis_from_rgbd_points(
                                points, min_surface_z=self.rim_z + 0.0005)
                            cap_surface = points[points[:, 2] > self.rim_z + 0.0005]
                            if len(cap_surface) >= 30:
                                high = cap_surface[
                                    cap_surface[:, 2] >= np.percentile(cap_surface[:, 2], 90)]
                                if len(high) >= 3:
                                    cap_high_point = np.mean(high, axis=0)
                targets.append({'class_name': name, 'confidence': float(confidence),
                                'bbox_xyxy': xyxy.tolist(), 'pixel': center.tolist(),
                                'base_pos_m': position.tolist(), 'tcp_pos_mm': base_to_uf0(position).tolist(),
                                'timestamp': timestamp, 'plane_base_z_m': plane,
                                'depth_median_m':depth_value,
                                'surface_top_base_z_m':surface_top,
                                'cap_axis_base':None if cap_axis is None else cap_axis.tolist(),
                                'cap_high_point_base_m':None if cap_high_point is None else cap_high_point.tolist(),
                                'depth_available':depth_value is not None,
                                'contact_qualified': False,
                                'localization_note': 'RGB box-center and optional aligned depth observation; feature refinement and contact error budget remain unvalidated'})
        return refine_defect_centres_from_grid(targets)
