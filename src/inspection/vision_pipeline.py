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


def _kmeans_1d(values, groups, iterations=30):
    """Small deterministic 1-D clustering helper; no learned tray coordinates."""
    values=np.asarray(values,dtype=float)
    centers=np.linspace(values.min(),values.max(),groups)
    for _ in range(iterations):
        labels=np.argmin(abs(values[:,None]-centers[None,:]),axis=1)
        updated=np.array([values[labels==i].mean() if np.any(labels==i) else centers[i] for i in range(groups)])
        if np.allclose(updated,centers): break
        centers=updated
    return np.sort(centers)


def refine_crooked_to_observed_grid(targets):
    """Replace a tilted cap centroid with its nearest grid intersection.

    The grid is fitted from current good/missing observations only. It is a
    perception-derived consistency prior, not a stored aperture coordinate map.
    """
    anchors=[t for t in targets if t['class_name'] in ('good_cap','missing_cap') and t['confidence']>=.9]
    if len(anchors)<18: return targets
    points=np.array([t['base_pos_m'][:2] for t in anchors],dtype=float)
    center=points.mean(axis=0)
    _,_,vt=np.linalg.svd(points-center,full_matrices=False)
    axes=vt[:2]
    uv=(points-center)@axes.T
    candidates=[]
    for rows,cols in ((6,5),(5,6)):
        u=_kmeans_1d(uv[:,0],rows); v=_kmeans_1d(uv[:,1],cols)
        grid=np.array([center+x*axes[0]+y*axes[1] for x in u for y in v])
        residual=np.mean([np.min(np.linalg.norm(grid-p,axis=1)) for p in points])
        candidates.append((residual,grid))
    _,grid=min(candidates,key=lambda item:item[0])
    refined=[]
    for target in targets:
        item=dict(target)
        if item['class_name']=='crooked_cap':
            old=np.asarray(item['base_pos_m'],dtype=float)
            nearest=grid[np.argmin(np.linalg.norm(grid-old[:2],axis=1))]
            correction=float(np.linalg.norm(nearest-old[:2]))
            # A poor lattice fit must never turn a detector estimate into a
            # distant command target. Expected tilted-cap correction is mm-scale.
            if correction<=.012:
                item['box_center_base_pos_m']=old.tolist()
                item['base_pos_m']=[float(nearest[0]),float(nearest[1]),float(old[2])]
                item['localization_note']='Tilted-cap box center refined to nearest TableCam-observed tray grid intersection; validation pending'
                item['feature']='observed_grid_slot'
            else:
                item['localization_note']='Observed grid fit disagreed with tilted-cap box center; retained coarse RGB estimate'
                item['grid_refinement_rejected_m']=correction
        refined.append(item)
    return refined


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
                 confidence=.6, device='cpu', rim_z=None):
        if model is None:
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError('A validated remediation detector checkpoint is required')
            from ultralytics import YOLO
            model = YOLO(str(checkpoint))
        self.model = model
        names = model.names
        actual = tuple(names[i] for i in range(len(names))) if isinstance(names, dict) else tuple(names)
        if actual != CLASS_NAMES:
            raise ValueError(f'Wrong detector class mapping: {actual}; expected {CLASS_NAMES}')
        self.table_cam = camera if camera is not None else load_cameras(calib_dir)['table']
        self.confidence, self.device = confidence, device
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
                                     imgsz=1280, device=self.device, verbose=False)
        targets = []
        for result in results:
            for xyxy, confidence, cls_id in zip(result.boxes.xyxy.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy(),
                                                result.boxes.cls.cpu().numpy()):
                name = CLASS_NAMES[int(cls_id)]
                center = (xyxy[:2]+xyxy[2:])/2
                # Object origins differ: spare caps rest on the plug, seated caps on the rim.
                plane = (SPEC.surface_base_z + SPEC.plug_depth + SPEC.flange_thickness
                         if name == 'table_cap' else self.rim_z + (0 if name == 'missing_cap' else SPEC.flange_thickness))
                position = pixel_to_base(self.table_cam, center, plane, (w, h))
                depth_value=None
                if depth_m is not None:
                    x0,y0,x1,y1=np.round(xyxy).astype(int)
                    crop=depth_m[max(y0,0):min(y1,h),max(x0,0):min(x1,w)]
                    valid=crop[np.isfinite(crop) & (crop>0)]
                    if len(valid): depth_value=float(np.median(valid))
                targets.append({'class_name': name, 'confidence': float(confidence),
                                'bbox_xyxy': xyxy.tolist(), 'pixel': center.tolist(),
                                'base_pos_m': position.tolist(), 'tcp_pos_mm': base_to_uf0(position).tolist(),
                                'timestamp': timestamp, 'plane_base_z_m': plane,
                                'depth_median_m':depth_value,
                                'depth_available':depth_value is not None,
                                'contact_qualified': False,
                                'localization_note': 'RGB box-center and optional aligned depth observation; feature refinement and contact error budget remain unvalidated'})
        return refine_crooked_to_observed_grid(targets)
