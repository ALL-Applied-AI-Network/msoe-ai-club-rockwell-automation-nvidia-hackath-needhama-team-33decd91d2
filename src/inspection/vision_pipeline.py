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
                targets.append({'class_name': name, 'confidence': float(confidence),
                                'bbox_xyxy': xyxy.tolist(), 'pixel': center.tolist(),
                                'base_pos_m': position.tolist(), 'tcp_pos_mm': base_to_uf0(position).tolist(),
                                'timestamp': timestamp, 'plane_base_z_m': plane,
                                'depth_median_m':depth_value,
                                'surface_top_base_z_m':surface_top,
                                'depth_available':depth_value is not None,
                                'contact_qualified': False,
                                'localization_note': 'RGB box-center and optional aligned depth observation; feature refinement and contact error budget remain unvalidated'})
        return targets
