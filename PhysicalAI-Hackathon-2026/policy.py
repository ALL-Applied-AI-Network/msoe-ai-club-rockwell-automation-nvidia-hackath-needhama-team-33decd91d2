"""Supervised, contact-free TableCam motion-validation policy.

Runs YOLO on a fresh TableCam image, projects a detected loose cap through the
real-cell TableCam calibration, and hovers 75 mm above it. It never closes the
gripper, touches a cap, presses a vial, or uses stored tray coordinates.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inspection.vision_pipeline import RemediationVisionDetector
from src.remediation.geometry import base_to_uf0, transit_base_z
from src.remediation.controller import ContactTool, RemediationController
from calibration.geometry import depth_to_base, load as load_cameras, project


DEFAULT_CHECKPOINT = ROOT / "shared/inspection_models/remediation_yolo_best.pt"
DOWN_WPR_DEG = (-180.0, 0.0, 0.0)
GRIPPER_OPEN = 0.0
TRANSIT_UF0_Z_MM = float(base_to_uf0([0.0, 0.0, transit_base_z()])[2])
HOVER_ABOVE_CAP_M = 0.075
# Only used by the local mock, which has no initial TCP pose.
MOCK_START_TCP = [400.0, 0.0, TRANSIT_UF0_Z_MM, *DOWN_WPR_DEG, GRIPPER_OPEN]

# ``REMEDIATION_MODE=full`` enables the deliberate pick/place/press trial.
# These zero offsets mean UTOOL1 is assumed to be the CAD cap-contact point.
# They are a temporary test assumption, replaced by measured values once they
# are available.  Hover mode (the default) never reaches this code path.
UNMEASURED_CONTACT_TOOL = ContactTool(0.0, 0.0, 0.0, validated=True)


class Policy:
    CAMERAS = ["table"]
    DEPTH = ["table"]
    IMAGE_SIZE = (1920, 1080)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False

    def __init__(self, checkpoint=None):
        weights = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        self.detector = RemediationVisionDetector(weights, confidence=0.80)
        self.cameras = load_cameras()
        self.full_remediation = os.environ.get("REMEDIATION_MODE", "hover").lower() == "full"
        self.controller = RemediationController(UNMEASURED_CONTACT_TOOL) if self.full_remediation else None
        label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_physical_policy")
        self.trace_dir = Path(os.environ.get("REMEDIATION_TRACE_DIR", ROOT / "runs/archive" / label))
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "tablecam_raw").mkdir(exist_ok=True)
        (self.trace_dir / "tablecam_annotated").mkdir(exist_ok=True)
        (self.trace_dir / "depth").mkdir(exist_ok=True)
        self.trace_file = self.trace_dir / "trace.jsonl"
        self.frame_count = 0
        self.last_saved_action = None
        self.last_saved_controller_state = None
        print(f"[remediation-policy] local trace: {self.trace_dir}", flush=True)
        self.reset()

    def reset(self):
        self.state = "inspect"
        self.target = None
        self.last_observation_time = None
        self.events = []
        if self.controller is not None:
            self.controller.reset()

    @staticmethod
    def _tcp(obs):
        tcp = obs.get("tcp_pose")
        if tcp is None:
            trial = obs.get("trial") or {}
            if str(trial.get("trial_id", "")).startswith("mock-"):
                return np.asarray(MOCK_START_TCP[:6], dtype=float)
            raise ValueError("Physical motion requires the current tcp_pose")
        tcp = np.asarray(tcp, dtype=float)
        if tcp.shape != (6,) or not np.isfinite(tcp).all():
            raise ValueError("Physical motion requires six finite tcp_pose values")
        return tcp

    @staticmethod
    def _command(xyz_mm):
        return [float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2]),
                *DOWN_WPR_DEG, GRIPPER_OPEN]

    def _hold(self, obs):
        return self._command(self._tcp(obs)[:3])

    def _choose_table_cap(self, image, stamp):
        detections = self.detector.detect(image, timestamp=stamp)
        candidates = [d for d in detections
                      if d["class_name"] == "table_cap" and d["confidence"] >= 0.80]
        if not candidates:
            self.events.append({"event": "hold", "reason": "no_confident_table_cap"})
            return None
        return max(candidates, key=lambda d: d["confidence"])

    @staticmethod
    def _json(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot encode {type(value).__name__}")

    def _write_trace(self, record):
        with self.trace_file.open("a") as f:
            f.write(json.dumps(record, default=self._json) + "\n")

    def _record_snapshot(self, obs, image, detections, action):
        """Persist compact, inspectable evidence for the decision that sent an action."""
        stamp = obs.get("t")
        state = self.controller.state if self.controller else self.state
        changed_action = (self.last_saved_action is None or
                          not np.allclose(np.asarray(action, dtype=float), self.last_saved_action, atol=1e-6))
        is_decision = changed_action or state != self.last_saved_controller_state
        # Limit an unattended 30-minute run to 600 decision image/depth
        # triplets while retaining every TCP command in trace.jsonl.
        if self.frame_count >= 600 or not is_decision:
            self._write_trace({"t": stamp, "controller_state": state,
                               "action_tcp": action, "detections": detections, "frame": None})
            return
        index = self.frame_count
        self.frame_count += 1
        self.last_saved_action = np.asarray(action, dtype=float)
        self.last_saved_controller_state = state
        raw_name = f"{index:04d}.png"
        annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for item in detections:
            x0, y0, x1, y1 = (int(round(v)) for v in item["bbox_xyxy"])
            color = {"table_cap": (255, 200, 0), "missing_cap": (0, 80, 255),
                     "crooked_cap": (0, 0, 255), "good_cap": (0, 220, 0)}.get(item["class_name"], (255, 255, 255))
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
            cv2.putText(annotated, f'{item["class_name"]} {item["confidence"]:.2f}', (x0, max(18, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(self.trace_dir / "tablecam_raw" / raw_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.trace_dir / "tablecam_annotated" / raw_name), annotated)
        depth = obs.get("depth", {}).get("table")
        if depth is not None:
            cv2.imwrite(str(self.trace_dir / "depth" / raw_name), np.asarray(depth))
        self._write_trace({"t": stamp, "controller_state": state,
                           "action_tcp": action, "tcp_pose_before": obs.get("tcp_pose"),
                           "detections": detections, "frame": raw_name})

    def _snapshot_detections(self, image, stamp):
        """Use one fresh TableCam snapshot to update a complete repair decision."""
        detections = self.detector.detect(image, timestamp=stamp)
        # This is an explicitly requested supervised test mode.  Contact is
        # enabled from the current camera observation, not fixture coordinates.
        return [dict(item, contact_qualified=True,
                     localization_note=item["localization_note"] + "; supervised full-remediation test mode")
                for item in detections]

    def _fuse_table_depth(self, detections, depth_mm):
        """Attach the calibrated RGB-D point inside each current YOLO box.

        The RGB and depth streams have different resolutions and optical
        centres. Depth pixels are first transformed into robot-base points,
        then projected into the RGB image. This avoids the unsafe assumption
        that a 1920x1080 RGB pixel maps by simple scaling to 640x400 depth.
        """
        if depth_mm is None:
            return detections
        depth_mm = np.asarray(depth_mm)
        depth_cam = self.cameras["table_depth"]
        if tuple(depth_mm.shape[:2][::-1]) != tuple(depth_cam["size"]):
            raise ValueError(f"Table depth has {depth_mm.shape[:2]}, expected {depth_cam['size'][::-1]}")
        # Half-resolution sampling still provides dozens of points inside a
        # cap box while keeping per-observation CPU work modest.
        points, pixels = depth_to_base(depth_cam, depth_mm, stride=2, with_pixels=True)
        rgb_pixels = project(self.cameras["table"], points)
        fused = []
        for item in detections:
            x0, y0, x1, y1 = np.asarray(item["bbox_xyxy"], dtype=float)
            # Prefer the central 70 percent of the box to avoid the background
            # around a cap or the neighbouring vial rim.
            mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
            inside = ((rgb_pixels[:, 0] >= x0 + mx) & (rgb_pixels[:, 0] <= x1 - mx) &
                      (rgb_pixels[:, 1] >= y0 + my) & (rgb_pixels[:, 1] <= y1 - my))
            if inside.sum() < 8:
                inside = ((rgb_pixels[:, 0] >= x0) & (rgb_pixels[:, 0] <= x1) &
                          (rgb_pixels[:, 1] >= y0) & (rgb_pixels[:, 1] <= y1))
            updated = dict(item)
            if inside.sum() >= 8:
                selected = points[inside]
                selected_depth = depth_mm[pixels[inside, 1], pixels[inside, 0]].astype(float) / 1000.0
                point = np.median(selected, axis=0)
                updated["base_pos_m"] = point.tolist()
                updated["tcp_pos_mm"] = base_to_uf0(point).tolist()
                updated["depth_median_m"] = float(np.median(selected_depth))
                updated["depth_available"] = True
                updated["depth_rgbd_point_count"] = int(inside.sum())
                updated["localization_note"] += "; calibrated RGB-D box median"
            else:
                updated["depth_available"] = False
                updated["localization_note"] += "; no valid projected depth points in box"
            fused.append(updated)
        return fused

    def _full_act(self, obs, image, stamp, fresh):
        """Pick a spare for a missing vial, then reseat a crooked cap.

        The controller emits one short TCP action per observation.  After the
        pickup route completes, a new TableCam snapshot must show that the cap
        is no longer at the original table location before placement is allowed.
        """
        observed = dict(obs)
        # Give the mock its deterministic initial TCP pose; the real-cell path
        # still rejects a missing pose in _tcp.
        if observed.get("tcp_pose") is None:
            observed["tcp_pose"] = self._tcp(observed).tolist()
        if image is None or not fresh:
            return self.controller.hold(observed)
        self.last_observation_time = stamp
        detections = self._snapshot_detections(image, stamp)
        detections = self._fuse_table_depth(detections, observed.get("depth", {}).get("table"))
        if self.controller.state == "pickup" and self.controller.pending is not None:
            spare = np.asarray(self.controller.pending["spare"]["base_pos_m"], dtype=float)
            still_on_table = any(
                d["class_name"] == "table_cap" and
                np.linalg.norm(np.asarray(d["base_pos_m"], dtype=float)[:2] - spare[:2]) < 0.040
                for d in detections
            )
            observed["grasp_verified"] = not still_on_table
            self.events.append({"event": "tablecam_grasp_check",
                                "verified": observed["grasp_verified"]})
        action = self.controller.act(observed, detections)
        self.events[:] = self.controller.events[-20:]
        self._record_snapshot(observed, image, detections, action)
        return action

    def act(self, obs):
        image = obs.get("images", {}).get("table")
        stamp = obs.get("t")
        fresh = stamp is None or self.last_observation_time is None or stamp > self.last_observation_time

        if self.full_remediation:
            return self._full_act(obs, image, stamp, fresh)

        if self.state == "inspect":
            if image is None or not fresh:
                return self._hold(obs)
            self.last_observation_time = stamp
            self.target = self._choose_table_cap(image, stamp)
            if self.target is None:
                return self._hold(obs)
            self.state = "raise"

        current = self._tcp(obs)
        cap_base = np.asarray(self.target["base_pos_m"], dtype=float)
        hover_uf0 = base_to_uf0(cap_base + np.array([0.0, 0.0, HOVER_ABOVE_CAP_M]))

        if self.state == "raise":
            self.state = "translate"
            return self._command([current[0], current[1], max(current[2], TRANSIT_UF0_Z_MM)])
        if self.state == "translate":
            self.state = "hover"
            return self._command([hover_uf0[0], hover_uf0[1], TRANSIT_UF0_Z_MM])
        if self.state == "hover":
            self.state = "hold"
            self.events.append({"event": "hovered_above_table_cap",
                                "pixel": self.target["pixel"],
                                "base_pos_m": cap_base.tolist(),
                                "hover_uf0_mm": hover_uf0.tolist()})
            return self._command(hover_uf0)
        return self._hold(obs)
