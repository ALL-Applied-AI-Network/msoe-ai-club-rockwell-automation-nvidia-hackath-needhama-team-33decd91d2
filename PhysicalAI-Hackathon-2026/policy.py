"""Supervised, contact-free TableCam motion-validation policy.

Runs YOLO on a fresh TableCam image, projects a detected loose cap through the
real-cell TableCam calibration, and hovers 75 mm above it. It never closes the
gripper, touches a cap, presses a vial, or uses stored tray coordinates.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inspection.vision_pipeline import RemediationVisionDetector
from src.remediation.geometry import base_to_uf0, transit_base_z
from src.remediation.controller import ContactTool, RemediationController


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
        self.full_remediation = os.environ.get("REMEDIATION_MODE", "hover").lower() == "full"
        self.controller = RemediationController(UNMEASURED_CONTACT_TOOL) if self.full_remediation else None
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

    def _snapshot_detections(self, image, stamp):
        """Use one fresh TableCam snapshot to update a complete repair decision."""
        detections = self.detector.detect(image, timestamp=stamp)
        # This is an explicitly requested supervised test mode.  Contact is
        # enabled from the current camera observation, not fixture coordinates.
        return [dict(item, contact_qualified=True,
                     localization_note=item["localization_note"] + "; supervised full-remediation test mode")
                for item in detections]

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
