"""Supervised, contact-free TableCam motion-validation policy.

Runs YOLO on a fresh TableCam image, projects a detected loose cap through the
real-cell TableCam calibration, and hovers 75 mm above it. It never closes the
gripper, touches a cap, presses a vial, or uses stored tray coordinates.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inspection.vision_pipeline import RemediationVisionDetector, cap_axis_from_rgbd_points
from src.inspection.depth_visualization import render_depth_heatmap
from src.inspection.height_plane import fit_parallel_work_surface
from src.remediation.geometry import (base_to_uf0, inspection_base_position,
                                      resolve_height_planes, SPEC, transit_base_z)
from src.remediation.controller import (DEFAULT_CONE_TIP_FROM_TCP_M, ContactTool,
                                        RemediationController, wpr_to_rotation)
from src.remediation.wrist_refinement import (project_base_to_wrist,
                                               refine_from_vlm_box,
                                               fit_local_support_plane,
                                               render_wrist_refinement_annotation)
from calibration.geometry import (depth_to_base, flange_pose,
                                  load as load_cameras, project)


DEFAULT_CHECKPOINT = ROOT / "shared/yolo26x/best.pt"
OFFSETS_PATH = Path(__file__).resolve().parent / "config" / "remediation_offsets.json"
_OFFSETS = json.loads(OFFSETS_PATH.read_text())
DEFAULT_PICKUP_Z_OFFSET_M = float(_OFFSETS["pickup_z_offset_m"])
DEFAULT_PICKUP_SLIDE_X_M = float(_OFFSETS["pickup_slide_x_m"])
DEFAULT_WORK_SURFACE_Z_OFFSET_M = float(_OFFSETS["work_surface_z_offset_m"])
DEFAULT_VIAL_RIM_PLANE_OFFSET_M = float(_OFFSETS["vial_rim_plane_offset_m"])
DEFAULT_VIAL_RIM_PLANE_OVERRIDE_M = float(_OFFSETS["vial_rim_plane_override_m"])
DEFAULT_VIAL_HEIGHT_OVERRIDE_M = float(_OFFSETS["vial_height_override_m"])
DEFAULT_MISSING_RELEASE_Z_OFFSET_M = float(_OFFSETS["missing_release_z_offset_m"])
DEFAULT_CONE_PRESS_Z_OFFSET_M = float(_OFFSETS["cone_press_z_offset_m"])
DEFAULT_WRIST_TABLE_CAP_XY_OFFSET_M = tuple(_OFFSETS["wrist_table_cap_xy_offset_m"])
DEFAULT_WRIST_MISSING_CAP_XY_OFFSET_M = tuple(_OFFSETS["wrist_missing_cap_xy_offset_m"])
DEFAULT_WRIST_CROOKED_CAP_XY_OFFSET_M = tuple(_OFFSETS["wrist_crooked_cap_xy_offset_m"])
DEFAULT_MISSING_RELEASE_SIDE_STEP_M = float(
    _OFFSETS["missing_release_side_step_m"])
DEFAULT_MISSING_RELEASE_EXIT_STEP_M = float(
    _OFFSETS["missing_release_exit_step_m"])
# TableCam supplies coarse candidates only when wrist refinement is enabled.
# Keep legacy base-frame controls available for non-wrist fallback operation,
# but do not bake a prior physical run's offsets into a new deployment.
DEFAULT_TABLE_CAP_XY_OFFSET_M = (0.0, 0.0)
DEFAULT_MISSING_CAP_XY_OFFSET_M = (0.0, 0.0)
DEFAULT_CROOKED_CAP_XY_OFFSET_M = (0.0, 0.0)
DOWN_WPR_DEG = (-180.0, 0.0, 0.0)
PHYSICAL_CONE_PRESS_W_DEG = -135.0
GRIPPER_OPEN = 0.0
TRANSIT_UF0_Z_MM = float(base_to_uf0([0.0, 0.0, transit_base_z()])[2])
HOVER_ABOVE_CAP_M = 0.075
INSPECTION_UF0_MM = base_to_uf0(inspection_base_position())
MAX_ROW_TRANSLATION_MM = 9.0
MAX_ROW_WPR_DEG = 5.0
MOTION_SETTLE_ROWS = 5
ARRIVAL_TRANSLATION_MM = 1.5
ARRIVAL_ORIENTATION_DEG = 2.0
# Only used by the local mock, which has no initial TCP pose.
MOCK_START_TCP = [400.0, 0.0, TRANSIT_UF0_Z_MM, *DOWN_WPR_DEG, GRIPPER_OPEN]

def provisional_contact_tool(allowed, pick_offset_m, place_offset_m, press_offset_m):
    """Build explicit provisional offsets; never silently bless the CAD assumption."""
    if not allowed:
        raise RuntimeError(
            "Full contact mode uses an unmeasured TCP. Pass --allow-provisional-contact "
            "only for an operator-supervised physical trial."
        )
    values = [pick_offset_m, place_offset_m, press_offset_m]
    for name, value in zip(("pick", "place", "press"), values):
        if not np.isfinite(value):
            raise ValueError(f"--{name}-offset-m must be finite")
    return ContactTool(*values, validated=True)


class Policy:
    # IMAGE_SIZE applies to every requested RGB stream.  Wrist calibration is
    # scaled from 1280x720 to this 1920x1080 deployment size while preserving
    # the physical camera's field of view.
    CAMERAS = ["table", "wrist"]
    DEPTH = ["table", "wrist"]
    IMAGE_SIZE = (1920, 1080)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False

    def __init__(self, checkpoint=None, *, mode="hover", confidence=0.70, imgsz=1280,
                 device="0", allow_provisional_contact=False,
                 pick_offset_m=None, place_offset_m=None, press_offset_m=None,
                 missing_release_z_offset_m=DEFAULT_MISSING_RELEASE_Z_OFFSET_M,
                 cone_press_z_offset_m=DEFAULT_CONE_PRESS_Z_OFFSET_M,
                 crooked_shoulder_depth_m=0.0326838,
                 crooked_tap_depth_m=0.005,
                 crooked_action="shoulder",
                 cone_tip_from_tcp_m=DEFAULT_CONE_TIP_FROM_TCP_M,
                 cone_press_below_rim_m=0.003,
                 cone_press_stroke_m=0.020,
                 cone_heading_deg=-90.0,
                 cone_table_pick_offset_m=None,
                 pickup_z_offset_m=DEFAULT_PICKUP_Z_OFFSET_M,
                 pickup_slide_x_m=DEFAULT_PICKUP_SLIDE_X_M,
                 work_surface_z_offset_m=DEFAULT_WORK_SURFACE_Z_OFFSET_M,
                 vial_rim_plane_offset_m=DEFAULT_VIAL_RIM_PLANE_OFFSET_M,
                 vial_rim_plane_override_m=DEFAULT_VIAL_RIM_PLANE_OVERRIDE_M,
                 vial_height_override_m=DEFAULT_VIAL_HEIGHT_OVERRIDE_M,
                 table_plane_offset_m=None,
                 table_cap_xy_offset_m=DEFAULT_TABLE_CAP_XY_OFFSET_M,
                 missing_cap_xy_offset_m=DEFAULT_MISSING_CAP_XY_OFFSET_M,
                 crooked_cap_xy_offset_m=DEFAULT_CROOKED_CAP_XY_OFFSET_M,
                 missing_release_side_step_m=DEFAULT_MISSING_RELEASE_SIDE_STEP_M,
                 missing_release_exit_step_m=DEFAULT_MISSING_RELEASE_EXIT_STEP_M,
                 crooked_wiggle_radius_m=0.00075,
                 crooked_wiggle_cycles=1,
                 pickup_wiggle_radius_m=0.0, pickup_wiggle_cycles=0,
                 work_surface_z_m=-0.0051, vial_rim_z_m=None,
                 wrist_refinement=False, wrist_view_distance_m=0.300,
                 wrist_crooked_view_distance_m=0.250,
                 wrist_max_correction_m=0.040,
                 wrist_table_cap_xy_offset_m=DEFAULT_WRIST_TABLE_CAP_XY_OFFSET_M,
                 wrist_missing_cap_xy_offset_m=DEFAULT_WRIST_MISSING_CAP_XY_OFFSET_M,
                 wrist_crooked_cap_xy_offset_m=DEFAULT_WRIST_CROOKED_CAP_XY_OFFSET_M,
                 trace_dir=None):
        weights = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if str(device).lower() != "cpu":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU inference requested but CUDA is unavailable. Claim a GPU (for example, "
                    "srun --gres=gpu:1 ...) or explicitly pass --device cpu."
                )
        self.nominal_work_surface_z = float(work_surface_z_m)
        self.work_surface_z_offset_m = float(work_surface_z_offset_m)
        # Backwards-compatible alias for launch commands written before the
        # run-level work-surface plane fit. It has exactly the new global
        # surface-offset meaning and cannot be combined with the new spelling.
        if table_plane_offset_m is not None:
            if abs(self.work_surface_z_offset_m) > 1e-12:
                raise ValueError(
                    "use either table_plane_offset_m or work_surface_z_offset_m, not both")
            self.work_surface_z_offset_m = float(table_plane_offset_m)
        self.vial_rim_plane_offset_m = float(vial_rim_plane_offset_m)
        self.vial_rim_plane_override_m = float(vial_rim_plane_override_m)
        self.vial_height_override_m = float(vial_height_override_m)
        if vial_rim_z_m is not None:
            if abs(self.vial_rim_plane_override_m) > 1e-12:
                raise ValueError(
                    "use either vial_rim_z_m or vial_rim_plane_override_m, not both")
            self.vial_rim_plane_override_m = float(vial_rim_z_m)
        for name, value in (
                ("work surface Z offset", self.work_surface_z_offset_m),
                ("vial-rim plane offset", self.vial_rim_plane_offset_m)):
            if not np.isfinite(value) or abs(value) > 0.025:
                raise ValueError(f"{name} must be within +/-0.025 m")
        if (not np.isfinite(self.vial_rim_plane_override_m) or
                (self.vial_rim_plane_override_m != 0.0 and
                 not 0.05 <= self.vial_rim_plane_override_m <= 0.45)):
            raise ValueError("vial-rim plane override must be zero or 0.05..0.45 m")
        if (not np.isfinite(self.vial_height_override_m) or
                (self.vial_height_override_m != 0.0 and
                 not 0.15 <= self.vial_height_override_m <= 0.25)):
            raise ValueError("vial-height override must be zero or 0.15..0.25 m")
        self.vial_height_m = (SPEC.vial_height if self.vial_height_override_m == 0.0
                              else self.vial_height_override_m)
        self.work_surface_z = self.nominal_work_surface_z + self.work_surface_z_offset_m
        self.derived_vial_rim_z = (self.work_surface_z + SPEC.tray_base_thickness +
                                   self.vial_height_m)
        self.vial_rim_z = (self.vial_rim_plane_override_m
                           if self.vial_rim_plane_override_m != 0.0
                           else self.derived_vial_rim_z + self.vial_rim_plane_offset_m)
        # Reuse the shared range validation without changing the fitted values.
        resolve_height_planes(self.work_surface_z, self.vial_rim_z)
        self.detector = RemediationVisionDetector(
            weights, confidence=confidence, imgsz=imgsz, device=device,
            work_surface_z_m=self.work_surface_z, rim_z=self.vial_rim_z)
        self.confidence = float(confidence)
        self.cameras = load_cameras()
        mode = mode.lower()
        if mode not in ("hover", "full"):
            raise ValueError("mode must be 'hover' or 'full'")
        self.full_remediation = mode == "full"
        # The user-facing pickup Z is an absolute fingertip clearance above
        # the local table plane. Convert it into the controller's legacy
        # cap-top-relative signed offset. Retain the old signed flags as
        # explicit compatibility overrides (positive-up).
        if crooked_action == "cone-press" and cone_table_pick_offset_m is not None:
            effective_pick_offset_m = float(cone_table_pick_offset_m)
            pickup_z_source = "legacy --cone-table-pick-offset-m (positive up)"
        elif pick_offset_m is not None:
            effective_pick_offset_m = float(pick_offset_m)
            pickup_z_source = "legacy --pick-offset-m (positive up)"
        else:
            effective_pick_offset_m = (float(pickup_z_offset_m) -
                                       SPEC.plug_depth - SPEC.flange_thickness)
            pickup_z_source = "pickup_z_offset_m (fingertip height above table)"
        if place_offset_m is not None:
            if abs(float(missing_release_z_offset_m)) > 1e-12:
                raise ValueError(
                    "use either place_offset_m or missing_release_z_offset_m, not both")
            effective_place_offset_m = float(place_offset_m)
            missing_release_z_source = "legacy place_offset_m"
        else:
            effective_place_offset_m = float(missing_release_z_offset_m)
            missing_release_z_source = "missing_release_z_offset_m"
        if press_offset_m is not None:
            if abs(float(cone_press_z_offset_m)) > 1e-12:
                raise ValueError(
                    "use either press_offset_m or cone_press_z_offset_m, not both")
            effective_press_offset_m = float(press_offset_m)
            cone_press_z_source = "legacy press_offset_m"
        else:
            effective_press_offset_m = float(cone_press_z_offset_m)
            cone_press_z_source = "cone_press_z_offset_m"
        if not np.isfinite(float(pickup_z_offset_m)):
            raise ValueError("pickup_z_offset_m must be finite")
        cone_heading_deg = float(cone_heading_deg)
        if not np.isfinite(cone_heading_deg) or not -180.0 <= cone_heading_deg <= 180.0:
            raise ValueError("cone_heading_deg must be between -180 and 180 degrees")
        self.controller = (RemediationController(provisional_contact_tool(
            allow_provisional_contact, effective_pick_offset_m,
            effective_place_offset_m, effective_press_offset_m),
            min_confidence=self.confidence,
            crooked_shoulder_depth_m=crooked_shoulder_depth_m,
            crooked_tap_depth_m=crooked_tap_depth_m,
            crooked_action=crooked_action,
            cone_tip_from_tcp_m=cone_tip_from_tcp_m,
            # The physical cone is mounted on the opposite FANUC-W side from
            # the retained Isaac visual.  The October 30 supervised trial
            # verified that +135 leans the real arm the wrong way.
            cone_press_wpr_deg=(PHYSICAL_CONE_PRESS_W_DEG, 0.0, cone_heading_deg),
            cone_wpr_selector=None,
            cone_press_below_rim_m=cone_press_below_rim_m,
            cone_press_stroke_m=cone_press_stroke_m,
            table_cap_xy_offset_m=table_cap_xy_offset_m,
            missing_cap_xy_offset_m=missing_cap_xy_offset_m,
            crooked_cap_xy_offset_m=crooked_cap_xy_offset_m,
            missing_release_side_step_m=missing_release_side_step_m,
            missing_release_exit_step_m=missing_release_exit_step_m,
            crooked_wiggle_radius_m=crooked_wiggle_radius_m,
            crooked_wiggle_cycles=crooked_wiggle_cycles,
            pickup_wiggle_radius_m=pickup_wiggle_radius_m,
            pickup_wiggle_cycles=pickup_wiggle_cycles,
            pickup_slide_x_m=pickup_slide_x_m,
            wrist_refinement=wrist_refinement,
            wrist_view_distance_m=wrist_view_distance_m,
            wrist_crooked_view_distance_m=wrist_crooked_view_distance_m,
            wrist_table_cap_xy_offset_m=wrist_table_cap_xy_offset_m,
            wrist_missing_cap_xy_offset_m=wrist_missing_cap_xy_offset_m,
            wrist_crooked_cap_xy_offset_m=wrist_crooked_cap_xy_offset_m,
            vial_rim_z_m=self.vial_rim_z)
            if self.full_remediation else None)
        self.wrist_refinement = bool(wrist_refinement and self.full_remediation)
        self.wrist_max_correction_m = float(wrist_max_correction_m)
        if not np.isfinite(self.wrist_max_correction_m):
            raise ValueError("wrist maximum correction must be finite")
        self.wrist_vlm = None
        self.wrist_pool = None
        if self.wrist_refinement:
            from src.inspection.locateanything_runtime import LocateAnythingRuntime
            self.wrist_vlm = LocateAnythingRuntime(dtype="float16")
            self.wrist_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wrist-vlm")
        label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_physical_policy")
        # Physical-policy traces are team artifacts, so share them by default.
        self.trace_dir = Path(trace_dir) if trace_dir else ROOT / "shared/runs" / label
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "tablecam_raw").mkdir(exist_ok=True)
        (self.trace_dir / "tablecam_annotated").mkdir(exist_ok=True)
        (self.trace_dir / "wristcam_raw").mkdir(exist_ok=True)
        (self.trace_dir / "wristcam_annotated").mkdir(exist_ok=True)
        (self.trace_dir / "wristcam_results").mkdir(exist_ok=True)
        self.height_planes_path = self.trace_dir / "height_planes.json"
        self.height_plane_history = []
        self._write_height_planes({
            "status": "pending_first_tablecam_depth_frame",
            "raw_fitted_work_surface_z_m": None,
            "effective_work_surface_z_m": self.work_surface_z,
            "derived_vial_rim_z_m": self.derived_vial_rim_z,
            "effective_vial_rim_z_m": self.vial_rim_z,
        })
        (self.trace_dir / "depth").mkdir(exist_ok=True)
        (self.trace_dir / "depth_raw").mkdir(exist_ok=True)
        config = {
            "checkpoint": str(weights.resolve()),
            "mode": mode,
            "confidence": self.confidence,
            "imgsz": int(imgsz),
            "device": str(device),
            "allow_provisional_contact": bool(allow_provisional_contact),
            "pick_offset_m": None if pick_offset_m is None else float(pick_offset_m),
            "effective_pick_offset_m": float(effective_pick_offset_m),
            "pickup_z_offset_m": float(pickup_z_offset_m),
            "pickup_slide_x_m": float(pickup_slide_x_m),
            "pickup_z_offset_convention": "fingertip clearance above local table plane",
            "pickup_z_source": pickup_z_source,
            "nominal_work_surface_z_m": self.nominal_work_surface_z,
            "work_surface_z_offset_m": self.work_surface_z_offset_m,
            "vial_rim_plane_offset_m": self.vial_rim_plane_offset_m,
            "vial_rim_plane_override_m": self.vial_rim_plane_override_m,
            "vial_height_override_m": self.vial_height_override_m,
            "effective_vial_height_m": self.vial_height_m,
            "legacy_table_plane_offset_m": (None if table_plane_offset_m is None
                                             else float(table_plane_offset_m)),
            "legacy_vial_rim_z_m": (None if vial_rim_z_m is None
                                     else float(vial_rim_z_m)),
            "offsets_file": str(OFFSETS_PATH),
            "cone_table_pick_offset_m": (None if cone_table_pick_offset_m is None
                                         else float(cone_table_pick_offset_m)),
            "place_offset_m": None if place_offset_m is None else float(place_offset_m),
            "press_offset_m": None if press_offset_m is None else float(press_offset_m),
            "missing_release_z_offset_m": float(effective_place_offset_m),
            "missing_release_z_source": missing_release_z_source,
            "cone_press_z_offset_m": float(effective_press_offset_m),
            "cone_press_z_source": cone_press_z_source,
            "table_cap_xy_offset_m": np.asarray(table_cap_xy_offset_m, dtype=float).tolist(),
            "missing_cap_xy_offset_m": np.asarray(missing_cap_xy_offset_m, dtype=float).tolist(),
            "crooked_cap_xy_offset_m": np.asarray(crooked_cap_xy_offset_m, dtype=float).tolist(),
            "crooked_action": crooked_action,
            "crooked_shoulder_depth_m": float(crooked_shoulder_depth_m),
            "crooked_tap_depth_m": float(crooked_tap_depth_m),
            "crooked_wiggle_radius_m": float(crooked_wiggle_radius_m),
            "crooked_wiggle_cycles": int(crooked_wiggle_cycles),
            "pickup_wiggle_radius_m": float(pickup_wiggle_radius_m),
            "pickup_wiggle_cycles": int(pickup_wiggle_cycles),
            "initial_work_surface_z_m": self.work_surface_z,
            "initial_vial_rim_z_m": self.vial_rim_z,
            "cone_tip_from_tcp_m": np.asarray(cone_tip_from_tcp_m, dtype=float).tolist(),
            "cone_press_below_rim_m": float(cone_press_below_rim_m),
            "cone_press_stroke_m": float(cone_press_stroke_m),
            "cone_heading_deg": cone_heading_deg,
            "cone_press_wpr_deg": list(self.controller.cone_press_wpr_deg),
            "missing_release_side_step_m": float(missing_release_side_step_m),
            "missing_release_exit_step_m": float(missing_release_exit_step_m),
            "wrist_refinement": self.wrist_refinement,
            "wrist_colour_size": list(self.IMAGE_SIZE),
            "wrist_view_distance_m": float(wrist_view_distance_m),
            "wrist_crooked_view_distance_m": float(wrist_crooked_view_distance_m),
            "wrist_max_correction_m": self.wrist_max_correction_m,
            "wrist_table_cap_xy_offset_m": np.asarray(
                wrist_table_cap_xy_offset_m, dtype=float).tolist(),
            "wrist_missing_cap_xy_offset_m": np.asarray(
                wrist_missing_cap_xy_offset_m, dtype=float).tolist(),
            "wrist_crooked_cap_xy_offset_m": np.asarray(
                wrist_crooked_cap_xy_offset_m, dtype=float).tolist(),
        }
        (self.trace_dir / "policy_config.json").write_text(json.dumps(config, indent=2) + "\n")
        self.trace_file = self.trace_dir / "trace.jsonl"
        self.frame_count = 0
        self.wrist_frame_count = 0
        self.last_saved_action = None
        self.last_saved_controller_state = None
        print(f"[remediation-policy] local trace: {self.trace_dir}", flush=True)
        self.reset()

    def _write_height_planes(self, current):
        payload = {
            "nominal_work_surface_z_m": self.nominal_work_surface_z,
            "work_surface_z_offset_m": self.work_surface_z_offset_m,
            "tray_base_thickness_m": SPEC.tray_base_thickness,
            "specified_vial_height_m": SPEC.vial_height,
            "vial_height_override_m": self.vial_height_override_m,
            "effective_vial_height_m": self.vial_height_m,
            "vial_rim_plane_offset_m": self.vial_rim_plane_offset_m,
            "vial_rim_plane_override_m": self.vial_rim_plane_override_m,
            "override_active": self.vial_rim_plane_override_m != 0.0,
            "current": current,
            "fits": self.height_plane_history,
        }
        self.height_planes_path.write_text(json.dumps(payload, indent=2) + "\n")

    def _set_effective_height_planes(self, raw_surface_z_m):
        self.work_surface_z = float(raw_surface_z_m) + self.work_surface_z_offset_m
        self.derived_vial_rim_z = (self.work_surface_z + SPEC.tray_base_thickness +
                                   self.vial_height_m)
        self.vial_rim_z = (self.vial_rim_plane_override_m
                           if self.vial_rim_plane_override_m != 0.0
                           else self.derived_vial_rim_z + self.vial_rim_plane_offset_m)
        resolve_height_planes(self.work_surface_z, self.vial_rim_z)
        self.detector.work_surface_z = self.work_surface_z
        self.detector.rim_z = self.vial_rim_z
        self.detector.derived_rim_z = self.derived_vial_rim_z
        if self.controller is not None:
            self.controller.vial_rim_z_m = self.vial_rim_z

    def _fit_run_height_planes(self, obs):
        """Fit and freeze one TableCam-depth Z plane for the current rollout."""
        if self.height_plane_fitted:
            return
        depth = obs.get("depth", {}).get("table")
        if depth is None:
            return
        depth = np.asarray(depth)
        depth_cam = self.cameras["table_depth"]
        if tuple(depth.shape[:2][::-1]) != tuple(depth_cam["size"]):
            raise ValueError(
                f"Table depth has {depth.shape[:2]}, expected {depth_cam['size'][::-1]}")
        fit = None
        fallback_reason = None
        try:
            points = depth_to_base(depth_cam, depth, stride=4)
            fit = fit_parallel_work_surface(points, self.nominal_work_surface_z)
            raw_surface = float(fit["fitted_z_base_m"])
            status = "fitted"
        except ValueError as error:
            raw_surface = self.nominal_work_surface_z
            fallback_reason = str(error)
            status = "nominal_fallback"
        self._set_effective_height_planes(raw_surface)
        trial = obs.get("trial") or {}
        record = {
            "status": status,
            "timestamp": obs.get("t"),
            "trial_id": trial.get("trial_id"),
            "fit": fit,
            "fallback_reason": fallback_reason,
            "raw_fitted_work_surface_z_m": (None if fit is None else raw_surface),
            "effective_work_surface_z_m": self.work_surface_z,
            "derived_vial_rim_z_m": self.derived_vial_rim_z,
            "effective_vial_rim_z_m": self.vial_rim_z,
            "vial_rim_source": ("nonzero absolute override"
                                if self.vial_rim_plane_override_m != 0.0
                                else "fitted work surface + tray base + vial height + rim offset"),
        }
        self.height_plane_history.append(record)
        self._write_height_planes(record)
        self.events.append({"event": "height_plane_calibrated", **record})
        self.height_plane_fitted = True

    def reset(self):
        self.state = "inspect"
        self.target = None
        self.last_observation_time = None
        self.events = []
        self.hover_detections = []
        self.motion_goal = None
        self.start_orientation_checked = False
        self.wrist_request = None
        self.height_plane_fitted = False
        self._set_effective_height_planes(self.nominal_work_surface_z)
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

    def _at_inspection(self, obs):
        return np.linalg.norm(self._tcp(obs)[:3] - INSPECTION_UF0_MM) <= 2.0

    def _inspection_action(self, obs):
        """Move to the camera-clear point without undoing a staged cone wrist."""
        joints = np.asarray(obs.get("joints", []), dtype=float)
        closed = float(joints[6]) if joints.size > 6 and np.isfinite(joints[6]) else GRIPPER_OPEN
        wpr = (self.controller.active_cone_press_wpr_deg
               if self.controller is not None and self.controller.cone_oriented
               else DOWN_WPR_DEG)
        return [*map(float, INSPECTION_UF0_MM), *map(float, wpr), closed]

    def _validate_start_orientation(self, obs):
        if self.start_orientation_checked:
            return
        current = self._tcp(obs)
        relative = wpr_to_rotation(current[3:6]).T @ wpr_to_rotation(DOWN_WPR_DEG)
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        if angle > 20.0:
            raise RuntimeError(
                f"starting tool orientation is {angle:.1f} deg from the outward-facing wrist-camera "
                f"home {DOWN_WPR_DEG}; jog to the approved home pose instead of allowing an automatic wrist turn"
            )
        self.start_orientation_checked = True

    def _choose_table_cap(self, image, stamp):
        detections = self.detector.detect(image, timestamp=stamp)
        self.hover_detections = detections
        candidates = [d for d in detections
                      if d["class_name"] == "table_cap" and d["confidence"] >= self.confidence]
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
        policy_decision = (self.controller.decision_records[-1]
                           if self.controller and self.controller.decision_records else None)
        # Limit an unattended 30-minute run to 600 decision image/depth
        # triplets while retaining every TCP command in trace.jsonl.
        if self.frame_count >= 600 or not is_decision:
            self._write_trace({"t": stamp, "controller_state": state,
                               "action_tcp": action, "detections": detections,
                               "events": self.events[-20:], "policy_decision": policy_decision,
                               "frame": None})
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
            depth = np.asarray(depth)
            cv2.imwrite(str(self.trace_dir / "depth_raw" / raw_name), depth)
            cv2.imwrite(str(self.trace_dir / "depth" / raw_name), render_depth_heatmap(depth))
        self._write_trace({"t": stamp, "controller_state": state,
                           "action_tcp": action, "tcp_pose_before": obs.get("tcp_pose"),
                           "detections": detections, "events": self.events[-20:],
                           "policy_decision": policy_decision, "frame": raw_name})

    def _snapshot_detections(self, image, stamp):
        """Use one fresh TableCam snapshot to update a complete repair decision."""
        detections = self.detector.detect(image, timestamp=stamp)
        # This is an explicitly requested supervised test mode.  Contact is
        # enabled from the current camera observation, not fixture coordinates.
        return [dict(item, contact_qualified=True,
                     localization_note=item["localization_note"] + "; supervised full-remediation test mode")
                for item in detections]

    def _attach_table_depth_observations(self, detections, depth_mm):
        """Attach RGB-D diagnostics without changing commanded target XYZ.

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
                updated["depth_median_m"] = float(np.median(selected_depth))
                updated["surface_top_base_z_m"] = float(np.percentile(selected[:, 2], 90))
                if updated["class_name"] == "crooked_cap":
                    axis = cap_axis_from_rgbd_points(selected)
                    updated["cap_axis_base"] = None if axis is None else axis.tolist()
                updated["depth_available"] = True
                updated["depth_rgbd_point_count"] = int(inside.sum())
                updated["localization_note"] += "; RGB-D observation only; commanded XYZ unchanged"
            else:
                updated["depth_available"] = False
                updated["localization_note"] += "; no valid projected depth points in box"
            fused.append(updated)
        return fused

    def _attach_wrist_refinement(self, observed, wrist_image, wrist_depth):
        """Start or poll eye-in-hand VLM localization requests.

        The first call at a refinement gate snapshots the camera and starts the
        GPU future.  The shared controller simultaneously emits a safe coarse
        pre-descent, or the one-time cone orientation.  A later call attaches
        the bounded result before any contact waypoint can be produced.  All
        crooked TableCam candidates are prompted against the same flat wrist
        snapshot so one cone orientation can be retained for the full batch.
        """
        states = {
            "refine_spare": "table_cap", "await_refine_spare": "table_cap",
            "refine_missing": "missing_cap", "await_refine_missing": "missing_cap",
            "refine_crooked": "crooked_cap", "await_refine_crooked": "crooked_cap",
        }
        class_name = states.get(self.controller.state)
        if class_name is None:
            return
        if wrist_image is None:
            raise RuntimeError("wrist refinement requested but obs['images']['wrist'] is absent")
        expected_item = (self.controller.pending["spare"] if class_name == "table_cap"
                         else self.controller.pending["target"])
        if self.wrist_request is None:
            snapshot_joints = list(observed["joints"])
            rgb = np.ascontiguousarray(wrist_image[:, :, :3]).copy()
            batch_items = [expected_item]
            if class_name == "crooked_cap":
                for track in self.controller.defect_tracks or []:
                    item = track.get("last_detection")
                    if (track.get("class_name") != "crooked_cap" or
                            track.get("attempts", 0) >= self.controller.max_attempts or
                            not isinstance(item, dict)):
                        continue
                    xy = np.asarray(item["base_pos_m"], dtype=float)[:2]
                    if any(np.linalg.norm(
                            xy - np.asarray(known["base_pos_m"], dtype=float)[:2]) < 0.003
                           for known in batch_items):
                        continue
                    batch_items.append(item)
            requests = []
            depth_copy = (None if wrist_depth is None else
                          np.ascontiguousarray(wrist_depth).copy())
            for item in batch_items:
                expected = np.asarray(item["base_pos_m"], dtype=float)
                if class_name == "missing_cap":
                    plane_z = self.vial_rim_z
                elif class_name == "crooked_cap":
                    observed_top = item.get("surface_top_base_z_m")
                    plane_z = (float(observed_top)
                               if observed_top is not None and np.isfinite(observed_top)
                               else self.vial_rim_z + 0.055)
                else:
                    plane_z = float(expected[2])
                expected_on_plane = expected.copy()
                expected_on_plane[2] = plane_z
                expected_pixel = project_base_to_wrist(
                    expected_on_plane, snapshot_joints, self.IMAGE_SIZE)
                index = self.wrist_frame_count
                self.wrist_frame_count += 1
                name = f"{index:04d}_{class_name}.png"
                cv2.imwrite(str(self.trace_dir / "wristcam_raw" / name),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                requests.append({
                    "future": self.wrist_pool.submit(
                        self.wrist_vlm.predict, rgb, class_name, expected_pixel),
                    "class_name": class_name, "image": name,
                    "joints": snapshot_joints,
                    "expected": expected_on_plane.tolist(),
                    "expected_pixel": expected_pixel.tolist(),
                    "plane_z_m": plane_z, "index": index,
                    "depth": depth_copy,
                })
            self.wrist_request = {
                "class_name": class_name, "requests": requests}
        if self.wrist_request["class_name"] != class_name:
            raise RuntimeError(
                f"stale wrist request for {self.wrist_request['class_name']} "
                f"while awaiting {class_name}")
        requests = self.wrist_request["requests"]
        if not all(request["future"].done() for request in requests):
            return
        refinements = []
        for request in requests:
            result = request["future"].result()
            public_request = {key: value for key, value in request.items()
                              if key not in ("future", "depth")}
            path = (self.trace_dir / "wristcam_results" /
                    f"{request['index']:04d}_{class_name}.json")
            annotated_relative = (Path("wristcam_annotated") /
                                  f"{request['index']:04d}_{class_name}.png")
            annotated_path = self.trace_dir / annotated_relative
            raw_path = self.trace_dir / "wristcam_raw" / request["image"]
            raw_bgr = cv2.imread(str(raw_path))
            annotation_written = False
            if raw_bgr is not None:
                annotated = render_wrist_refinement_annotation(
                    raw_bgr, class_name, result, request["expected_pixel"])
                annotation_written = cv2.imwrite(str(annotated_path), annotated)
            # Persist raw VLM evidence before plane fitting/backprojection so
            # an out-of-bounds correction still leaves an inspectable image.
            path.write_text(json.dumps({
                "request": public_request, "vlm": result,
                "annotated_image": (str(annotated_relative)
                                    if annotation_written else None),
            }, indent=2, default=self._json) + "\n")
            plane_z = request["plane_z_m"]
            support_plane = None
            support_plane_fallback = None
            if class_name == "table_cap" and request["depth"] is not None:
                try:
                    depth_cam = self.cameras["wrist_depth"]
                    depth = np.asarray(request["depth"])
                    if tuple(depth.shape[:2][::-1]) != tuple(depth_cam["size"]):
                        raise ValueError(
                            f"wrist depth has {depth.shape[:2]}, "
                            f"expected {depth_cam['size'][::-1]}")
                    points = depth_to_base(
                        depth_cam, depth, stride=2, joints=request["joints"])
                    flange = flange_pose(request["joints"])
                    points_flange = ((points - flange[:3, 3]) @ flange[:3, :3])
                    colour_pixels = project(self.cameras["wrist"], points_flange)
                    native = np.asarray(self.cameras["wrist"]["size"], dtype=float)
                    served = np.asarray(self.IMAGE_SIZE, dtype=float)
                    colour_pixels *= served / native
                    support_plane = fit_local_support_plane(
                        points, colour_pixels, result["bbox_xyxy"],
                        np.asarray(request["expected"], dtype=float)[:2],
                        self.work_surface_z)
                    plane_z = (support_plane["support_z_base_m"] +
                               self.work_surface_z_offset_m + SPEC.plug_depth +
                               SPEC.flange_thickness)
                except ValueError as error:
                    support_plane_fallback = str(error)
            refinement = refine_from_vlm_box(
                result["bbox_xyxy"], request["expected"], plane_z,
                request["joints"], self.IMAGE_SIZE, self.wrist_max_correction_m)
            refinement.update({
                "class_name": class_name, "vlm_latency_s": result["latency_s"],
                "expected_pixel": request["expected_pixel"],
                "image": request["image"], "support_plane": support_plane,
                "support_plane_fallback": support_plane_fallback,
            })
            if raw_bgr is not None:
                annotated = render_wrist_refinement_annotation(
                    raw_bgr, class_name, result, request["expected_pixel"],
                    refinement)
                annotation_written = cv2.imwrite(str(annotated_path), annotated)
            if annotation_written:
                refinement["annotated_image"] = str(annotated_relative)
            path.write_text(json.dumps({
                "request": public_request,
                "vlm": result, "refinement": refinement,
            }, indent=2, default=self._json) + "\n")
            self.events.append({"event": "wrist_vlm_refinement", **refinement})
            refinements.append(refinement)
        if class_name == "crooked_cap":
            observed["wrist_refinement_batch"] = refinements
        else:
            observed["wrist_refinement"] = refinements[0]
        self.wrist_request = None

    def _full_act(self, obs, image, stamp, fresh):
        """Pick a spare for a missing vial or press a crooked cap vertically.

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
        current = self._tcp(observed)
        # Execute a previously planned repair without rerunning YOLO between
        # ordinary waypoints. Every route ends at the shared inspection pose.
        if self.controller.queue:
            action = self.controller.act(observed, [])
            self.events[:] = self.controller.events[-20:]
            self._record_snapshot(observed, image, [], action)
            return action
        wrist_gate = (self.controller.state.startswith("refine_") or
                      self.controller.state.startswith("await_refine_"))
        if wrist_gate:
            self._attach_wrist_refinement(
                observed, obs.get("images", {}).get("wrist"),
                obs.get("depth", {}).get("wrist"))
            action = self.controller.act(observed, [])
            self.events[:] = self.controller.events[-20:] + self.events[-1:]
            self._record_snapshot(observed, image, [], action)
            return action
        if self.controller.continuing_cone_batch():
            # The first wrist image already supplied bounded corrections for
            # every crooked target. Continue directly without an occluded
            # TableCam decision; the final route returns to inspection and
            # verifies the complete batch from one fresh image.
            action = self.controller.act(observed, [])
            self.events[:] = self.controller.events[-20:]
            self._record_snapshot(observed, image, [], action)
            return action
        # Initial observation and every decision gate occur only from v47's
        # cable-safe, TableCam-clear starting pose.
        if not self._at_inspection(observed):
            action = self._inspection_action(observed)
            self.events.append({"event": "move_to_inspection_pose"})
            self._record_snapshot(observed, image, [], action)
            return action
        # Calibrate one parallel work-surface Z plane from the first clear
        # TableCam depth frame in each rollout. Freeze it for the complete
        # missing-then-crooked workflow so contact heights cannot jitter between
        # actions. Camera intrinsics/extrinsics and RGB processing are unchanged.
        self._fit_run_height_planes(observed)
        detections = self._snapshot_detections(image, stamp)
        detections = self._attach_table_depth_observations(
            detections, observed.get("depth", {}).get("table"))
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

    @staticmethod
    def _shortest_angle_delta_deg(target, current):
        return (np.asarray(target, float) - np.asarray(current, float) + 180.0) % 360.0 - 180.0

    def _goal_reached(self, obs, goal):
        current = self._tcp(obs)
        position_ok = np.linalg.norm(current[:3] - goal[:3]) <= ARRIVAL_TRANSLATION_MM
        relative = wpr_to_rotation(current[3:6]).T @ wpr_to_rotation(goal[3:6])
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        joints = np.asarray(obs.get("joints", []), dtype=float)
        jaw_ok = (joints.size <= 6 or not np.isfinite(joints[6]) or
                  (joints[6] >= 0.5) == (goal[6] >= 0.5))
        return position_ok and angle <= ARRIVAL_ORIENTATION_DEG and jaw_ok

    def _motion_chunk(self, obs, goal):
        current = self._tcp(obs)
        delta = goal[:3] - current[:3]
        distance = float(np.linalg.norm(delta))
        angular = self._shortest_angle_delta_deg(goal[3:6], current[3:6])
        largest = float(np.max(np.abs(angular)))
        rows = max(1, int(np.ceil(distance / MAX_ROW_TRANSLATION_MM)),
                   int(np.ceil(largest / MAX_ROW_WPR_DEG)))
        chunk = []
        for fraction in np.linspace(1.0 / rows, 1.0, rows):
            xyz = current[:3] + fraction * delta
            wpr = current[3:6] + fraction * angular
            chunk.append([*map(float, xyz), *map(float, wpr), float(goal[6])])
        # The real FANUC follows streamed targets slightly behind the nominal
        # 10 Hz sequence. Keep the final target queued for 0.5 s so the arm can
        # settle before the next observation is judged against the waypoint.
        # Without this tail, the policy repeatedly issued short correction
        # chunks from stale 5-37 mm-behind poses, producing stop/start motion.
        chunk.extend([chunk[-1].copy() for _ in range(MOTION_SETTLE_ROWS)])
        return chunk

    def _dispatch(self, obs, desired):
        self.motion_goal = np.asarray(desired, dtype=float)
        chunk = self._motion_chunk(obs, self.motion_goal)
        self._write_trace({"t": obs.get("t"), "controller_state": self.controller.state if self.controller else self.state,
                           "event": "motion_chunk", "tcp_pose_before": obs.get("tcp_pose"),
                           "motion_goal": self.motion_goal.tolist(), "row_count": len(chunk),
                           "action_tcp": chunk[-1]})
        return chunk

    def _decide_action(self, obs):
        self._validate_start_orientation(obs)
        image = obs.get("images", {}).get("table")
        stamp = obs.get("t")
        fresh = stamp is None or self.last_observation_time is None or stamp > self.last_observation_time

        if self.full_remediation:
            return self._full_act(obs, image, stamp, fresh)

        if self.state == "inspect":
            if not self._at_inspection(obs):
                action = self._command(INSPECTION_UF0_MM)
                if image is not None:
                    self._record_snapshot(obs, image, [], action)
                return action
            if image is None or not fresh:
                return self._hold(obs)
            self.last_observation_time = stamp
            self.target = self._choose_table_cap(image, stamp)
            if self.target is None:
                action = self._hold(obs)
                self._record_snapshot(obs, image, self.hover_detections, action)
                return action
            self.state = "raise"

        current = self._tcp(obs)
        cap_base = np.asarray(self.target["base_pos_m"], dtype=float)
        hover_uf0 = base_to_uf0(cap_base + np.array([0.0, 0.0, HOVER_ABOVE_CAP_M]))

        if self.state == "raise":
            self.state = "translate"
            action = self._command([current[0], current[1], max(current[2], TRANSIT_UF0_Z_MM)])
            self._record_snapshot(obs, image, self.hover_detections, action)
            return action
        if self.state == "translate":
            self.state = "hover"
            action = self._command([hover_uf0[0], hover_uf0[1], TRANSIT_UF0_Z_MM])
            self._record_snapshot(obs, image, self.hover_detections, action)
            return action
        if self.state == "hover":
            self.state = "hold"
            self.events.append({"event": "hovered_above_table_cap",
                                "pixel": self.target["pixel"],
                                "base_pos_m": cap_base.tolist(),
                                "hover_uf0_mm": hover_uf0.tolist()})
            action = self._command(hover_uf0)
            self._record_snapshot(obs, image, self.hover_detections, action)
            return action
        action = self._hold(obs)
        self._record_snapshot(obs, image, self.hover_detections, action)
        return action

    def act(self, obs):
        self._validate_start_orientation(obs)
        if self.motion_goal is not None:
            if not self._goal_reached(obs, self.motion_goal):
                chunk = self._motion_chunk(obs, self.motion_goal)
                self._write_trace({"t": obs.get("t"),
                                   "controller_state": self.controller.state if self.controller else self.state,
                                   "event": "motion_correction_chunk", "tcp_pose_before": obs.get("tcp_pose"),
                                   "motion_goal": self.motion_goal.tolist(), "row_count": len(chunk),
                                   "action_tcp": chunk[-1]})
                return chunk
            self.motion_goal = None
        return self._dispatch(obs, self._decide_action(obs))
