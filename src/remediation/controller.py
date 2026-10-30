"""Observation-driven remediation state machine, independent of Isaac and HTTP.

Contact actions require an explicitly accepted tool-offset contract and qualified
RGB-D localization.  No simulator pose or stored tray coordinate enters this module.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.remediation.geometry import (SPEC, base_to_uf0, inspection_base_position,
                                      rim_base_z, transit_base_z, uf0_to_base)


DEFECT_TRACK_GATE_M = 0.100

# The cone is on the outside of the negative-Y finger.  These values are from
# data/cone_gripper.step with the jaws at the cap-hold position used for a
# press.  The offset is expressed in the controller TCP frame, from the closed
# fingertip TCP to the cone nose.  At clearance the controller first rolls the
# straight vertical wrist clockwise, then applies the reach-limited wrist tilt.
# This keeps the arm in its conventional plane while placing the negative-Y
# (TableCam-left) jaw below the positive-Y jaw with its cone axis vertical.
DEFAULT_CONE_TIP_FROM_TCP_M = (0.0, -0.0360660669237695, -0.0107523335762304)
DEFAULT_CONE_WRIST_WPR_DEG = (-180.0, 0.0, -90.0)
# FANUC composes W/P/R as Rz(R) Ry(P) Rx(W). After the clockwise R=-90 wrist
# roll, decrease W through its wrapped equivalent from -180 to +135 so the
# wrist leans backward and lowers the cone-side jaw. Changing P instead bends
# across the arm and can sweep the wrist camera/cables through the work. Keep
# the physical cone axis vertical with the nominal 45-degree wrist lean.
# Reach adaptation changes the global heading first; reducing this lean is only
# a last resort after the complete heading sweep has failed.
DEFAULT_CONE_PRESS_WPR_DEG = (135.0, 0.0, -90.0)
# The rigid-cap PhysX proxy can remain balanced on the clear vial rim even after
# the useful impact.  Continue the TPU cone nose through the documented rim
# plane far enough that the real half-inch cap cannot stay hooked on the lip.
DEFAULT_CONE_PRESS_BELOW_RIM_M = 0.003
DEFAULT_CONE_PRESS_STROKE_M = 0.020
DEFAULT_CONE_TABLE_PICK_OFFSET_M = -0.006

# After a missing cap is released, move a few millimetres along the jaw spread
# direction and retreat diagonally toward the opposite side. This clears either
# flange lip from the compliant finger pockets without closing on the cap a
# second time.
DEFAULT_MISSING_RELEASE_SIDE_STEP_M = 0.001
DEFAULT_MISSING_RELEASE_EXIT_STEP_M = 0.002


# The shared policy boundary.  Adapters may own richer observations, but only
# these physical-cell fields may enter a controller decision.  In particular,
# no USD prim, PhysX pose/contact/load, generated label or true cap axis crosses
# this boundary.
ALLOWED_OBSERVATION_FIELDS = frozenset({
    "tcp_pose", "joints", "grasp_verified", "timestamp", "wrist_refinement",
    "wrist_refinement_batch",
})
ALLOWED_DETECTION_FIELDS = frozenset({
    "class_name", "confidence", "base_pos_m", "bbox_xyxy", "pixel",
    "depth_available", "surface_top_base_z_m", "cap_axis_base", "contact_qualified",
    "feature_base_pos_m", "cap_high_point_base_m", "localization_note", "timestamp",
})
FORBIDDEN_ORACLE_FIELDS = frozenset({
    "prim", "prim_path", "world_pose", "true_pose", "true_axis",
    "physx_pose", "contact_force", "contact_pairs", "scene_label",
    "ground_truth", "generated_state", "slot_truth",
})


def _reject_oracle(mapping, where):
    present = sorted(FORBIDDEN_ORACLE_FIELDS.intersection(mapping))
    if present:
        raise ValueError(f"Simulator oracle fields may not enter {where}: {present}")


def policy_observation(obs, detections):
    """Return the only observation data the shared controller is allowed to use."""
    _reject_oracle(obs, "policy observations")
    batch = obs.get("wrist_refinement_batch")
    if batch is not None:
        if not isinstance(batch, (list, tuple)):
            raise ValueError("wrist refinement batch must be a list")
        for index, item in enumerate(batch):
            if not isinstance(item, dict):
                raise ValueError(f"wrist refinement batch item {index} must be a mapping")
            _reject_oracle(item, f"wrist refinement batch item {index}")
    clean_obs = {key: obs[key] for key in ALLOWED_OBSERVATION_FIELDS if key in obs}
    clean_detections = []
    for index, detection in enumerate(detections):
        _reject_oracle(detection, f"detection {index}")
        clean_detections.append({key: detection[key] for key in ALLOWED_DETECTION_FIELDS
                                 if key in detection})
    return clean_obs, clean_detections


@dataclass(frozen=True)
class ContactTool:
    """Signed Z offsets from an observed feature to pick/place/press TCP targets."""
    pick_offset_m: float
    place_offset_m: float
    press_offset_m: float
    validated: bool = False


def checked_action(base_position, closed):
    # Physical-cell cable-safe vertical tool orientation.
    return checked_pose_action(base_position, (-180.0, 0.0, 0.0), closed)


def checked_pose_action(base_position, wpr_deg, closed):
    p = base_to_uf0(base_position)
    limits = np.array([[150, 650], [-450, 450], [-334, 300]])
    if not np.isfinite(p).all() or np.any(p < limits[:, 0]) or np.any(p > limits[:, 1]):
        raise ValueError(
            "Target outside calibrated workspace; refusing to clip it: "
            f"UF0 XYZ={np.asarray(p, dtype=float).round(3).tolist()} mm")
    wpr = np.asarray(wpr_deg, dtype=float)
    if wpr.shape != (3,) or not np.isfinite(wpr).all():
        raise ValueError("TCP orientation must contain three finite W/P/R angles")
    return [*map(float, p), *map(float, wpr), float(closed)]


def axis_to_wpr(axis_base):
    """Fanuc W/P/R whose tool +Z points down the observed cap axis.

    FANUC applies W/P/R as Rz(R) Ry(P) Rx(W); the physical-cell normal
    vertical tool pose is (-180, 0, 0) degrees.
    """
    axis = np.asarray(axis_base, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("crooked cap axis must be a finite nonzero 3-vector")
    axis /= np.linalg.norm(axis)
    if axis[2] <= 0:
        axis = -axis
    z = -axis
    reference = np.array([0.0, 0.0, 1.0])
    y = np.cross(reference, axis)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, -1.0, 0.0])
    else:
        y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, y, z])
    # R = Rz(r) Ry(p) Rx(w), including the gimbal-lock-safe branch.
    p = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    cp = np.cos(p)
    if abs(cp) > 1e-8:
        w = np.arctan2(R[2, 1], R[2, 2])
        r = np.arctan2(R[1, 0], R[0, 0])
    elif p > 0:
        w = np.arctan2(R[0, 1], R[0, 2])
        r = 0.0
    else:
        w = np.arctan2(-R[0, 1], -R[0, 2])
        r = 0.0
    degrees = (np.degrees([w, p, r]) + 180.0) % 360.0 - 180.0
    return tuple(degrees)


def wpr_to_rotation(wpr_deg):
    """FANUC W/P/R matrix, applied as Rz(R) Ry(P) Rx(W)."""
    w, p, r = np.radians(np.asarray(wpr_deg, dtype=float))
    cw, sw = np.cos(w), np.sin(w)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def command_rotation_to_j6(rotation):
    """Map controller TCP-frame orientation to the ER-4iA J6 frame.

    The controller's local +Z is the approach axis, while this robot model's
    physical tool axis is J6 +X; both use local +Y for finger spread.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    return np.column_stack((rotation[:, 2], rotation[:, 1], -rotation[:, 0]))


def j6_rotation_to_command(rotation):
    """Inverse of command_rotation_to_j6 for Cartesian interpolation."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    return np.column_stack((-rotation[:, 2], rotation[:, 1], rotation[:, 0]))


class RemediationController:
    def __init__(self, tool, max_attempts=1, min_confidence=0.70,
                 crooked_shoulder_depth_m=0.0326838, crooked_tap_depth_m=0.005,
                 crooked_action="shoulder", crooked_wiggle_radius_m=0.00075,
                 crooked_wiggle_cycles=1, pickup_wiggle_radius_m=0.0,
                 pickup_wiggle_cycles=0, vial_rim_z_m=None,
                 cone_tip_from_tcp_m=DEFAULT_CONE_TIP_FROM_TCP_M,
                 cone_wrist_wpr_deg=DEFAULT_CONE_WRIST_WPR_DEG,
                 cone_press_wpr_deg=DEFAULT_CONE_PRESS_WPR_DEG,
                 cone_wpr_selector=None,
                 cone_press_below_rim_m=DEFAULT_CONE_PRESS_BELOW_RIM_M,
                 cone_press_stroke_m=DEFAULT_CONE_PRESS_STROKE_M,
                 table_cap_xy_offset_m=(0.0, 0.0),
                 missing_cap_xy_offset_m=(0.0, 0.0),
                 crooked_cap_xy_offset_m=(0.0, 0.0),
                 missing_release_side_step_m=DEFAULT_MISSING_RELEASE_SIDE_STEP_M,
                 missing_release_exit_step_m=DEFAULT_MISSING_RELEASE_EXIT_STEP_M,
                 wrist_refinement=False,
                 wrist_view_distance_m=0.300,
                 wrist_crooked_view_distance_m=0.250,
                 wrist_table_cap_xy_offset_m=(0.0, 0.0),
                 wrist_missing_cap_xy_offset_m=(0.0, 0.0),
                 wrist_crooked_cap_xy_offset_m=(0.0, 0.0)):
        if not tool.validated:
            raise ValueError("Contact tool offsets have not been explicitly accepted")
        self.tool = tool
        self.max_attempts = max_attempts
        self.min_confidence = float(min_confidence)
        self.vial_rim_z_m = rim_base_z() if vial_rim_z_m is None else float(vial_rim_z_m)
        if not 0.0 < self.min_confidence < 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.crooked_shoulder_depth_m = float(crooked_shoulder_depth_m)
        if not 0.020 <= self.crooked_shoulder_depth_m <= 0.040:
            raise ValueError("crooked shoulder depth must be between 0.020 and 0.040 m")
        self.crooked_tap_depth_m = float(crooked_tap_depth_m)
        if not 0.0 < self.crooked_tap_depth_m <= 0.010:
            raise ValueError("crooked tap depth must be greater than 0 and at most 0.010 m")
        if crooked_action not in ("shoulder", "shoulder-close", "closed-tip", "regrip", "cone-press"):
            raise ValueError(
                "crooked_action must be shoulder, shoulder-close, closed-tip, regrip, or cone-press")
        self.crooked_action = crooked_action
        self.crooked_wiggle_radius_m = float(crooked_wiggle_radius_m)
        self.crooked_wiggle_cycles = int(crooked_wiggle_cycles)
        if not 0.0 <= self.crooked_wiggle_radius_m <= 0.002:
            raise ValueError("crooked wiggle radius must be between 0 and 0.002 m")
        if not 0 <= self.crooked_wiggle_cycles <= 3:
            raise ValueError("crooked wiggle cycles must be between 0 and 3")
        self.pickup_wiggle_radius_m = float(pickup_wiggle_radius_m)
        self.pickup_wiggle_cycles = int(pickup_wiggle_cycles)
        if not 0.0 <= self.pickup_wiggle_radius_m <= 0.005:
            raise ValueError("pickup wiggle radius must be between 0 and 0.005 m")
        if not 0 <= self.pickup_wiggle_cycles <= 2:
            raise ValueError("pickup wiggle cycles must be between 0 and 2")
        self.cone_tip_from_tcp_m = np.asarray(cone_tip_from_tcp_m, dtype=float)
        self.cone_wrist_wpr_deg = tuple(np.asarray(cone_wrist_wpr_deg, dtype=float))
        self.cone_press_wpr_deg = tuple(np.asarray(cone_press_wpr_deg, dtype=float))
        if cone_wpr_selector is not None and not callable(cone_wpr_selector):
            raise ValueError("cone W/P/R selector must be callable")
        self.cone_wpr_selector = cone_wpr_selector
        self.cone_press_below_rim_m = float(cone_press_below_rim_m)
        self.cone_press_stroke_m = float(cone_press_stroke_m)
        self.table_cap_xy_offset_m = self._checked_xy_offset(
            table_cap_xy_offset_m, "table-cap")
        self.missing_cap_xy_offset_m = self._checked_xy_offset(
            missing_cap_xy_offset_m, "missing-cap")
        self.crooked_cap_xy_offset_m = self._checked_xy_offset(
            crooked_cap_xy_offset_m, "crooked-cap")
        self.missing_release_side_step_m = float(missing_release_side_step_m)
        self.missing_release_exit_step_m = float(missing_release_exit_step_m)
        if (not np.isfinite(self.missing_release_side_step_m) or
                not 0.0 <= self.missing_release_side_step_m <= 0.010):
            raise ValueError(
                "missing release side step must be between 0 and 0.010 m")
        if (not np.isfinite(self.missing_release_exit_step_m) or
                not 0.0 <= self.missing_release_exit_step_m <= 0.015):
            raise ValueError(
                "missing release exit step must be between 0 and 0.015 m")
        self.wrist_refinement = bool(wrist_refinement)
        self.wrist_view_distance_m = float(wrist_view_distance_m)
        if not 0.120 <= self.wrist_view_distance_m <= 0.300:
            raise ValueError("wrist view distance must be between 0.120 and 0.300 m")
        self.wrist_crooked_view_distance_m = float(wrist_crooked_view_distance_m)
        if not 0.120 <= self.wrist_crooked_view_distance_m <= 0.300:
            raise ValueError(
                "wrist crooked view distance must be between 0.120 and 0.300 m")
        self.wrist_table_cap_xy_offset_m = self._checked_xy_offset(
            wrist_table_cap_xy_offset_m, "wrist table-cap")
        self.wrist_missing_cap_xy_offset_m = self._checked_xy_offset(
            wrist_missing_cap_xy_offset_m, "wrist missing-cap")
        self.wrist_crooked_cap_xy_offset_m = self._checked_xy_offset(
            wrist_crooked_cap_xy_offset_m, "wrist crooked-cap")
        if (self.cone_tip_from_tcp_m.shape != (3,) or
                not np.isfinite(self.cone_tip_from_tcp_m).all()):
            raise ValueError("cone tip offset must contain three finite coordinates")
        if (len(self.cone_wrist_wpr_deg) != 3 or
                not np.isfinite(self.cone_wrist_wpr_deg).all() or
                len(self.cone_press_wpr_deg) != 3 or
                not np.isfinite(self.cone_press_wpr_deg).all()):
            raise ValueError("cone orientations must contain three finite W/P/R angles")
        if (not np.isfinite(self.cone_press_below_rim_m) or
                not -0.010 <= self.cone_press_below_rim_m <= 0.010):
            raise ValueError("cone press depth must be between -0.010 and 0.010 m")
        if (not np.isfinite(self.cone_press_stroke_m) or
                not 0.005 <= self.cone_press_stroke_m <= 0.040):
            raise ValueError("cone press stroke must be between 0.005 and 0.040 m")
        self.reset()

    @staticmethod
    def _checked_xy_offset(value, name):
        offset = np.asarray(value, dtype=float)
        if (offset.shape != (2,) or not np.isfinite(offset).all() or
                np.any(np.abs(offset) > 0.025)):
            raise ValueError(f"{name} XY offset must contain two finite values within +/-0.025 m")
        return offset

    def _add_inspection(self, closed=False):
        self._add(inspection_base_position(), closed)

    def _add_missing_release(self, target_position, transit_height):
        """Release once, clear both flange lips, then retreat above the tray."""
        p = np.asarray(target_position, dtype=float)
        place = p + [0, 0, self.tool.place_offset_m]
        spread = wpr_to_rotation((-180.0, 0.0, 0.0))[:, 1]
        side = place + spread * self.missing_release_side_step_m
        opposite = (place - spread * self.missing_release_exit_step_m +
                    [0, 0, 0.025])
        self._add(p + [0, 0, 0.025], True)
        self._add(place, True)
        self._add(place, False)
        self._add(side, False)
        self._add(opposite, False)
        self._add([opposite[0], opposite[1], transit_height], False)

    def reset(self):
        self.queue = deque()
        self.pending = None
        self.attempts = {}
        self.defect_tracks = None
        self.events = []
        self.decision_records = []
        self.state = "inspect"
        self.confirmations = 0
        self.verify_frames = 0
        self.cone_oriented = False
        self.active_cone_press_wpr_deg = self.cone_press_wpr_deg
        self.crooked_refinement_cache = []
        self.cone_batch_targets = []

    def continuing_cone_batch(self):
        """True while cached crooked targets remain after the current press."""
        return bool(
            self.crooked_action == "cone-press" and self.cone_oriented and
            any(track.get("class_name") == "crooked_cap" and
                track.get("attempts", 0) < self.max_attempts
                for track in (self.defect_tracks or []))
        )

    def _valid_defect(self, detection):
        """Reject contact targets whose camera geometry is physically impossible."""
        localization_note = detection.get("localization_note", "")
        # Contact targets now come directly from the calibrated TableCam RGB
        # box-centre ray/plane intersection.  Do not require the retired tray-
        # grid refinement, but still reject detections with unknown provenance.
        known_localization = ("RGB box-center ray intersected with the calibrated object plane" in
                              localization_note)
        if localization_note and not known_localization:
            self.events.append({"event": "target_rejected",
                                "reason": "defect does not have calibrated RGB ray/plane localization"})
            return False
        if detection.get("class_name") != "crooked_cap":
            return True
        # Physical cone pressing uses a fixed calibrated 45-degree posture, a
        # wrist-RGB XY correction and the run-level rim plane. Missing or
        # unusable TableCam depth must not veto that workflow. When TableCam
        # *does* provide a finite surface height, however, retain the broad
        # physical-range check: it rejects false crooked-cap boxes on the
        # elevated wrist/gripper before they become wrist-camera requests.
        if (self.crooked_action == "cone-press" and self.wrist_refinement and
                (detection.get("depth_available") is not True or
                 not np.isfinite(float(detection.get(
                     "surface_top_base_z_m", np.nan))))):
            return True
        top = float(detection.get("surface_top_base_z_m", np.nan))
        if not self.vial_rim_z_m - 0.020 <= top <= self.vial_rim_z_m + 0.055:
            self.events.append({"event": "target_rejected",
                                "reason": "crooked RGB-D surface height outside physical range",
                                "surface_top_base_z_m": top})
            return False
        return True

    def _tracked_defects(self, defects):
        """Match camera defects to prior tracks and admit newly observed defects.

        Tracks are controller memory derived only from camera detections.  A cap
        that moves after contact remains the same attempted track instead of
        gaining a new attempt because its quantized XY coordinate changed.  A
        missing-cap track that becomes a crooked-cap detection after placement
        is deliberately a new repair task, so replacement-induced crooked caps
        are handled in the second phase of the workflow.
        """
        if self.defect_tracks is None and defects:
            self.defect_tracks = [{"class_name": d["class_name"],
                                   "anchor_xy": np.asarray(d["base_pos_m"][:2], float),
                                   "last_xy": np.asarray(d["base_pos_m"][:2], float),
                                   "last_detection": dict(d),
                                   "attempts": 0}
                                  for d in defects]
        if self.defect_tracks is None:
            return []
        pairs = []
        for ti, track in enumerate(self.defect_tracks):
            for di, detection in enumerate(defects):
                if detection["class_name"] != track["class_name"]:
                    continue
                distance = float(np.linalg.norm(
                    np.asarray(detection["base_pos_m"][:2], float) - track["last_xy"]))
                if distance <= DEFECT_TRACK_GATE_M:
                    pairs.append((distance, ti, di))
        assigned_tracks, assigned_detections, matches = set(), set(), []
        for distance, ti, di in sorted(pairs):
            if ti in assigned_tracks or di in assigned_detections:
                continue
            assigned_tracks.add(ti); assigned_detections.add(di)
            track = self.defect_tracks[ti]
            track["last_xy"] = np.asarray(defects[di]["base_pos_m"][:2], float)
            track["last_detection"] = dict(defects[di])
            matches.append((track, defects[di], distance))
        # A cone batch is established from one unobstructed TableCam view.
        # Retain every qualified, not-yet-attempted crooked target from that
        # view: after the first press the leaned arm can hide a farther vial
        # from TableCam even though its camera-derived XYZ/height/axis remain
        # valid.  This is perception memory, not a tray/G-slot lookup.
        if self.crooked_action == "cone-press":
            for ti, track in enumerate(self.defect_tracks):
                remembered = track.get("last_detection")
                if (ti not in assigned_tracks and track["attempts"] < self.max_attempts
                        and track["class_name"] == "crooked_cap"
                        and isinstance(remembered, dict)):
                    assigned_tracks.add(ti)
                    matches.append((track, dict(remembered), float("inf")))
        for di, detection in enumerate(defects):
            if di in assigned_detections:
                continue
            xy = np.asarray(detection["base_pos_m"][:2], float)
            # A duplicate of an already attempted same-class target is not a
            # new task.  Class transitions at the same aperture are new tasks.
            if any(ti not in assigned_tracks and
                   track["class_name"] == detection["class_name"] and
                   np.linalg.norm(xy - track["last_xy"]) <= DEFECT_TRACK_GATE_M
                   for ti, track in enumerate(self.defect_tracks)):
                continue
            track = {"class_name": detection["class_name"],
                     "anchor_xy": xy.copy(), "last_xy": xy.copy(), "attempts": 0}
            track["last_detection"] = dict(detection)
            self.defect_tracks.append(track)
            matches.append((track, detection, 0.0))
        matches.sort(key=lambda match: match[1]["class_name"] != "missing_cap")
        return matches

    def hold(self, obs):
        tcp = np.asarray(obs.get("tcp_pose"), dtype=float)
        if tcp.shape != (6,) or not np.isfinite(tcp).all():
            raise ValueError("Current TCP pose required to hold safely")
        joints = np.asarray(obs.get("joints", []), dtype=float)
        closed = float(joints[6]) if joints.size > 6 and np.isfinite(joints[6]) else 0.0
        return [*map(float, tcp), closed]

    def _add(self, position, closed):
        self.queue.append(checked_action(position, closed))

    def _add_pose(self, position, wpr, closed):
        self.queue.append(checked_pose_action(position, wpr, closed))

    def _cone_tcp_for_tip(self, cone_tip_position, wpr=None):
        """Fingertip TCP pose that places the CAD cone nose at a base point."""
        rotation = wpr_to_rotation(self.cone_press_wpr_deg if wpr is None else wpr)
        return np.asarray(cone_tip_position, dtype=float) - rotation @ self.cone_tip_from_tcp_m

    def _cone_wpr_for_target(self, target_position, obs):
        if self.cone_wpr_selector is None:
            return self.cone_press_wpr_deg
        selected = tuple(np.asarray(self.cone_wpr_selector(
            target_position, obs.get("joints")), dtype=float))
        if len(selected) != 3 or not np.isfinite(selected).all():
            raise ValueError("cone W/P/R selector returned an invalid orientation")
        return selected

    def _route(self, obs, target, spare=None):
        start = len(self.queue)
        try:
            self._route_checked(obs, target, spare)
        except ValueError:
            while len(self.queue) > start:
                self.queue.pop()
            raise

    def _wrist_view(self, position, obs, centre_camera=False, wpr=None,
                    use_measured_j6=True, distance_m=None):
        from src.remediation.wrist_refinement import (fingertip_tcp_from_joints,
                                                      flange_rotation,
                                                      wrist_view_tcp)
        joints = np.asarray(obs.get("joints", ()), dtype=float)
        j6_rotation = None
        if use_measured_j6 and joints.size >= 6 and np.isfinite(joints[:6]).all():
            # Unit tests and degraded physical observations can contain a
            # stale/default joint vector alongside a valid TCP.  Only use the
            # measured J6 branch when the two independent observations agree.
            observed_tcp = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
            if np.linalg.norm(fingertip_tcp_from_joints(joints) - observed_tcp) <= 0.030:
                j6_rotation = flange_rotation(joints)
        command_wpr = (-180.0, 0.0, 0.0) if wpr is None else wpr
        distance = (self.wrist_view_distance_m if distance_m is None
                    else float(distance_m))
        view = wrist_view_tcp(position, wpr_to_rotation(command_wpr),
                              distance, j6_rotation=j6_rotation,
                              centre_camera=centre_camera)
        # The camera hand-eye offset can place its ideal centred TCP outside
        # the calibrated robot envelope even though the object itself is
        # reachable (notably the near-X row).  Clamp only this non-contact
        # camera staging pose.  Projection of the unchanged target XYZ then
        # supplies the correct off-centre VLM prompt and backprojection; all
        # pickup, placement, and press poses retain strict rejection in
        # checked_pose_action.
        view[:2] = np.clip(view[:2], [0.155, -0.445], [0.645, 0.445])
        return view

    def _remember_crooked_refinement_batch(self, batch):
        if batch is None:
            return
        for index, item in enumerate(batch):
            if not isinstance(item, dict) or item.get("class_name") != "crooked_cap":
                raise ValueError(
                    f"wrist refinement batch item {index} is not a crooked_cap")
            position = np.asarray(item.get("base_pos_m"), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(
                    f"wrist refinement batch item {index} must contain finite base coordinates")
            if not any(np.linalg.norm(position[:2] - np.asarray(cached["base_pos_m"])[:2])
                       < 0.003 for cached in self.crooked_refinement_cache):
                self.crooked_refinement_cache.append(dict(item))

    def _consume_crooked_refinement(self, expected):
        if not self.crooked_refinement_cache:
            return None
        expected = np.asarray(expected, dtype=float)
        distances = [np.linalg.norm(np.asarray(item["base_pos_m"], dtype=float)[:2]
                                    - expected[:2])
                     for item in self.crooked_refinement_cache]
        index = int(np.argmin(distances))
        return np.asarray(self.crooked_refinement_cache.pop(index)["base_pos_m"], dtype=float)

    def _refinement(self, obs, class_name, expected):
        if class_name == "crooked_cap":
            self._remember_crooked_refinement_batch(obs.get("wrist_refinement_batch"))
            batched = self._consume_crooked_refinement(expected)
            if batched is not None:
                return batched
        item = obs.get("wrist_refinement")
        if item is None:
            return None
        if not isinstance(item, dict) or item.get("class_name") != class_name:
            raise ValueError(f"expected a {class_name} wrist refinement")
        position = np.asarray(item.get("base_pos_m"), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("wrist refinement must contain three finite base coordinates")
        return position

    def _route_checked(self, obs, target, spare=None, wrist_refined=False):
        current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
        height = max(transit_base_z(), current[2])
        raw_target_pos = np.asarray(target["base_pos_m"], dtype=float)
        if (self.wrist_refinement and not wrist_refined and spare is None and
                target.get("class_name") == "crooked_cap"):
            cached = self._consume_crooked_refinement(raw_target_pos)
            if cached is not None:
                updated = dict(target)
                updated["base_pos_m"] = cached.tolist()
                updated["feature_base_pos_m"] = cached.tolist()
                updated["localization_note"] = (
                    str(updated.get("localization_note", "")) +
                    "; cached wrist LocateAnything batch refinement")
                return self._route_checked(
                    obs, updated, spare, wrist_refined=True)
        target_pos = raw_target_pos.copy()
        crooked_xy_offset = (self.wrist_crooked_cap_xy_offset_m
                             if wrist_refined else self.crooked_cap_xy_offset_m)
        if target.get("class_name") == "crooked_cap":
            target_pos = target_pos + np.r_[crooked_xy_offset, 0.0]
        cone_batch = spare is None and self.crooked_action == "cone-press"
        if self.wrist_refinement and not wrist_refined:
            # Stop at a target-centred, calibrated wrist-camera view.  The
            # adapter starts LocateAnything here.  On the next decision the
            # arm can make a safe coarse pre-descent (or stage cone posture)
            # while inference runs, but no contact waypoint is emitted until
            # a finite wrist result has arrived.
            self.pending = {"target": target, "spare": spare, "height": height}
            rolled_cone_view = cone_batch and self.cone_oriented
            view_wpr = self.cone_wrist_wpr_deg if rolled_cone_view else None
            view_closed = bool(rolled_cone_view)
            add_view = (lambda point: self._add_pose(point, view_wpr, view_closed)) \
                if rolled_cone_view else (lambda point: self._add(point, False))
            add_view([current[0], current[1], height])
            expected = (np.asarray(spare["base_pos_m"], dtype=float)
                        if spare is not None else raw_target_pos)
            if cone_batch:
                # One flat wrist image serves the complete crooked batch.
                # Centre it on the perception-derived target cloud so every
                # TableCam projection remains inside the calibrated 16:9 FOV;
                # no tray/slot coordinate enters this calculation.
                batch_points = [raw_target_pos]
                for track in self.defect_tracks or []:
                    remembered = track.get("last_detection")
                    if (track.get("class_name") != "crooked_cap" or
                            track.get("attempts", 0) >= self.max_attempts or
                            not isinstance(remembered, dict)):
                        continue
                    point = np.asarray(remembered.get("base_pos_m"), dtype=float)
                    if (point.shape != (3,) or not np.isfinite(point).all() or
                            any(np.linalg.norm(point[:2] - known[:2]) < 0.003
                                for known in batch_points)):
                        continue
                    batch_points.append(point)
                expected = np.mean(np.asarray(batch_points), axis=0)
            view = self._wrist_view(
                expected, obs, wpr=view_wpr,
                use_measured_j6=not rolled_cone_view,
                distance_m=(self.wrist_crooked_view_distance_m
                            if cone_batch else None))
            # The camera is about 130 mm above the fingertip, so a useful
            # close view can put the TCP below tray transit height.  Translate
            # at clearance first and make only the final camera approach
            # vertical; never take a low diagonal shortcut across the tray.
            add_view([view[0], view[1], height])
            add_view(view)
            if rolled_cone_view:
                # The cable-safe J6 quarter-turn remains in place.  Only the
                # target-dependent lean is removed so the wrist camera is
                # parallel to the work plane for the next local correction.
                self.active_cone_press_wpr_deg = self.cone_wrist_wpr_deg
            self.state = "refine_spare" if spare is not None else "refine_crooked"
            return
        if not cone_batch:
            self._add([current[0], current[1], height], False)
        if spare is None:
            raw_surface_top = target.get("surface_top_base_z_m")
            surface_top = (float(raw_surface_top) if raw_surface_top is not None
                           else float("nan"))
            if not np.isfinite(surface_top):
                if (self.crooked_action == "cone-press" and wrist_refined):
                    # Conservative clearance over the entire supported physical
                    # lean range. Contact Z itself comes from the fitted rim.
                    surface_top = self.vial_rim_z_m + 0.055
                else:
                    raise ValueError("crooked cap requires an RGB-D surface-top height")
            if not cone_batch:
                self._add([target_pos[0], target_pos[1], height], False)
            if self.crooked_action in ("shoulder", "shoulder-close"):
                # Open fingers surround the vial; the CAD-measured internal
                # shoulder taps from the observed cap-surface height.
                contact = np.array([target_pos[0], target_pos[1],
                                    surface_top - self.crooked_shoulder_depth_m])
                self._add(contact + [0, 0, 0.025], False)
                press_closed = self.crooked_action == "shoulder-close"
                if press_closed:
                    # Close with the internal shoulder still clear of the cap.
                    # Opening at the bottom releases any flange captured by the
                    # pockets before the vertical retreat.
                    preclose = contact + [0, 0, self.tool.press_offset_m + 0.003]
                    self._add(preclose, False)
                    self._add(preclose, True)
                # The physical cap balances at four rim contacts. Descend in
                # bounded increments and use a tiny square orbit to break that
                # static-friction balance without estimating the cap axis.
                for fraction in (1.0 / 3.0, 2.0 / 3.0, 1.0):
                    low = contact + [0, 0, self.tool.press_offset_m
                                     - fraction * self.crooked_tap_depth_m]
                    self._add(low, press_closed)
                    radius = self.crooked_wiggle_radius_m
                    for _ in range(self.crooked_wiggle_cycles):
                        for dx, dy in ((radius, 0), (0, radius), (-radius, 0),
                                       (0, -radius), (0, 0)):
                            self._add(low + [dx, dy, 0], press_closed)
                if press_closed:
                    self._add(low, False)
                self._add(contact + [0, 0, 0.025], False)
            elif self.crooked_action == "closed-tip":
                # Close in free space and use the two fingertip end faces as a
                # compact press. Never change jaw state while touching the cap.
                contact = np.array([target_pos[0], target_pos[1], surface_top])
                self._add(contact + [0, 0, 0.025], False)
                self._add(contact + [0, 0, 0.025], True)
                self._add(contact + [0, 0, self.tool.press_offset_m - self.crooked_tap_depth_m], True)
                self._add(contact + [0, 0, 0.025], True)
                self._add(contact + [0, 0, 0.025], False)
            elif self.crooked_action == "cone-press":
                # Close and orient only at vial-clearance height.  The CAD cone
                # nose is then placed from the RGB-D cap centre and surface
                # height.  Jaw state never changes during cap contact.
                vertical_high = np.array([target_pos[0], target_pos[1], height])
                # The coarse TableCam position identifies the vial; a leaning
                # cap's observed centre can be several millimetres away. Centre
                # the STEP-measured broad cone on the bounded wrist observation.
                cap_center_xy = raw_target_pos[:2]
                observed_center = np.asarray(target.get("feature_base_pos_m", []), dtype=float)
                if (observed_center.shape == (3,) and np.isfinite(observed_center).all() and
                        np.linalg.norm(observed_center[:2] - raw_target_pos[:2]) <= 0.016):
                    cap_center_xy = observed_center[:2]
                feature_xy = cap_center_xy + crooked_xy_offset
                high_point = np.asarray(target.get("cap_high_point_base_m", []), dtype=float)
                if (high_point.shape == (3,) and np.isfinite(high_point).all() and
                        np.linalg.norm(high_point[:2] - raw_target_pos[:2]) >= 0.001 and
                        np.linalg.norm(high_point[:2] - raw_target_pos[:2]) <= 0.016 and
                        self.vial_rim_z_m - 0.020 <= high_point[2] <= self.vial_rim_z_m + 0.055):
                    contact_surface_top = float(high_point[2])
                else:
                    contact_surface_top = surface_top
                selected_wpr = self._cone_wpr_for_target(
                    [feature_xy[0], feature_xy[1], contact_surface_top], obs)
                cone_high = self._cone_tcp_for_tip(vertical_high, selected_wpr)
                # Advance only through the cone-led interval from the contact
                # trace, retreat, and reacquire the cap surface before the next
                # tap.  This prevents the broad finger proxy from continuing
                # through the old 45 mm sweep after the useful initial impact.
                # This endpoint is a directly calibratable physical quantity:
                # positive depth moves below the rim and positive press offset
                # raises it. The old surface-relative stroke is deliberately
                # not allowed to override a clearance-only physical trial.
                cone_contact_z = (self.vial_rim_z_m - self.cone_press_below_rim_m +
                                  self.tool.press_offset_m)
                cone_contact = self._cone_tcp_for_tip(
                    [feature_xy[0], feature_xy[1], cone_contact_z], selected_wpr)
                cone_above_z = max(contact_surface_top + 0.005,
                                   cone_contact_z + self.cone_press_stroke_m)
                cone_above = self._cone_tcp_for_tip(
                    [feature_xy[0], feature_xy[1], cone_above_z], selected_wpr)
                if not self.cone_oriented:
                    current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
                    orientation_point = np.array([current[0], current[1], height])
                    self._add(orientation_point, True)
                    # Stage the physical handoff exactly once for the batch:
                    # remain straight while J6 rolls clockwise, then use the
                    # wrist bend to lean the gripper on its reachable axis. Every later
                    # crooked-cap waypoint retains the final posture.
                    self._add_pose(orientation_point, self.cone_wrist_wpr_deg, True)
                    self._add_pose(orientation_point, selected_wpr, True)
                    self.cone_oriented = True
                elif not np.allclose(selected_wpr, self.active_cone_press_wpr_deg,
                                     atol=1e-8):
                    current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
                    self._add_pose([current[0], current[1], height], selected_wpr, True)
                self.active_cone_press_wpr_deg = selected_wpr
                selected_lean_deg = 180.0 - float(selected_wpr[0])
                if selected_lean_deg > 180.0:
                    selected_lean_deg -= 360.0
                self.events.append({"event": "cone_lean_selected",
                                    "target_base_m": [float(feature_xy[0]),
                                                      float(feature_xy[1]),
                                                      float(contact_surface_top)],
                                    "wpr_deg": list(map(float, selected_wpr)),
                                    "selected_lean_deg": selected_lean_deg,
                                    "selected_heading_deg": float(selected_wpr[2]),
                                    "nominal_lean_deg": 45.0,
                                    "press_stroke_m": float(cone_above_z - cone_contact_z)})
                aperture = np.asarray(
                    target.get("aperture_base_pos_m", raw_target_pos), dtype=float)
                self.cone_batch_targets.append(
                    [float(aperture[0]), float(aperture[1]),
                     float(self.vial_rim_z_m)])
                remaining = sum(
                    track.get("class_name") == "crooked_cap" and
                    track.get("attempts", 0) < self.max_attempts
                    for track in (self.defect_tracks or []))
                final_cone_target = self.defect_tracks is None or remaining <= 1
                self._add_pose(cone_high, selected_wpr, True)
                self._add_pose(cone_above, selected_wpr, True)
                self._add_pose(cone_contact, selected_wpr, True)
                self._add_pose(cone_above, selected_wpr, True)
                self._add_pose(cone_high, selected_wpr, True)
                if final_cone_target:
                    # Clear TableCam exactly once after the complete cached
                    # batch. Intermediate targets transfer directly from
                    # cone-tip transit height with the same orientation.
                    self._add_pose(inspection_base_position(), selected_wpr, True)
            else:
                # Reproduce the useful part of the original regrip: approach
                # and capture coaxially with the observed crooked cap, lift
                # along that axis, then turn upright at clearance.  Unlike the
                # old wrist-camera/table sequence, the axis comes from the same
                # TableCam RGB-D crop on both systems and the cap is replaced
                # immediately over its vial.
                axis = np.asarray(target.get("cap_axis_base"), dtype=float)
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
                    raise ValueError("regrip requires a TableCam RGB-D cap axis")
                axis /= np.linalg.norm(axis)
                if axis[2] < 0:
                    axis = -axis
                tilted_wpr = axis_to_wpr(axis)
                pick = target_pos + [0, 0, self.tool.pick_offset_m]
                contact = np.array([target_pos[0], target_pos[1], surface_top])
                # Establish the tilted wrist pose above the vial before any
                # descent.  Do not combine a branch-sensitive wrist rotation
                # with the low cap approach.
                self._add_pose([target_pos[0], target_pos[1], height], tilted_wpr, False)
                self._add_pose(pick + axis * 0.025, tilted_wpr, False)
                self._add_pose(pick, tilted_wpr, False)
                self._add_pose(pick, tilted_wpr, True)
                self._add_pose(pick + axis * 0.050, tilted_wpr, True)
                self._add_pose([target_pos[0], target_pos[1], height], tilted_wpr, True)
                self._add([target_pos[0], target_pos[1], height], True)
                self._add(target_pos + [0, 0, 0.025], True)
                self._add(target_pos + [0, 0, self.tool.place_offset_m], True)
                self._add(target_pos + [0, 0, self.tool.place_offset_m], False)
                self._add(contact + [0, 0, 0.025], False)
                self._add(contact + [0, 0, 0.025], True)
                self._add(contact + [0, 0, self.tool.press_offset_m - self.crooked_tap_depth_m], True)
                self._add(contact + [0, 0, 0.025], True)
                self._add(contact + [0, 0, 0.025], False)
            if not cone_batch:
                self._add([target_pos[0], target_pos[1], height], False)
                self._add_inspection(False)
            if self.crooked_action == "cone-press":
                self.state = ("repair_cone_batch" if final_cone_target
                              else "inspect")
            else:
                self.state = "repair"
        else:
            spare_pos = (np.asarray(spare["base_pos_m"], dtype=float) +
                         np.r_[self.table_cap_xy_offset_m, 0.0])
            self._add([spare_pos[0], spare_pos[1], height], False)
            self._add(spare_pos + [0, 0, 0.025], False)
            pick = spare_pos + [0, 0, self.tool.pick_offset_m]
            self._add(pick, False)
            radius = self.pickup_wiggle_radius_m
            for _ in range(self.pickup_wiggle_cycles):
                for dx, dy in ((radius, 0), (0, radius), (-radius, 0),
                               (0, -radius), (0, 0)):
                    self._add(pick + [dx, dy, 0], False)
            self._add(pick, True)
            self._add([spare_pos[0], spare_pos[1], height], True)
            self._add_inspection(True)
            self.state = "pickup"
        self.pending = {"target": target, "spare": spare, "height": height}

    def act(self, obs, detections):
        obs, detections = policy_observation(obs, detections)
        state_before = self.state
        action = self._act(obs, detections)
        self.decision_records.append({
            "state_before": state_before,
            "state_after": self.state,
            "observation_fields": sorted(obs),
            "detection_fields": sorted({key for item in detections for key in item}),
            "oracle_inputs_used": [],
            "action_tcp": list(map(float, action)),
        })
        return action

    def _act(self, obs, detections):
        if self.queue:
            return self.queue.popleft()
        if self.state == "failed":
            return self.hold(obs)
        if self.state == "refine_spare":
            # Start descending to a still-clear 25 mm pre-pick point while the
            # adapter's VLM future runs.  The final XY correction and pickup
            # are deliberately deferred to await_refine_spare.
            spare = np.asarray(self.pending["spare"]["base_pos_m"], dtype=float)
            current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
            # At the default 300 mm optical working distance the fingertips
            # remain about 170 mm above a table cap.  Use that already-safe
            # height to align the TCP over the coarse target, then descend.
            # Raising back to tray transit here wasted the VLM overlap window
            # and crossed an avoidable ER-4iA wrist singularity.
            align_z = max(current[2], spare[2] + 0.050)
            self._add([spare[0], spare[1], align_z], False)
            self._add(spare + [0, 0, 0.025], False)
            self.state = "await_refine_spare"
            return self.queue.popleft()
        if self.state == "await_refine_spare":
            expected = np.asarray(self.pending["spare"]["base_pos_m"], dtype=float)
            refined = self._refinement(obs, "table_cap", expected)
            if refined is None:
                return self.hold(obs)
            spare_pos = refined + np.r_[self.wrist_table_cap_xy_offset_m, 0.0]
            pick = spare_pos + [0, 0, self.tool.pick_offset_m]
            self._add(spare_pos + [0, 0, 0.025], False)
            self._add(pick, False)
            radius = self.pickup_wiggle_radius_m
            for _ in range(self.pickup_wiggle_cycles):
                for dx, dy in ((radius, 0), (0, radius), (-radius, 0),
                               (0, -radius), (0, 0)):
                    self._add(pick + [dx, dy, 0], False)
            self._add(pick, True)
            self._add([spare_pos[0], spare_pos[1], self.pending["height"]], True)
            self._add_inspection(True)
            self.pending["spare"] = dict(self.pending["spare"], base_pos_m=refined.tolist())
            self.state = "pickup"
            return self.queue.popleft()
        if self.state == "pickup":
            if obs.get("grasp_verified") is not True:
                self.state = "failed"
                self.events.append({"event": "pickup_unverified"})
                return self.hold(obs)
            p = (np.asarray(self.pending["target"]["base_pos_m"], dtype=float) +
                 np.r_[self.missing_cap_xy_offset_m, 0.0])
            if self.wrist_refinement:
                # Re-centre the missing-vial mouth from the wrist before any
                # place/contact waypoint.  The old TableCam empirical offset
                # is not applied to the wrist result.
                expected = np.asarray(self.pending["target"]["base_pos_m"], dtype=float)
                # Unlike a round loose cap, an open clear-vial mouth has a
                # weak/asymmetric boundary. Put it on the optical axis rather
                # than relying on an off-centre VLM correction.
                self._add(self._wrist_view(expected, obs, centre_camera=True), True)
                self.state = "refine_missing"
                return self.queue.popleft()
            self._add([p[0], p[1], self.pending["height"]], True)
            self._add_missing_release(p, self.pending["height"])
            self._add_inspection(False)
            self.state = "repair"
            return self.queue.popleft()
        if self.state == "refine_missing":
            target = np.asarray(self.pending["target"]["base_pos_m"], dtype=float)
            current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
            align_z = max(current[2], self.pending["height"])
            self._add([target[0], target[1], align_z], True)
            self._add(target + [0, 0, 0.025], True)
            self.state = "await_refine_missing"
            return self.queue.popleft()
        if self.state == "await_refine_missing":
            expected = np.asarray(self.pending["target"]["base_pos_m"], dtype=float)
            refined = self._refinement(obs, "missing_cap", expected)
            if refined is None:
                return self.hold(obs)
            p = refined + np.r_[self.wrist_missing_cap_xy_offset_m, 0.0]
            self._add_missing_release(p, self.pending["height"])
            self._add_inspection(False)
            self.pending["target"] = dict(self.pending["target"], base_pos_m=refined.tolist())
            self.state = "repair"
            return self.queue.popleft()
        if self.state == "refine_crooked":
            # Use the VLM wait to perform the one-time cable-safe cone handoff.
            # This batch is emitted only once; later crooked caps reuse it.
            current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
            raised = np.array([current[0], current[1], self.pending["height"]])
            inspection = inspection_base_position()
            point = np.array([inspection[0], inspection[1], self.pending["height"]])
            if not self.cone_oriented and self.crooked_action == "cone-press":
                selected_wpr = self._cone_wpr_for_target(
                    self.pending["target"]["base_pos_m"], obs)
                # First clear the vials vertically, then move laterally to the
                # TableCam-clear inspection XY before rolling and leaning.
                # No later crooked target repeats either orientation change.
                self._add(raised, True)
                self._add(point, True)
                self._add_pose(point, self.cone_wrist_wpr_deg, True)
                self._add_pose(point, selected_wpr, True)
                self.active_cone_press_wpr_deg = selected_wpr
                self.cone_oriented = True
            self.state = "await_refine_crooked"
            return self.queue.popleft() if self.queue else self.hold(obs)
        if self.state == "await_refine_crooked":
            target = self.pending["target"]
            expected = np.asarray(target["base_pos_m"], dtype=float)
            refined = self._refinement(obs, "crooked_cap", expected)
            if refined is None:
                return self.hold(obs)
            updated = dict(target)
            updated["aperture_base_pos_m"] = list(target["base_pos_m"])
            updated["base_pos_m"] = refined.tolist()
            updated["feature_base_pos_m"] = refined.tolist()
            updated["localization_note"] = (
                str(updated.get("localization_note", "")) +
                "; wrist LocateAnything bounded local refinement")
            # A dedicated residual handles hand-eye/TCP error.  Do not reuse
            # the old TableCam correction after local wrist backprojection.
            self.pending["target"] = updated
            self._route_checked(obs, updated, None, wrist_refined=True)
            return self.queue.popleft()
        if self.state == "repair_cone_batch":
            self.state = "verify_cone_batch"
        if self.state == "verify_cone_batch":
            results = []
            for target in self.cone_batch_targets:
                good = any(
                    d["class_name"] == "good_cap" and
                    d["confidence"] >= self.min_confidence and
                    np.linalg.norm(np.asarray(d["base_pos_m"])[:2] -
                                   np.asarray(target)[:2]) < 0.003
                    for d in detections)
                results.append({"target_base_m": list(map(float, target)),
                                "good_cap_detected": bool(good)})
            self.events.append({
                "event": "cone_batch_verification",
                "all_fixed": bool(results and all(
                    result["good_cap_detected"] for result in results)),
                "targets": results,
            })
            self.state, self.pending = "inspect", None
            self.cone_batch_targets = []
            return self.hold(obs)
        if self.state == "repair":
            self.state = "verify"
            self.confirmations = 0
            self.verify_frames = 0
        if self.state == "verify":
            target = np.asarray(self.pending["target"]["base_pos_m"], dtype=float)
            good = any(d["class_name"] == "good_cap" and d["confidence"] >= self.min_confidence
                       and np.linalg.norm(np.asarray(d["base_pos_m"])[:2] - target[:2]) < 0.003
                       for d in detections)
            self.confirmations = self.confirmations + 1 if good else 0
            self.verify_frames += 1
            incremental_cone = (self.crooked_action == "cone-press" and
                                self.pending["target"]["class_name"] == "crooked_cap")
            required_confirmations = 1 if incremental_cone else 2
            maximum_verify_frames = 1 if incremental_cone else 5
            if self.confirmations >= required_confirmations:
                self.events.append({"event": "visually_verified_repair", "target": target.tolist()})
                self.state, self.pending = "inspect", None
            elif self.verify_frames >= maximum_verify_frames:
                self.events.append({"event": "repair_not_verified", "target": target.tolist()})
                self.state, self.pending = "inspect", None
            return self.hold(obs)

        qualified = [d for d in detections if d.get("contact_qualified")
                     and d["confidence"] >= self.min_confidence]
        # Clear/yellow loose caps can return no usable TableCam depth. Their
        # RGB box centre is already ray/plane-backprojected onto the calibrated
        # cap plane, so depth is an optional XY refinement for pickup. Defect
        # contact remains depth-qualified: missing/crooked targets use depth to
        # confirm slot state and crooked-cap surface geometry.
        defects = sorted((d for d in qualified
                          if d["class_name"] in ("crooked_cap", "missing_cap")
                          and (d.get("depth_available") is True or self.wrist_refinement)
                          and self._valid_defect(d)),
                         key=lambda d: d["class_name"] != "missing_cap")
        spares = [d for d in qualified if d["class_name"] == "table_cap"]
        for track, target, _distance in self._tracked_defects(defects):
            if track["attempts"] >= self.max_attempts:
                continue
            spare = None
            if target["class_name"] == "missing_cap":
                if not spares:
                    continue
                # The bbox centre is backprojected onto the calibrated vial-rim
                # plane by both adapters. Do not derive placement Z from hollow-
                # vial depth or require a neighbouring good cap.
                target = dict(target)
                target["base_pos_m"] = list(target["base_pos_m"])
                target["base_pos_m"][2] = self.vial_rim_z_m
                spare = min(spares, key=lambda s: np.linalg.norm(
                    np.asarray(s["base_pos_m"]) - np.asarray(target["base_pos_m"])))
            try:
                self._route(obs, target, spare)
            except ValueError as exc:
                track["attempts"] = self.max_attempts
                self.events.append({"event": "target_rejected", "reason": str(exc),
                                    "target": np.asarray(target["base_pos_m"]).tolist()})
                continue
            track["attempts"] += 1
            return self.queue.popleft()
        self.events.append({"event": "hold", "reason": "no_qualified_repair_target"})
        return self.hold(obs)
