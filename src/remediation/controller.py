"""Observation-driven remediation state machine, independent of Isaac and HTTP.

Contact actions require an explicitly accepted tool-offset contract and qualified
RGB-D localization.  No simulator pose or stored tray coordinate enters this module.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.remediation.geometry import SPEC, base_to_uf0, rim_base_z, transit_base_z, uf0_to_base


# Camera-derived cap surfaces outside this generous band cannot belong to a
# vial cap.  The band is relative to the calibrated vial rim, not an XY slot.
CROOKED_SURFACE_MIN_M = rim_base_z() - 0.020
CROOKED_SURFACE_MAX_M = rim_base_z() + 0.055
DEFECT_TRACK_GATE_M = 0.100


# The shared policy boundary.  Adapters may own richer observations, but only
# these physical-cell fields may enter a controller decision.  In particular,
# no USD prim, PhysX pose/contact/load, generated label or true cap axis crosses
# this boundary.
ALLOWED_OBSERVATION_FIELDS = frozenset({
    "tcp_pose", "joints", "grasp_verified", "timestamp",
})
ALLOWED_DETECTION_FIELDS = frozenset({
    "class_name", "confidence", "base_pos_m", "bbox_xyxy", "pixel",
    "depth_available", "surface_top_base_z_m", "cap_axis_base", "contact_qualified",
    "localization_note", "timestamp",
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
    return checked_pose_action(base_position, (-180.0, 0.0, 0.0), closed)


def checked_pose_action(base_position, wpr_deg, closed):
    p = base_to_uf0(base_position)
    limits = np.array([[150, 650], [-450, 450], [-334, 300]])
    if not np.isfinite(p).all() or np.any(p < limits[:, 0]) or np.any(p > limits[:, 1]):
        raise ValueError("Target outside calibrated workspace; refusing to clip it")
    wpr = np.asarray(wpr_deg, dtype=float)
    if wpr.shape != (3,) or not np.isfinite(wpr).all():
        raise ValueError("TCP orientation must contain three finite W/P/R angles")
    return [*map(float, p), *map(float, wpr), float(closed)]


def axis_to_wpr(axis_base):
    """Fanuc W/P/R whose tool +Z points down the observed cap axis.

    The roll-pitch-yaw convention is the same fixed XYZ convention for which
    the normal vertical tool pose is (-180, 0, 0) degrees.
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
    # R = Rx(w) Ry(p) Rz(r), including the gimbal-lock-safe branch.
    p = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    cp = np.cos(p)
    if abs(cp) > 1e-8:
        w = np.arctan2(-R[1, 2], R[2, 2])
        r = np.arctan2(-R[0, 1], R[0, 0])
    else:
        w = np.arctan2(R[2, 1], R[1, 1])
        r = 0.0
    degrees = (np.degrees([w, p, r]) + 180.0) % 360.0 - 180.0
    return tuple(degrees)


def wpr_to_rotation(wpr_deg):
    """Fixed XYZ Fanuc W/P/R rotation matrix used by the shared adapters."""
    w, p, r = np.radians(np.asarray(wpr_deg, dtype=float))
    cw, sw = np.cos(w), np.sin(w)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


class RemediationController:
    def __init__(self, tool, max_attempts=1, min_confidence=0.70,
                 crooked_shoulder_depth_m=0.0326838, crooked_tap_depth_m=0.005,
                 crooked_action="shoulder", crooked_wiggle_radius_m=0.00075,
                 crooked_wiggle_cycles=1, vial_rim_z_m=None):
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
        if crooked_action not in ("shoulder", "shoulder-close", "closed-tip", "regrip"):
            raise ValueError("crooked_action must be shoulder, shoulder-close, closed-tip, or regrip")
        self.crooked_action = crooked_action
        self.crooked_wiggle_radius_m = float(crooked_wiggle_radius_m)
        self.crooked_wiggle_cycles = int(crooked_wiggle_cycles)
        if not 0.0 <= self.crooked_wiggle_radius_m <= 0.002:
            raise ValueError("crooked wiggle radius must be between 0 and 0.002 m")
        if not 0 <= self.crooked_wiggle_cycles <= 3:
            raise ValueError("crooked wiggle cycles must be between 0 and 3")
        self.reset()

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

    def _valid_defect(self, detection):
        """Reject contact targets whose camera geometry is physically impossible."""
        if detection.get("class_name") != "crooked_cap":
            return True
        top = float(detection.get("surface_top_base_z_m", np.nan))
        if not CROOKED_SURFACE_MIN_M <= top <= CROOKED_SURFACE_MAX_M:
            self.events.append({"event": "target_rejected",
                                "reason": "crooked RGB-D surface height outside physical range",
                                "surface_top_base_z_m": top})
            return False
        return True

    def _tracked_defects(self, defects):
        """Match fresh camera defects to the initial defect set.

        Tracks are controller memory derived only from camera detections.  A cap
        that moves after contact remains the same attempted track instead of
        gaining a new attempt because its quantized XY coordinate changed.
        """
        if self.defect_tracks is None and defects:
            self.defect_tracks = [{"class_name": d["class_name"],
                                   "anchor_xy": np.asarray(d["base_pos_m"][:2], float),
                                   "last_xy": np.asarray(d["base_pos_m"][:2], float),
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
            matches.append((track, defects[di], distance))
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

    def _route(self, obs, target, spare=None):
        start = len(self.queue)
        try:
            self._route_checked(obs, target, spare)
        except ValueError:
            while len(self.queue) > start:
                self.queue.pop()
            raise

    def _route_checked(self, obs, target, spare=None):
        current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
        height = max(transit_base_z(), current[2])
        target_pos = np.asarray(target["base_pos_m"], dtype=float)
        self._add([current[0], current[1], height], False)
        if spare is None:
            surface_top = float(target.get("surface_top_base_z_m", np.nan))
            if not np.isfinite(surface_top):
                raise ValueError("crooked cap requires an RGB-D surface-top height")
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
            self._add([target_pos[0], target_pos[1], height], False)
            self.state = "repair"
        else:
            spare_pos = np.asarray(spare["base_pos_m"], dtype=float)
            self._add([spare_pos[0], spare_pos[1], height], False)
            self._add(spare_pos + [0, 0, 0.025], False)
            self._add(spare_pos + [0, 0, self.tool.pick_offset_m], False)
            self._add(spare_pos + [0, 0, self.tool.pick_offset_m], True)
            self._add([spare_pos[0], spare_pos[1], height], True)
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
        if self.state == "pickup":
            if obs.get("grasp_verified") is not True:
                self.state = "failed"
                self.events.append({"event": "pickup_unverified"})
                return self.hold(obs)
            p = np.asarray(self.pending["target"]["base_pos_m"], dtype=float)
            self._add([p[0], p[1], self.pending["height"]], True)
            self._add(p + [0, 0, 0.025], True)
            self._add(p + [0, 0, self.tool.place_offset_m], True)
            self._add(p + [0, 0, self.tool.place_offset_m], False)
            self._add(p + [0, 0, self.tool.press_offset_m], True)
            self._add(p + [0, 0, self.tool.press_offset_m], False)
            self._add([p[0], p[1], self.pending["height"]], False)
            self.state = "repair"
            return self.queue.popleft()
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
            if self.confirmations >= 2:
                self.events.append({"event": "visually_verified_repair", "target": target.tolist()})
                self.state, self.pending = "inspect", None
            elif self.verify_frames >= 5:
                self.events.append({"event": "repair_not_verified", "target": target.tolist()})
                self.state, self.pending = "inspect", None
            return self.hold(obs)

        eligible = [d for d in detections if d.get("contact_qualified")
                    and d.get("depth_available") is True
                    and d["confidence"] >= self.min_confidence]
        defects = sorted((d for d in eligible if d["class_name"] in ("crooked_cap", "missing_cap")
                          and self._valid_defect(d)),
                         key=lambda d: d["class_name"] != "missing_cap")
        spares = [d for d in eligible if d["class_name"] == "table_cap"]
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
