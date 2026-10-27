"""Observation-driven remediation state machine, independent of Isaac and HTTP.

Contact actions require an explicitly accepted tool-offset contract and qualified
RGB-D localization.  No simulator pose or stored tray coordinate enters this module.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.remediation.geometry import base_to_uf0, transit_base_z, uf0_to_base


# The shared policy boundary.  Adapters may own richer observations, but only
# these physical-cell fields may enter a controller decision.  In particular,
# no USD prim, PhysX pose/contact/load, generated label or true cap axis crosses
# this boundary.
ALLOWED_OBSERVATION_FIELDS = frozenset({
    "tcp_pose", "joints", "grasp_verified", "crooked_axis_base",
    "upright_cap_base_pos", "table_regrasp_base_pos", "timestamp",
})
ALLOWED_DETECTION_FIELDS = frozenset({
    "class_name", "confidence", "base_pos_m", "bbox_xyxy", "pixel",
    "depth_available", "contact_qualified", "localization_note", "timestamp",
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
    def __init__(self, tool, max_attempts=1, min_confidence=0.70):
        if not tool.validated:
            raise ValueError("Contact tool offsets have not been explicitly accepted")
        self.tool = tool
        self.max_attempts = max_attempts
        self.min_confidence = float(min_confidence)
        if not 0.0 < self.min_confidence < 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.reset()

    def reset(self):
        self.queue = deque()
        self.pending = None
        self.attempts = {}
        self.events = []
        self.decision_records = []
        self.camera_verified_free_table_spots = []
        self.state = "inspect"
        self.confirmations = 0
        self.verify_frames = 0

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
            self._add([target_pos[0], target_pos[1], height], True)
            self._add(target_pos + [0, 0, 0.025], True)
            self._add(target_pos + [0, 0, self.tool.press_offset_m], True)
            self._add(target_pos + [0, 0, self.tool.press_offset_m], False)
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
        if self.state == "crooked_observe_axis":
            axis = obs.get("crooked_axis_base")
            table = obs.get("table_regrasp_base_pos")
            if table is None and self.camera_verified_free_table_spots:
                table = self.camera_verified_free_table_spots[-1]
            if axis is None or table is None:
                self.events.append({"event": "hold", "reason": "crooked_camera_observation_required"})
                return self.hold(obs)
            target = np.asarray(self.pending["target"]["base_pos_m"], dtype=float)
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            if axis[2] < 0:
                axis = -axis
            table = np.asarray(table, dtype=float)
            wpr = axis_to_wpr(axis)
            height = self.pending["height"]
            grasp = target + axis * self.tool.pick_offset_m
            self._add_pose(target + axis * 0.025, wpr, False)
            self._add_pose(grasp, wpr, False)
            self._add_pose(grasp, wpr, True)
            self._add_pose(target + axis * 0.050, wpr, True)
            self._add([target[0], target[1], height], True)
            self._add([table[0], table[1], height], True)
            self._add(table + [0, 0, self.tool.place_offset_m], True)
            self._add(table + [0, 0, self.tool.place_offset_m], False)
            self.state = "crooked_verify_table"
            self.pending["table_regrasp_base_pos"] = table.tolist()
            self.pending["crooked_axis_base"] = axis.tolist()
            return self.queue.popleft()
        if self.state == "crooked_verify_table":
            upright = obs.get("upright_cap_base_pos")
            if upright is None:
                self.state = "failed"
                self.events.append({"event": "crooked_table_regrasp_aborted",
                                    "reason": "upright cap not confirmed by camera"})
                return self.hold(obs)
            upright = np.asarray(upright, dtype=float)
            target = np.asarray(self.pending["target"]["base_pos_m"], dtype=float)
            height = self.pending["height"]
            self._add(upright + [0, 0, 0.025], False)
            self._add(upright + [0, 0, self.tool.pick_offset_m], False)
            self._add(upright + [0, 0, self.tool.pick_offset_m], True)
            self._add([upright[0], upright[1], height], True)
            self._add([target[0], target[1], height], True)
            self._add(target + [0, 0, 0.025], True)
            self._add(target + [0, 0, self.tool.place_offset_m], True)
            self._add(target + [0, 0, self.tool.place_offset_m], False)
            self._add(target + [0, 0, self.tool.press_offset_m], True)
            self._add(target + [0, 0, self.tool.press_offset_m], False)
            self._add([target[0], target[1], height], False)
            self.state = "repair"
            return self.queue.popleft()
        if self.state == "pickup":
            if obs.get("grasp_verified") is not True:
                self.state = "failed"
                self.events.append({"event": "pickup_unverified"})
                return self.hold(obs)
            self.camera_verified_free_table_spots.append(
                list(map(float, self.pending["spare"]["base_pos_m"])))
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
        defects = sorted((d for d in eligible if d["class_name"] in ("crooked_cap", "missing_cap")),
                         key=lambda d: d["class_name"] != "missing_cap")
        spares = [d for d in eligible if d["class_name"] == "table_cap"]
        for target in defects:
            key = tuple(np.round(np.asarray(target["base_pos_m"])[:2] / 0.005).astype(int))
            if self.attempts.get(key, 0) >= self.max_attempts:
                continue
            spare = None
            if target["class_name"] == "missing_cap":
                if not spares:
                    continue
                spare = min(spares, key=lambda s: np.linalg.norm(
                    np.asarray(s["base_pos_m"]) - np.asarray(target["base_pos_m"])))
            try:
                if target["class_name"] == "crooked_cap":
                    current = uf0_to_base(np.asarray(obs["tcp_pose"][:3], dtype=float))
                    height = max(transit_base_z(), current[2])
                    target_pos = np.asarray(target["base_pos_m"], dtype=float)
                    self._add([current[0], current[1], height], False)
                    self._add([target_pos[0], target_pos[1], height], False)
                    self._add(target_pos + [0, 0, 0.15], False)
                    self.pending = {"target": target, "spare": None, "height": height}
                    self.state = "crooked_observe_axis"
                else:
                    self._route(obs, target, spare)
            except ValueError as exc:
                self.attempts[key] = self.max_attempts
                self.events.append({"event": "target_rejected", "reason": str(exc),
                                    "target": np.asarray(target["base_pos_m"]).tolist()})
                continue
            self.attempts[key] = self.attempts.get(key, 0) + 1
            return self.queue.popleft()
        self.events.append({"event": "hold", "reason": "no_qualified_repair_target"})
        return self.hold(obs)
