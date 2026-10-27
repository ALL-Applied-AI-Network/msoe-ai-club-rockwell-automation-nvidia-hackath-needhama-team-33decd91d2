"""Observation-driven remediation state machine, independent of Isaac and HTTP.

Contact actions require an explicitly accepted tool-offset contract and qualified
RGB-D localization.  No simulator pose or stored tray coordinate enters this module.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.remediation.geometry import base_to_uf0, transit_base_z, uf0_to_base


@dataclass(frozen=True)
class ContactTool:
    """Signed Z offsets from an observed feature to pick/place/press TCP targets."""
    pick_offset_m: float
    place_offset_m: float
    press_offset_m: float
    validated: bool = False


def checked_action(base_position, closed):
    p = base_to_uf0(base_position)
    limits = np.array([[150, 650], [-450, 450], [-334, 300]])
    if not np.isfinite(p).all() or np.any(p < limits[:, 0]) or np.any(p > limits[:, 1]):
        raise ValueError("Target outside calibrated workspace; refusing to clip it")
    return [*map(float, p), -180.0, 0.0, 0.0, float(closed)]


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
        defects = [d for d in eligible if d["class_name"] in ("crooked_cap", "missing_cap")]
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
