"""Perception-coordinate cone lean selection for the calibrated ER-4iA.

The cone CAD axis is exactly 45 degrees from the gripper axis.  We request the
full 45-degree W lean first, then reduce it only when the observed base-frame
contact coordinate is outside the arm's reachable envelope.  No tray slot or
fixture label enters this calculation.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from src.remediation.controller import wpr_to_rotation
from src.remediation.geometry import transit_base_z


ROOT = Path(__file__).resolve().parents[2]
KINEMATICS = ROOT / "data" / "fanuc_er4ia_sim"
if str(KINEMATICS) not in sys.path:
    sys.path.insert(0, str(KINEMATICS))

import ik  # noqa: E402
import scene_config as C  # noqa: E402


NOMINAL_CONE_LEAN_DEG = 45
CONE_LEAN_CANDIDATES_DEG = tuple(range(NOMINAL_CONE_LEAN_DEG, -1, -5))


class ConeReachSelector:
    """Choose the steepest reachable cone lean from continuous base XYZ."""

    def __init__(self, fingertip_tcp_j6_m, cone_tip_from_tcp_m,
                 vial_rim_z_m, cone_press_below_rim_m,
                 press_offset_m=0.0):
        self.tool = tuple(np.asarray(fingertip_tcp_j6_m, dtype=float).reshape(3))
        self.cone_offset = np.asarray(cone_tip_from_tcp_m, dtype=float).reshape(3)
        self.rim_z = float(vial_rim_z_m)
        self.press_below = float(cone_press_below_rim_m)
        self.press_offset = float(press_offset_m)
        self.limits = np.deg2rad(
            [C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        self.fallback_seeds = [np.deg2rad(values) for values in (
            (46.2, 12.7, 3.3, 0.0, -80.6, -136.2),
            (0.0, 20.0, 20.0, 0.0, -90.0, -90.0),
            (30.0, 10.0, 0.0, 0.0, -60.0, -90.0),
            (-30.0, 10.0, 0.0, 0.0, -60.0, -90.0),
        )]

    @staticmethod
    def wpr_for_lean(lean_deg):
        w = 180.0 - float(lean_deg)
        if abs(w - 180.0) < 1e-9:
            w = -180.0
        return (w, 0.0, -90.0)

    def _reachable(self, cone_tip, wpr, seeds):
        rotation = wpr_to_rotation(wpr)
        tcp = np.asarray(cone_tip, dtype=float) - rotation @ self.cone_offset
        for seed in seeds:
            _q, ok = ik.solve(
                seed, tcp, rotation[:, 2], self.limits,
                tool_offset=self.tool, spread_dir=rotation[:, 1])
            if ok:
                return True
        return False

    def __call__(self, target_base_m, joints_deg=None):
        target = np.asarray(target_base_m, dtype=float).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError("cone target must contain three finite base coordinates")
        seeds = []
        measured = np.asarray(joints_deg if joints_deg is not None else (), dtype=float)
        if measured.size >= 6 and np.isfinite(measured[:6]).all():
            seeds.append(np.deg2rad(measured[:6]))
        seeds.extend(self.fallback_seeds)
        high_tip = np.array([target[0], target[1], transit_base_z()])
        contact_tip = np.array([
            target[0], target[1],
            self.rim_z - self.press_below + self.press_offset])
        for lean in CONE_LEAN_CANDIDATES_DEG:
            wpr = self.wpr_for_lean(lean)
            if (self._reachable(high_tip, wpr, seeds) and
                    self._reachable(contact_tip, wpr, seeds)):
                return wpr
        raise ValueError(
            "no cable-safe R=-90 cone orientation reaches the observed base coordinate")
