#!/usr/bin/env python3
"""Physics-based, perception-driven cap remediation in Isaac Sim.

Every cap moves only through PhysX contact with the custom fingers, glass and
tray; no cap pose is ever written by this script.

Targets come from the current TableCam detections: the tray grid is fitted to
the cap detections (no hardcoded tray coordinates) and refined with the
missing-cap detection.  Dimensions follow docs/PHYSICAL_SPECIFICATIONS.md.

Missing cap (tool pointing straight down):
  1. Open fingers descend around the spare table cap until the pocket ledge is
     1 mm under its flange; closing the jaws captures the flange between the
     pocket ledge and shoulder.
  2. The cap is carried over the empty vial and lowered until the plug is 6 mm
     into the mouth (fingertips stay clear of the rim); the jaws open and the
     cap drops under gravity.
  3. The closed fingertip face presses the flange down, force limited.

Crooked cap (leaning on the rim, sec. 5 of the spec: flange dipped into the mouth,
plug bottom corner on the opposite rim), default `--crooked-press regrip`:
  1. The wrist camera is servoed onto the cap axis (servo_crooked_cap.servo_cap_axis);
     TableCam supplies only the vial position.
  2. Open fingers approach along that axis, spread axis horizontal, until the pocket
     ledge is REGRIP_LEDGE_GAP_M under the flange; closing the jaws to the coaxial
     grip value traps the flange as in the spare-cap pick.
  3. The cap is lifted straight up clear of the rim and turned upright in free air
     about the grip point.
  4. It is regrasped on the table (the side grip leaves it rocked in the pocket):
     stood on a free table spot, released, located with a straight-down wrist view,
     and picked again like the spare cap.
  5. It is placed and pressed exactly like the missing cap.
  The crooked-cap motions are planned after the servo, since they depend on its result.

Older strategies, kept for comparison (`--crooked-press tilted|vertical`):
  The cap's high side is measured from TableCam depth.  The tool tilts
  PRESS_TILT_DEG off vertical toward the cap's high side, rolled so one finger's
  solid outer back face points down; it presses with that face PRESS_FACE_UP_M up
  from the tip, so the fingertip and its cap pocket pass beyond the raised edge
  (the SMC body stays above the neighbouring caps).
  It presses straight down just inboard of the raised flange edge, force limited.

Vial loads are measured from PhysX contact reports (impulse / dt) and checked
against the fracture limits below.  After both actions the caps settle and their
measured poses are classified with geometry.cap_state.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / 'data/fanuc_er4ia_sim'),
    str(ROOT / 'shared/data/fanuc_er4ia_sim'),
]

# Physical glass vial fracture threshold parameters
VIAL_AXIAL_FRACTURE_LIMIT_N = 65.0
VIAL_LATERAL_FRACTURE_LIMIT_N = 40.0
# Force at which a press stops advancing (well under the axial fracture limit).
PRESS_FORCE_LIMIT_N = 30.0
# Tool axis tilt from vertical for the crooked-cap press.  45 deg keeps the SMC
# body ~21 mm above neighbouring caps and is reachable for the tested slots.
PRESS_TILT_DEG = 45.0
# The crooked cap is pressed with the solid outer back face of the -Y finger this far
# up from the fingertip, so the fingertip (and its cap-pocket mouth) passes beyond the
# cap's raised edge instead of landing on it; the pocket otherwise catches the flange.
PRESS_FACE_UP_M = 0.010
# Mid-radius of the solid fingertip crescents (pocket mouth 13.6 mm, finger back 17.6 mm).
TIP_CRESCENT_MID_M = 0.0156
# Regrip: pocket ledge this far under the flange underside (along the cap axis) when the
# jaws close; the 4.76 mm flange slot leaves room either way.
REGRIP_LEDGE_GAP_M = 0.0005
# Regrip approach: start up to this far up the cap axis (shorter where the arm cannot
# reach, e.g. far tray edge), slow over the last REGRIP_SLOW_M.
REGRIP_STANDOFFS_M = (0.08, 0.06, 0.04, 0.03)
REGRIP_SLOW_M = 0.02
# Near the base a cap leaning toward the robot needs J3 past its limit for a coaxial
# grip; the tool is then tilted toward vertical (about the spread axis) in 2 deg steps
# up to this much, still centred on the cap origin.
REGRIP_MAX_MISALIGN_DEG = 10.
# Per-jaw closing past the coaxial grip value while a cap is held.  At the coaxial value
# the flange has 0.125 mm radial play in its 4.76 mm slot and rocks ~5 deg onto one lip;
# this squeeze (~11 N per jaw at the 4e4 N/m jaw drive) clamps the flange rim square.
JAW_CLAMP_M = 0.0004
# Regrasp on the table: the held cap is lowered until its plug bottom would be this far
# below the work surface (negative = into the table), so the table pushes a cap that is
# rocked in the loose side grip flat within the 2.7 mm flange-slot play before release.
# Table spots are free if TableCam depth shows nothing taller than TABLE_FREE_HEIGHT_M
# within TABLE_FREE_RADIUS_M.
PUTDOWN_DROP_M = -0.0015
TABLE_FREE_RADIUS_M = 0.030
TABLE_FREE_HEIGHT_M = 0.003
# Tray-grid fit: detections farther than this from their nearest slot are outliers.
FIT_INLIER_M = 0.015
DT = 1.0 / 60.0
DOWN = np.array([0., 0., -1.])
SPREAD_Y = np.array([0., 1., 0.])


def choose_candidate(detections, class_name):
    """Select the highest-confidence prediction for the given class."""
    candidates = [d for d in detections if d.get('class_name') == class_name]
    if not candidates:
        raise RuntimeError(f'YOLO did not report any detection for {class_name}')
    return max(candidates, key=lambda item: item['confidence'])


def s_curve_pts(p0, p1, steps):
    """Generate smooth cosine-eased Cartesian waypoints."""
    pts = []
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    for i in range(steps):
        t = (i + 1) / float(steps)
        s = 0.5 * (1.0 - np.cos(np.pi * t))
        pts.append(p0 + s * (p1 - p0))
    return pts


def s_curve_joints(q0, q1, steps):
    """Generate smooth cosine-eased joint waypoints."""
    pts = []
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    for i in range(steps):
        t = (i + 1) / float(steps)
        s = 0.5 * (1.0 - np.cos(np.pi * t))
        pts.append(q0 + s * (q1 - q0))
    return pts


def fit_tray_center(observed, layout):
    """Estimate the fixture XY in the robot base frame from TableCam cap detections.

    Slides the documented 6 x 5 slot pattern over the detected caps (good and crooked)
    and keeps the translation with the smallest mean nearest-slot distance, then
    refines it with the matched residuals. Distances are capped at FIT_INLIER_M and only
    inliers refine the fit, so a false detection off the tray cannot drag the grid.
    No tray coordinates are hardcoded.
    """
    slots = np.array([v[:2] for v in layout['slots_m'].values()])
    pts = np.array([d['base_pos_m'][:2] for d in observed if d['class_name'] in ('good_cap', 'crooked_cap')])
    if len(pts) < 10:
        raise RuntimeError('too few cap detections to localise the tray')

    def score(c):
        d = np.linalg.norm(pts[:, None, :] - (slots + c)[None, :, :], axis=2)
        return np.minimum(d.min(axis=1), FIT_INLIER_M).mean()

    xs = np.arange(0.05, 0.60, 0.002)
    ys = np.arange(-0.25, 0.30, 0.002)
    best = min(((score(np.array([x, y])), x, y) for x in xs for y in ys))
    c = np.array([best[1], best[2]])
    for _ in range(5):
        d = np.linalg.norm(pts[:, None, :] - (slots + c)[None, :, :], axis=2)
        idx = d.argmin(axis=1)
        inlier = d.min(axis=1) < FIT_INLIER_M
        c = c + (pts[inlier] - (slots[idx[inlier]] + c)).mean(axis=0)
    return c, float(score(c))


def set_jaw_separation(arm, names, jaw_val):
    """Set gripper prismatic jaw positions for a requested jaw opening value."""
    import cell
    positions = arm.get_joint_positions().copy()
    for joint, value in cell.jaw_targets(jaw_val).items():
        positions[names.index(joint)] = value
    return positions


def configure_stage_collisions(stage):
    """Deconflict EOAT colliders and configure sub-mm physics via CollisionSubAgent."""
    from src.remediation.collision_manager import CollisionSubAgent
    agent = CollisionSubAgent()
    return agent.configure_stage(stage)


def find_closest_cap_prim(stage, target_base_pos):
    """Find the cap prim in /World/RemediationCaps closest to a robot-base XY coordinate."""
    from pxr import UsdGeom
    from src.remediation.geometry import SPEC
    target_world = np.array([target_base_pos[0], target_base_pos[1], target_base_pos[2] + SPEC.world_base_z])
    scope = stage.GetPrimAtPath('/World/RemediationCaps')
    if not scope.IsValid():
        return None
    closest_prim = None
    min_dist = float('inf')
    for prim in scope.GetChildren():
        xf = UsdGeom.Xformable(prim)
        mat = np.asarray(xf.ComputeLocalToWorldTransform(0.), dtype=float)
        pos = mat[3, :3]
        d = float(np.linalg.norm(pos[:2] - target_world[:2]))
        if d < min_dist:
            min_dist = d
            closest_prim = prim
    return closest_prim


def get_cap_pose(cap_prim):
    """World position and (w, x, y, z) orientation of a cap, read from PhysX (USD may lag)."""
    from omni.physx import get_physx_interface
    t = get_physx_interface().get_rigidbody_transformation(str(cap_prim.GetPath()))
    if t and t.get('ret_val', True):
        x, y, z, w = t['rotation']
        return np.asarray(t['position'], float), (float(w), float(x), float(y), float(z))
    from pxr import UsdGeom
    m = np.asarray(UsdGeom.Xformable(cap_prim).ComputeLocalToWorldTransform(0.), dtype=float)
    w, x, y, z = ScipyRot.from_matrix(m[:3, :3].T).as_quat()[[3, 0, 1, 2]]
    return m[3, :3], (float(w), float(x), float(y), float(z))


def on_tray(detections, class_name, tray_center, layout, tol_m=0.025):
    """Keep only detections of `class_name` that sit within `tol_m` of a documented slot."""
    slots = np.array([v[:2] for v in layout['slots_m'].values()]) + tray_center
    return [d for d in detections if d.get('class_name') == class_name
            and np.linalg.norm(slots - np.asarray(d['base_pos_m'][:2]), axis=1).min() <= tol_m]


def tablecam_points(depth, capture_report):
    """Base-frame point per TableCam depth pixel (H x W x 3, NaN where depth is invalid)."""
    from src.inspection.vision_pipeline import camera_from_usd_capture
    cam = camera_from_usd_capture(capture_report['camera'])
    w, h = cam['size']
    scale = w / depth.shape[1]
    vs, us = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(float)
    pc = np.stack([(us * scale - w / 2) / cam['fx'] * z, (vs * scale - h / 2) / cam['fx'] * z, z], -1)
    return pc @ cam['T'][:3, :3].T + cam['T'][:3, 3], scale


def refine_caps_with_depth(observed, depth, capture_report):
    """Replace each good-cap position with the centroid of its flange top in TableCam depth.

    The box centre of an obliquely viewed cap is biased by ~1.4 mm; the flange top
    (depth points within 0.8 mm of the seated flange-top height) is a full annulus
    about the cap axis, so its centroid is the bore centre.
    """
    from src.remediation.geometry import SPEC, rim_base_z
    pts, scale = tablecam_points(depth, capture_report)
    top = rim_base_z() + SPEC.flange_thickness
    out = []
    for d in observed:
        d = dict(d)
        if d.get('class_name') == 'good_cap' and 'bbox_xyxy' in d:
            x0, y0, x1, y1 = [int(round(v / scale)) for v in d['bbox_xyxy']]
            q = pts[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
            q = q[np.isfinite(q).all(1) & (np.abs(q[:, 2] - top) < 0.0008)]
            if len(q) > 30:
                d['base_pos_m'] = [*q[:, :2].mean(0).tolist(), top]
                d['localization'] = 'tablecam_depth_flange_top_centroid'
        out.append(d)
    return out


def choose_targets(observed, layout, depth_refined=False):
    """Spare cap, missing slot and crooked slot from the current TableCam detections.

    With `depth_refined` cap positions the fitted grid is used directly (within ~0.7 mm);
    otherwise it is corrected with the empty-bore detection.  With no missing-cap
    detection on the tray the missing entries (and the spare) are None.
    """
    center, fit_err = fit_tray_center(observed, layout)
    slots = lambda c: {s: c + np.array(v[:2]) for s, v in layout['slots_m'].items()}
    missing_dets = on_tray(observed, 'missing_cap', center, layout)
    missing = choose_candidate(missing_dets, 'missing_cap') if missing_dets else None
    spare = choose_candidate(observed, 'table_cap') if missing else None
    crooked = choose_candidate(on_tray(observed, 'crooked_cap', center, layout), 'crooked_cap')
    grid = slots(center)
    nearest = lambda xy: min(grid, key=lambda s: np.linalg.norm(grid[s] - xy))
    missing_slot = nearest(np.asarray(missing['base_pos_m'][:2])) if missing else None
    if missing and not depth_refined:
        # Without depth the box-centre grid is biased; the empty-bore detection corrects
        # it (to ~1 mm), and the crooked bore is then taken from the refined grid.
        residual = np.asarray(missing['base_pos_m'][:2]) - grid[missing_slot]
        if np.linalg.norm(residual) < 0.010:
            center = center + residual
            grid = slots(center)
    crooked_slot = nearest(np.asarray(crooked['base_pos_m'][:2]))
    missing_xy = None
    if missing:
        missing_xy = grid[missing_slot] if depth_refined else np.asarray(missing['base_pos_m'][:2])
    return {'tray_center_base_m': center.tolist(), 'tray_fit_error_m': fit_err, 'depth_refined': depth_refined,
            'spare': spare, 'missing': missing, 'crooked': crooked,
            'spare_xy': np.asarray(spare['base_pos_m'][:2]) if spare else None,
            'missing_slot': missing_slot, 'missing_xy': missing_xy,
            'crooked_slot': crooked_slot, 'crooked_xy': grid[crooked_slot]}


class Planner:
    """IK waypoint generation for one tool point and tool orientation at a time."""

    def __init__(self):
        import ik
        import scene_config as C
        self.ik = ik
        self.limits = np.deg2rad([C.JOINT_LIMITS_DEG[f'J{i}'] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)

    def solve(self, q, p, d, s, tool):
        q_new, ok = self.ik.solve(q, p, d, self.limits, tool_offset=tuple(tool), spread_dir=s)
        if not ok:
            raise RuntimeError(f'unreachable waypoint at {np.round(p, 4)}, dir={np.round(d, 3)}')
        return q_new

    def line(self, q, p0, p1, d, s, tool, steps):
        path = []
        for p in s_curve_pts(p0, p1, steps):
            q = self.solve(q, p, d, s, tool)
            path.append(q.copy())
        return path

    def steady(self, q, p0, p1, d, s, tool, step_m):
        """Constant-speed straight line, `step_m` per physics step (used for pressing)."""
        n = max(1, int(np.ceil(np.linalg.norm(np.asarray(p1) - p0) / step_m)))
        path = []
        for t in np.linspace(0, 1, n + 1)[1:]:
            q = self.solve(q, np.asarray(p0) + t * (np.asarray(p1) - p0), d, s, tool)
            path.append(q.copy())
        return path


def cap_tilt_from_depth(depth, capture_report, bbox_xyxy, bore_xy, rim_z):
    """High-side azimuth (rad) and tilt (deg) of a crooked cap from aligned TableCam depth.

    Backprojects the depth pixels in the detection box, keeps cap surface above the
    rim within the flange radius of the bore, and fits a plane with outlier rejection.
    """
    from src.inspection.vision_pipeline import camera_from_usd_capture
    cam = camera_from_usd_capture(capture_report['camera'])
    w, h = cam['size']
    scale = w / depth.shape[1]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    vs, us = np.mgrid[max(0, y0 - 6):min(depth.shape[0], y1 + 6), max(0, x0 - 6):min(depth.shape[1], x1 + 6)]
    z = depth[vs, us].astype(float)
    pc = np.stack([(us * scale - w / 2) / cam['fx'] * z, (vs * scale - h / 2) / cam['fx'] * z, z], -1).reshape(-1, 3)
    pc = pc[np.isfinite(pc).all(1) & (pc[:, 2] > 0)]
    pts = pc @ cam['T'][:3, :3].T + cam['T'][:3, 3]
    r = np.linalg.norm(pts[:, :2] - bore_xy, axis=1)
    q = pts[(r < 0.017) & (pts[:, 2] > rim_z + 0.0005)]
    if len(q) < 30:
        raise RuntimeError(f'only {len(q)} depth points on the crooked cap')
    A = np.c_[q[:, :2] - bore_xy, np.ones(len(q))]
    for _ in range(3):
        c, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        res = q[:, 2] - A @ c
        keep = np.abs(res) < max(0.0015, 2 * res.std())
        A, q = A[keep], q[keep]
    return float(np.arctan2(c[1], c[0])), float(np.degrees(np.arctan(np.hypot(c[0], c[1])))), int(len(q))


def tilted_press_pose(planner, q, xy, z_high, z_low, tool_for, azimuth=None):
    """Tool axis, spread axis and tool point for the tilted back-edge press.

    The tool axis leans PRESS_TILT_DEG from vertical toward heading h and is rolled so
    J6 -Y points down-back: the -Y finger's back-face fingertip edge (along J6 Z) is
    then the lowest point, and the cap pocket mouth lies just forward (along h) of it.
    With a measured tilt, h points to the cap's high side and the edge runs across the
    tilt direction, so the edge lands on the outermost high flange and the pocket mouth
    is beyond the cap.  `tool_for(along)` returns the J6 tool point on that edge.
    """
    b = np.radians(PRESS_TILT_DEG)
    up = np.array([0., 0., 1.])
    candidates = []
    if azimuth is not None:
        a = np.array([np.cos(azimuth), np.sin(azimuth), 0.])
        candidates.append((np.cross(a, up), True))           # h = up x z_dir = a
    radial = np.r_[xy / np.linalg.norm(xy), 0.]
    candidates += [(z, False) for z in (radial, -radial, np.cross(up, radial), -np.cross(up, radial))]
    tool = tool_for(0.0)
    for z_dir, aligned in candidates:
        h = np.cross(up, z_dir)                # gives J6 Z = d x Y = z_dir
        d = np.sin(b) * h - np.cos(b) * up
        y = np.cos(b) * h + np.sin(b) * up     # J6 +Y up-forward, so -Y (back face) is down-back
        try:
            planner.solve(q, np.r_[xy, z_high], d, y, tool)
            planner.solve(q, np.r_[xy, z_low], d, y, tool)
            return d, y, tool, aligned
        except RuntimeError:
            continue
    raise RuntimeError(f'no reachable {PRESS_TILT_DEG:.0f} deg press heading at {np.round(xy, 4)}')


def high_flange_edge(bore_xy, rim_z, azimuth, tilt_deg):
    """Horizontal reach (m, toward the high side) and height above the rim of the
    raised flange edge of a cap leaning on the rim (geometry.leaning_cap_pose)."""
    from src.remediation.geometry import CROOKED_TILT_RANGE_DEG, SPEC, leaning_cap_pose
    tilt = float(np.clip(tilt_deg, *CROOKED_TILT_RANGE_DEG))
    rim = np.r_[bore_xy, rim_z]
    # rotation(tilt, yaw) puts the cap's high side (local +Y) at azimuth yaw + 90 deg.
    p, R = leaning_cap_pose(rim, tilt, np.degrees(azimuth) - 90.)
    edge = R @ np.array([0., SPEC.flange_radius, SPEC.flange_thickness]) + p
    a = np.array([np.cos(azimuth), np.sin(azimuth)])
    return float((edge[:2] - bore_xy) @ a), float(edge[2] - rim_z)


def plan_place_and_press(P, q_high, xy, spread, ledge, tip):
    """Place a cap held by its flange (tool vertical, ledge tool point at [xy, transit])
    into the vial at `xy`, then press it flush with the closed fingertips.

    Lowers until the plug is 6 mm into the mouth (fingertips stay clear of the rim),
    slowing to 0.2 mm per step from 3 mm above the rim so the tapered plug can comply
    as it enters; after the jaws open the cap drops under gravity and the fingertip
    face presses it.
    """
    from src.remediation.geometry import SPEC, rim_base_z, transit_base_z
    rim_z, transit_z = rim_base_z(), transit_base_z()
    seated_top_z = rim_z + SPEC.flange_thickness
    plan = {}
    high, release = np.r_[xy, transit_z], np.r_[xy, rim_z + SPEC.plug_depth - 0.006]
    entry = np.r_[xy, rim_z + SPEC.plug_depth + 0.003]
    plan['lower_into_mouth'] = P.line(q_high, high, entry, DOWN, spread, ledge, 50)
    plan['lower_into_mouth'] += P.steady(plan['lower_into_mouth'][-1], entry, release, DOWN, spread, ledge, 0.0002)
    # Switch the tool point to the fingertip face for the press.
    q_rel = plan['lower_into_mouth'][-1]
    tip_above = np.r_[xy, seated_top_z + 0.012]
    q_tip_above = P.solve(q_rel, tip_above, DOWN, spread, tip)
    plan['reposition_for_press'] = s_curve_joints(q_rel, q_tip_above, 30)
    plan['press'] = P.steady(q_tip_above, tip_above, np.r_[xy, seated_top_z - 0.0005], DOWN, spread, tip, 0.0002)
    q_up = P.solve(plan['press'][-1], high, DOWN, spread, tip)
    plan['retreat'] = s_curve_joints(plan['press'][-1], q_up, 40)
    return plan


def place_and_press(ex, plan, slot, jaw_grip, jaw_hold):
    """Execute plan_place_and_press for a cap held at `jaw_hold`; returns the press record."""
    import scene_config as C
    ex.run(f'lower_plug_into_{slot}', plan['lower_into_mouth'], jaw_hold)
    ex.jaws(f'release_into_{slot}', jaw_hold, C.JAW_OPEN)
    ex.hold(f'release_into_{slot}', C.JAW_OPEN, 40)
    ex.run('reposition_for_press', plan['reposition_for_press'], C.JAW_OPEN)
    ex.jaws('close_fingertips', C.JAW_OPEN, jaw_grip)
    press = ex.press(f'press_seat_{slot}', plan['press'], jaw_grip, slot)
    ex.hold(f'press_seat_{slot}', jaw_grip, 15)
    ex.run('retreat', plan['retreat'], jaw_grip)
    return press


def plan_table_pick(P, q, xy, spread, ledge):
    """Pick an upright cap standing on the table at `xy` by its flange (tool vertical).

    Open fingers descend around it until the pocket ledge is 1 mm under the flange;
    closing the jaws traps the flange between the pocket ledge and shoulder.
    """
    from src.remediation.geometry import SPEC, transit_base_z
    high, low = np.r_[xy, transit_base_z()], np.r_[xy, SPEC.surface_base_z + SPEC.plug_depth - 0.001]
    plan = {'approach': s_curve_joints(q, P.solve(q, high, DOWN, spread, ledge), 40)}
    plan['descend'] = P.line(plan['approach'][-1], high, low, DOWN, spread, ledge, 60)
    plan['lift'] = P.line(plan['descend'][-1], low, high, DOWN, spread, ledge, 50)
    return plan


def plan_table_putdown(P, q, xy, spread, ledge):
    """Stand a held cap upright on the table at `xy`, then raise the open fingers."""
    from src.remediation.geometry import SPEC, transit_base_z
    high = np.r_[xy, transit_base_z()]
    low = np.r_[xy, SPEC.surface_base_z + SPEC.plug_depth + PUTDOWN_DROP_M]
    plan = {'carry': s_curve_joints(q, P.solve(q, high, DOWN, spread, ledge), 50)}
    plan['lower'] = P.line(plan['carry'][-1], high, low, DOWN, spread, ledge, 60)
    plan['raise'] = P.line(plan['lower'][-1], low, high, DOWN, spread, ledge, 40)
    return plan


def free_table_spot(depth, capture_report, near_xy, away_xy):
    """A bare table spot near `near_xy` from TableCam depth, searched outward along the
    direction from `away_xy` to `near_xy` (away from the tray)."""
    from src.remediation.geometry import SPEC
    pts, _ = tablecam_points(depth, capture_report)
    pts = pts.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    pts = pts[np.linalg.norm(pts[:, :2] - near_xy, axis=1) < 0.15]
    out = (near_xy - away_xy) / np.linalg.norm(near_xy - away_xy)
    for dist in (0.065, 0.080, 0.095, 0.110):
        for ang in (0., 30., -30., 60., -60.):
            a = np.radians(ang)
            xy = near_xy + dist * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ out
            q = pts[np.linalg.norm(pts[:, :2] - xy, axis=1) < TABLE_FREE_RADIUS_M]
            if len(q) > 50 and np.all(q[:, 2] < SPEC.surface_base_z + TABLE_FREE_HEIGHT_M) \
                    and np.all(q[:, 2] > SPEC.surface_base_z - TABLE_FREE_HEIGHT_M):
                return xy
    raise RuntimeError(f'no free table spot near {np.round(near_xy, 3)} in TableCam depth')


def plan_regrasp_and_place(P, q, cap_xy, bore_xy, spread, ledge, tip):
    """Re-pick the cap standing on the table at `cap_xy`, carry it over the vial at
    `bore_xy`, place it and press it (plan_table_pick + plan_place_and_press)."""
    from src.remediation.geometry import transit_base_z
    plan = plan_table_pick(P, q, cap_xy, spread, ledge)
    q_high = P.solve(plan['lift'][-1], np.r_[bore_xy, transit_base_z()], DOWN, spread, ledge)
    plan['carry_to_vial'] = s_curve_joints(plan['lift'][-1], q_high, 50)
    plan.update(plan_place_and_press(P, q_high, bore_xy, spread, ledge, tip))
    return plan


def plan_regrip(P, q, bore_xy, axis, ledge, tip, putdown_xy):
    """Grip a leaning cap coaxially, lift it clear, turn it upright and re-place it.

    `axis` is the measured cap axis (base frame).  The tool points down the axis with
    the spread axis horizontal, so the fingers take the flange at its sides (the low
    flange edge is dipped into the mouth); the cap origin comes from the leaning-pose
    model for that axis.  The cap is then lifted straight up (the dipped edge leaves
    the mouth vertically) to transit height above the vial and rotated about the
    horizontal spread axis until the tool is vertical.

    This side grip can hold the cap up to ~2.5 mm off the pocket axis and ~15 deg
    rocked in the 4.76 mm flange slot, which catches the plug on the rim (0.13 mm
    radial clearance at the plug top).  So the cap is regrasped on the table: stood
    upright at `putdown_xy` and released; the caller then locates it with the wrist
    camera and picks it again exactly like the spare cap (plan_regrasp_and_place).
    """
    from src.remediation.geometry import transit_base_z
    from src.remediation.servo_crooked_cap import leaning_cap_from_axis
    p_cap, R_cap = leaning_cap_from_axis(bore_xy, axis)
    cap_a = R_cap[:, 2]
    high = np.r_[bore_xy, transit_base_z()]
    s0 = np.cross([0., 0., 1.], cap_a)
    s0 /= np.linalg.norm(s0)
    toward_up = np.sign(np.cross(cap_a, [0., 0., 1.]) @ s0)
    errors = []
    options = [(m, st, sg) for m in np.arange(0., REGRIP_MAX_MISALIGN_DEG + 1e-6, 2.)
               for st in REGRIP_STANDOFFS_M for sg in (1., -1.)]   # either finger may take either side
    for misalign, standoff, sg in options:
        s = sg * s0
        a = ScipyRot.from_rotvec(s0 * toward_up * np.radians(misalign)).apply(cap_a)
        d = -a
        grasp = p_cap - REGRIP_LEDGE_GAP_M * a
        slow, far = grasp + REGRIP_SLOW_M * a, grasp + standoff * a
        try:
            P.solve(q, grasp, d, s, ledge)      # cheap reachability check first
            plan = {}
            q_far = P.solve(q, far, d, s, ledge)
            plan['approach'] = s_curve_joints(q, q_far, 60)
            plan['descend'] = P.line(q_far, far, slow, d, s, ledge, 50)
            plan['descend'] += P.steady(plan['descend'][-1], slow, grasp, d, s, ledge, 0.0002)
            plan['lift'] = P.line(plan['descend'][-1], grasp, high, d, s, ledge, 90)
            # Rotate about the spread axis (horizontal, normal to both d and DOWN).
            ang = np.arctan2(np.cross(d, DOWN) @ s, d @ DOWN)
            plan['upright'] = []
            qq = plan['lift'][-1]
            for t in s_curve_pts([0.], [1.], 70):
                r = ScipyRot.from_rotvec(s * ang * t[0])
                qq = P.solve(qq, high, r.apply(d), s, ledge)
                plan['upright'].append(qq.copy())
            down = plan_table_putdown(P, plan['upright'][-1], putdown_xy, s, ledge)
            plan.update({'carry_to_table': down['carry'], 'lower_to_table': down['lower'], 'raise_open': down['raise']})
            # Checked here so an unreachable regrasp fails before any motion; the
            # re-pick itself is replanned once the wrist camera has located the cap.
            plan_regrasp_and_place(P, down['raise'][-1], putdown_xy, bore_xy, s, ledge, tip)
            return plan, {'cap_origin_model_base_m': p_cap.tolist(), 'cap_axis': cap_a.tolist(),
                          'tool_misalignment_deg': float(misalign), 'standoff_m': float(standoff),
                          'tool_axis': d.tolist(), 'spread_axis': s.tolist(), 'grasp_tool_point_base_m': grasp.tolist(),
                          'regrasp_table_xy_base_m': np.asarray(putdown_xy).tolist(),
                          'model_tilt_deg': float(np.degrees(np.arccos(np.clip(cap_a[2], -1, 1)))),
                          'tool_tilt_deg': float(np.degrees(np.arccos(np.clip(a[2], -1, 1))))}
        except RuntimeError as exc:
            errors.append(f'misalign {misalign:.0f} deg, standoff {standoff * 100:.0f} cm: {exc}')
    raise RuntimeError(f'regrip unreachable: {errors[-4:]}')


class ContactMeter:
    """Per-step contact forces from PhysX contact reports (impulse / dt)."""

    def __init__(self, stage, body_paths):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema
        for path in body_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
        self.sim = get_physx_simulation_interface()

    def read(self):
        """List of (path_a, path_b, force_vector_N) for every contact pair this step."""
        from pxr import PhysicsSchemaTools
        headers, data = self.sim.get_contact_report()
        pairs = []
        for h in headers:
            impulse = np.zeros(3)
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                impulse += np.asarray(data[i].impulse, float)
            a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
            b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            pairs.append((a, b, impulse / DT))
        return pairs


class VialBreakabilityMonitor:
    """Glass vial fracture check from measured contact loads.

    A vial fractures if the measured axial (vertical) load exceeds
    VIAL_AXIAL_FRACTURE_LIMIT_N or the lateral load exceeds VIAL_LATERAL_FRACTURE_LIMIT_N.
    """

    def __init__(self, stage):
        self.stage = stage
        self.vial_states = {}

    def state(self, slot):
        return self.vial_states.setdefault(slot, {'status': 'INTACT', 'peak_axial_N': 0.0,
                                                  'peak_lateral_N': 0.0, 'fractured': False})

    def record(self, pairs):
        """Accumulate this step's tube loads; returns {slot: (axial_N, lateral_N)}."""
        loads = {}
        for a, b, f in pairs:
            for path in (a, b):
                if path.startswith('/World/RemediationTray/') and '_tube' in path:
                    slot = path.split('/')[3].split('_tube')[0]
                    loads[slot] = loads.get(slot, np.zeros(3)) + f
        out = {}
        for slot, f in loads.items():
            axial, lateral = abs(float(f[2])), float(np.linalg.norm(f[:2]))
            vial = self.state(slot)
            vial['peak_axial_N'] = max(vial['peak_axial_N'], axial)
            vial['peak_lateral_N'] = max(vial['peak_lateral_N'], lateral)
            if not vial['fractured'] and (axial > VIAL_AXIAL_FRACTURE_LIMIT_N or lateral > VIAL_LATERAL_FRACTURE_LIMIT_N):
                vial['fractured'] = True
                vial['status'] = 'FRACTURED'
                self._trigger_fracture_geometry(slot)
            out[slot] = (axial, lateral)
        return out

    def _trigger_fracture_geometry(self, slot_id):
        """Displace and tilt the broken vial mesh to visually reflect the damaged_vial state."""
        from pxr import Gf, UsdGeom
        tube_prim = self.stage.GetPrimAtPath(f'/World/RemediationTray/{slot_id}_tube')
        if tube_prim.IsValid():
            xf = UsdGeom.Xformable(tube_prim)
            rot_op = next((op for op in xf.GetOrderedXformOps() if 'rotate' in op.GetOpName()), None)
            if rot_op is None:
                rot_op = xf.AddRotateXYZOp()
            rot_op.Set(Gf.Vec3f(18.5, -12.0, 45.0))


class Executor:
    """Drives the arm through PD targets, steps PhysX, measures loads and records frames."""

    def __init__(self, world, arm, names, arm_idx, camera, frame_dir, meter, monitor, slots, tracked=()):
        self.world, self.arm, self.names, self.arm_idx = world, arm, names, arm_idx
        self.tracked = list(tracked)   # cap prims whose pose and gripper contacts are logged each step
        self.trace = []
        self.camera, self.frame_dir, self.meter, self.monitor = camera, frame_dir, meter, monitor
        self.slots = slots
        self.q = arm.get_joint_positions()[arm_idx].copy()
        self.frames = 0
        self.loads = {}
        self.status = ''

    def step(self, phase, q, jaw):
        positions = set_jaw_separation(self.arm, self.names, jaw)
        positions[self.arm_idx] = q
        velocities = np.zeros(len(self.names))
        velocities[self.arm_idx] = (q - self.q) / DT
        self.q = np.asarray(q, float).copy()
        from isaacsim.core.utils.types import ArticulationAction
        self.arm.apply_action(ArticulationAction(joint_positions=positions, joint_velocities=velocities))
        self.world.step(render=True)
        pairs = self.meter.read()
        self.loads = self.monitor.record(pairs)
        self._log(phase, jaw, pairs)
        self._frame(phase)

    def _log(self, phase, jaw, pairs):
        entry = {'step': len(self.trace), 'phase': phase, 'jaw_m': float(jaw), 'caps': {},
                 'q_cmd': np.round(self.q, 5).tolist(),
                 'q_act': np.round(self.arm.get_joint_positions()[self.arm_idx], 5).tolist()}
        for prim in self.tracked:
            path = str(prim.GetPath())
            pos, quat = get_cap_pose(prim)
            touching = sorted({(b if a.startswith(path) else a).replace('/World/ER4iA', '')
                               for a, b, f in pairs if (a.startswith(path) or b.startswith(path))
                               and '/World/ER4iA' in a + b and np.linalg.norm(f) > 1e-6})
            entry['caps'][path] = {'pos_world_m': np.round(np.asarray(pos), 5).tolist(), 'quat_wxyz': np.round(quat, 4).tolist(),
                                   'gripper_contacts': touching}
        self.trace.append(entry)

    def run(self, phase, path, jaw):
        for q in path:
            self.step(phase, q, jaw)

    def hold(self, phase, jaw, n):
        for _ in range(n):
            self.step(phase, self.q, jaw)

    def jaws(self, phase, j0, j1, n=20):
        for j in np.linspace(j0, j1, n):
            self.step(phase, self.q, j)

    def press(self, phase, path, jaw, slot, max_hold=90):
        """Advance along `path` only while the measured vial load is under the press limit."""
        i, held, peak = 0, 0, 0.0
        while i < len(path):
            self.step(phase, path[i], jaw)
            axial = self.loads.get(slot, (0.0, 0.0))[0]
            peak = max(peak, axial)
            if axial < PRESS_FORCE_LIMIT_N:
                i += 1
            else:
                held += 1
                if held > max_hold:
                    break
        return {'completed_waypoints': i, 'waypoints': len(path), 'force_limited_steps': held, 'peak_axial_N': peak}

    def _frame(self, phase):
        rgba = self.camera.get_rgba()
        if rgba is None or not rgba.size:
            return
        frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        cv2.putText(frame, 'PHYSICS-BASED REMEDIATION | TableCam', (20, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 230, 255), 2)
        cv2.putText(frame, phase.replace('_', ' ').upper(), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
        for k, slot in enumerate(self.slots):
            axial, lateral = self.loads.get(slot, (0.0, 0.0))
            st = self.monitor.state(slot)['status']
            color = (100, 255, 100) if st == 'INTACT' else (50, 50, 255)
            cv2.putText(frame, f'Vial {slot} measured load: {axial:5.1f} N axial, {lateral:5.1f} N lateral [{st}]',
                        (20, 90 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, .50, color, 1)
        if not cv2.imwrite(str(self.frame_dir / f'{self.frames:04d}.png'), frame):
            raise RuntimeError('could not write output frame')
        self.frames += 1


def measure_cap(prim, rim_world, slot):
    from src.remediation.geometry import cap_state
    pos, (w, x, y, z) = get_cap_pose(prim)
    R = ScipyRot.from_quat([x, y, z, w]).as_matrix()
    state = cap_state(np.asarray(pos), R, rim_world)
    state.update({'prim': str(prim.GetPath()), 'slot': slot, 'position_world_m': np.asarray(pos).tolist(),
                  'seated': state['class_name'] == 'good_cap'})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--depth', type=Path, help='aligned TableCam depth .npy (metres) for the crooked-cap tilt')
    parser.add_argument('--capture-report', type=Path, help='capture_report.json with the TableCam model')
    parser.add_argument('--crooked-press', choices=('regrip', 'vertical', 'tilted'), default='regrip',
                        help='regrip: wrist-camera servo onto the cap axis, coaxial grip, lift, turn upright and '
                             're-place; vertical: closed fingertips pressed straight down; tilted: PRESS_TILT_DEG '
                             'back-face press toward the TableCam high side')
    args = parser.parse_args()

    current_observation = json.loads(args.detections.read_text())
    if current_observation.get('source') != 'current_loaded_scene_tablecam':
        raise ValueError('--detections must come from capture_live_tablecam.py and current-scene YOLO inference')
    observed = current_observation.get('targets', [])
    args.out.mkdir(parents=True, exist_ok=True)

    from src.remediation.geometry import load_layout
    have_depth = bool(args.depth and args.capture_report)
    if have_depth:
        observed = refine_caps_with_depth(observed, np.load(args.depth), json.loads(args.capture_report.read_text()))
    targets = choose_targets(observed, load_layout(), depth_refined=have_depth)
    missing_slot, crooked_slot = targets['missing_slot'], targets['crooked_slot']
    crooked_tilt = None
    # The regrip measures the cap axis with the wrist camera; the TableCam lean is only
    # used by the older press strategies.
    if have_depth and args.crooked_press != 'regrip':
        from src.remediation.geometry import rim_base_z
        az, tilt, n = cap_tilt_from_depth(np.load(args.depth), json.loads(args.capture_report.read_text()),
                                          targets['crooked']['bbox_xyxy'], targets['crooked_xy'], rim_base_z())
        crooked_tilt = {'high_side_azimuth_deg': float(np.degrees(az)), 'tilt_deg': tilt, 'depth_points': n}
        print(f'[remediation] crooked cap from depth: high side {np.degrees(az):.1f} deg, tilt {tilt:.1f} deg ({n} points)', flush=True)
    putdown_xy = None
    if args.crooked_press == 'regrip':
        # Regrasp spot: the spare's spot once it has been picked, else bare table beside it.
        if missing_slot is not None:
            putdown_xy = targets['spare_xy']
        elif have_depth:
            near = np.asarray(choose_candidate(observed, 'table_cap')['base_pos_m'][:2])
            putdown_xy = free_table_spot(np.load(args.depth), json.loads(args.capture_report.read_text()), near,
                                         np.asarray(targets['tray_center_base_m']))
        else:
            raise RuntimeError('regrip needs a free table spot: no missing cap and no TableCam depth')
        targets['regrasp_table_xy'] = putdown_xy
        print(f'[remediation] regrasp table spot {np.round(putdown_xy, 4)}', flush=True)
    print(f"[remediation] tray centre {np.round(targets['tray_center_base_m'], 4)} "
          f"(fit error {targets['tray_fit_error_m']*1000:.1f} mm); missing={missing_slot}, crooked={crooked_slot}", flush=True)

    # Prevent argparse flags being passed to Kit
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.sensors.camera import Camera
        import cell
        import scene_config as C
        from src.remediation.custom_gripper import (attach_collision_proxies, attach_visual_pair,
                                                    finger_back_face_tcp, finger_tip_tcp)
        from src.remediation.geometry import SPEC, rim_base_z, transit_base_z
        from src.remediation.servo_crooked_cap import locate_upright_cap, read_wrist, servo_cap_axis, wrist_camera

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
        mount = attach_visual_pair(stage)
        attach_collision_proxies(stage)
        col_report = configure_stage_collisions(stage)

        has_missing = missing_slot is not None
        spare_prim = find_closest_cap_prim(stage, np.r_[targets['spare_xy'], 0.]) if has_missing else None
        crooked_prim = find_closest_cap_prim(stage, np.r_[targets['crooked_xy'], 0.])
        caps = [c for c in (spare_prim, crooked_prim) if c is not None]
        meter = ContactMeter(stage, [str(c.GetPath()) for c in caps] +
                             [cell.JAW_PRIMS['L'], cell.JAW_PRIMS['R'], cell.FLANGE])
        world.reset()
        print(f'[remediation] gripper mounted; collisions {col_report}; spare {spare_prim.GetPath() if spare_prim else None}, '
              f'crooked {crooked_prim.GetPath()}', flush=True)

        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f'J{i}') for i in range(1, 7)]

        ledge = tuple(mount['cap_seat_tcp_j6_m'])      # held cap origin (flange underside)
        tip = finger_tip_tcp(ledge)                    # closed fingertip face centre
        jaw_grip = mount['cap_grip_jaw_m']
        jaw_hold = jaw_grip - JAW_CLAMP_M
        press_face = lambda along: finger_back_face_tcp(ledge, C.JAW_CLOSED, jaw_grip, along, up=PRESS_FACE_UP_M)
        rim_z = rim_base_z()
        seated_top_z = rim_z + SPEC.flange_thickness
        transit_z = transit_base_z()
        spare_xy, missing_xy, crooked_xy = targets['spare_xy'], targets['missing_xy'], targets['crooked_xy']
        spare_flange_z = SPEC.surface_base_z + SPEC.plug_depth  # cap stands on its plug bottom

        # ---- plan (raises before any motion if a waypoint is unreachable) ----
        P = Planner()
        try:
            q0 = P.solve(np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)]),
                         np.array([.24, .25, transit_z]), DOWN, SPREAD_Y, ledge)
            plan = {}
            q = q_up = q0
            if has_missing:
                high_sp, low_sp = np.r_[spare_xy, transit_z], np.r_[spare_xy, spare_flange_z - 0.001]
                q_high_sp = P.solve(q, high_sp, DOWN, SPREAD_Y, ledge)
                plan['approach_spare'] = s_curve_joints(q, q_high_sp, 40)
                plan['descend_spare'] = P.line(q_high_sp, high_sp, low_sp, DOWN, SPREAD_Y, ledge, 60)
                plan['lift_spare'] = P.line(plan['descend_spare'][-1], low_sp, high_sp, DOWN, SPREAD_Y, ledge, 50)
                q_high_mi = P.solve(plan['lift_spare'][-1], np.r_[missing_xy, transit_z], DOWN, SPREAD_Y, ledge)
                plan['carry_spare'] = s_curve_joints(plan['lift_spare'][-1], q_high_mi, 50)
                plan['place_missing'] = plan_place_and_press(P, q_high_mi, missing_xy, SPREAD_Y, ledge, tip)
                q_up = plan['place_missing']['retreat'][-1]
            if args.crooked_press == 'regrip':
                # Planned after the wrist servo; only the view above the vial is checked here.
                P.solve(q_up, np.r_[crooked_xy, transit_z], DOWN, SPREAD_Y, ledge)
                d_t, s_t, edge, aligned = DOWN, SPREAD_Y, ledge, False
            elif args.crooked_press == 'vertical':
                # Closed fingertips straight down, centred on the vial.  The closed pair is
                # wider than the cap's 21 mm opening in both directions, so it cannot thread
                # into it; the pocket lip (r 13.6 mm) clears the glass, and the raised side of
                # the flange is pushed down until the cap pivots level.  Needs no tilt estimate.
                # At the tip the pocket mouth (r 13.6 mm) is wider than the finger (11.1 mm
                # half-width), so the closed tip face is two solid crescents along the
                # spread axis, 13.6-17.6 mm from the centre.  Align the spread axis with the
                # cap's high side and offset the tool so the raised flange edge sits under
                # the middle of one crescent; centred, the cap slides up into the pocket.
                spread, centre = SPREAD_Y, crooked_xy
                if crooked_tilt:
                    az = np.radians(crooked_tilt['high_side_azimuth_deg'])
                    a2 = np.array([np.cos(az), np.sin(az)])
                    reach, _ = high_flange_edge(crooked_xy, rim_z, az, crooked_tilt['tilt_deg'])
                    spread = np.r_[a2, 0.]
                    centre = crooked_xy + (reach - TIP_CRESCENT_MID_M) * a2
                d_t, s_t, edge, aligned = DOWN, spread, tip, crooked_tilt is not None
                high_cr = np.r_[centre, transit_z]
                start_cr = np.r_[centre, rim_z + 0.040]
                end_cr = np.r_[centre, seated_top_z - 0.0005]
                for sgn in (1., -1.):   # either finger can be the pressing crescent
                    try:
                        q_high_cr = P.solve(q_up, high_cr, DOWN, sgn * spread, tip)
                        P.solve(q_high_cr, end_cr, DOWN, sgn * spread, tip)
                        spread = sgn * spread
                        break
                    except RuntimeError:
                        if sgn < 0:
                            raise
                s_t = spread
                plan['transit_to_crooked'] = s_curve_joints(q_up, q_high_cr, 40)
                plan['descend_crooked'] = P.line(q_high_cr, high_cr, start_cr, DOWN, spread, tip, 40)
                plan['press_crooked'] = P.steady(plan['descend_crooked'][-1], start_cr, end_cr, DOWN, spread, tip, 0.00025)
                plan['retreat_crooked'] = P.line(plan['press_crooked'][-1], end_cr, high_cr, DOWN, spread, tip, 40)
            else:
                d_t, s_t, edge, aligned = tilted_press_pose(P, q_up, crooked_xy, transit_z, seated_top_z, press_face,
                                                            np.radians(crooked_tilt['high_side_azimuth_deg']) if crooked_tilt else None)
                if crooked_tilt:
                    az = np.radians(crooked_tilt['high_side_azimuth_deg'])
                    a2 = np.array([np.cos(az), np.sin(az)])
                    reach, top = high_flange_edge(crooked_xy, rim_z, az, crooked_tilt['tilt_deg'])
                else:
                    a2, reach, top = np.zeros(2), 0.0, 0.020
                # The pressing face slopes down toward the high side.  Start with the face
                # point just beyond the raised flange edge (the face meets the edge higher
                # up the face, well away from the fingertip pocket) and finish with it at
                # the seated flange's outer edge, 0.5 mm below its seated top: a level
                # flange is first touched there, and the fingertip ends outside the glass.
                land_xy = crooked_xy + (reach + 0.002) * a2
                end_xy = crooked_xy + SPEC.flange_radius * a2
                high_cr = np.r_[land_xy, transit_z]
                start_cr = np.r_[land_xy, rim_z + top + 0.008]
                touch_cr = np.r_[land_xy, rim_z + top + 0.001]
                end_cr = np.r_[end_xy, seated_top_z - 0.0005]
                q_high_cr = P.solve(q_up, high_cr, d_t, s_t, edge)
                plan['tilt_to_crooked'] = s_curve_joints(q_up, q_high_cr, 50)
                plan['descend_crooked'] = P.line(q_high_cr, high_cr, start_cr, d_t, s_t, edge, 40)
                plan['press_crooked'] = P.steady(plan['descend_crooked'][-1], start_cr, touch_cr, d_t, s_t, edge, 0.00025)
                plan['press_crooked'] += P.steady(plan['press_crooked'][-1], touch_cr, end_cr, d_t, s_t, edge, 0.00025)
                # Back away along +Y (the back face normal points down-back) before lifting.
                away_cr = end_cr + 0.010 * s_t
                plan['retreat_crooked'] = P.line(plan['press_crooked'][-1], end_cr, away_cr, d_t, s_t, edge, 25)
                plan['retreat_crooked'] += P.line(plan['retreat_crooked'][-1], away_cr, np.r_[away_cr[:2], transit_z],
                                                  d_t, s_t, edge, 40)
        except BaseException as exc:
            failure = {'stage': 'trajectory_planning', 'scene': str(args.scene), 'error_type': type(exc).__name__,
                       'error': str(exc), 'targets': {k: v for k, v in targets.items() if k in ('spare', 'missing', 'crooked')}}
            (args.out / 'failure_report.json').write_text(json.dumps(failure, indent=2, default=str) + '\n')
            raise
        print(f'[remediation] all motions planned; crooked press heading {np.round(d_t, 3)}', flush=True)

        camera = Camera(cell.TABLE_CAM_PATH, resolution=(1280, 720))
        camera.initialize()
        wrist = wrist_camera(stage) if args.crooked_press == 'regrip' else None
        positions = set_jaw_separation(arm, names, C.JAW_OPEN)
        positions[arm_idx] = q0
        arm.set_joint_positions(positions)
        arm.set_joint_velocities(np.zeros(len(names)))
        for _ in range(30):
            world.step(render=True)

        frame_dir = args.out / 'frames'
        frame_dir.mkdir(exist_ok=True)
        monitor = VialBreakabilityMonitor(stage)
        ex = Executor(world, arm, names, arm_idx, camera, frame_dir, meter, monitor,
                      tuple(s for s in (missing_slot, crooked_slot) if s), tracked=caps)
        presses = {}

        if has_missing:
            ex.run('approach_spare', plan['approach_spare'], C.JAW_OPEN)
            ex.run('descend_around_spare', plan['descend_spare'], C.JAW_OPEN)
            ex.jaws('capture_spare_flange', C.JAW_OPEN, jaw_hold)
            ex.hold('capture_spare_flange', jaw_hold, 15)
            ex.run('lift_spare', plan['lift_spare'], jaw_hold)
            ex.run(f'carry_spare_to_{missing_slot}', plan['carry_spare'], jaw_hold)
            presses['missing'] = place_and_press(ex, plan['place_missing'], missing_slot, jaw_grip, jaw_hold)
        else:
            print('[remediation] no missing-cap detection on the tray; skipping the spare-cap placement', flush=True)
            ex.jaws('close_fingers', C.JAW_OPEN, jaw_grip)
        regrip = None
        if args.crooked_press == 'regrip':
            servo_dir = args.out / 'wrist_servo'
            servo_dir.mkdir(exist_ok=True)
            ex.jaws('close_for_wrist_view', jaw_grip, C.JAW_CLOSED)

            def move(q_to):
                ex.run(f'wrist_servo_{crooked_slot}', s_curve_joints(ex.q, q_to, 60), C.JAW_CLOSED)

            def capture():
                ex.hold(f'wrist_servo_{crooked_slot}', C.JAW_CLOSED, 20)
                rgb, depth = read_wrist(wrist)
                return rgb, depth, arm.get_joint_positions()[arm_idx]

            def held_cap_offset():
                """Measured cap pose against the ledge tool point (diagnostic only)."""
                import fk
                Rj, pj = fk.fk(arm.get_joint_positions()[arm_idx])
                pos, (w, x, y, z) = get_cap_pose(crooked_prim)
                held = np.asarray(pos) - np.array([0., 0., SPEC.world_base_z]) - (pj + Rj @ np.asarray(ledge))
                tilt = np.degrees(np.arccos(np.clip(ScipyRot.from_quat([x, y, z, w]).as_matrix()[2, 2], -1, 1)))
                return {'cap_minus_ledge_tool_point_m': held.tolist(), 'cap_tilt_deg': float(tilt)}

            def true_axis():   # scoring only; never fed back into the estimate
                _, (w, x, y, z) = get_cap_pose(crooked_prim)
                return ScipyRot.from_quat([x, y, z, w]).as_matrix()[:, 2]

            regrip = {}
            try:
                axis, q_view, servo_report = servo_cap_axis(P, ex.q, crooked_xy, wrist, move, capture, servo_dir,
                                                            truth=true_axis)
                plan_r, info = plan_regrip(P, q_view, crooked_xy, axis, ledge, tip, putdown_xy)
                regrip.update(info, servo=servo_report)
                d_t, s_t = np.asarray(regrip['tool_axis']), np.asarray(regrip['spread_axis'])
                ex.run(f'approach_along_cap_axis_{crooked_slot}', plan_r['approach'], C.JAW_CLOSED)
                ex.jaws(f'open_for_regrip_{crooked_slot}', C.JAW_CLOSED, C.JAW_OPEN)
                ex.run(f'descend_along_cap_axis_{crooked_slot}', plan_r['descend'], C.JAW_OPEN)
                ex.jaws(f'capture_crooked_flange_{crooked_slot}', C.JAW_OPEN, jaw_grip, n=30)
                ex.hold(f'capture_crooked_flange_{crooked_slot}', jaw_grip, 15)
                ex.run(f'lift_crooked_{crooked_slot}', plan_r['lift'], jaw_grip)
                ex.run(f'turn_cap_upright_{crooked_slot}', plan_r['upright'], jaw_grip)
                ex.hold(f'turn_cap_upright_{crooked_slot}', jaw_grip, 15)
                regrip['held_after_side_grip'] = held_cap_offset()
                print(f"[remediation] side-grip held cap: {regrip['held_after_side_grip']}", flush=True)
                # Regrasp on the table so the cap sits square in the pocket for placement.
                ex.run('carry_to_regrasp_spot', plan_r['carry_to_table'], jaw_grip)
                ex.run('stand_cap_on_table', plan_r['lower_to_table'], jaw_grip)
                ex.jaws('release_on_table', jaw_grip, C.JAW_OPEN)
                ex.hold('release_on_table', C.JAW_OPEN, 30)
                ex.run('raise_open', plan_r['raise_open'], C.JAW_OPEN)
                ex.jaws('close_for_wrist_view', C.JAW_OPEN, C.JAW_CLOSED)
                cap_xy, q_look = locate_upright_cap(P, ex.q, putdown_xy, SPEC.surface_base_z + SPEC.plug_depth + SPEC.flange_thickness,
                                                    wrist, move, capture, servo_dir / 'regrasp_view.png')
                regrip['regrasp_cap_xy_wrist_base_m'] = cap_xy.tolist()
                print(f'[remediation] cap on table located at {np.round(cap_xy, 4)} (released at {np.round(putdown_xy, 4)})', flush=True)
                plan_p = plan_regrasp_and_place(P, q_look, cap_xy, crooked_xy, s_t, ledge, tip)
                ex.run('regrasp_approach', plan_p['approach'], C.JAW_CLOSED)
                ex.jaws('open_for_regrasp', C.JAW_CLOSED, C.JAW_OPEN)
                ex.run('regrasp_descend', plan_p['descend'], C.JAW_OPEN)
                ex.jaws('regrasp_flange', C.JAW_OPEN, jaw_hold)
                ex.hold('regrasp_flange', jaw_hold, 15)
                ex.run('regrasp_lift', plan_p['lift'], jaw_hold)
                ex.run(f'carry_to_{crooked_slot}', plan_p['carry_to_vial'], jaw_hold)
                ex.hold(f'carry_to_{crooked_slot}', jaw_hold, 10)
                regrip['held_after_regrasp'] = held_cap_offset()
                print(f"[remediation] regrasped held cap: {regrip['held_after_regrasp']}", flush=True)
                presses['crooked'] = place_and_press(ex, plan_p, crooked_slot, jaw_grip, jaw_hold)
            except RuntimeError as exc:
                # Report the failed regrip and still run the seating check.
                regrip['error'] = str(exc)
                print(f'[remediation] regrip failed: {exc}', flush=True)
            ex.hold('settle', jaw_grip, 30)
        elif args.crooked_press == 'vertical':
            ex.run(f'transit_to_{crooked_slot}', plan['transit_to_crooked'], jaw_grip)
            ex.run(f'descend_to_{crooked_slot}', plan['descend_crooked'], jaw_grip)
            presses['crooked'] = ex.press(f'vertical_press_{crooked_slot}', plan['press_crooked'], jaw_grip, crooked_slot)
            ex.hold(f'vertical_press_{crooked_slot}', jaw_grip, 15)
            # Open the jaws so the pocket lets go of the seated flange, then lift.
            ex.jaws(f'release_{crooked_slot}', jaw_grip, C.JAW_OPEN)
            ex.run('retreat', plan['retreat_crooked'], C.JAW_OPEN)
            ex.hold('settle', C.JAW_OPEN, 30)
        else:
            ex.jaws('close_for_edge_press', jaw_grip, C.JAW_CLOSED)
            ex.run(f'tilt_{PRESS_TILT_DEG:.0f}deg_over_{crooked_slot}', plan['tilt_to_crooked'], C.JAW_CLOSED)
            ex.run(f'descend_to_{crooked_slot}', plan['descend_crooked'], C.JAW_CLOSED)
            presses['crooked'] = ex.press(f'edge_press_{crooked_slot}', plan['press_crooked'], C.JAW_CLOSED, crooked_slot)
            ex.hold(f'edge_press_{crooked_slot}', C.JAW_CLOSED, 15)
            # Jaws stay closed: opening would drive the pressing back face further into the cap.
            ex.run('retreat', plan['retreat_crooked'], C.JAW_CLOSED)
            ex.hold('settle', C.JAW_CLOSED, 30)
        print(f'[remediation] wrote {ex.frames} frames; presses {presses}', flush=True)

        for _ in range(120):
            world.step(render=False)
        # Classify against the actual tube rims in the scene, not the perceived ones.
        def tube_rim(slot):
            from pxr import UsdGeom
            m = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(f'/World/RemediationTray/{slot}_tube'))
                           .ComputeLocalToWorldTransform(0.))
            return m[3, :3] + np.array([0., 0., SPEC.vial_height])
        seating = {'crooked_slot_cap': measure_cap(crooked_prim, tube_rim(crooked_slot), crooked_slot)}
        if has_missing:
            seating['missing_slot_cap'] = measure_cap(spare_prim, tube_rim(missing_slot), missing_slot)
        for k, v in seating.items():
            print(f'[remediation] {k}: {v}', flush=True)

        report = {
            'mode': 'physics_contact_remediation',
            'physical_robot_authorized': False,
            'scene': str(args.scene),
            'current_scene_detections': str(args.detections),
            'targets': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in targets.items()},
            'missing_slot': missing_slot,
            'crooked_slot': crooked_slot,
            'tool_points_j6_m': {'pocket_ledge': list(ledge), 'fingertip_face': list(tip), 'crooked_press_point': list(edge)},
            'crooked_regrip': regrip,
            'crooked_press': {'strategy': args.crooked_press,
                              'tilt_from_vertical_deg': PRESS_TILT_DEG if args.crooked_press == 'tilted' else 0.0,
                              'tool_axis': np.asarray(d_t).tolist(), 'edge_direction': np.asarray(s_t).tolist(),
                              'edge_aligned_with_cap_tilt': aligned, 'cap_tilt_from_depth': crooked_tilt,
                              'contact': ('coaxial flange grip, upright re-place, fingertip press' if args.crooked_press == 'regrip' else
                                          'closed fingertip crescent over the raised flange edge' if args.crooked_press == 'vertical' else
                                          f'-Y finger outer back face, {PRESS_FACE_UP_M*1000:.0f} mm up from the tip')},
            'cap_motion': 'PhysX contact only; no cap pose is written by the script',
            'press_force_limit_N': PRESS_FORCE_LIMIT_N,
            'presses': presses,
            'vial_breakability': {'axial_fracture_threshold_N': VIAL_AXIAL_FRACTURE_LIMIT_N,
                                  'lateral_fracture_threshold_N': VIAL_LATERAL_FRACTURE_LIMIT_N,
                                  'load_source': 'PhysX contact report impulses / dt',
                                  'vials': monitor.vial_states,
                                  'all_vials_intact': not any(v['fractured'] for v in monitor.vial_states.values())},
            'seating_check_after_120_settle_steps': seating,
            'frame_directory': str(frame_dir),
            'frames_written': ex.frames,
        }
        (args.out / 'trace.json').write_text(json.dumps(ex.trace) + '\n')
        (args.out / 'attempt_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    except BaseException:
        # app.close() exits the process, so print the traceback before it.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
