#!/usr/bin/env python3
"""Physics-based, perception-driven cap remediation in Isaac Sim.

Every cap moves only through PhysX contact with the custom fingers, glass and
tray; no cap pose is ever written by this script.

Targets come from the current TableCam detections: the tray grid is fitted to
the cap detections (no hardcoded tray coordinates) and refined with the
missing-cap detections.  Dimensions follow docs/PHYSICAL_SPECIFICATIONS.md.
Every detected missing cap is filled first (each from the nearest unused table cap),
then every crooked cap is remediated, one after the other.

Missing cap (tool pointing straight down):
  1. Open fingers descend around the spare table cap until the pocket ledge is
     1 mm under its flange; closing the jaws captures the flange between the
     pocket ledge and shoulder.
  2. The cap is carried over the empty vial and lowered until the plug is 6 mm
     into the mouth (fingertips stay clear of the rim); the jaws open and the
     cap drops under gravity.
  3. The closed fingertip face presses the flange down, force limited.

The supported shared controller handles a crooked cap with a straight vertical,
position-limited press while holding W/P/R at (-180, 0, 0). Its target comes
from the current TableCam RGB-D detection and it does not request a wrist camera.

Legacy crooked-cap comparison mode (`--controller-policy legacy`, with
`--crooked-press regrip`) remains available for archived experiments:
  1. The wrist camera is servoed onto the cap axis (servo_crooked_cap.servo_cap_axis);
     TableCam supplies only the vial position.
  2. Open fingers approach along that axis, spread axis horizontal, until the pocket
     ledge is REGRIP_LEDGE_GAP_M under the flange; closing the jaws to the coaxial
     grip value traps the flange as in the spare-cap pick.
  3. The cap is lifted straight up clear of the rim and turned upright in free air
     about the grip point.
  4. It is regrasped on the table (the side grip leaves it rocked in the pocket):
     stood on a free table spot, released so it settles flat, and gripped again in
     place by the still-surrounding fingers (no camera view).
  5. It is placed and pressed exactly like the missing cap.
  The crooked-cap motions are planned after the servo, since they depend on its result.

Other legacy strategies, kept for comparison (`--crooked-press tilted|vertical`):
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
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot, Slerp

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
# At the simulation's 60 Hz control rate, limit shared-policy TCP transitions
# to 1 mm and 0.5 degree per step.  A physical-policy TCP action is a
# destination; the Isaac adapter must not snap to its final wrist orientation.
POLICY_MAX_LINEAR_STEP_M = 0.001
POLICY_MAX_ANGULAR_STEP_DEG = 0.5
POLICY_MAX_JOINT_STEP_DEG = 0.5
# Closed-loop following-error limits for the simulated articulation.  PhysX's
# drive retains roughly 0.3--0.4 degree steady-state bias under gravity, so the
# trajectory may stream while error is bounded, pauses above 2 degrees, and
# resumes/finally settles below 0.5 degree.
POLICY_MAX_FOLLOWING_ERROR_DEG = 2.0
POLICY_RESUME_ERROR_DEG = 0.5
# A position-limited contact endpoint can retain a small elastic drive error
# even after settling (the TPU tool/contact stack is deliberately compliant).
# Keep it below the normal bounded-following limit without requiring the
# evaluator's contact force in the controller decision.
POLICY_ENDPOINT_ERROR_DEG = 1.0
# Table-cap detections backproject to the 15.492 mm cap top.  The STEP finger
# pocket ledge is 2.7454 mm behind the fingertip TCP and must pass the 2.032 mm
# flange thickness, requiring 4.7774 mm downward travel.  Bias another 1.22 mm
# low inside the pocket's 4.7625 mm flange band; the TPU fingers tolerate the
# low miss and this remains exclusive to the opt-in cone workflow.
CONE_TABLE_CAP_PICK_OFFSET_M = -0.006
# Near the base a cap leaning toward the robot needs J3 past its limit for a coaxial
# grip, and at the far tray edge a cap leaning away needs the wrist beyond reach; the
# tool is then tilted toward vertical (about the spread axis) in 2 deg steps up to this
# much, still centred on the cap origin.  The smallest reachable tilt is used.  An
# offline check on 31 qualified crooked caps: 21 reachable within 10 deg, all but one
# within 30 deg, the last at 40 deg.  Past ~8 deg the 4.76 mm flange slot no longer
# matches the flange plane, so the closing jaws must rotate the cap into the pocket.
REGRIP_MAX_MISALIGN_DEG = 40.
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
# Grid search half-width about the depth-measured plate centre (under half a slot pitch).
PLATE_SEARCH_M = 0.020
# Fewest good/crooked cap detections the tray-grid fit accepts (partly loaded trays).
FIT_MIN_CAPS = 6
# Per-slot TableCam depth check (point counts at the 1920 x 1080 depth resolution).  A
# seated cap hides the vial rim; an exposed rim (>= SLOT_RIM_MIN_PTS points on the glass
# lip at rim height) means the cap is missing, or crooked if cap surface stands over the
# bore (>= SLOT_ABOVE_MIN_PTS points above the rim inside the bore).  No rim, no cap top
# and nothing over the bore is an empty aperture (no vial).
SLOT_RIM_MIN_PTS = 60
SLOT_ABOVE_MIN_PTS = 200
SLOT_CAP_TOP_MIN_PTS = 20
SLOT_ABOVE_MAX_M = 0.040
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


def pose_interpolation(p0, R0, p1, R1, min_steps=60):
    """Cosine-eased position and shortest-path orientation TCP samples."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    R0, R1 = np.asarray(R0, float), np.asarray(R1, float)
    relative = ScipyRot.from_matrix(R0).inv() * ScipyRot.from_matrix(R1)
    angle_deg = float(np.degrees(relative.magnitude()))
    # Cosine easing has a peak derivative of pi/2 times its average.
    easing_peak = np.pi / 2.0
    steps = max(int(min_steps),
                int(np.ceil(easing_peak * np.linalg.norm(p1 - p0) /
                            POLICY_MAX_LINEAR_STEP_M)),
                int(np.ceil(easing_peak * angle_deg / POLICY_MAX_ANGULAR_STEP_DEG)))
    u = np.asarray(s_curve_pts([0.0], [1.0], steps), float).reshape(-1)
    rotations = Slerp([0.0, 1.0], ScipyRot.from_matrix([R0, R1]))(u).as_matrix()
    positions = p0[None, :] + u[:, None] * (p1 - p0)[None, :]
    return list(zip(positions, rotations))


def speed_limited_joints(q0, q1, min_steps=60):
    """Cosine-eased joint transition whose peak step respects the policy limit."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    peak_delta_deg = float(np.degrees(np.max(np.abs(q1 - q0))))
    steps = max(int(min_steps), int(np.ceil((np.pi / 2.0) * peak_delta_deg /
                                            POLICY_MAX_JOINT_STEP_DEG)))
    return s_curve_joints(q0, q1, steps)


def nearest_periodic_joints(q, reference, limits):
    """Choose the nearest equivalent command for full-turn robot joints.

    The ER-4iA J6 limits admit periodic representations of the same physical
    pose (for example 0 and 360 degrees).  Numerical IK can converge onto a
    limit representation even when the adjacent equivalent is a short move.
    Normalize only joints whose configured range spans a full revolution;
    non-periodic arm joints retain the IK result exactly.
    """
    q = np.asarray(q, float).copy()
    reference = np.asarray(reference, float)
    limits = np.asarray(limits, float)
    full_turn = 2.0 * np.pi
    for index, (low, high) in enumerate(limits):
        if high - low < full_turn - 1e-9:
            continue
        first = int(np.ceil((low - q[index]) / full_turn - 1e-12))
        last = int(np.floor((high - q[index]) / full_turn + 1e-12))
        candidates = q[index] + full_turn * np.arange(first, last + 1)
        if len(candidates):
            q[index] = candidates[np.argmin(np.abs(candidates - reference[index]))]
    return q


def single_joint_offset(q, joint_index, delta_rad, limits):
    """Return an exact one-joint move, rejecting rather than wrapping limits."""
    q = np.asarray(q, float)
    limits = np.asarray(limits, float)
    target = q.copy()
    target[joint_index] += float(delta_rad)
    low, high = limits[joint_index]
    if target[joint_index] < low - 1e-9 or target[joint_index] > high + 1e-9:
        raise RuntimeError(
            f'joint {joint_index + 1} offset would exceed limits: '
            f'{np.degrees(target[joint_index]):.1f} deg not in '
            f'[{np.degrees(low):.1f}, {np.degrees(high):.1f}]')
    return target


def is_default_vertical_wpr(wpr):
    return bool(np.allclose(wpr, (-180.0, 0.0, 0.0), atol=1e-8))


def should_preserve_missing_wrist(pending_class, wpr):
    """Round missing-cap actions retain current yaw instead of swapping jaws."""
    return pending_class == 'missing_cap' and is_default_vertical_wpr(wpr)


def is_cone_prepare_action(pending_class, cone_wrist_staged,
                           current_jaw, target_jaw, wpr):
    """Identify only the jaw-close immediately before the crooked-cap handoff."""
    return (pending_class == 'crooked_cap' and not cone_wrist_staged and
            target_jaw != current_jaw and is_default_vertical_wpr(wpr))


def joint_step_deg(q0, q1):
    """Largest shortest-angle change between two six-axis commands."""
    delta = (np.asarray(q1, float) - np.asarray(q0, float) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(np.max(np.abs(delta))))


def plate_center_from_depth(depth, capture_report):
    """Fixture XY from TableCam depth: midpoint of the 1st-99th percentile extent of the
    points at the silver plate's top height (within ~3 mm in the qualified scenes)."""
    from src.remediation.geometry import SPEC
    z = SPEC.surface_base_z + SPEC.tray_base_thickness + SPEC.tray_top_clearance + SPEC.tray_top_thickness
    pts, _ = tablecam_points(depth, capture_report)
    pts = pts.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1) & (np.abs(pts[:, 2] - z) < 0.0015)]
    if len(pts) < 1000:
        raise RuntimeError('tray plate not found in TableCam depth')
    return (np.percentile(pts[:, :2], 1, axis=0) + np.percentile(pts[:, :2], 99, axis=0)) / 2


def fit_tray_center(observed, layout, plate_xy=None):
    """Estimate the fixture XY in the robot base frame from TableCam cap detections.

    Slides the documented 6 x 5 slot pattern over the detected caps (good and crooked)
    and keeps the translation with the smallest mean nearest-slot distance, then
    refines it with the matched residuals. Distances are capped at FIT_INLIER_M and only
    inliers refine the fit, so a false detection off the tray cannot drag the grid.
    No tray coordinates are hardcoded.  With few caps (a partly loaded tray) a grid
    shifted by one slot pitch can fit as well, so with `plate_xy` (plate_center_from_depth)
    the search is limited to within PLATE_SEARCH_M of the measured plate centre.
    """
    slots = np.array([v[:2] for v in layout['slots_m'].values()])
    pts = np.array([d['base_pos_m'][:2] for d in observed if d['class_name'] in ('good_cap', 'crooked_cap')])
    if len(pts) < FIT_MIN_CAPS:
        raise RuntimeError('too few cap detections to localise the tray')

    def score(c):
        d = np.linalg.norm(pts[:, None, :] - (slots + c)[None, :, :], axis=2)
        return np.minimum(d.min(axis=1), FIT_INLIER_M).mean()

    if plate_xy is None:
        xs, ys = np.arange(0.05, 0.60, 0.002), np.arange(-0.25, 0.30, 0.002)
    else:
        xs = np.arange(plate_xy[0] - PLATE_SEARCH_M, plate_xy[0] + PLATE_SEARCH_M, 0.002)
        ys = np.arange(plate_xy[1] - PLATE_SEARCH_M, plate_xy[1] + PLATE_SEARCH_M, 0.002)
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
    """Replace each good-cap and table-cap position with the centroid of its flange top
    in TableCam depth.

    The box centre of an obliquely viewed cap is biased by ~1.4 mm (table caps
    1.9-2.6 mm, enough to grip a spare off-centre so its plug catches the rim); the
    flange top (depth points within 0.8 mm of the seated, or standing, flange-top
    height) is a full annulus about the cap axis, so its centroid is the cap axis
    (within 0.5 mm).
    """
    from src.remediation.geometry import SPEC, rim_base_z
    pts, scale = tablecam_points(depth, capture_report)
    tops = {'good_cap': rim_base_z() + SPEC.flange_thickness,
            'table_cap': SPEC.surface_base_z + SPEC.plug_depth + SPEC.flange_thickness}
    out = []
    for d in observed:
        d = dict(d)
        class_name = d.get('class_name')
        target_z = tops.get(class_name)
        if target_z is not None and 'bbox_xyxy' in d:
            # A loose cap settles on the paper with small physics/contact
            # offsets, so its actual flange top can be about 1 mm above the
            # nominal construction plane.  Sampling at the nominal height then
            # selects only a partial side ring and biases XY several mm.  Use
            # the camera-observed top to choose depth points, but retain the
            # nominal target Z so the proven pickup TCP height is unchanged.
            sample_z = target_z
            tolerance = 0.0008
            observed_top = float(d.get('surface_top_base_z_m', np.nan))
            if (class_name == 'table_cap' and np.isfinite(observed_top) and
                    abs(observed_top - target_z) < 0.004):
                sample_z = observed_top
                tolerance = 0.0005
            x0, y0, x1, y1 = [int(round(v / scale)) for v in d['bbox_xyxy']]
            q = pts[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
            q = q[np.isfinite(q).all(1) & (np.abs(q[:, 2] - sample_z) < tolerance)]
            if len(q) > 30:
                d['base_pos_m'] = [*q[:, :2].mean(0).tolist(), target_z]
                d['localization'] = 'tablecam_depth_flange_top_centroid'
        out.append(d)
    return out


def classify_slots_from_depth(grid, depth, capture_report):
    """Per-slot state from TableCam depth on the fitted grid (see SLOT_* above):
    {slot: {'state': 'seated' | 'crooked' | 'missing' | 'empty', point counts}}."""
    from src.remediation.geometry import SPEC, rim_base_z
    pts, _ = tablecam_points(depth, capture_report)
    pts = pts.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    rim_z = rim_base_z()
    out = {}
    for slot, xy in grid.items():
        r = np.linalg.norm(pts[:, :2] - xy, axis=1)
        rim = int(((r > SPEC.vial_inner_radius - 0.0005) & (r < SPEC.vial_outer_radius + 0.0008)
                   & (np.abs(pts[:, 2] - rim_z) < 0.0015)).sum())
        top = int(((r < SPEC.flange_radius + 0.001) & (np.abs(pts[:, 2] - rim_z - SPEC.flange_thickness) < 0.0015)).sum())
        # Up to SLOT_ABOVE_MAX_M over the rim: a leaning cap stays lower, while the parked
        # gripper hanging over the back row would otherwise count as cap material.
        above = int(((r < SPEC.vial_inner_radius) & (pts[:, 2] > rim_z + 0.003)
                     & (pts[:, 2] < rim_z + SLOT_ABOVE_MAX_M)).sum())
        if rim >= SLOT_RIM_MIN_PTS:
            state = 'crooked' if above >= SLOT_ABOVE_MIN_PTS else 'missing' if top < SLOT_CAP_TOP_MIN_PTS else 'crooked'
        else:
            state = 'seated' if top >= SLOT_CAP_TOP_MIN_PTS or above >= SLOT_ABOVE_MIN_PTS else 'empty'
        out[slot] = {'state': state, 'rim_pts': rim, 'cap_top_pts': top, 'above_bore_pts': above}
    return out


def choose_targets(observed, layout, depth_refined=False, depth=None, capture_report=None):
    """Every missing and crooked slot, and a spare table cap per missing slot, from the
    current TableCam detections.

    Detections are assigned to their nearest slot; a slot with several missing/crooked
    detections keeps the most confident one.  Each missing slot (in slot order) takes the
    nearest unused table cap; missing slots beyond the available spares are listed in
    `unfilled_missing_slots`.  With `depth_refined` cap positions the fitted grid is used
    directly (within ~0.7 mm); otherwise it is corrected with the empty-bore detections.
    """
    from src.remediation.geometry import rim_base_z
    plate_xy = plate_center_from_depth(depth, capture_report) if depth is not None else None
    center, fit_err = fit_tray_center(observed, layout, plate_xy)
    slots = lambda c: {s: c + np.array(v[:2]) for s, v in layout['slots_m'].items()}
    grid = slots(center)
    nearest = lambda xy: min(grid, key=lambda s: np.linalg.norm(grid[s] - xy))
    per_slot = {}
    for d in on_tray(observed, 'missing_cap', center, layout) + on_tray(observed, 'crooked_cap', center, layout):
        s = nearest(np.asarray(d['base_pos_m'][:2]))
        if s not in per_slot or d['confidence'] > per_slot[s]['confidence']:
            per_slot[s] = d
    slot_depth = None
    if depth is not None:
        # Depth decides each slot's state: it finds the defects YOLO misses or calls good,
        # and rejects missing-cap detections on apertures without a vial.
        slot_depth = classify_slots_from_depth(grid, depth, capture_report)
        for s, v in slot_depth.items():
            cls = {'missing': 'missing_cap', 'crooked': 'crooked_cap'}.get(v['state'])
            if cls is None:
                per_slot.pop(s, None)
            elif s not in per_slot:
                per_slot[s] = {'class_name': cls, 'confidence': 0.0, 'base_pos_m': [*grid[s].tolist(), rim_base_z()],
                               'source': 'tablecam_depth_slot_check'}
            elif per_slot[s]['class_name'] != cls:
                per_slot[s] = {**per_slot[s], 'class_name': cls, 'yolo_class_name': per_slot[s]['class_name'],
                               'source': 'tablecam_depth_slot_check'}
    missing_slots = sorted(s for s, d in per_slot.items() if d['class_name'] == 'missing_cap')
    crooked_slots = sorted(s for s, d in per_slot.items() if d['class_name'] == 'crooked_cap')
    if missing_slots and not depth_refined:
        # Without depth the box-centre grid is biased; the empty-bore detections correct
        # it (to ~1 mm), and the crooked bores are then taken from the refined grid.
        res = [np.asarray(per_slot[s]['base_pos_m'][:2]) - grid[s] for s in missing_slots]
        res = [r for r in res if np.linalg.norm(r) < 0.010]
        if res:
            center = center + np.mean(res, axis=0)
            grid = slots(center)
    spares_all = sorted((d for d in observed if d.get('class_name') == 'table_cap'), key=lambda d: -d['confidence'])
    spares = list(spares_all)
    missing = []
    for s in missing_slots:
        if not spares:
            break
        xy = grid[s] if depth_refined else np.asarray(per_slot[s]['base_pos_m'][:2])
        spare = min(spares, key=lambda d: np.linalg.norm(np.asarray(d['base_pos_m'][:2]) - xy))
        spares.remove(spare)
        missing.append({'slot': s, 'xy': xy, 'detection': per_slot[s],
                        'spare': spare, 'spare_xy': np.asarray(spare['base_pos_m'][:2])})
    crooked = [{'slot': s, 'xy': grid[s], 'detection': per_slot[s]} for s in crooked_slots]
    return {'tray_center_base_m': center.tolist(), 'tray_fit_error_m': fit_err, 'depth_refined': depth_refined,
            'plate_center_depth_base_m': None if plate_xy is None else plate_xy.tolist(),
            'missing': missing, 'crooked': crooked, 'spares': spares_all, 'slot_depth_check': slot_depth,
            'unfilled_missing_slots': [s for s in missing_slots if s not in {m['slot'] for m in missing}]}


def align_policy_defects_to_grid(observed, target_summary):
    """Return one camera defect per fitted aperture, centred on that aperture.

    RGB boxes propose missing/crooked candidates and aligned depth decides the
    slot state in :func:`choose_targets`.  The runner must then command the
    perception-fitted grid centre, not the oblique RGB box centre: the latter is
    biased toward a visible vial-mouth edge by several millimetres.  Preserve
    the observed cap feature/high point for cone contact while replacing only
    the vial-aperture XY used for association and placement.
    """
    detections = [dict(item) for item in observed
                  if item.get('class_name') not in ('missing_cap', 'crooked_cap')]
    for class_name, key in (('missing_cap', 'missing'), ('crooked_cap', 'crooked')):
        for target in target_summary[key]:
            detection = dict(target['detection'])
            observed_pos = np.asarray(detection['base_pos_m'], float)
            grid_xy = np.asarray(target['xy'], float)
            detection.update(
                class_name=class_name,
                base_pos_m=[*grid_xy.tolist(), float(observed_pos[2])],
                tray_slot=target['slot'],
                localization_note=(
                    'Contact XY assigned to the perception-derived tray-grid fit; '
                    'cap surface and feature geometry remain aligned RGB-D observations'),
                source='live_tablecam_depth_tray_grid')
            detections.append(detection)
    return detections


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
        return nearest_periodic_joints(q_new, q, self.limits)

    def line(self, q, p0, p1, d, s, tool, steps):
        path = []
        for p in s_curve_pts(p0, p1, steps):
            q = self.solve(q, p, d, s, tool)
            path.append(q.copy())
        return path

    def pose_line(self, q, p0, R0, p1, R1, tool, min_steps=60):
        """Cartesian path with simultaneous speed-limited orientation change."""
        q_start = np.asarray(q, float).copy()
        requested_steps = int(min_steps)
        last_step_deg = float('inf')
        for _ in range(4):
            q = q_start.copy()
            path = []
            samples = pose_interpolation(p0, R0, p1, R1, requested_steps)
            for p, R in samples:
                q_next = self.solve(q, p, R[:, 2], R[:, 1], tool)
                last_step_deg = joint_step_deg(q, q_next)
                if last_step_deg > POLICY_MAX_JOINT_STEP_DEG + 1e-6:
                    break
                q = q_next
                path.append(q.copy())
            else:
                return path
            # A genuine alternate IK branch remains a large jump regardless of
            # sampling density.  Small excesses are ordinary nonlinear joint
            # amplification, so refine the still-unexecuted Cartesian path.
            if last_step_deg > 5.0:
                break
            requested_steps = max(requested_steps * 2, len(samples) * 2)
        raise RuntimeError(f'Cartesian IK changed joint branch ({last_step_deg:.1f} deg step)')

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


def plan_crooked_press(P, q_up, crooked_xy, strategy, crooked_tilt, tip, press_face):
    """Older crooked-cap press strategies (`vertical` / `tilted`) for one cap.

    Returns (plan, tool axis, spread axis, tool point, edge aligned with the measured tilt).
    """
    from src.remediation.geometry import SPEC, rim_base_z, transit_base_z
    rim_z, transit_z = rim_base_z(), transit_base_z()
    seated_top_z = rim_z + SPEC.flange_thickness
    plan = {}
    if strategy == 'vertical':
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
        plan['transit_to_crooked'] = s_curve_joints(q_up, q_high_cr, 40)
        plan['descend_crooked'] = P.line(q_high_cr, high_cr, start_cr, DOWN, spread, tip, 40)
        plan['press_crooked'] = P.steady(plan['descend_crooked'][-1], start_cr, end_cr, DOWN, spread, tip, 0.00025)
        plan['retreat_crooked'] = P.line(plan['press_crooked'][-1], end_cr, high_cr, DOWN, spread, tip, 40)
        return plan, DOWN, spread, tip, crooked_tilt is not None
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
    return plan, d_t, s_t, edge, aligned


def plan_table_putdown(P, q, xy, spread, ledge):
    """Carry a held cap over the table at `xy` and stand it upright there."""
    from src.remediation.geometry import SPEC, transit_base_z
    high = np.r_[xy, transit_base_z()]
    low = np.r_[xy, SPEC.surface_base_z + SPEC.plug_depth + PUTDOWN_DROP_M]
    plan = {'carry': s_curve_joints(q, P.solve(q, high, DOWN, spread, ledge), 50)}
    plan['lower'] = P.line(plan['carry'][-1], high, low, DOWN, spread, ledge, 60)
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
    """Re-close on the cap just released on the table at `cap_xy`, carry it over the
    vial at `bore_xy`, place it and press it (plan_place_and_press).

    The open fingers are still around the released cap (put-down height), so they only
    rise until the pocket ledge is 1 mm under its flange, as in the spare-cap pick; the
    closing jaws centre the cap.  No camera view is needed.
    """
    from src.remediation.geometry import SPEC, transit_base_z
    put = np.r_[cap_xy, SPEC.surface_base_z + SPEC.plug_depth + PUTDOWN_DROP_M]
    low, high = np.r_[cap_xy, SPEC.surface_base_z + SPEC.plug_depth - 0.001], np.r_[cap_xy, transit_base_z()]
    plan = {'rise_to_flange': P.line(q, put, low, DOWN, spread, ledge, 10)}
    plan['lift'] = P.line(plan['rise_to_flange'][-1], low, high, DOWN, spread, ledge, 50)
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
    upright at `putdown_xy`, released so it settles flat, and gripped again in place
    by the same fingers (plan_regrasp_and_place).
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
            plan.update({'carry_to_table': down['carry'], 'lower_to_table': down['lower']})
            # Checked here so an unreachable regrasp fails before any motion; the
            # re-pick itself is replanned from the measured release point.
            plan_regrasp_and_place(P, down['lower'][-1], putdown_xy, bore_xy, s, ledge, tip)
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

    def __init__(self, world, arm, names, arm_idx, camera, frame_dir, meter, monitor, slots,
                 tracked=(), press_control='position-limited', eoat_paths=(),
                 closed_loop_tracking=False):
        self.world, self.arm, self.names, self.arm_idx = world, arm, names, arm_idx
        self.tracked = list(tracked)   # cap prims whose pose and gripper contacts are logged each step
        self.trace = []
        self.camera, self.frame_dir, self.meter, self.monitor = camera, frame_dir, meter, monitor
        self.slots = slots
        self.q = arm.get_joint_positions()[arm_idx].copy()
        self.frames = 0
        self.loads = {}
        self.status = ''
        self.press_control = press_control
        self.eoat_paths = tuple(eoat_paths)
        self.eoat_reference = self._eoat_distances() if self.eoat_paths else ()
        self.closed_loop_tracking = bool(closed_loop_tracking)

    def _eoat_distances(self):
        """Distances from each moving jaw root to the flange in world space."""
        from pxr import UsdGeom
        if len(self.eoat_paths) != 3:
            return ()
        positions = []
        for path in self.eoat_paths:
            prim = self.monitor.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f'EOAT coherence prim is missing: {path}')
            matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.), float)
            positions.append(matrix[3, :3])
        return tuple(float(np.linalg.norm(position - positions[2])) for position in positions[:2])

    def _tracking_error_deg(self):
        actual = np.asarray(self.arm.get_joint_positions()[self.arm_idx], float)
        if not np.isfinite(actual).all():
            raise RuntimeError('robot articulation produced non-finite joint positions')
        delta = (actual - self.q + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.degrees(np.max(np.abs(delta))))

    def _check_eoat(self):
        if self.eoat_reference:
            current = self._eoat_distances()
            drift = max(abs(a - b) for a, b in zip(current, self.eoat_reference))
            if not np.isfinite(current).all() or drift > 0.050:
                raise RuntimeError(f'EOAT detached from flange (relative-distance drift {drift:.3f} m)')

    def step(self, phase, q, jaw):
        command_step_deg = joint_step_deg(self.q, q)
        if self.closed_loop_tracking and command_step_deg > POLICY_MAX_JOINT_STEP_DEG + 1e-6:
            raise RuntimeError(f'unsafe joint command discontinuity ({command_step_deg:.1f} deg)')
        positions = set_jaw_separation(self.arm, self.names, jaw)
        positions[self.arm_idx] = q
        velocities = np.zeros(len(self.names))
        if not self.closed_loop_tracking:
            velocities[self.arm_idx] = (q - self.q) / DT
        self.q = np.asarray(q, float).copy()
        from isaacsim.core.utils.types import ArticulationAction
        action = ArticulationAction(joint_positions=positions, joint_velocities=velocities)
        settle_steps = 30 if self.closed_loop_tracking else 1
        for settle in range(settle_steps):
            self.arm.apply_action(action)
            self.world.step(render=True)
            self._check_eoat()
            error_deg = self._tracking_error_deg()
            pairs = self.meter.read()
            self.loads = self.monitor.record(pairs)
            self._log(phase, jaw, pairs)
            self.trace[-1]['tracking_error_deg'] = error_deg
            self.trace[-1]['waypoint_settle_step'] = settle
            self._frame(phase)
            if not self.closed_loop_tracking:
                if error_deg > 15.0:
                    raise RuntimeError(f'robot articulation lost command tracking ({error_deg:.1f} deg)')
                return
            # Stream one sample per waypoint while following error is bounded.
            if settle == 0 and error_deg < POLICY_MAX_FOLLOWING_ERROR_DEG:
                return
            # Once backpressure engages, do not advance until the drive has
            # recovered close to the trajectory.
            if settle > 0 and error_deg <= POLICY_RESUME_ERROR_DEG:
                return
        raise RuntimeError(f'robot articulation did not settle on waypoint within {settle_steps} steps '
                           f'({error_deg:.1f} deg error)')

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
        if self.closed_loop_tracking and path:
            # Endpoint accuracy matters even when intermediate following error
            # was bounded. Hold the final command until the physical drive bias
            # is reached or fail rather than claiming the TCP destination.
            for _ in range(60):
                if self._tracking_error_deg() <= POLICY_ENDPOINT_ERROR_DEG:
                    break
                self.step(phase + '_endpoint_settle', self.q, jaw)
            else:
                raise RuntimeError('robot articulation did not settle at action endpoint '
                                   f'({self._tracking_error_deg():.1f} deg error)')

    def hold(self, phase, jaw, n):
        for _ in range(n):
            self.step(phase, self.q, jaw)

    def jaws(self, phase, j0, j1, n=20):
        for j in np.linspace(j0, j1, n):
            self.step(phase, self.q, j)

    def press(self, phase, path, jaw, slot, max_hold=90):
        """Execute a press; force is evaluator-only in physical-parity mode."""
        i, held, peak = 0, 0, 0.0
        while i < len(path):
            self.step(phase, path[i], jaw)
            axial = self.loads.get(slot, (0.0, 0.0))[0]
            peak = max(peak, axial)
            if self.press_control == 'position-limited' or axial < PRESS_FORCE_LIMIT_N:
                i += 1
            else:
                held += 1
                if held > max_hold:
                    break
        return {'control': self.press_control,
                'force_used_for_control': self.press_control == 'force-limited',
                'completed_waypoints': i, 'waypoints': len(path),
                'force_limited_steps': held, 'peak_axial_N': peak,
                'would_exceed_operational_limit': peak >= PRESS_FORCE_LIMIT_N}

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


def run_shared_policy_isaac(args, stage, world, mount, collision_report, initial_observed,
                            capture, target_summary):
    """Drive Isaac exclusively through the same observation-driven controller as the cell."""
    import cell
    import fk
    import scene_config as C
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.sensors.camera import Camera
    from src.remediation.controller import (ContactTool, RemediationController,
                                            command_rotation_to_j6,
                                            j6_rotation_to_command, wpr_to_rotation)
    from src.remediation.geometry import SPEC, base_to_uf0, transit_base_z, uf0_to_base
    from src.remediation.geometry import load_layout
    from src.remediation.policy_driver import PolicyDriver

    P = Planner()
    tip = tuple(__import__('src.remediation.custom_gripper', fromlist=['finger_tip_tcp'])
                .finger_tip_tcp(tuple(mount['cap_seat_tcp_j6_m'])))
    q0 = P.solve(np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)]),
                 np.array([.24, .25, transit_base_z()]), DOWN, SPREAD_Y, tip)

    # Oracle association is constructed for evaluator/tracing only and is never
    # included in the controller observation or detection dictionaries.
    evaluator_caps = []
    for item in target_summary['missing']:
        prim = find_closest_cap_prim(stage, np.r_[item['spare_xy'], 0.])
        if prim is not None:
            evaluator_caps.append((f"missing_{item['slot']}", item['slot'], prim))
    for item in target_summary['crooked']:
        prim = find_closest_cap_prim(stage, np.r_[item['xy'], 0.])
        if prim is not None:
            evaluator_caps.append((f"crooked_{item['slot']}", item['slot'], prim))
    tracked = [item[2] for item in evaluator_caps]
    meter = ContactMeter(stage, [str(prim.GetPath()) for prim in tracked] +
                         [cell.JAW_PRIMS['L'], cell.JAW_PRIMS['R'], cell.FLANGE])

    world.reset()
    arm = SingleArticulation(C.ROBOT_PRIM)
    arm.initialize()
    names = list(arm.dof_names)
    arm_idx = [names.index(f'J{i}') for i in range(1, 7)]
    table = Camera(cell.TABLE_CAM_PATH, resolution=(1920, 1080))
    table.initialize()
    table.add_distance_to_image_plane_to_frame()
    positions = set_jaw_separation(arm, names, C.JAW_OPEN)
    positions[arm_idx] = q0
    arm.set_joint_positions(positions)
    arm.set_joint_velocities(np.zeros(len(names)))
    for _ in range(30):
        world.step(render=True)

    frame_dir = args.out / 'frames'
    frame_dir.mkdir(exist_ok=True)
    monitor = VialBreakabilityMonitor(stage)
    slots = tuple(item[1] for item in evaluator_caps)
    ex = Executor(world, arm, names, arm_idx, table, frame_dir, meter, monitor, slots,
                  tracked=tracked, press_control='position-limited',
                  eoat_paths=(cell.JAW_PRIMS['L'], cell.JAW_PRIMS['R'], cell.FLANGE),
                  closed_loop_tracking=True)
    effective_pick_offset = (args.cone_table_pick_offset_m
                             if args.crooked_action == 'cone-press'
                             else args.pick_offset_m)
    controller = RemediationController(
        ContactTool(effective_pick_offset, args.place_offset_m,
                    args.press_offset_m, validated=True),
        # The STEP-measured broad TPU cone performs one centred seating press;
        # do not repeat the wrist rotation after contact.
        max_attempts=1,
        min_confidence=args.confidence,
        crooked_shoulder_depth_m=args.crooked_shoulder_depth_m,
        crooked_tap_depth_m=args.crooked_tap_depth_m,
        crooked_action=args.crooked_action,
        cone_tip_from_tcp_m=mount.get('cone_tip_from_tcp_m',
                                      (0.0, -0.0369851212, -0.0098332793)),
        cone_wrist_wpr_deg=mount.get('cone_wrist_wpr_deg',
                                     (-180.0, 0.0, -90.0)),
        cone_press_wpr_deg=mount.get('cone_press_wpr_deg',
                                     (-180.0, 45.0, -90.0)),
        cone_press_below_rim_m=args.cone_press_below_rim_m)
    decision_dir = args.out / 'policy_observations'
    decision_dir.mkdir(exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    conda_python = Path(os.environ.get(
        'REMEDIATION_PYTHON', str(ROOT / '.conda_env/bin/python')))
    if not conda_python.is_absolute():
        conda_python = ROOT / conda_python
    if not conda_python.is_file():
        raise FileNotFoundError(
            f'YOLO Python not found at {conda_python}; set REMEDIATION_PYTHON')
    inspect_script = ROOT / 'src/inspection/inspect_remediation_capture.py'
    capture_report = Path(args.capture_report).resolve()
    last_action = [*base_to_uf0((np.array([.24, .25, transit_base_z()]))), -180., 0., 0.]
    current_jaw = C.JAW_OPEN
    cone_wrist_staged = False
    inference_index = 0
    cached = {'detections': None, 'depth': None}

    def fresh_perception():
        nonlocal inference_index
        for _ in range(5):
            world.step(render=True)
        rgba = table.get_rgba()
        depth = np.asarray(table.get_depth(device='cpu'), np.float32)
        image_path = decision_dir / f'table_{inference_index:04d}.png'
        depth_path = decision_dir / f'table_{inference_index:04d}_depth.npy'
        result_path = decision_dir / f'table_{inference_index:04d}_detections.json'
        if rgba is None or not rgba.size or depth.shape[:2] != rgba.shape[:2]:
            raise RuntimeError('physical-parity TableCam RGB-D capture failed')
        cv2.imwrite(str(image_path), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
        np.save(depth_path, depth)
        command = [str(conda_python), str(inspect_script), '--checkpoint', str(checkpoint),
                   '--image', str(image_path), '--depth', str(depth_path),
                   '--capture-report', str(capture_report), '--out', str(result_path),
                   '--device', args.inference_device]
        # Isaac sets its embedded Python paths in the process environment.  Do
        # not leak those paths into the separate conda interpreter used for
        # YOLO, or its extension modules and stdlib can come from incompatible
        # Python builds (for example, Python 3.12's re with conda's _sre).
        inference_env = os.environ.copy()
        inference_env.pop('PYTHONHOME', None)
        inference_env.pop('PYTHONPATH', None)
        subprocess.run(command, cwd=ROOT, check=True, env=inference_env)
        raw = json.loads(result_path.read_text())['targets']
        refined = refine_caps_with_depth(raw, depth, capture)
        fitted = choose_targets(refined, load_layout(), depth_refined=True,
                                depth=depth, capture_report=capture)
        detections = []
        for item in align_policy_defects_to_grid(refined, fitted):
            clean = dict(item)
            clean['contact_qualified'] = True
            clean['depth_available'] = bool(item.get('depth_available', True))
            detections.append(clean)
        inference_index += 1
        cached.update(detections=detections, depth=depth)
        return detections, depth

    def tcp_observation():
        return {'tcp_pose': list(last_action),
                'joints': [*np.degrees(ex.q).tolist(),
                           1.0 if current_jaw != C.JAW_OPEN else 0.0],
                'timestamp': len(controller.decision_records)}

    def observe(ctrl):
        # Queue steps do not inspect detections; decision gates always receive a
        # fresh rendered RGB-D observation and real YOLO inference.
        if ctrl.queue and cached['detections'] is not None:
            detections, depth = cached['detections'], cached['depth']
        else:
            detections, depth = fresh_perception()
        obs = tcp_observation()
        if ctrl.state == 'pickup' and not ctrl.queue and ctrl.pending is not None:
            spare = np.asarray(ctrl.pending['spare']['base_pos_m'], float)
            still_present = any(item['class_name'] == 'table_cap' and
                                np.linalg.norm(np.asarray(item['base_pos_m'])[:2] - spare[:2]) < .040
                                for item in detections)
            obs['grasp_verified'] = not still_present
        return obs, detections

    def execute(action, state):
        nonlocal last_action, current_jaw, cone_wrist_staged
        action = np.asarray(action, dtype=float)
        target = uf0_to_base(action[:3])
        Rj, pj = fk.fk(ex.q)
        pending_class = (controller.pending or {}).get('target', {}).get('class_name')
        preserve_missing_wrist = should_preserve_missing_wrist(
            pending_class, action[3:6])
        # Round-cap pick/place has no meaningful yaw.  Preserve the measured
        # straight wrist orientation instead of asking IK for its 180-degree
        # jaw-swapped equivalent; this removes the otherwise unnecessary turn
        # before the table-cap pickup without changing any TCP position.
        rotation = (j6_rotation_to_command(Rj) if preserve_missing_wrist
                    else wpr_to_rotation(action[3:6]))
        target_j6_rotation = command_rotation_to_j6(rotation)
        start = pj + Rj @ np.asarray(tip)
        target_jaw = mount['cap_grip_jaw_m'] - JAW_CLAMP_M if action[6] >= .5 else C.JAW_OPEN
        same_position = np.linalg.norm(target - start) < 1e-5
        same_orientation = (ScipyRot.from_matrix(Rj).inv() *
                            ScipyRot.from_matrix(target_j6_rotation)).magnitude() < 1e-5
        cone_wrist_stage = (
            args.controller_policy == 'shared' and
            args.crooked_action == 'cone-press' and
            not cone_wrist_staged and
            np.allclose(action[3:6], mount.get('cone_wrist_wpr_deg',
                                               (-180.0, 0.0, -90.0)), atol=1e-8))
        cone_prepare_stage = (
            args.controller_policy == 'shared' and
            args.crooked_action == 'cone-press' and
            is_cone_prepare_action(pending_class, cone_wrist_staged,
                                   current_jaw, target_jaw, action[3:6]))
        if cone_prepare_stage:
            # The arm is already at the straight clearance pose left by the
            # missing-cap workflow.  Re-solving this nominal TCP orientation
            # admits an equivalent 180-degree J6 representation before the
            # requested quarter-turn.  This action is intentionally jaw-only.
            ex.jaws(f'policy_{state}_cone_prepare', current_jaw, target_jaw, n=20)
        elif cone_wrist_stage:
            # A Cartesian IK solve can represent this pure wrist roll by the
            # periodic +270-degree J6 solution and redistribute it through the
            # arm.  The real motion is exactly one clockwise quarter-turn at
            # clearance: hold J1--J5, move only J6 by -90 degrees, and retain
            # normal collision/tracking checks while executing the path.
            q_target = single_joint_offset(
                ex.q, 5, np.radians(-90.0), P.limits)
            path = speed_limited_joints(ex.q, q_target, min_steps=60)
            ex.run(f'policy_{state}_cone_wrist_90', path, target_jaw)
            cone_wrist_staged = True
        elif same_position and same_orientation and target_jaw != current_jaw:
            ex.jaws(f'policy_{state}_gripper', current_jaw, target_jaw, n=20)
        else:
            phase = f'policy_{state}'
            try:
                # pose_line consumes controller-frame rotations because its IK
                # extracts command +Z as the physical J6 +X tool axis.  Convert
                # the measured flange orientation back into that frame before
                # interpolating; mixing the two frames causes artificial 100+
                # degree wrist branch jumps even on a no-op waypoint.
                path = P.pose_line(ex.q, start, j6_rotation_to_command(Rj),
                                   target, rotation, tip, min_steps=60)
            except RuntimeError as cartesian_error:
                # Some reachable endpoint pairs straddle an ER-4iA wrist IK
                # singularity.  Do not snap orientation or fail on an artificial
                # intermediate pose: interpolate the reachable joint endpoints
                # with the same bounded angular step and retain physics collision
                # checking throughout the executed path.
                clearance = transit_base_z()
                upward_clearance_retreat = (
                    float(target[2]) >= clearance - 1e-6
                    and float(target[2]) > float(start[2])
                    and np.linalg.norm(np.asarray(target[:2]) - np.asarray(start[:2])) < 1e-4)
                if min(float(start[2]), float(target[2])) < clearance and not upward_clearance_retreat:
                    raise RuntimeError(
                        'refusing joint-space fallback below tray clearance; '
                        'the Cartesian contact route is not safely executable') from cartesian_error
                q_target = P.solve(ex.q, target, rotation[:, 2], rotation[:, 1], tip)
                path = speed_limited_joints(ex.q, q_target, min_steps=60)
                phase += '_joint_fallback'
                print(f'[remediation] Cartesian pose interpolation unavailable; '
                      f'using speed-limited joint fallback: {cartesian_error}', flush=True)
            ex.run(phase, path, target_jaw)
        current_jaw = target_jaw
        last_action = action[:6].tolist()

    driver = PolicyDriver(controller, observe, execute, max_decisions=240, max_idle=8)
    driver.run()
    ex.hold('policy_settle', current_jaw, 30)
    for _ in range(120):
        world.step(render=False)

    def tube_rim(slot):
        from pxr import UsdGeom
        matrix = np.asarray(UsdGeom.Xformable(
            stage.GetPrimAtPath(f'/World/RemediationTray/{slot}_tube')).ComputeLocalToWorldTransform(0.))
        return matrix[3, :3] + np.array([0., 0., SPEC.vial_height])

    seating = {label: measure_cap(prim, tube_rim(slot), slot)
               for label, slot, prim in evaluator_caps}
    report = {
        'mode': 'shared_physical_policy_in_isaac',
        'controller_module': 'src.remediation.controller.RemediationController',
        'force_available_to_controller': False,
        'press_control': 'position-limited',
        'controller_input_contract': 'camera RGB-D, detections, joints, TCP and gripper only',
        'decision_provenance': controller.decision_records,
        'oracle_evaluator_only': True,
        'physical_robot_authorized': False,
        'collision_configuration': collision_report,
        'command_count': len(controller.decision_records),
        'vial_breakability_evaluator': {'vials': monitor.vial_states,
                                        'all_vials_intact': not any(
                                            value['fractured'] for value in monitor.vial_states.values())},
        'seating_evaluator': seating,
        'seated_count': sum(value['seated'] for value in seating.values()),
        'remediated_count': len(seating),
        'frames_written': ex.frames,
    }
    # The deployable trace contains policy inputs/actions only. Simulator poses,
    # contacts and loads are deliberately segregated into an oracle evaluator file.
    (args.out / 'trace.json').write_text(json.dumps(controller.decision_records) + '\n')
    (args.out / 'oracle_evaluation_trace.json').write_text(json.dumps(ex.trace) + '\n')
    (args.out / 'policy_decisions.json').write_text(json.dumps(driver.records, indent=2) + '\n')
    (args.out / 'attempt_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--depth', type=Path, help='aligned TableCam depth .npy (metres) for the crooked-cap tilt')
    parser.add_argument('--capture-report', type=Path, help='capture_report.json with the TableCam model')
    parser.add_argument('--controller-policy', choices=('shared', 'legacy'), default='shared',
                        help='shared uses the physical robot controller; legacy is retained for comparison only')
    parser.add_argument('--checkpoint', type=Path,
                        help='YOLO checkpoint required by the shared policy for fresh camera observations')
    parser.add_argument('--confidence', type=float, default=.70)
    parser.add_argument('--inference-device', default='0')
    parser.add_argument('--pick-offset-m', type=float, default=0.0)
    parser.add_argument('--cone-table-pick-offset-m', type=float,
                        default=CONE_TABLE_CAP_PICK_OFFSET_M,
                        help='cone-press-only table-cap pickup TCP offset; default is 6 mm low '
                             'to put the STEP finger pocket around the cap flange')
    parser.add_argument('--place-offset-m', type=float, default=0.0)
    parser.add_argument('--press-offset-m', type=float, default=0.0)
    parser.add_argument('--cone-press-below-rim-m', type=float, default=0.003,
                        help='cone nose endpoint below the calibrated vial rim; positive is lower '
                             '(default: 0.003 m; use a negative value for a clearance-only trial)')
    parser.add_argument('--crooked-shoulder-depth-m', type=float, default=0.0326838)
    parser.add_argument('--crooked-tap-depth-m', type=float, default=0.005,
                        help='descent beyond RGB-D-measured shoulder contact (default: 0.005 m)')
    parser.add_argument('--crooked-action',
                        choices=('shoulder', 'closed-tip', 'regrip', 'cone-press'),
                        default='shoulder',
                        help='shared-controller crooked-cap action; cone-press selects the separate '
                             'side-cone CAD tool (default remains shoulder)')
    parser.add_argument('--crooked-press', choices=('regrip', 'vertical', 'tilted'), default='regrip',
                        help='regrip: wrist-camera servo onto the cap axis, coaxial grip, lift, turn upright and '
                             're-place; vertical: closed fingertips pressed straight down; tilted: PRESS_TILT_DEG '
                             'back-face press toward the TableCam high side')
    parser.add_argument('--press-control', choices=('position-limited', 'force-limited'),
                        default='position-limited',
                        help='position-limited matches the physical robot; force remains recorded for evaluation')
    args = parser.parse_args()
    if args.controller_policy == 'shared' and not args.checkpoint:
        parser.error('--checkpoint is required with --controller-policy shared')

    current_observation = json.loads(args.detections.read_text())
    if current_observation.get('source') != 'current_loaded_scene_tablecam':
        raise ValueError('--detections must come from capture_live_tablecam.py and current-scene YOLO inference')
    observed = current_observation.get('targets', [])
    args.out.mkdir(parents=True, exist_ok=True)

    from src.remediation.geometry import load_layout
    have_depth = bool(args.depth and args.capture_report)
    if have_depth:
        depth_img, capture = np.load(args.depth), json.loads(args.capture_report.read_text())
        observed = refine_caps_with_depth(observed, depth_img, capture)
    targets = choose_targets(observed, load_layout(), depth_refined=have_depth,
                             depth=depth_img if have_depth else None, capture_report=capture if have_depth else None)
    missing, crooked = targets['missing'], targets['crooked']
    if not missing and not crooked:
        raise RuntimeError('no missing or crooked cap detected on the tray')
    # The regrip measures each cap axis with the wrist camera; the TableCam lean is only
    # used by the older press strategies.
    for c in crooked:
        c['tilt'] = None
        if have_depth and args.crooked_press != 'regrip' and 'bbox_xyxy' in c['detection']:
            from src.remediation.geometry import rim_base_z
            az, tilt, n = cap_tilt_from_depth(depth_img, capture, c['detection']['bbox_xyxy'], c['xy'], rim_base_z())
            c['tilt'] = {'high_side_azimuth_deg': float(np.degrees(az)), 'tilt_deg': tilt, 'depth_points': n}
            print(f"[remediation] crooked cap {c['slot']} from depth: high side {np.degrees(az):.1f} deg, "
                  f"tilt {tilt:.1f} deg ({n} points)", flush=True)
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
        import fk
        import scene_config as C
        from src.remediation.custom_gripper import (CONE_ASSET, attach_collision_proxies,
                                                    attach_visual_pair, cone_tip_from_fingertip_tcp,
                                                    finger_back_face_tcp, finger_tip_tcp)
        from src.remediation.geometry import SPEC, rim_base_z, transit_base_z
        from src.remediation.servo_crooked_cap import read_wrist, servo_cap_axis, wrist_camera

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
        cone_press = args.controller_policy == 'shared' and args.crooked_action == 'cone-press'
        mount = (attach_visual_pair(stage, asset=CONE_ASSET) if cone_press
                 else attach_visual_pair(stage))
        attach_collision_proxies(stage, cone_press=cone_press)
        col_report = configure_stage_collisions(stage)

        ledge = tuple(mount['cap_seat_tcp_j6_m'])      # held cap origin (flange underside)
        tip = finger_tip_tcp(ledge)                    # closed fingertip face centre
        jaw_grip = mount['cap_grip_jaw_m']
        jaw_hold = jaw_grip - JAW_CLAMP_M
        if cone_press:
            mount['cone_tip_from_tcp_m'] = list(
                cone_tip_from_fingertip_tcp(jaw_hold, jaw_grip, side=-1))
            mount['cone_wrist_wpr_deg'] = [-180.0, 0.0, -90.0]
            mount['cone_press_wpr_deg'] = [-180.0, 45.0, -90.0]
        press_face = lambda along: finger_back_face_tcp(ledge, C.JAW_CLOSED, jaw_grip, along, up=PRESS_FACE_UP_M)
        transit_z = transit_base_z()
        spare_flange_z = SPEC.surface_base_z + SPEC.plug_depth  # cap stands on its plug bottom

        # ---- plan (raises before any motion if a waypoint is unreachable) ----
        # The crooked-cap regrips are planned after each wrist servo, since they depend
        # on its result; only the view above each vial is checked here.
        P = Planner()
        if args.controller_policy == 'shared':
            report = run_shared_policy_isaac(args, stage, world, mount, col_report,
                                             observed, capture, targets)
            print(f"[remediation] shared physical policy completed: "
                  f"{report['seated_count']}/{report['remediated_count']} seated", flush=True)
            return
        try:
            q0 = P.solve(np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)]),
                         np.array([.24, .25, transit_z]), DOWN, SPREAD_Y, ledge)
            q_up = q0
            def plan_spare(q, spare_xy, missing_xy):
                high_sp, low_sp = np.r_[spare_xy, transit_z], np.r_[spare_xy, spare_flange_z - 0.001]
                q_high_sp = P.solve(q, high_sp, DOWN, SPREAD_Y, ledge)
                plan = {'approach_spare': s_curve_joints(q, q_high_sp, 40)}
                plan['descend_spare'] = P.line(q_high_sp, high_sp, low_sp, DOWN, SPREAD_Y, ledge, 60)
                plan['lift_spare'] = P.line(plan['descend_spare'][-1], low_sp, high_sp, DOWN, SPREAD_Y, ledge, 50)
                q_high_mi = P.solve(plan['lift_spare'][-1], np.r_[missing_xy, transit_z], DOWN, SPREAD_Y, ledge)
                plan['carry_spare'] = s_curve_joints(plan['lift_spare'][-1], q_high_mi, 50)
                plan['place_missing'] = plan_place_and_press(P, q_high_mi, missing_xy, SPREAD_Y, ledge, tip)
                return plan

            # Each missing slot takes its assigned spare, else the nearest unused spare the
            # arm can reach; slots left without a reachable spare are reported unfilled.
            unused = [d for d in targets['spares'] if not any(d is m['spare'] for m in missing)]
            filled, unfilled = [], list(targets['unfilled_missing_slots'])
            for m in missing:
                options = [m['spare']] + sorted(unused, key=lambda d: np.linalg.norm(np.asarray(d['base_pos_m'][:2]) - m['xy']))
                for spare in options:
                    try:
                        plan = plan_spare(q_up, np.asarray(spare['base_pos_m'][:2]), m['xy'])
                    except RuntimeError as exc:
                        print(f"[remediation] spare at {np.round(spare['base_pos_m'][:2], 4)} for {m['slot']}: {exc}", flush=True)
                        continue
                    if spare is not m['spare']:
                        unused.remove(spare)
                        unused.append(m['spare'])
                    m.update(spare=spare, spare_xy=np.asarray(spare['base_pos_m'][:2]), plan=plan)
                    q_up = plan['place_missing']['retreat'][-1]
                    filled.append(m)
                    break
                else:
                    unfilled.append(m['slot'])
            missing = filled
            for c in crooked:
                if args.crooked_press == 'regrip':
                    P.solve(q_up, np.r_[c['xy'], transit_z], DOWN, SPREAD_Y, ledge)
                    c['press_pose'] = (DOWN, SPREAD_Y, ledge, False)
                else:
                    plan, *pose = plan_crooked_press(P, q_up, c['xy'], args.crooked_press, c['tilt'], tip, press_face)
                    c['plan'], c['press_pose'] = plan, tuple(pose)
                    q_up = plan['retreat_crooked'][-1]
        except BaseException as exc:
            failure = {'stage': 'trajectory_planning', 'scene': str(args.scene), 'error_type': type(exc).__name__,
                       'error': str(exc), 'targets': {'missing': [{k: m[k] for k in ('slot', 'detection', 'spare')} for m in missing],
                                                      'crooked': [{k: c[k] for k in ('slot', 'detection')} for c in crooked]}}
            (args.out / 'failure_report.json').write_text(json.dumps(failure, indent=2, default=str) + '\n')
            raise
        putdown_xy = None
        if args.crooked_press == 'regrip' and crooked:
            # Regrasp spot (reused for every crooked cap, which is picked back up from it):
            # the first spare's spot once the spares have been placed, else bare table beside it.
            if missing:
                putdown_xy = missing[0]['spare_xy']
            elif have_depth:
                near = np.asarray(choose_candidate(observed, 'table_cap')['base_pos_m'][:2])
                putdown_xy = free_table_spot(depth_img, capture, near, np.asarray(targets['tray_center_base_m']))
            else:
                raise RuntimeError('regrip needs a free table spot: no missing cap and no TableCam depth')
            targets['regrasp_table_xy'] = putdown_xy
            print(f'[remediation] regrasp table spot {np.round(putdown_xy, 4)}', flush=True)
        missing_slots, crooked_slots = [m['slot'] for m in missing], [c['slot'] for c in crooked]
        print(f"[remediation] tray centre {np.round(targets['tray_center_base_m'], 4)} "
              f"(fit error {targets['tray_fit_error_m']*1000:.1f} mm); missing={missing_slots}, crooked={crooked_slots}, "
              f"unfilled (no spare)={unfilled}", flush=True)

        for m in missing:
            m['prim'] = find_closest_cap_prim(stage, np.r_[m['spare_xy'], 0.])
        for c in crooked:
            c['prim'] = find_closest_cap_prim(stage, np.r_[c['xy'], 0.])
        caps = [x['prim'] for x in missing + crooked if x['prim'] is not None]
        meter = ContactMeter(stage, [str(c.GetPath()) for c in caps] +
                             [cell.JAW_PRIMS['L'], cell.JAW_PRIMS['R'], cell.FLANGE])
        world.reset()
        print(f"[remediation] gripper mounted; collisions {col_report}; "
              f"spares {[str(m['prim'].GetPath()) for m in missing]}, crooked {[str(c['prim'].GetPath()) for c in crooked]}",
              flush=True)

        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f'J{i}') for i in range(1, 7)]

        print(f'[remediation] all motions planned', flush=True)

        camera = Camera(cell.TABLE_CAM_PATH, resolution=(1280, 720))
        camera.initialize()
        wrist = wrist_camera(stage) if args.crooked_press == 'regrip' and crooked else None
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
                      tuple(missing_slots + crooked_slots), tracked=caps,
                      press_control=args.press_control,
                      eoat_paths=(cell.JAW_PRIMS['L'], cell.JAW_PRIMS['R'], cell.FLANGE))
        presses = {}

        jaw = C.JAW_OPEN
        for m in missing:
            slot, plan = m['slot'], m['plan']
            ex.jaws(f'open_for_spare_{slot}', jaw, C.JAW_OPEN)
            ex.run(f'approach_spare_{slot}', plan['approach_spare'], C.JAW_OPEN)
            ex.run('descend_around_spare', plan['descend_spare'], C.JAW_OPEN)
            ex.jaws('capture_spare_flange', C.JAW_OPEN, jaw_hold)
            ex.hold('capture_spare_flange', jaw_hold, 15)
            ex.run('lift_spare', plan['lift_spare'], jaw_hold)
            ex.run(f'carry_spare_to_{slot}', plan['carry_spare'], jaw_hold)
            presses[f'missing_{slot}'] = place_and_press(ex, plan['place_missing'], slot, jaw_grip, jaw_hold)
            jaw = jaw_grip
        if not missing:
            print('[remediation] no missing-cap detection on the tray; skipping the spare-cap placement', flush=True)
        ex.jaws('close_fingers', jaw, jaw_grip)

        def lift_clear(tool):
            """Straight up to transit height, keeping the current tool orientation (after a failed attempt)."""
            Rj, pj = fk.fk(ex.q)
            d, s = Rj @ np.asarray(C.TOOL_AXIS, float), Rj @ np.asarray(C.FINGER_SPREAD_AXIS, float)
            p = pj + Rj @ np.asarray(tool)
            ex.run('lift_clear', P.line(ex.q, p, np.r_[p[:2], max(p[2], transit_z)], d, s, tool, 60), C.JAW_OPEN)

        regrips = {}
        for c in crooked:
            crooked_slot, crooked_xy, crooked_prim = c['slot'], c['xy'], c['prim']
            if args.crooked_press == 'vertical':
                plan = c['plan']
                ex.run(f'transit_to_{crooked_slot}', plan['transit_to_crooked'], jaw_grip)
                ex.run(f'descend_to_{crooked_slot}', plan['descend_crooked'], jaw_grip)
                presses[f'crooked_{crooked_slot}'] = ex.press(f'vertical_press_{crooked_slot}', plan['press_crooked'],
                                                              jaw_grip, crooked_slot)
                ex.hold(f'vertical_press_{crooked_slot}', jaw_grip, 15)
                # Open the jaws so the pocket lets go of the seated flange, then lift.
                ex.jaws(f'release_{crooked_slot}', jaw_grip, C.JAW_OPEN)
                ex.run('retreat', plan['retreat_crooked'], C.JAW_OPEN)
                ex.jaws('close_fingers', C.JAW_OPEN, jaw_grip)
                continue
            if args.crooked_press == 'tilted':
                plan = c['plan']
                ex.jaws('close_for_edge_press', jaw_grip, C.JAW_CLOSED)
                ex.run(f'tilt_{PRESS_TILT_DEG:.0f}deg_over_{crooked_slot}', plan['tilt_to_crooked'], C.JAW_CLOSED)
                ex.run(f'descend_to_{crooked_slot}', plan['descend_crooked'], C.JAW_CLOSED)
                presses[f'crooked_{crooked_slot}'] = ex.press(f'edge_press_{crooked_slot}', plan['press_crooked'],
                                                              C.JAW_CLOSED, crooked_slot)
                ex.hold(f'edge_press_{crooked_slot}', C.JAW_CLOSED, 15)
                # Jaws stay closed: opening would drive the pressing back face further into the cap.
                ex.run('retreat', plan['retreat_crooked'], C.JAW_CLOSED)
                ex.jaws('reopen_to_grip', C.JAW_CLOSED, jaw_grip)
                continue

            servo_dir = args.out / 'wrist_servo' / crooked_slot
            servo_dir.mkdir(parents=True, exist_ok=True)
            ex.jaws('close_for_wrist_view', jaw_grip, C.JAW_CLOSED)

            def move(q_to):
                ex.run(f'wrist_servo_{crooked_slot}', s_curve_joints(ex.q, q_to, 60), C.JAW_CLOSED)

            def capture():
                ex.hold(f'wrist_servo_{crooked_slot}', C.JAW_CLOSED, 20)
                rgb, depth = read_wrist(wrist)
                return rgb, depth, arm.get_joint_positions()[arm_idx]

            def held_cap_offset():
                """Measured cap pose against the ledge tool point (diagnostic only)."""
                Rj, pj = fk.fk(arm.get_joint_positions()[arm_idx])
                pos, (w, x, y, z) = get_cap_pose(crooked_prim)
                held = np.asarray(pos) - np.array([0., 0., SPEC.world_base_z]) - (pj + Rj @ np.asarray(ledge))
                tilt = np.degrees(np.arccos(np.clip(ScipyRot.from_quat([x, y, z, w]).as_matrix()[2, 2], -1, 1)))
                return {'cap_minus_ledge_tool_point_m': held.tolist(), 'cap_tilt_deg': float(tilt)}

            regrip = regrips[crooked_slot] = {}
            moved = False
            try:
                axis, q_view, servo_report = servo_cap_axis(
                    P, ex.q, crooked_xy, wrist, move, capture, servo_dir)
                plan_r, info = plan_regrip(P, q_view, crooked_xy, axis, ledge, tip, putdown_xy)
                regrip.update(info, servo=servo_report)
                c['press_pose'] = (np.asarray(info['tool_axis']), np.asarray(info['spread_axis']), ledge, False)
                s_t = np.asarray(info['spread_axis'])
                moved = True
                ex.run(f'approach_along_cap_axis_{crooked_slot}', plan_r['approach'], C.JAW_CLOSED)
                ex.jaws(f'open_for_regrip_{crooked_slot}', C.JAW_CLOSED, C.JAW_OPEN)
                ex.run(f'descend_along_cap_axis_{crooked_slot}', plan_r['descend'], C.JAW_OPEN)
                ex.jaws(f'capture_crooked_flange_{crooked_slot}', C.JAW_OPEN, jaw_grip, n=30)
                ex.hold(f'capture_crooked_flange_{crooked_slot}', jaw_grip, 15)
                ex.run(f'lift_crooked_{crooked_slot}', plan_r['lift'], jaw_grip)
                ex.run(f'turn_cap_upright_{crooked_slot}', plan_r['upright'], jaw_grip)
                ex.hold(f'turn_cap_upright_{crooked_slot}', jaw_grip, 15)
                regrip['held_after_side_grip'] = held_cap_offset()
                print(f"[remediation] {crooked_slot} side-grip held cap: {regrip['held_after_side_grip']}", flush=True)
                # Regrasp on the table so the cap sits square in the pocket for placement.
                ex.run('carry_to_regrasp_spot', plan_r['carry_to_table'], jaw_grip)
                ex.run('stand_cap_on_table', plan_r['lower_to_table'], jaw_grip)
                ex.jaws('release_on_table', jaw_grip, C.JAW_OPEN)
                ex.hold('release_on_table', C.JAW_OPEN, 30)
                # Re-close in place on the measured (FK) release point; no camera view.
                Rj, pj = fk.fk(arm.get_joint_positions()[arm_idx])
                release_xy = (pj + Rj @ np.asarray(ledge))[:2]
                regrip['regrasp_release_xy_fk_base_m'] = release_xy.tolist()
                plan_p = plan_regrasp_and_place(P, ex.q, release_xy, crooked_xy, s_t, ledge, tip)
                ex.run('regrasp_rise_to_flange', plan_p['rise_to_flange'], C.JAW_OPEN)
                ex.jaws('regrasp_flange', C.JAW_OPEN, jaw_hold)
                ex.hold('regrasp_flange', jaw_hold, 15)
                ex.run('regrasp_lift', plan_p['lift'], jaw_hold)
                ex.run(f'carry_to_{crooked_slot}', plan_p['carry_to_vial'], jaw_hold)
                ex.hold(f'carry_to_{crooked_slot}', jaw_hold, 10)
                regrip['held_after_regrasp'] = held_cap_offset()
                print(f"[remediation] {crooked_slot} regrasped held cap: {regrip['held_after_regrasp']}", flush=True)
                presses[f'crooked_{crooked_slot}'] = place_and_press(ex, plan_p, crooked_slot, jaw_grip, jaw_hold)
            except RuntimeError as exc:
                # Report the failed regrip, clear the tray and go on to the next cap.
                regrip['error'] = str(exc)
                print(f'[remediation] {crooked_slot} regrip failed: {exc}', flush=True)
                if moved:
                    ex.jaws('release_after_failure', jaw_grip, C.JAW_OPEN)
                    try:
                        lift_clear(ledge)
                    except RuntimeError as lift_exc:
                        regrip['recovery_error'] = str(lift_exc)
                        print(f'[remediation] cannot lift clear ({lift_exc}); stopping the crooked caps', flush=True)
                        break
                ex.jaws('close_fingers', C.JAW_OPEN if moved else C.JAW_CLOSED, jaw_grip)
        ex.hold('settle', jaw_grip, 30)
        print(f'[remediation] wrote {ex.frames} frames; presses {presses}', flush=True)

        for _ in range(120):
            world.step(render=False)
        # Classify against the actual tube rims in the scene, not the perceived ones.
        def tube_rim(slot):
            from pxr import UsdGeom
            m = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(f'/World/RemediationTray/{slot}_tube'))
                           .ComputeLocalToWorldTransform(0.))
            return m[3, :3] + np.array([0., 0., SPEC.vial_height])
        seating = {}
        for c in crooked:
            seating[f"crooked_slot_cap_{c['slot']}"] = measure_cap(c['prim'], tube_rim(c['slot']), c['slot'])
        for m in missing:
            seating[f"missing_slot_cap_{m['slot']}"] = measure_cap(m['prim'], tube_rim(m['slot']), m['slot'])
        for k, v in seating.items():
            print(f'[remediation] {k}: {v}', flush=True)
        seated = sum(v['seated'] for v in seating.values())
        print(f'[remediation] seated {seated}/{len(seating)} remediated caps', flush=True)

        clean = lambda x: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items()
                           if k not in ('prim', 'plan', 'press_pose')}
        report = {
            'mode': 'physics_contact_remediation',
            'physical_robot_authorized': False,
            'scene': str(args.scene),
            'current_scene_detections': str(args.detections),
            'targets': {**{k: v for k, v in targets.items() if k not in ('missing', 'crooked', 'unfilled_missing_slots')},
                        'missing': [clean(m) for m in missing], 'crooked': [clean(c) for c in crooked]},
            'missing_slots': missing_slots,
            'crooked_slots': crooked_slots,
            'unfilled_missing_slots': unfilled,
            'tool_points_j6_m': {'pocket_ledge': list(ledge), 'fingertip_face': list(tip)},
            'crooked_regrip': regrips,
            'crooked_press': {'strategy': args.crooked_press,
                              'tilt_from_vertical_deg': PRESS_TILT_DEG if args.crooked_press == 'tilted' else 0.0,
                              'per_cap': {c['slot']: {'tool_axis': np.asarray(c['press_pose'][0]).tolist(),
                                                      'edge_direction': np.asarray(c['press_pose'][1]).tolist(),
                                                      'press_point_j6_m': list(c['press_pose'][2]),
                                                      'edge_aligned_with_cap_tilt': c['press_pose'][3],
                                                      'cap_tilt_from_depth': c['tilt']} for c in crooked},
                              'contact': ('coaxial flange grip, upright re-place, fingertip press' if args.crooked_press == 'regrip' else
                                          'closed fingertip crescent over the raised flange edge' if args.crooked_press == 'vertical' else
                                          f'-Y finger outer back face, {PRESS_FACE_UP_M*1000:.0f} mm up from the tip')},
            'cap_motion': 'PhysX contact only; no cap pose is written by the script',
            'press_force_limit_N': PRESS_FORCE_LIMIT_N,
            'press_control': args.press_control,
            'force_available_to_controller': args.press_control == 'force-limited',
            'presses': presses,
            'vial_breakability': {'axial_fracture_threshold_N': VIAL_AXIAL_FRACTURE_LIMIT_N,
                                  'lateral_fracture_threshold_N': VIAL_LATERAL_FRACTURE_LIMIT_N,
                                  'load_source': 'PhysX contact report impulses / dt',
                                  'vials': monitor.vial_states,
                                  'all_vials_intact': not any(v['fractured'] for v in monitor.vial_states.values())},
            'seating_check_after_120_settle_steps': seating,
            'seated_count': seated,
            'remediated_count': len(seating),
            'frame_directory': str(frame_dir),
            'frames_written': ex.frames,
        }
        (args.out / 'trace.json').write_text(json.dumps(ex.trace) + '\n')
        (args.out / 'attempt_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    except BaseException:
        # SimulationApp.close() terminates Kit with status zero on this cluster,
        # which would mask a policy failure.  Persist the failure and terminate
        # nonzero before close so Slurm and the pipeline cannot report success.
        import traceback
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'failure_report.json').write_text(json.dumps({
            'mode': args.controller_policy,
            'error': detail,
        }, indent=2) + '\n')
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == '__main__':
    main()
