#!/usr/bin/env python3
"""Servo the wrist camera onto a crooked cap's axis and score the result.

TableCam gives only the vial position (its lean estimate proved unreliable).  The
first wrist view looks straight down over the vial; after that the camera is
pointed down the current estimate of the cap axis.  Each wrist image is searched for the cap's dark open top;
the fitted ellipse gives two candidate plane normals (wrist_servo.circle_normals),
the one closest to the current estimate becomes the new estimate, and the wrist is
turned to it.  Depth on the flange ring picks between the two candidates.  It stops
at the first view where the opening looks round and reports that camera axis.  The true axis is read from
PhysX only to score each iteration; it never feeds back into the estimate.

    sbatch scripts/submit_isaac_job.slurm src/remediation/servo_crooked_cap.py \\
        --scene <scene.usda> --perception <run>/perception --out shared/runs/<run>_wrist_servo
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'data/fanuc_er4ia_sim'), str(ROOT / 'shared/data/fanuc_er4ia_sim')]

VIEW_DISTANCE_M = 0.15     # camera to cap opening along the estimated axis
MAX_ITERATIONS = 6
ROUND_ENOUGH = 0.995       # minor/major ratio treated as round
MOVE_STEPS = 60
UPRIGHT_ROUNDNESS = 0.95   # straight-down view of a standing cap (cos 18 deg)


def wrist_camera(stage, seed=0):
    """The wrist camera as the real cell has it (wrist_camera_model): the colour camera
    and a separate depth camera on J6_link at their calibrated poses, rendering ideal
    pinhole images that read_wrist turns into the cell's colour and depth streams.
    K (undistorted colour pixels) and the colour camera's pose in the J6 frame (CV
    convention: x right, y down, z forward) are what the servo maths uses."""
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, UsdGeom
    import cell
    from src.remediation import wrist_camera_model as M
    cams = {}
    for name, path, T in (('colour', cell.WRIST_CAM_PATH, M.T_FLANGE_COLOUR),
                          ('depth', cell.WRIST_CAM_PATH.rsplit('/', 1)[0] + '/WristDepthCam', M.T_FLANGE_DEPTH)):
        W, H, fx = M.render_spec()[name]
        usd = UsdGeom.Camera.Define(stage, path)
        # USD cameras look down -z with +y up; rows of a USD matrix are the local axes.
        R_usd = T[:3, :3] @ np.diag([1., -1., -1.])
        M4 = np.eye(4)
        M4[:3, :3], M4[3, :3] = R_usd.T, T[:3, 3]
        UsdGeom.Xformable(usd).MakeMatrixXform().Set(Gf.Matrix4d(M4.tolist()))
        focal = usd.GetFocalLengthAttr().Get() or 10.0
        usd.GetFocalLengthAttr().Set(focal)
        usd.GetHorizontalApertureAttr().Set(W * focal / fx)
        usd.GetVerticalApertureAttr().Set(H * focal / fx)
        usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))
        cam = Camera(path, resolution=(W, H))
        cam.initialize()
        if name == 'depth':
            cam.add_distance_to_image_plane_to_frame()
        cams[name] = cam
    return {'camera': cams['colour'], 'depth_camera': cams['depth'], 'rng': np.random.default_rng(seed),
            'K': M.colour_K(), 'R_cam_j6': M.T_FLANGE_COLOUR[:3, :3], 't_cam_j6': M.T_FLANGE_COLOUR[:3, 3]}


def read_wrist(cam):
    """(rgb, depth) as the cell serves them: 1280 x 720 colour and 848 x 480 uint16 mm depth."""
    from src.remediation import wrist_camera_model as M
    rgb = M.real_colour(np.ascontiguousarray(cam['camera'].get_rgba()[:, :, :3]))
    depth = M.real_depth(np.asarray(cam['depth_camera'].get_depth(device='cpu'), np.float32), cam['rng'])
    return rgb, depth


def expected_px(cam, R_cam_w, eye_w, point):
    """Served colour pixel of a base-frame point."""
    from src.remediation.wrist_camera_model import distort_px
    p = cam['K'] @ (R_cam_w.T @ (point - eye_w))
    return distort_px(p[:2] / p[2])[0]


def draw_ellipse(img, ellipse, colour=(0, 255, 0)):
    """Draw an ellipse fitted in undistorted pixels onto the served image."""
    from src.remediation.wrist_camera_model import distort_px
    (cx, cy), (w, h), ang = ellipse
    pts = cv2.ellipse2Poly((int(round(cx)), int(round(cy))), (int(round(w / 2)), int(round(h / 2))), int(round(ang)), 0, 360, 4)
    cv2.polylines(img, [np.rint(distort_px(pts)).astype(np.int32)], True, colour, 2)


def servo_cap_axis(P, q, bore, cam, move, capture, out_dir, truth=None):
    """Turn the wrist camera onto the crooked cap's axis at vial `bore` (base XY).

    `move(q_to)` drives the arm to a joint pose, `capture()` returns (rgb, depth,
    q_actual) from the wrist camera, `truth()` (optional, scoring only) the true
    cap axis.  Returns (axis estimate in the base frame, joints at the last view,
    report).  Raises RuntimeError if no estimate could be made.
    """
    import fk
    from src.remediation.geometry import CROOKED_TILT_RANGE_DEG, SPEC, leaning_cap_pose, rim_base_z
    from src.remediation.wrist_camera_model import register_depth, undistort_px
    from src.remediation.wrist_servo import (choose_normal, circle_normals, depth_rise_direction, find_cap_opening,
                                             look_rotation, roundness)
    K, R_cam_j6, t_cam_j6 = cam['K'], cam['R_cam_j6'], cam['t_cam_j6']
    # First look straight down from above the vial: needs no lean estimate (TableCam's
    # heading proved unreliable).  The opening of a leaning cap sits ~10-20 mm above the rim.
    n_est = np.array([0., 0., 1.])
    opening = np.r_[bore, rim_base_z() + 0.015]
    report = {'bore_base_m': list(map(float, bore)), 'first_view': 'straight down over the vial', 'iterations': []}
    angle = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))

    def view_pose(n, distance, hint):
        """Joint solution putting the camera `distance` up the axis `n`, looking down it."""
        R_j6 = look_rotation(-n, up_hint=hint) @ R_cam_j6.T
        return P.solve(q, opening + n * distance, R_j6[:, 0], R_j6[:, 1], tuple(t_cam_j6))

    def view_options(n, first):
        # Image up = world down keeps the fingers above the line of sight.  Looking
        # straight down that is undefined, so try image-up headings around the vial.
        if abs(n[2]) < 0.95:
            hints = [(0., 0., -1.)]
        else:
            out = np.r_[bore / np.linalg.norm(bore), 0.]
            side = np.cross([0., 0., 1.], out)
            hints = [-out, out, side, -side]
        dists = (0.18, 0.15, 0.22, 0.26) if first else (VIEW_DISTANCE_M, 0.20, 0.25)
        return [(d, h) for d in dists for h in hints]

    n_prev = n_est.copy()
    for it in range(MAX_ITERATIONS):
        q_view = None
        for dist, hint in view_options(n_est, it == 0):
            try:
                q_view = view_pose(n_est, dist, hint)
                break
            except RuntimeError:
                continue
        if q_view is None:
            report['iterations'].append({'iteration': it, 'error': 'view pose unreachable', 'estimate': n_est.tolist()})
            break
        move(q_view)
        q = q_view
        rgb, depth, q_act = capture()
        cv2.imwrite(str(out_dir / f'raw_{it}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(out_dir / f'depth_{it}.npy', depth)
        Rj, pj = fk.fk(q_act)
        R_cam_w = Rj @ R_cam_j6
        eye_w = pj + Rj @ t_cam_j6
        ellipse, ring = find_cap_opening(rgb, expected_px(cam, R_cam_w, eye_w, opening), to_pinhole=undistort_px)
        entry = {'iteration': it, 'camera_axis': (-R_cam_w[:, 2]).tolist()}
        t_axis = truth() if truth else None
        if t_axis is not None:
            entry.update({'camera_axis_error_vs_true_deg': angle(-R_cam_w[:, 2], t_axis),
                          'estimate_error_vs_true_deg': angle(n_est, t_axis)})
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ellipse is None:
            entry['error'] = 'cap opening not found'
            report['iterations'].append(entry)
            cv2.imwrite(str(out_dir / f'iter_{it}.png'), img)
            print(f'[servo] iteration {it}: {entry}', flush=True)
            break
        cands_cam = circle_normals(ellipse, K)
        n_cam = choose_normal(cands_cam, depth_rise_direction(register_depth(depth), ring))
        if n_cam is None:                      # no depth: fall back to the prior side
            n_cam = max(cands_cam, key=lambda c: (R_cam_w @ c) @ n_prev)
        n_new = R_cam_w @ n_cam
        entry.update({'new_estimate': n_new.tolist(), 'roundness': roundness(ellipse),
                      'ellipse': [list(ellipse[0]), list(ellipse[1]), ellipse[2]], 'step_deg': angle(n_new, n_est)})
        if t_axis is not None:
            others = [R_cam_w @ c for c in cands_cam if not np.allclose(c, n_cam)]
            entry.update({'new_estimate_error_vs_true_deg': angle(n_new, t_axis),
                          'other_candidate_error_vs_true_deg': min(angle(c, t_axis) for c in others) if others else None})
        img[ring > 0] = (0.6 * img[ring > 0] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
        draw_ellipse(img, ellipse)
        label = f'iter {it}: roundness {entry["roundness"]:.3f}'
        if t_axis is not None:
            label += f', camera axis {entry["camera_axis_error_vs_true_deg"]:.1f} deg off true'
        cv2.putText(img, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f'iter_{it}.png'), img)
        report['iterations'].append(entry)
        print(f'[servo] iteration {it}: {entry}', flush=True)
        n_prev, n_est = n_est, n_new
        # Opening centre for the next view: a leaning cap with the estimated axis.
        tilt_e = float(np.degrees(np.arccos(np.clip(n_est[2], -1, 1))))
        if CROOKED_TILT_RANGE_DEG[0] - 10 <= tilt_e <= CROOKED_TILT_RANGE_DEG[1] + 10:
            pc, Rc = leaning_cap_from_axis(bore, n_est)
            opening = pc + Rc @ np.array([0., 0., SPEC.flange_thickness])
        # A near-round ellipse gives a noise-limited estimate (it can be several
        # degrees off), so stop at the first round view and keep that camera axis.
        if entry['roundness'] >= ROUND_ENOUGH:
            break
    if not any('new_estimate' in e for e in report['iterations']):
        raise RuntimeError(f'wrist servo made no cap-axis estimate: {report["iterations"]}')
    # Once the opening looks round, the camera axis that produced that view is the
    # best estimate (a single near-round ellipse is noise-limited to a few degrees).
    round_axes = [np.asarray(e['camera_axis']) for e in report['iterations'] if e.get('roundness', 0) >= ROUND_ENOUGH]
    if round_axes:
        n_est = round_axes[0]
    report['final'] = {'estimate': n_est.tolist(), 'estimate_tilt_deg': float(np.degrees(np.arccos(np.clip(n_est[2], -1, 1))))}
    t_axis = truth() if truth else None
    if t_axis is not None:
        report['final'].update({'true_axis': t_axis.tolist(), 'error_deg': angle(n_est, t_axis),
                                'true_tilt_deg': float(np.degrees(np.arccos(np.clip(t_axis[2], -1, 1))))})
    print(f"[servo] final: {report['final']}", flush=True)
    return n_est, q, report


def locate_upright_cap(P, q, xy_guess, z_top, cam, move, capture, out_path=None):
    """Base XY of an upright cap whose flange top is at height `z_top`, from one
    straight-down wrist view: the centre of the fitted opening ellipse is back-projected
    onto the flange-top plane.  Returns (xy, joints at the view)."""
    import fk
    from src.remediation.wrist_camera_model import undistort_px
    from src.remediation.wrist_servo import find_cap_opening, look_rotation, roundness
    K, R_cam_j6, t_cam_j6 = cam['K'], cam['R_cam_j6'], cam['t_cam_j6']
    target = np.r_[xy_guess, z_top]
    out = np.r_[np.asarray(xy_guess) / np.linalg.norm(xy_guess), 0.]
    side = np.cross([0., 0., 1.], out)
    q_view = None
    for dist in (0.15, 0.18, 0.22):
        for hint in (-out, out, side, -side):
            R_j6 = look_rotation([0., 0., -1.], up_hint=hint) @ R_cam_j6.T
            try:
                q_view = P.solve(q, target + np.array([0., 0., dist]), R_j6[:, 0], R_j6[:, 1], tuple(t_cam_j6))
                break
            except RuntimeError:
                continue
        if q_view is not None:
            break
    if q_view is None:
        raise RuntimeError(f'no straight-down wrist view of {np.round(target, 4)}')
    move(q_view)
    rgb, _, q_act = capture()
    Rj, pj = fk.fk(q_act)
    R_cam_w, eye = Rj @ R_cam_j6, pj + Rj @ t_cam_j6
    ellipse, _ = find_cap_opening(rgb, expected_px(cam, R_cam_w, eye, target), to_pinhole=undistort_px)
    if out_path is not None:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ellipse is not None:
            draw_ellipse(img, ellipse)
        cv2.imwrite(str(out_path), img)
    if ellipse is None:
        raise RuntimeError(f'upright cap not found in the wrist view near {np.round(target, 4)}')
    if roundness(ellipse) < UPRIGHT_ROUNDNESS:
        # Seen straight down, a standing cap's opening is round; an ellipse means it fell over.
        raise RuntimeError(f'cap on the table is not upright (opening roundness {roundness(ellipse):.2f})')
    ray = R_cam_w @ (np.linalg.inv(K) @ np.r_[ellipse[0], 1.])
    hit = eye + ray * (z_top - eye[2]) / ray[2]
    return hit[:2], q_view


def leaning_cap_from_axis(bore, n):
    """Pose (origin, rotation) of a cap leaning on the rim of vial `bore` with axis `n`
    (geometry.leaning_cap_pose, tilt clipped to CROOKED_TILT_RANGE_DEG)."""
    from src.remediation.geometry import CROOKED_TILT_RANGE_DEG, leaning_cap_pose, rim_base_z
    tilt = float(np.clip(np.degrees(np.arccos(np.clip(n[2], -1, 1))), *CROOKED_TILT_RANGE_DEG))
    high = -np.asarray(n[:2]) / np.linalg.norm(n[:2])
    # rotation(tilt, yaw) puts the cap's high side (local +Y) at azimuth yaw + 90 deg.
    return leaning_cap_pose(np.r_[bore, rim_base_z()], tilt, np.degrees(np.arctan2(high[1], high[0])) - 90.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--perception', type=Path, required=True,
                   help='perception/ dir with current_scene_detections.json, depth and capture_report.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from src.remediation.geometry import load_layout, transit_base_z
    from src.remediation import run_compliant_breakable_remediation as R

    det = json.loads((args.perception / 'current_scene_detections.json').read_text())
    layout = load_layout()
    centre, _ = R.fit_tray_center(det['targets'], layout)
    crooked = R.choose_candidate(R.on_tray(det['targets'], 'crooked_cap', centre, layout), 'crooked_cap')
    grid = {s_: centre + np.array(v[:2]) for s_, v in layout['slots_m'].items()}
    slot = min(grid, key=lambda s_: np.linalg.norm(grid[s_] - np.asarray(crooked['base_pos_m'][:2])))
    bore = grid[slot]
    print(f'[servo] crooked cap at {slot} {np.round(bore, 4)}; first view straight down', flush=True)

    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from scipy.spatial.transform import Rotation as Rot
        import scene_config as C
        from src.remediation.custom_gripper import attach_collision_proxies, attach_visual_pair

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0, physics_dt=R.DT, rendering_dt=R.DT)
        mount = attach_visual_pair(stage)
        attach_collision_proxies(stage)
        R.configure_stage_collisions(stage)
        cap_prim = R.find_closest_cap_prim(stage, np.r_[bore, 0.])
        world.reset()
        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        names = list(arm.dof_names)
        idx = [names.index(f'J{i}') for i in range(1, 7)]
        cam = wrist_camera(stage)

        P = R.Planner()
        ledge = tuple(mount['cap_seat_tcp_j6_m'])
        q = P.solve(np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)]),
                    np.array([.24, .25, transit_base_z()]), R.DOWN, R.SPREAD_Y, ledge)
        pos = R.set_jaw_separation(arm, names, C.JAW_CLOSED)
        pos[idx] = q
        arm.set_joint_positions(pos)
        for _ in range(30):
            world.step(render=True)
        state = {'q': q}

        def move(q_to):
            for qq in R.s_curve_joints(state['q'], q_to, MOVE_STEPS):
                pp = R.set_jaw_separation(arm, names, C.JAW_CLOSED)
                pp[idx] = qq
                arm.apply_action(ArticulationAction(joint_positions=pp))
                world.step(render=True)
            state['q'] = q_to

        def capture():
            for _ in range(20):
                world.step(render=True)
            rgb, depth = read_wrist(cam)
            return rgb, depth, arm.get_joint_positions()[idx]

        def true_axis():
            _, (w, x, y, z) = R.get_cap_pose(cap_prim)
            return Rot.from_quat([x, y, z, w]).as_matrix()[:, 2]

        try:
            _, _, report = servo_cap_axis(P, q, bore, cam, move, capture, args.out, truth=true_axis)
        except RuntimeError as exc:
            report = {'error': str(exc)}
        report.update({'scene': str(args.scene), 'slot': slot})
        (args.out / 'servo_report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        app.close()


if __name__ == '__main__':
    main()
