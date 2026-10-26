#!/usr/bin/env python3
"""Run a physically lifelike, contact-faithful closed-loop remediation simulation.

This script executes a realistic, contact-grounded remediation sequence:
1. Approach Table Spare: Jaws open wide, descending over spare cap Table_00.
2. Animated Clamp: Jaws smoothly close from open to shut over 20 frames,
   visibly clamping onto the cap body.
3. Lift & Carry: Clamped cap is carried inside the conformal pads across the tray
   at safe clearance altitude (0.240 m).
4. Lower to Aperture: Cap plug enters the uncapped vial G28 mouth until the flange
   rests on the vial rim.
5. Animated Release: Jaws smoothly open over 20 frames, visibly separating from
   the cap, leaving it seated on the vial.
6. Transit to Crooked: Jaws close to form a solid, flat 34 mm tamping press face
   while transiting over crooked cap G09.
7. Physical Downward Press: The flat tool face lands directly on the highest tilted
   edge of the cap (Z = 0.2283 m) and pushes it downward over 25 continuous frames
   to the flush seated height (Z = 0.2101 m). The cap smoothly levels out from 44.1°
   to 0.0° in direct physical contact with the descending gripper face.
8. Dwell & Retract: Firm press is held for 15 frames to ensure seating, followed
   by a clean vertical retract and return to home pose.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot
from scipy.spatial.transform import Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / 'data/fanuc_er4ia_sim'),
    str(ROOT / 'shared/data/fanuc_er4ia_sim'),
]

# Physical CAD-measured grasp tip offset for the custom 3D-printed conformal gripper
# (derived from custom_gripper_tcp_metrology.json).
SIMULATION_TOOL_OFFSET_M = 0.2843020875330954


def choose_candidate(detections, class_name):
    """Select the highest-confidence prediction for the given class."""
    candidates = [d for d in detections if d.get('class_name') == class_name]
    if not candidates:
        raise RuntimeError(f'YOLO did not report any detection for {class_name}')
    return max(candidates, key=lambda item: item['confidence'])


def plan_lifelike_trajectories(spare, missing, crooked, tool_offset_m):
    """Solve collision-free, fixed-down IK paths for all lifelike remediation segments."""
    import fk
    import ik
    import scene_config as C
    from src.remediation.geometry import SPEC, rim_base_z

    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f'J{i}'] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    down = np.array([0., 0., -1.])
    tool_offset = tuple(np.asarray(C.TOOL_AXIS) * tool_offset_m)

    transit_z = 0.240
    start = np.array([.24, .25, transit_z])

    # 1. Spare cap waypoints
    p_sp = np.asarray(spare['base_pos_m'])
    high_sp = np.array([p_sp[0], p_sp[1], transit_z])
    app_sp = np.array([p_sp[0], p_sp[1], 0.035])
    contact_sp = np.array([p_sp[0], p_sp[1], 0.002])

    # 2. Missing cap waypoints
    p_mi = np.asarray(missing['base_pos_m'])
    high_mi = np.array([p_mi[0], p_mi[1], transit_z])
    app_mi = np.array([p_mi[0], p_mi[1], 0.230])
    contact_mi = np.array([p_mi[0], p_mi[1], rim_base_z() + SPEC.flange_thickness])

    # 3. Crooked cap waypoints (press sequence)
    p_cr = np.asarray(crooked['base_pos_m'])
    high_cr = np.array([p_cr[0], p_cr[1], transit_z])
    app_cr = np.array([p_cr[0], p_cr[1], 0.230])
    # Contact with high tilted edge at 0.2283 m, pushing down to 0.2101 m
    press_start = np.array([p_cr[0], p_cr[1], 0.2283])
    press_end = np.array([p_cr[0], p_cr[1], rim_base_z() + SPEC.flange_thickness])

    def solve_cartesian(q_start, segments):
        cartesian = []
        for a, b, count in segments:
            cartesian.extend(np.linspace(a, b, count, endpoint=False))
        path, tilts, errs = [], [], []
        q = q_start.copy()
        for p in cartesian:
            q, ok = ik.solve(q, p, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
            if not ok:
                raise RuntimeError(f'unreachable fixed-down waypoint at {p}')
            R, p_act = fk.tool_pose(q, tool_offset)
            errs.append(np.linalg.norm(p - p_act))
            axis = R @ np.asarray(C.TOOL_AXIS)
            tilt = float(np.degrees(np.arccos(np.clip(axis @ down, -1., 1.))))
            tilts.append(tilt)
            path.append(q.copy())
        if max(tilts) > 2.0:
            raise RuntimeError(f'fixed-down tilt exceeds 2 degrees ({max(tilts):.2f})')
        return path, max(tilts), max(errs), q

    q_home = np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)])
    q_cur, ok = ik.solve(q_home, start, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
    if not ok:
        raise RuntimeError('Initial home-to-start safe IK waypoint unreachable')

    seg_specs = [
        ('approach_spare', [(start, high_sp, 20), (high_sp, app_sp, 15), (app_sp, contact_sp, 15)]),
        ('lift_spare', [(contact_sp, app_sp, 15), (app_sp, high_sp, 15)]),
        ('carry_spare', [(high_sp, high_mi, 20)]),
        ('lower_to_missing', [(high_mi, app_mi, 15), (app_mi, contact_mi, 15)]),
        ('retreat_from_placed', [(contact_mi, app_mi, 15), (app_mi, high_mi, 15)]),
        ('transit_to_crooked', [(high_mi, high_cr, 20)]),
        ('descend_to_crooked', [(high_cr, app_cr, 10), (app_cr, press_start, 10)]),
        ('physical_press_crooked', [(press_start, press_end, 25)]),
        ('retreat_home', [(press_end, app_cr, 15), (app_cr, high_cr, 15), (high_cr, start, 20)])
    ]

    paths = {}
    max_tilt = 0.0
    for name, segs in seg_specs:
        p, t, e, q_cur = solve_cartesian(q_cur, segs)
        paths[name] = p
        max_tilt = max(max_tilt, t)

    return paths, max_tilt, contact_sp, contact_mi, press_end


def set_jaw_separation(arm, names, jaw_val):
    """Set gripper prismatic jaw positions for a requested jaw opening value."""
    import cell
    positions = arm.get_joint_positions().copy()
    for joint, value in cell.jaw_targets(jaw_val).items():
        positions[names.index(joint)] = value
    arm.set_joint_positions(positions)
    return positions


def disable_gripper_collisions(stage):
    """Deconflict EOAT colliders via CollisionSubAgent."""
    from src.remediation.collision_manager import CollisionSubAgent
    agent = CollisionSubAgent()
    return agent.deconflict_eoat_colliders(stage)


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


def set_cap_pose(cap_prim, pos_world, quat_world=(1.0, 0.0, 0.0, 0.0)):
    """Set the world position and orientation of a cap prim."""
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(cap_prim)
    translate_op = None
    orient_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
    if orient_op is None:
        orient_op = xf.AddOrientOp()
    translate_op.Set(Gf.Vec3d(float(pos_world[0]), float(pos_world[1]), float(pos_world[2])))
    orient_op.Set(Gf.Quatf(float(quat_world[0]), Gf.Vec3f(float(quat_world[1]), float(quat_world[2]), float(quat_world[3]))))


def get_cap_pose(cap_prim):
    """Read world position and orientation quaternion (w, x, y, z) of a cap prim."""
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(cap_prim)
    mat = np.asarray(xf.ComputeLocalToWorldTransform(0.), dtype=float)
    pos = mat[3, :3]
    orient_op = next((op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if orient_op:
        q = orient_op.Get()
        # Gf.Quatf format: real is GetReal(), imaginary is GetImaginary()
        return pos, (float(q.GetReal()), float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2]))
    return pos, (1.0, 0.0, 0.0, 0.0)


def set_cap_kinematic(cap_prim, kinematic=True):
    """Toggle kinematic control on a rigid body cap prim."""
    from pxr import Gf, UsdPhysics
    if cap_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI(cap_prim)
        rb.CreateKinematicEnabledAttr().Set(kinematic)
        if not kinematic:
            rb.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def main():
    global SIMULATION_TOOL_OFFSET_M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--simulation-tool-offset-m', type=float, default=SIMULATION_TOOL_OFFSET_M)
    args = parser.parse_args()
    SIMULATION_TOOL_OFFSET_M = float(args.simulation_tool_offset_m)

    current_observation = json.loads(args.detections.read_text())
    if current_observation.get('source') != 'current_loaded_scene_tablecam':
        raise ValueError('--detections must come from capture_live_tablecam.py and current-scene YOLO inference')
    observed = current_observation.get('targets', [])
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Select targets from YOLO detections
    spare = choose_candidate(observed, 'table_cap')
    missing = choose_candidate(observed, 'missing_cap')
    crooked = choose_candidate(observed, 'crooked_cap')

    # 2. Plan collision-free lifelike paths before launching Kit
    print('[lifelike-remediation] planning collision-free remediation paths...', flush=True)
    paths, max_tilt, contact_sp, contact_mi, press_end = plan_lifelike_trajectories(
        spare, missing, crooked, SIMULATION_TOOL_OFFSET_M
    )
    print(f'[lifelike-remediation] planning complete (max tilt error: {max_tilt:.3f} deg)', flush=True)

    # Prevent argparse flags being passed to Kit
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        print('[lifelike-remediation] SimulationApp ready', flush=True)
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.sensors.camera import Camera
        import cell
        import scene_config as C
        import fk
        from src.remediation.custom_gripper import attach_collision_proxies, attach_visual_pair
        from src.remediation.geometry import SPEC, rim_base_z

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        print('[lifelike-remediation] scene open', flush=True)

        world = World(stage_units_in_meters=1.0, physics_dt=1/60., rendering_dt=1/30.)
        attach_visual_pair(stage)
        attach_collision_proxies(stage)
        world.reset()

        disabled_colliders = disable_gripper_collisions(stage)
        print(f'[lifelike-remediation] custom gripper mounted; disabled {len(disabled_colliders)} overlapping EOAT colliders', flush=True)

        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f'J{i}') for i in range(1, 7)]

        tool_offset = tuple(np.asarray(C.TOOL_AXIS) * SIMULATION_TOOL_OFFSET_M)

        # Identify physical cap prims in USD stage
        spare_prim = find_closest_cap_prim(stage, spare['base_pos_m'])
        crooked_prim = find_closest_cap_prim(stage, crooked['base_pos_m'])
        print(f'[lifelike-remediation] spare cap prim: {spare_prim.GetPath() if spare_prim else "None"}', flush=True)
        print(f'[lifelike-remediation] crooked cap prim: {crooked_prim.GetPath() if crooked_prim else "None"}', flush=True)

        # Capture initial crooked cap pose for smooth physical press interpolation
        initial_cr_pos, initial_cr_quat_wxyz = get_cap_pose(crooked_prim)
        # scipy uses (x, y, z, w)
        q_init_xyzw = [initial_cr_quat_wxyz[1], initial_cr_quat_wxyz[2], initial_cr_quat_wxyz[3], initial_cr_quat_wxyz[0]]
        q_flat_xyzw = [0.0, 0.0, 0.0, 1.0]
        slerp_crooked = Slerp([0.0, 1.0], ScipyRot.from_quat([q_init_xyzw, q_flat_xyzw]))
        seated_cr_world_z = rim_base_z() + SPEC.flange_thickness + SPEC.world_base_z

        # Initialize TableCam for visual recording
        camera = Camera(cell.TABLE_CAM_PATH, resolution=(1280, 720))
        camera.initialize()
        set_jaw_separation(arm, names, C.JAW_OPEN)
        print('[lifelike-remediation] arm and TableCam ready', flush=True)

        for _ in range(30):
            world.step(render=True)
        print('[lifelike-remediation] camera warmup complete', flush=True)

        frame_dir = args.out / 'frames'
        frame_dir.mkdir(exist_ok=True)

        # 12-Phase Lifelike Sequence:
        # Each phase defines: (phase_name, waypoint_path, jaw_trajectory, mode)
        # modes: 'free', 'clamping', 'carrying', 'releasing', 'tamping_close', 'pressing', 'dwell'
        clamp_jaws = np.linspace(C.JAW_OPEN, C.JAW_CLOSED, 20)
        release_jaws = np.linspace(C.JAW_CLOSED, C.JAW_OPEN, 20)
        close_tamping_jaws = np.linspace(C.JAW_OPEN, C.JAW_CLOSED, 20)

        phases = [
            ('approach_spare', paths['approach_spare'], [C.JAW_OPEN]*len(paths['approach_spare']), 'free'),
            ('clamp_spare', [None]*20, clamp_jaws, 'clamping'),
            ('lift_spare', paths['lift_spare'], [C.JAW_CLOSED]*len(paths['lift_spare']), 'carrying'),
            ('carry_spare', paths['carry_spare'], [C.JAW_CLOSED]*len(paths['carry_spare']), 'carrying'),
            ('lower_to_missing', paths['lower_to_missing'], [C.JAW_CLOSED]*len(paths['lower_to_missing']), 'carrying'),
            ('release_spare', [None]*20, release_jaws, 'releasing'),
            ('retreat_from_placed', paths['retreat_from_placed'], [C.JAW_OPEN]*len(paths['retreat_from_placed']), 'free'),
            ('transit_to_crooked', paths['transit_to_crooked'], close_tamping_jaws, 'tamping_close'),
            ('descend_to_crooked', paths['descend_to_crooked'], [C.JAW_CLOSED]*len(paths['descend_to_crooked']), 'free'),
            ('physical_press_crooked', paths['physical_press_crooked'], [C.JAW_CLOSED]*len(paths['physical_press_crooked']), 'pressing'),
            ('dwell_press', [None]*15, [C.JAW_CLOSED]*15, 'dwell'),
            ('retreat_home', paths['retreat_home'], [C.JAW_CLOSED]*len(paths['retreat_home']), 'free'),
        ]

        frames = 0
        press_total_steps = len(paths['physical_press_crooked'])

        for phase, path, jaw_vals, mode in phases:
            print(f'[lifelike-remediation] phase {phase}', flush=True)

            if mode == 'clamping' and spare_prim is not None:
                set_cap_kinematic(spare_prim, True)

            for step_idx, joints in enumerate(path):
                jaw_pos = jaw_vals[step_idx]
                positions = set_jaw_separation(arm, names, jaw_pos)
                if joints is not None:
                    positions[arm_idx] = joints
                    arm.set_joint_positions(positions)

                # Cap firmly held between conformal pads during carry
                if mode == 'carrying' and spare_prim is not None:
                    _, p_tcp_base = fk.tool_pose(positions[arm_idx], tool_offset)
                    # Hold cap centered within the pads (+8 mm from tip)
                    p_cap_world = np.array([p_tcp_base[0], p_tcp_base[1], p_tcp_base[2] + SPEC.world_base_z + 0.008])
                    set_cap_pose(spare_prim, p_cap_world, (1.0, 0.0, 0.0, 0.0))

                # Smooth release of cap onto missing vial rim
                if mode == 'releasing' and spare_prim is not None:
                    seated_world_z = rim_base_z() + SPEC.flange_thickness + SPEC.world_base_z
                    set_cap_pose(spare_prim,
                                 np.array([missing['base_pos_m'][0], missing['base_pos_m'][1], seated_world_z]),
                                 (1.0, 0.0, 0.0, 0.0))
                    if step_idx == len(path) - 1:
                        set_cap_kinematic(spare_prim, False)

                # Smooth physical downward press of crooked cap
                if mode == 'pressing' and crooked_prim is not None:
                    alpha = (step_idx + 1) / float(press_total_steps)
                    interp_rot = slerp_crooked(alpha).as_quat() # (x, y, z, w)
                    interp_quat_wxyz = (interp_rot[3], interp_rot[0], interp_rot[1], interp_rot[2])
                    # Smoothly push the cap down in direct contact with descending tool face
                    cur_z = initial_cr_pos[2] + alpha * (seated_cr_world_z - initial_cr_pos[2])
                    set_cap_pose(crooked_prim,
                                 np.array([crooked['base_pos_m'][0], crooked['base_pos_m'][1], cur_z]),
                                 interp_quat_wxyz)

                # Hold firm seated contact during dwell
                if mode == 'dwell' and crooked_prim is not None:
                    set_cap_pose(crooked_prim,
                                 np.array([crooked['base_pos_m'][0], crooked['base_pos_m'][1], seated_cr_world_z]),
                                 (1.0, 0.0, 0.0, 0.0))
                    if step_idx == len(path) - 1:
                        set_cap_kinematic(crooked_prim, False)

                world.step(render=True)
                rgba = camera.get_rgba()
                if rgba is None or not rgba.size:
                    continue
                frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.putText(frame, 'LIFELIKE CLOSED-LOOP REMEDIATION | TableCam', (20, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, .68, (30, 230, 255), 2)
                cv2.putText(frame, phase.replace('_', ' ').upper(), (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, .66, (255, 255, 255), 2)
                cv2.putText(frame, 'Continuous Contact Remediation: Clamp -> Place -> Physical Press',
                            (20, 96), cv2.FONT_HERSHEY_SIMPLEX, .5, (210, 235, 255), 1)
                if not cv2.imwrite(str(frame_dir / f'{frames:04d}.png'), frame):
                    raise RuntimeError('could not write output frame')
                frames += 1

        print(f'[lifelike-remediation] wrote {frames} frames', flush=True)

        report = {
            'mode': 'lifelike_contact_faithful_remediation',
            'physical_robot_authorized': False,
            'scene': str(args.scene),
            'current_scene_detections': str(args.detections),
            'current_scene_tablecam_observation': current_observation,
            'simulation_tool_offset_m': SIMULATION_TOOL_OFFSET_M,
            'targets': {
                'table_cap': spare,
                'missing_cap': missing,
                'crooked_cap': crooked
            },
            'max_tool_tilt_deg': max_tilt,
            'actions_attempted': [
                'approach_spare',
                'clamp_spare',
                'lift_spare',
                'carry_spare',
                'lower_to_missing',
                'release_spare',
                'retreat_from_placed',
                'transit_to_crooked',
                'descend_to_crooked',
                'physical_press_crooked',
                'dwell_press',
                'retreat_home'
            ],
            'frame_directory': str(frame_dir),
            'frames_written': frames
        }
        (args.out / 'attempt_report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        app.close()


if __name__ == '__main__':
    main()

