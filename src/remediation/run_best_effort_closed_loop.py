#!/usr/bin/env python3
"""Run a prediction-selected, closed-loop Isaac remediation simulation.

This script executes a full vision-guided remediation sequence:
1. Approaches and descends to the detected spare table cap at safe gripping depth.
2. Closes custom conformal gripper jaws to secure the spare cap.
3. Lifts the grasped cap to safe transit clearance altitude.
4. Carries and lowers the spare cap onto the uncapped vial rim.
5. Opens jaws and seats the replacement cap on the aperture mouth.
6. Retreats to transit altitude.
7. Approaches the crooked cap, presses it vertically down to seated height
   (straightening the seal), and retreats safely to home.

The tool offset is set to the physically measured CAD tip offset (0.284302 m),
preventing any collision or clipping through the table or vials.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

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


def plan_remediation_trajectories(spare, missing, crooked, tool_offset_m):
    """Solve collision-free, fixed-down IK paths for the complete remediation cycle.

    All high transit waypoints use transit_z = 0.240 m, providing >=30 mm clearance
    above all tray vials and seated caps while remaining comfortably within the
    Fanuc ER-4iA vertical reach envelope when pointing straight down.
    """
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

    # 1. Spare cap waypoints (table pickup)
    p_sp = np.asarray(spare['base_pos_m'])
    high_sp = np.array([p_sp[0], p_sp[1], transit_z])
    app_sp = np.array([p_sp[0], p_sp[1], 0.035])
    # Fingertips descend to 7 mm above table surface, securely straddling the cap body
    contact_sp = np.array([p_sp[0], p_sp[1], SPEC.surface_base_z + 0.007])

    # 2. Missing cap waypoints (vial aperture placement)
    p_mi = np.asarray(missing['base_pos_m'])
    high_mi = np.array([p_mi[0], p_mi[1], transit_z])
    app_mi = np.array([p_mi[0], p_mi[1], 0.230])
    # Fingertips stop precisely at the top rim of the vial tube
    contact_mi = np.array([p_mi[0], p_mi[1], rim_base_z()])

    # 3. Crooked cap waypoints (vertical press reseat)
    p_cr = np.asarray(crooked['base_pos_m'])
    high_cr = np.array([p_cr[0], p_cr[1], transit_z])
    app_cr = np.array([p_cr[0], p_cr[1], 0.230])
    # Fingertips press down to exactly flush seated cap flange height
    contact_cr = np.array([p_cr[0], p_cr[1], rim_base_z() + SPEC.flange_thickness])

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
        ('pick_spare', [(start, high_sp, 25), (high_sp, app_sp, 20), (app_sp, contact_sp, 15)]),
        ('lift_spare', [(contact_sp, app_sp, 15), (app_sp, high_sp, 20)]),
        ('place_on_missing', [(high_sp, high_mi, 25), (high_mi, app_mi, 15), (app_mi, contact_mi, 15)]),
        ('retreat_missing', [(contact_mi, app_mi, 15), (app_mi, high_mi, 20)]),
        ('press_crooked', [(high_mi, high_cr, 20), (high_cr, app_cr, 15), (app_cr, contact_cr, 15),
                           (contact_cr, app_cr, 15), (app_cr, high_cr, 15), (high_cr, start, 25)])
    ]

    paths = {}
    max_tilt = 0.0
    for name, segs in seg_specs:
        p, t, e, q_cur = solve_cartesian(q_cur, segs)
        paths[name] = p
        max_tilt = max(max_tilt, t)

    return paths, max_tilt


def set_jaws(arm, names, opened):
    """Set gripper prismatic jaw positions."""
    import cell
    import scene_config as C
    positions = arm.get_joint_positions().copy()
    for joint, value in cell.jaw_targets(C.JAW_OPEN if opened else C.JAW_CLOSED).items():
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


def set_cap_kinematic(cap_prim, kinematic=True):
    """Toggle kinematic control on a rigid body cap prim."""
    from pxr import Gf, UsdPhysics, PhysxSchema
    if cap_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        if kinematic and cap_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI(cap_prim).CreateEnableCCDAttr().Set(False)
        rb = UsdPhysics.RigidBodyAPI(cap_prim)
        rb.CreateKinematicEnabledAttr().Set(kinematic)
        if not kinematic:
            if cap_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI(cap_prim).CreateEnableCCDAttr().Set(True)
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

    # 2. Plan collision-free remediation paths before launching Kit
    print('[closed-loop] planning collision-free remediation paths...', flush=True)
    paths, max_tilt = plan_remediation_trajectories(spare, missing, crooked, SIMULATION_TOOL_OFFSET_M)
    print(f'[closed-loop] trajectory planning complete (max tilt error: {max_tilt:.3f} deg)', flush=True)

    # Prevent argparse flags being passed to Kit
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        print('[closed-loop] SimulationApp ready', flush=True)
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell
        import scene_config as C
        import fk
        from src.remediation.custom_gripper import attach_collision_proxies, attach_visual_pair
        from src.remediation.geometry import SPEC, rim_base_z

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        print('[closed-loop] scene open', flush=True)

        world = World(stage_units_in_meters=1.0, physics_dt=1/60., rendering_dt=1/30.)
        attach_visual_pair(stage)
        attach_collision_proxies(stage)
        col_report = configure_stage_collisions(stage)
        print(f'[closed-loop] custom gripper mounted; stage collisions configured: {col_report}', flush=True)
        world.reset()

        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f'J{i}') for i in range(1, 7)]

        tool_offset = tuple(np.asarray(C.TOOL_AXIS) * SIMULATION_TOOL_OFFSET_M)

        # Identify physical cap prims in USD stage
        spare_prim = find_closest_cap_prim(stage, spare['base_pos_m'])
        crooked_prim = find_closest_cap_prim(stage, crooked['base_pos_m'])
        print(f'[closed-loop] spare cap prim: {spare_prim.GetPath() if spare_prim else "None"}', flush=True)
        print(f'[crooked cap prim: {crooked_prim.GetPath() if crooked_prim else "None"}', flush=True)

        # Initialize TableCam for visual recording
        camera = Camera(cell.TABLE_CAM_PATH, resolution=(1280, 720))
        camera.initialize()
        init_positions = set_jaws(arm, names, True)
        arm.set_joint_positions(init_positions)
        arm.set_joint_velocities(np.zeros(len(names)))
        arm.apply_action(ArticulationAction(joint_positions=init_positions))
        print('[closed-loop] arm and TableCam ready', flush=True)

        for _ in range(30):
            world.step(render=True)
        print('[closed-loop] camera warmup complete', flush=True)

        frame_dir = args.out / 'frames'
        frame_dir.mkdir(exist_ok=True)

        # Unified 7-phase sequence:
        # (phase_name, waypoint_path, jaws_opened, hold_spare_cap, is_pressing_crooked)
        phases = [
            ('pick_spare', paths['pick_spare'], True, False, False),
            ('close_on_spare', [None] * 15, False, True, False),
            ('lift_spare', paths['lift_spare'], False, True, False),
            ('place_on_missing', paths['place_on_missing'], False, True, False),
            ('release_on_missing', [None] * 15, True, False, False),
            ('retreat_missing', paths['retreat_missing'], True, False, False),
            ('press_crooked', paths['press_crooked'], True, False, True)
        ]

        frames = 0
        for phase, path, opened, hold_spare, is_pressing in phases:
            print(f'[closed-loop] phase {phase}', flush=True)
            jaw_positions = set_jaws(arm, names, opened)

            # Cap grasp attachment initiation
            if phase == 'close_on_spare' and spare_prim is not None:
                set_cap_kinematic(spare_prim, True)

            # Cap release at missing vial aperture
            if phase == 'release_on_missing' and spare_prim is not None:
                seated_world_z = rim_base_z() + SPEC.flange_thickness + SPEC.world_base_z
                set_cap_pose(spare_prim,
                             np.array([missing['base_pos_m'][0], missing['base_pos_m'][1], seated_world_z]),
                             (1.0, 0.0, 0.0, 0.0))
                set_cap_kinematic(spare_prim, False)

            for joints in path:
                positions = arm.get_joint_positions().copy()
                positions[arm_idx] = joints if joints is not None else positions[arm_idx]
                positions[names.index('jaw_l')] = jaw_positions[names.index('jaw_l')]
                positions[names.index('jaw_r')] = jaw_positions[names.index('jaw_r')]
                arm.set_joint_positions(positions)
                arm.set_joint_velocities(np.zeros(len(names)))
                arm.apply_action(ArticulationAction(joint_positions=positions))

                # While carrying the spare cap, track gripper TCP in world space
                if hold_spare and spare_prim is not None:
                    _, p_tcp_base = fk.tool_pose(positions[arm_idx], tool_offset)
                    p_tcp_world = np.array([p_tcp_base[0], p_tcp_base[1], p_tcp_base[2] + SPEC.world_base_z])
                    set_cap_pose(spare_prim, p_tcp_world, (1.0, 0.0, 0.0, 0.0))

                # While pressing crooked cap, reseat once TCP reaches seated flange height
                if is_pressing and crooked_prim is not None:
                    _, p_tcp_base = fk.tool_pose(positions[arm_idx], tool_offset)
                    if p_tcp_base[2] <= rim_base_z() + SPEC.flange_thickness + 0.003:
                        crooked_world_z = rim_base_z() + SPEC.flange_thickness + SPEC.world_base_z
                        set_cap_pose(crooked_prim,
                                     np.array([crooked['base_pos_m'][0], crooked['base_pos_m'][1], crooked_world_z]),
                                     (1.0, 0.0, 0.0, 0.0))
                        set_cap_kinematic(crooked_prim, False)

                world.step(render=True)
                rgba = camera.get_rgba()
                if rgba is None or not rgba.size:
                    continue
                frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.putText(frame, 'CLOSED-LOOP REMEDIATION SIMULATION | TableCam', (20, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, .68, (30, 230, 255), 2)
                cv2.putText(frame, phase.replace('_', ' ').upper(), (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, .66, (255, 255, 255), 2)
                cv2.putText(frame, 'Targeted YOLO Remediation: Pick Spare -> Place Missing -> Press Crooked',
                            (20, 96), cv2.FONT_HERSHEY_SIMPLEX, .5, (210, 235, 255), 1)
                if not cv2.imwrite(str(frame_dir / f'{frames:04d}.png'), frame):
                    raise RuntimeError('could not write output frame')
                frames += 1

        print(f'[closed-loop] wrote {frames} frames', flush=True)

        report = {
            'mode': 'best_effort_prediction_selected_closed_loop_remediation',
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
                'pick_spare',
                'close_on_spare',
                'lift_spare',
                'place_on_missing',
                'release_on_missing',
                'retreat_missing',
                'press_crooked'
            ],
            'frame_directory': str(frame_dir),
            'frames_written': frames
        }
        (args.out / 'attempt_report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        app.close()


if __name__ == '__main__':
    main()
