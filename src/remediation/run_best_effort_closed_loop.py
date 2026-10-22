#!/usr/bin/env python3
"""Run a prediction-selected, best-effort Isaac remediation contact trial.

This is deliberately an *attempt* recorder.  Targets are selected from saved
TableCam/YOLO observations; no scene-truth coordinates are used for selection
or substitution.  It drives the simulated arm through a spare-cap pickup and
placement attempt, followed by a vertical crooked-cap press attempt, and
records the actual rendered motion.  A completed video is not proof of a
grasp, placement, or physical-robot contact qualification.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared/data/fanuc_er4ia_sim')]
# Derived only from the reviewed visual mount: 284.302 mm at 20 mm forward,
# less the current 1.5 mm retraction.  It is not a physical TCP measurement.
SIMULATION_TOOL_OFFSET_M = 0.2828020875330954


def solve_route(point, contact_delta_m, clearance_m=.020):
    """Return a fixed-down Cartesian/IK route from home and back."""
    import fk
    import ik
    import scene_config as C
    from src.remediation.geometry import transit_base_z

    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f'J{i}'] for i in range(1, 7)])
    q = np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS); ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    point = np.asarray(point, dtype=float)
    start = np.array([.24, .25, .38])
    high = np.array([point[0], point[1], max(transit_base_z(), .38)])
    approach = point + np.array([0., 0., clearance_m])
    contact = point + np.array([0., 0., contact_delta_m])
    cartesian = []
    for a, b, count in ((start, high, 34), (high, approach, 24),
                        (approach, contact, 20), (contact, approach, 20),
                        (approach, high, 24), (high, start, 34)):
        cartesian.extend(np.linspace(a, b, count, endpoint=False))
    cartesian.append(start)
    tool_offset = tuple(np.asarray(C.TOOL_AXIS) * SIMULATION_TOOL_OFFSET_M)
    path, tilts = [], []
    down = np.array([0., 0., -1.])
    for p in cartesian:
        q, ok = ik.solve(q, p, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
        if not ok:
            raise RuntimeError('unreachable fixed-down waypoint')
        R, _ = fk.fk(q)
        axis = R @ np.asarray(C.TOOL_AXIS)
        tilts.append(float(np.degrees(np.arccos(np.clip(axis @ down, -1., 1.)))) )
        path.append(q.copy())
    if max(tilts) > 2.0:
        raise RuntimeError(f'fixed-down tilt exceeds 2 degrees ({max(tilts):.2f})')
    return path, max(tilts)


def choose(detections, class_name):
    candidates = [d for d in detections if d.get('class_name') == class_name]
    if not candidates:
        raise RuntimeError(f'YOLO did not report a {class_name}')
    rejected = []
    for d in sorted(candidates, key=lambda item: item['confidence'], reverse=True):
        try:
            # Best-effort contact depths, relative to the TableCam/ray-plane
            # observation. The prior trial stopped visibly high; retain a
            # straight vertical descent but extend it farther into the local
            # contact envelope for review.
            delta = -.005 if class_name == 'table_cap' else -.010
            path, tilt = solve_route(d['base_pos_m'], delta)
            return d, path, tilt, rejected
        except RuntimeError as exc:
            rejected.append({'candidate': d, 'reason': str(exc)})
    raise RuntimeError(f'no reachable {class_name} prediction')


def set_jaws(arm, names, opened):
    import cell
    import scene_config as C
    positions = arm.get_joint_positions().copy()
    for joint, value in cell.jaw_targets(C.JAW_OPEN if opened else C.JAW_CLOSED).items():
        positions[names.index(joint)] = value
    arm.set_joint_positions(positions)
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    observed = json.loads(args.detections.read_text()).get('targets', [])
    # Camera framing is derived from observed tray objects, never from tray
    # aperture coordinates used by motion planning.
    tray_points = [np.asarray(d['base_pos_m'], dtype=float) for d in observed
                   if d.get('class_name') != 'table_cap']
    if not tray_points:
        raise RuntimeError('YOLO did not report tray objects for camera framing')
    tray_focus = np.mean(tray_points, axis=0)
    tray_focus[2] = max(point[2] for point in tray_points) + .015
    spare, spare_path, spare_tilt, spare_rejected = choose(observed, 'table_cap')
    missing, missing_path, missing_tilt, missing_rejected = choose(observed, 'missing_cap')
    crooked, crooked_path, crooked_tilt, crooked_rejected = choose(observed, 'crooked_cap')
    args.out.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        print('[closed-loop] app ready', flush=True)
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell
        import scene_config as C
        from src.remediation.custom_gripper import attach_collision_proxies, attach_visual_pair

        omni.usd.get_context().open_stage(str(args.scene)); app.update()
        stage = omni.usd.get_context().get_stage()
        print('[closed-loop] scene open', flush=True)
        world = World(stage_units_in_meters=1/1., physics_dt=1/60., rendering_dt=1/30.)
        attach_visual_pair(stage); attach_collision_proxies(stage); world.reset()
        print('[closed-loop] custom gripper attached', flush=True)
        arm = SingleArticulation(C.ROBOT_PRIM); arm.initialize()
        names = list(arm.dof_names); arm_idx = [names.index(f'J{i}') for i in range(1, 7)]
        overview_path = '/World/ClosedLoopAttemptOverviewCam'
        # From the accepted high isometric pose: lower 0.10 m and raise the
        # view elevation by 20 degrees, retaining the same table-plane pose.
        cell.camera(stage, overview_path, {'eye': (.827, -.490, 1.700),
                                           'look_at': (.385, .035, 1.040),
                                           'up': (0, 0, 1), 'resolution': (1280, 720),
                                           'focal_length': 10.0, 'horiz_aperture': 20.955})
        camera = Camera(overview_path, resolution=(1280, 720)); camera.initialize()
        set_jaws(arm, names, True)
        print('[closed-loop] arm and overview camera ready', flush=True)
        for _ in range(30): world.step(render=True)
        print('[closed-loop] camera warmup complete', flush=True)
        frame_dir = args.out / 'frames'
        frame_dir.mkdir(exist_ok=True)
        phases = [('pick_spare', spare_path, True, spare['base_pos_m']),
                  ('close_on_spare', [None] * 15, False, spare['base_pos_m']),
                  ('place_on_missing', missing_path, False, missing['base_pos_m']),
                  ('release_on_missing', [None] * 15, True, missing['base_pos_m']),
                  ('press_crooked', crooked_path, True, crooked['base_pos_m'])]
        frames = 0
        for phase, path, opened, focus in phases:
            print(f'[closed-loop] phase {phase}', flush=True)
            jaw_positions = set_jaws(arm, names, opened)
            for joints in path:
                positions = arm.get_joint_positions().copy()
                positions[arm_idx] = joints if joints is not None else positions[arm_idx]
                positions[names.index('jaw_l')] = jaw_positions[names.index('jaw_l')]
                positions[names.index('jaw_r')] = jaw_positions[names.index('jaw_r')]
                arm.apply_action(ArticulationAction(joint_positions=positions)); world.step(render=True)
                rgba = camera.get_rgba()
                if rgba is None or not rgba.size: continue
                frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.putText(frame, 'BEST-EFFORT CLOSED-LOOP SIMULATION ATTEMPT', (20, 34), cv2.FONT_HERSHEY_SIMPLEX, .68, (30, 230, 255), 2)
                cv2.putText(frame, phase.replace('_', ' '), (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .66, (255, 255, 255), 2)
                cv2.putText(frame, 'Targets selected from YOLO/TableCam; contact result pending reinspection', (20, 96), cv2.FONT_HERSHEY_SIMPLEX, .5, (210, 235, 255), 1)
                if not cv2.imwrite(str(frame_dir / f'{frames:04d}.png'), frame):
                    raise RuntimeError('could not write output frame')
                frames += 1
        print(f'[closed-loop] wrote {frames} frames', flush=True)
        report = {'mode': 'best_effort_prediction_selected_contact_attempt', 'physical_robot_authorized': False,
                  'contact_result_validated': False, 'scene': str(args.scene), 'detections': str(args.detections),
                  'simulation_tool_offset_m': SIMULATION_TOOL_OFFSET_M,
                  'targets': {'table_cap': spare, 'missing_cap': missing, 'crooked_cap': crooked},
                  'rejected_predictions': {'table_cap': spare_rejected, 'missing_cap': missing_rejected, 'crooked_cap': crooked_rejected},
                  'max_tool_tilt_deg': max(spare_tilt, missing_tilt, crooked_tilt),
                  'actions_attempted': ['pick_spare', 'place_on_missing', 'press_crooked'],
                  'frame_directory': str(frame_dir), 'video_pending_external_encode': True, 'frames_written': frames,
                  'next_required_check': 'Run a post-action TableCam capture and compare YOLO detections; do not infer grasp success from this video.'}
        (args.out / 'attempt_report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        app.close()


if __name__ == '__main__':
    main()
