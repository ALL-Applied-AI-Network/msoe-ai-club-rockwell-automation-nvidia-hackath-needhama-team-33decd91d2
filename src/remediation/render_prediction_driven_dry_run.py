#!/usr/bin/env python3
"""Render a prediction-driven, non-contact Isaac arm dry run.

The target comes from YOLO on a TableCam capture.  Scene truth is not used for
target selection; the arm only approaches the predicted point at the mandated
clearance height and returns home.  This is deliberately not a repair trial.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared/data/fanuc_er4ia_sim')]
# From the Isaac CAD metrology artifact. Kept here so preflight IK does not
# import pxr/Isaac-only modules before SimulationApp starts.
SIMULATION_GRASP_TIP_OFFSET_M = 0.2843020875330954


def fixed_down_path(predicted, approach_clearance_m=0.020):
    """Solve the review path before Kit starts, isolating IK from Isaac runtime."""
    import scene_config as C
    import ik
    import fk
    from src.remediation.geometry import transit_base_z

    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f'J{i}'] for i in range(1, 7)])
    q_current = np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    predicted = np.asarray(predicted, dtype=float)
    overhead = np.array([predicted[0], predicted[1], transit_base_z()])
    approach = np.array([predicted[0], predicted[1], predicted[2] + approach_clearance_m])
    start = np.array([.24, .25, .38])
    high_target = np.array([predicted[0], predicted[1], .38])
    cartesian = []
    for a, b, count in ((start, high_target, 35), (high_target, overhead, 25),
                        (overhead, approach, 20), (approach, overhead, 20),
                        (overhead, high_target, 25), (high_target, start, 35)):
        cartesian.extend(np.linspace(a, b, count, endpoint=False))
    cartesian.append(start)
    path, tilt_errors = [], []
    down = np.array([0., 0., -1.])
    tool_offset = tuple(np.asarray(C.TOOL_AXIS) * SIMULATION_GRASP_TIP_OFFSET_M)
    for point in cartesian:
        q_current, converged = ik.solve(q_current, point, down, limits,
                                         tool_offset=tool_offset, spread_dir=[0, 1, 0])
        if not converged:
            raise RuntimeError('A fixed-down safe waypoint is unreachable; refusing to substitute scene truth')
        R, _ = fk.fk(q_current)
        tool_axis = R @ np.asarray(C.TOOL_AXIS)
        tilt_errors.append(float(np.degrees(np.arccos(np.clip(np.dot(tool_axis, down), -1, 1)))))
        path.append(q_current.copy())
    if max(tilt_errors) > 2.0:
        raise RuntimeError(f'Fixed-down orientation error exceeds 2 degrees: {max(tilt_errors):.3f}')
    return path, overhead, approach, max(tilt_errors), SIMULATION_GRASP_TIP_OFFSET_M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True,
                        help='Saved YOLO detections from inspect_remediation_capture.py')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    scene_dir = args.scene.parent
    detections = json.loads(args.detections.read_text())
    targets = detections.get('targets', [])
    candidates = [item for item in targets if item['class_name'] in ('crooked_cap', 'missing_cap')]
    if not candidates:
        raise RuntimeError('YOLO found no crooked or missing cap for dry run')
    # Select from predictions only.  A high-confidence point outside the
    # fixed-down workspace must be rejected rather than changing orientation
    # or silently substituting a tray coordinate.
    target = path = overhead = approach = max_tilt_deg = tool_length_m = None
    rejected = []
    for candidate in sorted(candidates, key=lambda item: item['confidence'], reverse=True):
        try:
            path, overhead, approach, max_tilt_deg, tool_length_m = fixed_down_path(candidate['base_pos_m'])
            target = candidate
            break
        except RuntimeError as exc:
            rejected.append({'class_name': candidate['class_name'], 'confidence': candidate['confidence'],
                             'base_pos_m': candidate['base_pos_m'], 'reason': str(exc)})
    if target is None:
        raise RuntimeError('No YOLO defect prediction is reachable with fixed-down safety constraints')
    # Prevent argparse flags being passed to Kit at startup.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        print('[dry-run] app ready', flush=True)
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell
        import scene_config as C
        from src.remediation.custom_gripper import attach_visual_pair, attach_collision_proxies
        from src.remediation.geometry import SPEC, transit_base_z

        print('[dry-run] imports ready', flush=True)
        print('[dry-run] target detected', flush=True)

        omni.usd.get_context().open_stage(str(args.scene)); app.update()
        stage = omni.usd.get_context().get_stage()
        print('[dry-run] scene open', flush=True)
        world = World(stage_units_in_meters=1.0, physics_dt=1/60., rendering_dt=1/30.)
        print('[dry-run] world created', flush=True)
        attach_visual_pair(stage); attach_collision_proxies(stage)
        print('[dry-run] custom gripper attached', flush=True)
        world.reset()
        print('[dry-run] world reset', flush=True)
        arm = SingleArticulation(C.ROBOT_PRIM); arm.initialize()
        print('[dry-run] arm initialized', flush=True)
        names = list(arm.dof_names); arm_indices = [names.index(f'J{i}') for i in range(1, 7)]
        print('[dry-run] arm dofs resolved', flush=True)
        print(f'[dry-run] preflight solved {len(path)} fixed-down waypoints', flush=True)
        # The calibrated side camera intentionally concentrates on the tray
        # and clips the arm. This review-only overview keeps both arm and tray
        # visible; it is not used for perception or localization.
        overview_path = '/World/PredictionDryRunOverviewCam'
        cell.camera(stage, overview_path, {
            'eye': (1.05, -1.00, 1.15), 'look_at': (.36, .02, .58), 'up': (0, 0, 1),
            'resolution': (1280, 720), 'focal_length': 8.0, 'horiz_aperture': 20.955,
        })
        camera = Camera(overview_path, resolution=(1280, 720)); camera.initialize()
        args.out.mkdir(parents=True, exist_ok=True)
        # RTX camera annotators need several render frames after a referenced
        # stage is opened. Do not attempt to encode an uninitialized frame.
        rgba = None
        for _ in range(20):
            world.step(render=True)
            candidate = camera.get_rgba()
            if candidate is not None and candidate.size:
                rgba = candidate
        if rgba is None:
            raise RuntimeError('Side camera did not produce an RGB frame after warmup')
        last_rgba = rgba
        frame_dir = args.out / 'frames'
        frame_dir.mkdir(exist_ok=True)
        for frame_index, joints in enumerate(path):
            positions = arm.get_joint_positions().copy(); positions[arm_indices] = joints
            arm.apply_action(ArticulationAction(joint_positions=positions))
            world.step(render=True)
            candidate = camera.get_rgba()
            if candidate is not None and candidate.size:
                last_rgba = candidate
            frame = cv2.cvtColor(last_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            text = f"YOLO {target['class_name']} {target['confidence']:.2f} | measured-CAD approach"
            cv2.putText(frame, text, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Vertical approach stops 20 mm above prediction; no contact action', (18, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (190, 225, 255), 2, cv2.LINE_AA)
            if frame_index % 3 == 0:
                if not cv2.imwrite(str(frame_dir / f'{frame_index // 3:04d}.png'), frame):
                    raise RuntimeError('Could not write dry-run frame')
        report = {'schema_version': 1, 'mode': 'prediction_driven_isaac_dry_run', 'motion_enabled': True,
                  'contact_enabled': False, 'scene': str(args.scene), 'detections': str(args.detections), 'target': target,
                  'unreachable_predictions_rejected': rejected,
                  'overhead_base_pos_m': overhead.tolist(), 'clearance_above_surface_m': transit_base_z() - SPEC.surface_base_z,
                  'approach_base_pos_m': approach.tolist(), 'approach_clearance_m': 0.020,
                  'simulation_cad_grasp_tip_offset_m': tool_length_m,
                  'max_commanded_tool_tilt_deg': max_tilt_deg,
                  'frame_directory': str(frame_dir), 'video_pending_external_encode': True}
        (args.out / 'dry_run_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == '__main__':
    main()
