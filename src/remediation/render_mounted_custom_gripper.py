"""Render the visual-only custom gripper pair attached to the live Isaac jaws."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FANUC = ROOT / 'shared/data/fanuc_er4ia_sim'
if FANUC.is_dir():
    sys.path.insert(0, str(FANUC))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Isaac forwards the parent process command line to Kit.  Keep this
    # preview's output argument out of Kit so it does not interpret it as a
    # launcher request and close before the Python scene is built.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        # Let Kit finish initializing before requesting the live robot stage.
        # The batch runner otherwise observes a clean early shutdown.
        app.update()
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Usd, UsdGeom
        import cv2
        import cell
        from src.remediation.custom_gripper import attach_visual_pair, attach_collision_proxies

        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0)
        cell.build(stage, with_objects=False)
        mount = attach_visual_pair(stage)
        collision_proxies = attach_collision_proxies(stage)
        world.reset()
        # The custom fingers achieve their reviewed 15 mm separation at this
        # commanded mechanical endpoint, not at Kit's uncontrolled default.
        import scene_config as C
        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        positions = arm.get_joint_positions().copy()
        names = list(arm.dof_names)
        for joint, target in cell.jaw_targets(C.JAW_CLOSED).items():
            positions[names.index(joint)] = target
        arm.apply_action(ArticulationAction(joint_positions=positions))
        # Allow the position drive to settle before measuring or capturing.
        for _ in range(35):
            world.step(render=True)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        boxes = [cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox() for path in mount['paths']]
        low = np.min([np.array(box.GetMin()) for box in boxes], axis=0)
        high = np.max([np.array(box.GetMax()) for box in boxes], axis=0)
        target = (low + high) / 2
        views = {
            'top': (target + np.array((0.0, 0.0, .24)), (0, 1, 0)),
            # Look down the tool axis at the finger faces. This is the true
            # front view; the previous -Y view was a left-side view.
            'front': (target + np.array((.24, 0.0, .03)), (0, 0, 1)),
            'side': (target + np.array((0.0, -.24, .03)), (0, 0, 1)),
            'perspective': (target + np.array((.24, -.24, .16)), (0, 0, 1)),
        }
        for name, (eye, up) in views.items():
            camera_path = '/World/CustomGripperInspectionCam_' + name
            cell.camera(stage, camera_path, {'eye': eye, 'look_at': target, 'up': up,
                                             'resolution': (1280, 960), 'focal_length': 14.0,
                                             'horiz_aperture': 20.955, 'fisheye': False})
            camera = Camera(camera_path, resolution=(1280, 960))
            camera.initialize()
            for _ in range(18):
                world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is None or rgba.size == 0:
                raise RuntimeError('Empty mounted gripper ' + name + ' capture')
            if not cv2.imwrite(str(args.out / ('mounted_custom_gripper_' + name + '.png')),
                               cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Could not write mounted gripper ' + name + ' preview')
        stage.GetRootLayer().Export(str(args.out / 'mounted_custom_gripper_scene.usda'))
        descendants = []
        for root_path in mount['paths'][:1]:
            root_prim = stage.GetPrimAtPath(root_path)
            for prim in Usd.PrimRange(root_prim):
                descendants.append(str(prim.GetPath()))
        per_finger = {}
        for path in mount['paths']:
            box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
            per_finger[path] = {'min': list(box.GetMin()), 'max': list(box.GetMax())}
        from scene_config import JAW_STROKE
        wrist_detail_paths = [str(prim.GetPath()) for prim in stage.Traverse()
                              if any(token in str(prim.GetPath()).lower()
                                     for token in ('laser', 'photoblacksensorlead'))]
        gap_contract = {'minimum_mesh_gap_m': 0.003, 'maximum_mesh_gap_m': 0.015,
                        'jaw_travel_m': 2 * float(JAW_STROKE), 'verified_travel_span_m': 0.012,
                        'status': 'contracted_to_SMC_6mm_per_jaw_travel'}
        (args.out / 'mount_report.json').write_text(json.dumps({**mount, 'collision_proxy_paths': collision_proxies,
            'source_descendants': descendants,
            'bounds_world_m': {'min': low.tolist(), 'max': high.tolist()},
            'per_finger_bounds_world_m': per_finger, 'finger_gap_contract': gap_contract,
            'wrist_detail_paths': wrist_detail_paths}, indent=2) + '\n')
    finally:
        app.close()


if __name__ == '__main__':
    main()
