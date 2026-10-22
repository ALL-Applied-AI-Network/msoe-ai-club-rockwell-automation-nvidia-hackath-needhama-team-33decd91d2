"""Run an Isaac-stage proxy sweep report for the custom gripper."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'shared/data/fanuc_er4ia_sim'))


def run(args):
    # Kit parses process arguments during SimulationApp construction. Strip
    # this script's argparse flags so they are not treated as Kit options.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        print('[sweep] app ready', flush=True)
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Usd, UsdGeom
        from src.remediation.custom_gripper import attach_visual_pair, attach_collision_proxies
        import cell
        import scene_config as C

        scene_file = ROOT / 'runs/archive/2026-10-21_remediation/qualification_002/scene.usda'
        omni.usd.get_context().open_stage(str(scene_file))
        app.update()
        stage = omni.usd.get_context().get_stage()
        print('[sweep] imports/stage ready', flush=True)
        world = World(stage_units_in_meters=1.0)
        mount = attach_visual_pair(stage)
        proxies = attach_collision_proxies(stage)
        print('[sweep] gripper attached', flush=True)
        world.reset()
        for _ in range(30):
            world.step(render=False)
        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        dof_names = list(arm.dof_names)
        jaw_indices = {name: dof_names.index(name) for name in cell.JAW_JOINTS.values()}
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        def bounds(path):
            box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
            return np.asarray(box.GetMin(), float), np.asarray(box.GetMax(), float)

        object_paths = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith('/World/RemediationCaps/') or '/_tube' in path:
                if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cylinder):
                    object_paths.append(path)
        commanded_states = []
        overlaps = []
        for extension in (0.0, 0.003, 0.006):
            requested_opening = C.JAW_CLOSED + extension
            full = arm.get_joint_positions().copy()
            targets = cell.jaw_targets(requested_opening)
            for name, target in targets.items():
                full[jaw_indices[name]] = target
            arm.apply_action(ArticulationAction(joint_positions=full))
            for _ in range(30):
                world.step(render=False)
            cache.Clear()
            p_bounds = [bounds(proxy) for proxy in proxies]
            mesh_bounds = [bounds(path) for path in mount['paths']]
            # Report the actual imported CAD-reference separation. The proxies
            # deliberately include a thicker conservative envelope and are only
            # used for collision screening.
            left, right = sorted(mesh_bounds, key=lambda item: (item[0][1] + item[1][1]) / 2)
            measured_gap = max(0.0, float(right[0][1] - left[1][1]))
            state_overlaps = []
            for proxy, (pmin, pmax) in zip(proxies, p_bounds):
                for obj in object_paths:
                    omin, omax = bounds(obj)
                    if np.all(np.minimum(pmax, omax) >= np.maximum(pmin, omin)):
                        state_overlaps.append({'proxy': proxy, 'object': obj})
            overlaps.extend(state_overlaps)
            commanded_states.append({
                'extension_per_side_m': extension,
                'requested_joint_targets_m': targets,
                'minimum_mesh_gap_m': measured_gap,
                'aabb_overlaps': state_overlaps,
            })
        mesh_gaps = [state['minimum_mesh_gap_m'] for state in commanded_states]
        gap_contract = {
            'required_min_m': 0.003,
            'required_max_m': 0.015,
            'tolerance_m': 0.001,
            'measured_min_m': min(mesh_gaps),
            'measured_max_m': max(mesh_gaps),
        }
        gap_contract['accepted'] = (abs(gap_contract['measured_min_m'] - gap_contract['required_min_m']) <= gap_contract['tolerance_m']
                                    and abs(gap_contract['measured_max_m'] - gap_contract['required_max_m']) <= gap_contract['tolerance_m'])
        report = {
            'schema_version': 1,
            'mode': 'isaac_stage_proxy_sweep',
            'accepted': not overlaps and gap_contract['accepted'],
            'contact_enabled': False,
            'jaw_states': commanded_states,
            'proxy_paths': proxies,
            'mesh_gap_contract': gap_contract,
            'inventory_object_count': len(object_paths),
            'aabb_overlaps_at_inspection_pose': overlaps,
            'limitations': [
                'Jaw joints were position-commanded; actuator forces and object contact were not evaluated',
                'AABB overlap is a conservative screen, not a swept PhysX contact certificate',
                'TCP and table-clearance validation remain separate gates',
            ],
        }
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    run(p.parse_args())
