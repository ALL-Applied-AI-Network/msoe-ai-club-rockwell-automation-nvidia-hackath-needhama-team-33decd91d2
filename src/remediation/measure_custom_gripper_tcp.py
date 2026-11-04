#!/usr/bin/env python3
"""Measure the mounted custom CAD's conservative grasp-tip offset in Isaac.

This is CAD/scene metrology only.  It does not validate physical TCP, contact,
or allow a real robot to perform a contact move.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared/data/fanuc_er4ia_sim')]


def corners(minimum, maximum):
    return np.array([[x, y, z] for x in (minimum[0], maximum[0])
                      for y in (minimum[1], maximum[1])
                      for z in (minimum[2], maximum[2])], dtype=float)


def run(args):
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd, UsdGeom
        import cell
        import scene_config as C
        from src.remediation.custom_gripper import attach_visual_pair
        from src.remediation.geometry import SPEC

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0)
        mount = attach_visual_pair(stage)
        world.reset()
        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        for _ in range(5):
            world.step(render=False)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        flange = UsdGeom.Xformable(stage.GetPrimAtPath(cell.FLANGE))
        flange_matrix = np.asarray(flange.ComputeLocalToWorldTransform(0.), dtype=float)
        flange_position = flange_matrix[3, :3]
        # USD uses row vectors; transpose gives the usual local-to-world basis.
        flange_rotation = flange_matrix[:3, :3].T
        grasp_axis = flange_rotation @ np.asarray(C.TOOL_AXIS, dtype=float)
        grasp_axis /= np.linalg.norm(grasp_axis)

        each = []
        all_points = []
        for path in mount['paths']:
            box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
            lower, upper = np.asarray(box.GetMin(), dtype=float), np.asarray(box.GetMax(), dtype=float)
            points = corners(lower, upper)
            all_points.extend(points)
            each.append({'path': path, 'world_aabb_min_m': lower.tolist(), 'world_aabb_max_m': upper.tolist(),
                         'furthest_grasp_projection_m': float(np.max(points @ grasp_axis))})
        furthest = max(float(point @ grasp_axis) for point in all_points)
        flange_projection = float(flange_position @ grasp_axis)
        offset = furthest - flange_projection
        report = {
            'schema_version': 1,
            'mode': 'isaac_cad_tcp_metrology',
            'contact_enabled': False,
            'physical_tcp_validated': False,
            'flange_world_m': flange_position.tolist(),
            'grasp_axis_world': grasp_axis.tolist(),
            'cad_grasp_tip_offset_from_flange_m': offset,
            'current_planner_tool_length_m': SPEC.tool_length,
            'planner_length_delta_m': offset - SPEC.tool_length,
            'cad_halves': each,
            'limitations': [
                'Uses conservative CAD world AABBs rather than measured contact points',
                'Does not establish cap pick, place, or press offsets',
                'Does not authorize physical-robot contact',
            ],
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
