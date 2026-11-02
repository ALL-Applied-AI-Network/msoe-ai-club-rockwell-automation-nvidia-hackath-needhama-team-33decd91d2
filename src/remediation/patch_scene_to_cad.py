#!/usr/bin/env python3
"""Rebuild an existing remediation scene's vial and cap geometry to the supplied CAD.

Source of truth: jeremiah_given_context/csi-vial.usda (CAD 0890-01) and
csi-vial-cap.usda (MOCAP TWF.820/.930 YW1), mirrored in geometry.SPEC.  Older
scenes were generated with 1.00/1.05 in x 8.00 in vials and 1.24 in caps.

Slot layout, tube origins, cap origins/orientations, materials and the robot cell
are kept.  Dynamic caps are re-settled under PhysX on the corrected geometry and
their settled poses are written back.  The input scene is never modified.

    ./scripts/run_isaac.sh src/remediation/patch_scene_to_cad.py \
        --scene shared/scenes/remediation_dataset1_v1.usda \
        --out shared/scenes/remediation_dataset1_v2_cad.usda
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared/data/fanuc_er4ia_sim')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--settle-steps', type=int, default=180)
    args = parser.parse_args()
    if args.out.resolve() == args.scene.resolve():
        raise ValueError('--out must differ from --scene')
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim
        from src.remediation.geometry import SPEC, spawn_height
        from src.remediation import scene as S

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        looks = '/World/RemediationLooks/'
        glass = UsdShade.Material(stage.GetPrimAtPath(looks + 'Glass'))
        yellow = UsdShade.Material(stage.GetPrimAtPath(looks + 'Cap'))
        cavity_prim = stage.GetPrimAtPath(looks + 'CapCavity')
        cavity = UsdShade.Material(cavity_prim) if cavity_prim.IsValid() else S.material(stage, 'CapCavity', (.5, .37, .01), roughness=.58)

        rims = {}
        for tube in stage.GetPrimAtPath('/World/RemediationTray').GetChildren():
            if not tube.GetName().endswith('_tube'):
                continue
            path = str(tube.GetPath())
            for child in ('Wall', 'Bottom'):
                stage.RemovePrim(path + '/' + child)
            S.annulus(stage, path + '/Wall', SPEC.vial_outer_radius, SPEC.vial_outer_radius,
                      SPEC.vial_inner_radius, 0, SPEC.vial_height, glass)
            bottom = UsdGeom.Cylinder.Define(stage, path + '/Bottom')
            bottom.CreateRadiusAttr(SPEC.vial_outer_radius); bottom.CreateHeightAttr(SPEC.vial_bottom)
            bottom.AddTranslateOp().Set(Gf.Vec3d(0, 0, SPEC.vial_bottom / 2)); S.collision(bottom.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(bottom.GetPrim()).Bind(glass)
            base = np.asarray(UsdGeom.Xformable(tube).ComputeLocalToWorldTransform(0.), float)[3, :3]
            rims[tube.GetName()[:-5]] = base + [0, 0, SPEC.vial_height]

        dynamic = []
        for cap in stage.GetPrimAtPath('/World/RemediationCaps').GetChildren():
            path = str(cap.GetPath())
            for child in list(cap.GetChildren()):
                stage.RemovePrim(child.GetPath())
            S.cap_geometry(stage, path, yellow, cavity)
            xf = UsdGeom.Xformable(cap)
            m = np.asarray(xf.ComputeLocalToWorldTransform(0.), float)
            slot = cap.GetName()
            if not cap.HasAPI(UsdPhysics.RigidBodyAPI):
                if slot in rims:
                    # Static seated cap: same 0.2 mm rim offset as scene.build_scene.
                    for op in xf.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            op.Set(Gf.Vec3d(m[3, 0], m[3, 1], rims[slot][2] + .0002))
                continue
            dynamic.append(path)
            if slot in rims:
                # Re-drop a crooked cap onto the corrected rim, as scene.build_scene does.
                R = m[:3, :3].T
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(m[3, 0], m[3, 1], spawn_height(rims[slot][2], R)))
            for attr in ('physics:velocity', 'physics:angularVelocity'):
                if cap.HasAttribute(attr):
                    cap.GetAttribute(attr).Set(Gf.Vec3f(0, 0, 0))

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60., rendering_dt=1 / 60.)
        world.reset()
        caps = RigidPrim(dynamic)
        for _ in range(args.settle_steps):
            world.step(render=False)
        pos, quat = caps.get_world_poses()
        lin = np.linalg.norm(caps.get_linear_velocities(), axis=1)
        world.stop()
        report = {'source_scene': str(args.scene), 'cad': {
            'vial': 'jeremiah_given_context/csi-vial.usda', 'cap': 'jeremiah_given_context/csi-vial-cap.usda'},
            'vial_od_m': 2 * SPEC.vial_outer_radius, 'vial_id_m': 2 * SPEC.vial_inner_radius,
            'vial_length_m': SPEC.vial_height, 'rim_world_z_m': float(next(iter(rims.values()))[2]),
            'settled_dynamic_caps': {}}
        for k, path in enumerate(dynamic):
            prim = stage.GetPrimAtPath(path)
            xf = UsdGeom.Xformable(prim)
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*map(float, pos[k])))
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    op.Set(Gf.Quatf(float(quat[k][0]), Gf.Vec3f(*map(float, quat[k][1:]))))
            for attr in ('physics:velocity', 'physics:angularVelocity'):
                if prim.HasAttribute(attr):
                    prim.GetAttribute(attr).Set(Gf.Vec3f(0, 0, 0))
            w, x, y, z = map(float, quat[k])
            tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
            report['settled_dynamic_caps'][path] = {'position_world_m': pos[k].tolist(), 'quat_wxyz': [w, x, y, z],
                                                    'tilt_deg': tilt, 'residual_speed_m_s': float(lin[k])}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(args.out))
        args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
