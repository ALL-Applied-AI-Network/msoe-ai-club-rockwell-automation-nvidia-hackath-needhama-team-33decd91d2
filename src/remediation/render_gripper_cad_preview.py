"""Render the converted custom gripper CAD alone for mounting-interface review."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def look_at(eye, target, up=(0, 0, 1)):
    from pxr import Gf
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=ROOT / 'data/gripper.usd')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.asset.is_file():
        raise FileNotFoundError(args.asset)
    args.out.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Usd, UsdGeom, UsdLux
        import cv2

        world = World(stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, '/World/CustomGripper')
        root.GetPrim().GetReferences().AddReference(str(args.asset))
        UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(-.018, .009, .002))
        mesh_paths = []
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(.08, .32, .78)])
                gprim.CreateVisibilityAttr().Set('inherited')
                mesh_paths.append(str(prim.GetPath()))
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root.GetPrim()).ComputeAlignedBox()
        floor = UsdGeom.Cube.Define(stage, '/World/Floor')
        floor.CreateSizeAttr(1.0)
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
        floor.AddScaleOp().Set(Gf.Vec3f(.4, .4, .01))
        light = UsdLux.DistantLight.Define(stage, '/World/Key')
        light.CreateIntensityAttr(2500)
        light.AddRotateYOp().Set(35)
        light.AddRotateZOp().Set(-35)
        dome = UsdLux.DomeLight.Define(stage, '/World/Fill')
        dome.CreateIntensityAttr(600)
        dome.CreateColorAttr(Gf.Vec3f(.72, .76, .82))
        camera_path = '/World/PreviewCam'
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        camera_prim.CreateFocalLengthAttr().Set(28.0)
        camera_prim.CreateHorizontalApertureAttr().Set(20.955)
        camera_prim.CreateVerticalApertureAttr().Set(15.716)
        UsdGeom.Xformable(camera_prim).AddTransformOp().Set(look_at((.16, -.16, .11), (0, 0, 0)))
        world.reset()
        camera = Camera(camera_path, resolution=(1280, 960))
        camera.initialize()
        for _ in range(30):
            world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            raise RuntimeError('Empty CAD preview capture')
        if not cv2.imwrite(str(args.out / 'gripper_cad_preview.png'), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Could not write CAD preview')
        stage.GetRootLayer().Export(str(args.out / 'gripper_preview_scene.usda'))
        np.save(args.out / 'raw_asset_bounds_note.npy', np.array([[0.0, .0356293480873969],[-.0205384609807311,.00175355771760241],[-.0477762395223592,.0437506223631226]]))
        (args.out / 'asset_preview_report.json').write_text(json.dumps({
            'mesh_count': len(mesh_paths), 'mesh_paths': mesh_paths,
            'world_bounds_m': {'min': list(bounds.GetMin()), 'max': list(bounds.GetMax())},
        }, indent=2) + '\n')
    finally:
        app.close()


if __name__ == '__main__':
    main()
