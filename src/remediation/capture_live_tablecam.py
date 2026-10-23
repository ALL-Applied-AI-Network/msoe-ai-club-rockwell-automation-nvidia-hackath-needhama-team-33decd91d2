#!/usr/bin/env python3
"""Capture RGB, aligned depth, and calibration metadata from a loaded Isaac scene."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared/data/fanuc_er4ia_sim')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        import omni.usd
        from pxr import UsdGeom
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        import cell
        omni.usd.get_context().open_stage(str(args.scene)); app.update()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=1/60., rendering_dt=1/30.)
        world.reset()
        camera = Camera(cell.TABLE_CAM_PATH, resolution=(1920, 1080)); camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        for _ in range(20): world.step(render=True)
        rgba = camera.get_rgba(); depth = np.asarray(camera.get_depth(device='cpu'), dtype=np.float32)
        if rgba is None or not rgba.size or depth.shape[:2] != rgba.shape[:2] or not np.isfinite(depth).any():
            raise RuntimeError('TableCam RGB/depth capture is unavailable or misaligned')
        image = args.out / 'current_scene_tablecam.png'; depth_path = args.out / 'current_scene_tablecam_depth_m.npy'
        cv2.imwrite(str(image), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)); np.save(depth_path, depth)
        cam = UsdGeom.Camera(stage.GetPrimAtPath(cell.TABLE_CAM_PATH))
        report = {'source': 'current_loaded_scene_tablecam', 'scene': str(args.scene),
                  'image': str(image), 'depth': str(depth_path),
                  'camera': {'resolution': [1920, 1080], 'focal_length_mm': float(cam.GetFocalLengthAttr().Get()),
                  'horizontal_aperture_mm': float(cam.GetHorizontalApertureAttr().Get()),
                  'vertical_aperture_mm': float(cam.GetVerticalApertureAttr().Get()),
                  'camera_to_world': np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0.)).tolist()}}
        (args.out / 'capture_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == '__main__':
    main()
