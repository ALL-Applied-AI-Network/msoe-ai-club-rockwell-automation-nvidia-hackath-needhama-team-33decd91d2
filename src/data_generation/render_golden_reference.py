"""
Render a clean, nominal Golden Reference image of the tray inspection station.

Renders through CapSealCam with real lens distortion and saves directly to:
  shared/inspection_models/golden_reference.png
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSI_DIR = REPO_ROOT / "shared" / "data" / "csi_digital_twin_v1"
DEFAULT_OUT = REPO_ROOT / "shared" / "inspection_models" / "golden_reference.png"


def render_golden(out_path=None, subframes=32):
    out_file = Path(out_path) if out_path else DEFAULT_OUT
    out_file.parent.mkdir(parents=True, exist_ok=True)

    stage_file = CSI_DIR / "stations" / "tray_inspection" / "tray_inspection.usdc"
    cam_json_path = CSI_DIR / "cameras" / "CapSealCam.json"
    with open(cam_json_path) as f:
        cam_cfg = json.load(f)

    rw, rh = cam_cfg["render_px"]

    print(f"[render_golden] Initializing Isaac Sim (resolution: {rw}x{rh})...")
    from isaacsim import SimulationApp
    kit = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": rw, "height": rh})

    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, Gf

        ctx = omni.usd.get_context()
        ctx.open_stage(str(stage_file))
        for _ in range(15):
            kit.update()

        stage = ctx.get_stage()

        # Ensure all 30 caps and tubes are fully nominal and visible
        for i in range(1, 31):
            cap_prim = stage.GetPrimAtPath(f"/TrayInspection/G{i:02d}_cap")
            tube_prim = stage.GetPrimAtPath(f"/TrayInspection/G{i:02d}_tube")

            if cap_prim.IsValid():
                UsdGeom.Imageable(cap_prim).MakeVisible()
                xf = UsdGeom.Xformable(cap_prim)
                rot_op = next((op for op in xf.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                if rot_op:
                    rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))

            if tube_prim.IsValid():
                UsdGeom.Imageable(tube_prim).MakeVisible()
                xf_t = UsdGeom.Xformable(tube_prim)
                rot_op_t = next((op for op in xf_t.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                if rot_op_t:
                    rot_op_t.Set(Gf.Vec3f(0.0, 0.0, 0.0))

        # Setup render product
        prim_path = "/TrayInspection/CapSealCam/CapSealCam"
        rp = rep.create.render_product(prim_path, (rw, rh))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        rep.orchestrator.set_capture_on_play(False)

        for _ in range(10):
            kit.update()
            rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0)

        data = np.asarray(ann.get_data())
        img_bgr = data[..., :3][..., ::-1].copy()

        # Apply camera lens distortion
        tmp_pinhole = str(out_file) + ".pinhole.tmp.png"
        cv2.imwrite(tmp_pinhole, img_bgr)

        subprocess.run([
            sys.executable,
            str(CSI_DIR / "tools" / "apply_lens.py"),
            tmp_pinhole,
            str(cam_json_path),
            str(out_file)
        ], check=True)

        if os.path.exists(tmp_pinhole):
            os.remove(tmp_pinhole)

        print(f"[render_golden] Successfully generated golden reference at: {out_file}")

    finally:
        kit.close()


def main():
    parser = argparse.ArgumentParser(description="Render clean golden reference frame")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output image file path")
    parser.add_argument("--subframes", type=int, default=32, help="Path-tracing / lighting subframes")
    args = parser.parse_args()

    render_golden(out_path=args.out, subframes=args.subframes)


if __name__ == "__main__":
    main()
