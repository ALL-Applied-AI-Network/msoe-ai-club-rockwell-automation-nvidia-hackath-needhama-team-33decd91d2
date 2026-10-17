"""
Synthetic Defect Injector & Renderer for CSI Tray Inspection.

Modifies the tray inspection USD stage in-memory during render:
- Injects tilted/crooked caps (simulating cross-threading: rotX/Y 15° to 35°).
- Injects severely dislodged caps (tilt >= 45°).
- Injects missing caps (hides the cap mesh).
- Renders through CapSealCam with real lens distortion.
"""

import argparse
from datetime import datetime
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSI_DIR = REPO_ROOT / "shared" / "data" / "csi_digital_twin_v1"


def render_defective_tray(out_dir, num_crooked=3, num_missing=2, num_dislodged=2, seed=42):
    random.seed(seed)
    run_dir = Path(out_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    
    stage_file = CSI_DIR / "stations" / "tray_inspection" / "tray_inspection.usdc"
    cam_json_path = CSI_DIR / "cameras" / "CapSealCam.json"
    with open(cam_json_path) as f:
        cam_cfg = json.load(f)

    rw, rh = cam_cfg["render_px"]

    from isaacsim import SimulationApp
    kit = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": rw, "height": rh})
    
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import numpy as np, cv2
        import omni.usd, omni.timeline
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, Gf

        ctx = omni.usd.get_context()
        ctx.open_stage(str(stage_file))
        for _ in range(15):
            kit.update()

        stage = ctx.get_stage()
        
        # Collect tray caps (G01_cap through G30_cap)
        tray_caps = []
        for i in range(1, 31):
            p = stage.GetPrimAtPath(f"/TrayInspection/G{i:02d}_cap")
            if p.IsValid():
                tray_caps.append((i, p))

        print(f"[inject_defects] Found {len(tray_caps)} fixture tray caps.")
        
        # Shuffle and select targets for synthetic defects
        indices = list(range(len(tray_caps)))
        random.shuffle(indices)
        
        crooked_indices = indices[:num_crooked]
        dislodged_indices = indices[num_crooked:num_crooked + num_dislodged]
        missing_indices = indices[num_crooked + num_dislodged:num_crooked + num_dislodged + num_missing]

        injected_records = []

        # 1. Inject Crooked Caps (Tilt 18° to 30°)
        for idx in crooked_indices:
            num, prim = tray_caps[idx]
            xf = UsdGeom.Xformable(prim)
            rot_op = next((op for op in xf.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
            if rot_op:
                orig_rot = rot_op.Get()
                tilt_x = random.choice([-1, 1]) * random.uniform(18.0, 30.0)
                tilt_y = random.choice([-1, 1]) * random.uniform(18.0, 30.0)
                rot_op.Set(Gf.Vec3f(orig_rot[0] + tilt_x, orig_rot[1] + tilt_y, orig_rot[2]))
                injected_records.append({
                    "slot": num,
                    "defect": "crooked_cap",
                    "action": "press_reseat",
                    "induced_tilt_deg": round(float(max(abs(tilt_x), abs(tilt_y))), 1)
                })

        # 2. Inject Severely Dislodged Caps (Tilt 45° to 60°)
        for idx in dislodged_indices:
            num, prim = tray_caps[idx]
            xf = UsdGeom.Xformable(prim)
            rot_op = next((op for op in xf.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
            trans_op = next((op for op in xf.GetOrderedXformOps() if "translate" in op.GetOpName()), None)
            if rot_op:
                orig_rot = rot_op.Get()
                tilt = random.uniform(45.0, 60.0)
                rot_op.Set(Gf.Vec3f(orig_rot[0] + tilt, orig_rot[1], orig_rot[2]))
            if trans_op:
                orig_trans = trans_op.Get()
                # Lift cap slightly off rim
                trans_op.Set(Gf.Vec3f(orig_trans[0], orig_trans[1], orig_trans[2] + 0.005))
            injected_records.append({
                "slot": num,
                "defect": "severely_dislodged_cap",
                "action": "scrap_and_backfill",
                "induced_tilt_deg": 50.0
            })

        # 3. Inject Missing Caps (Hide cap geometry)
        for idx in missing_indices:
            num, prim = tray_caps[idx]
            img = UsdGeom.Imageable(prim)
            img.MakeInvisible()
            injected_records.append({
                "slot": num,
                "defect": "missing_cap",
                "action": "scrap_and_backfill",
                "induced_tilt_deg": None
            })

        print(f"[inject_defects] Injected {len(injected_records)} defects: {injected_records}")

        # Render stage through CapSealCam
        prim_path = "/TrayInspection/CapSealCam/CapSealCam"
        rp = rep.create.render_product(prim_path, (rw, rh))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        rep.orchestrator.set_capture_on_play(False)

        for _ in range(8):
            kit.update()
            rep.orchestrator.step(rt_subframes=16, delta_time=0.0)

        data = np.asarray(ann.get_data())
        img = data[..., :3].copy()

        out_path = run_dir / "render.png"
        tmp_pinhole = str(out_path) + ".pinhole.png"
        cv2.imwrite(tmp_pinhole, img[..., ::-1])

        # Apply lens distortion
        subprocess.run([
            sys.executable,
            str(CSI_DIR / "tools" / "apply_lens.py"),
            tmp_pinhole,
            str(cam_json_path),
            str(out_path)
        ], check=True)
        if os.path.exists(tmp_pinhole):
            os.remove(tmp_pinhole)

        with open(run_dir / "ground_truth_defects.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "seed": seed,
                "injected_defects": injected_records
            }, f, indent=2)

        print(f"[inject_defects] Successfully rendered defective scene to: {out_path}")
    finally:
        kit.close()


def main():
    parser = argparse.ArgumentParser(description="Render synthetic defective tray")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--crooked", type=int, default=3, help="Number of crooked caps to inject")
    parser.add_argument("--dislodged", type=int, default=2, help="Number of severely dislodged caps to inject")
    parser.add_argument("--missing", type=int, default=2, help="Number of missing caps to inject")
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir or str(REPO_ROOT / "runs" / f"{timestamp}_cap_defects_synthetic")
    render_defective_tray(
        out_dir=out_dir,
        num_crooked=args.crooked,
        num_missing=args.missing,
        num_dislodged=args.dislodged,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
