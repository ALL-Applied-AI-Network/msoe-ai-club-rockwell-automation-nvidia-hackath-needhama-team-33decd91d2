"""
Procedural Synthetic Crop Generator for YOLOv11 & TAO Training.

Renders randomized scenarios with Isaac Sim:
- Nominal vials (pass)
- Missing caps (missing_cap)
- Crooked/tilted caps (crooked_cap)
- Empty apertures (no_vial)
- Damaged / misaligned vials (damaged_vial)

Cuts aperture crops directly into standard classification dataset directories:
  data/crop_dataset/{train,val}/{pass,missing_cap,crooked_cap,damaged_vial,no_vial}/
"""

import argparse
from datetime import datetime
import json
import os
import random
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSI_DIR = REPO_ROOT / "shared" / "data" / "csi_digital_twin_v1"
DATASET_DIR = REPO_ROOT / "shared" / "inspection_models" / "crop_dataset"


def setup_dataset_dirs():
    splits = ["train", "val"]
    classes = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
    for s in splits:
        for c in classes:
            (DATASET_DIR / s / c).mkdir(parents=True, exist_ok=True)


def generate_batch(num_scenes=10, seed=42):
    setup_dataset_dirs()
    random.seed(seed)
    np.random.seed(seed)

    apertures_path = REPO_ROOT / "src" / "inspection" / "config" / "fixture_g_apertures.json"
    with open(apertures_path) as f:
        apertures = json.load(f)

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
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, Gf

        ctx = omni.usd.get_context()
        ctx.open_stage(str(stage_file))
        for _ in range(15):
            kit.update()

        stage = ctx.get_stage()

        # Render setup
        prim_path = "/TrayInspection/CapSealCam/CapSealCam"
        rp = rep.create.render_product(prim_path, (rw, rh))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        rep.orchestrator.set_capture_on_play(False)

        crop_r = 16
        total_crops = {c: 0 for c in ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]}

        for scene_idx in range(num_scenes):
            split = "val" if (scene_idx % 5 == 0) else "train"
            print(f"\n--- [Synthetic Generator] Rendering Scene {scene_idx + 1}/{num_scenes} ({split}) ---")

            # Reset all caps and tubes to default transforms
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

            # Decide defect assignments for this scene
            slot_labels = {}
            for i in range(1, 31):
                slot_labels[i] = "pass"

            # Randomly select slots for defects
            all_slots = list(range(1, 31))
            random.shuffle(all_slots)

            # Assign 2 missing caps
            missing_slots = all_slots[:2]
            for s in missing_slots:
                slot_labels[s] = "missing_cap"
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                if cap.IsValid():
                    UsdGeom.Imageable(cap).MakeInvisible()

            # Assign 3 crooked caps
            crooked_slots = all_slots[2:5]
            for s in crooked_slots:
                slot_labels[s] = "crooked_cap"
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                if cap.IsValid():
                    xf = UsdGeom.Xformable(cap)
                    rot_op = next((op for op in xf.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                    if rot_op:
                        tilt_x = random.choice([-1, 1]) * random.uniform(15.0, 32.0)
                        tilt_y = random.choice([-1, 1]) * random.uniform(15.0, 32.0)
                        rot_op.Set(Gf.Vec3f(tilt_x, tilt_y, 0.0))

            # Assign 2 empty slots (no_vial)
            empty_slots = all_slots[5:7]
            for s in empty_slots:
                slot_labels[s] = "no_vial"
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                tube = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_tube")
                if cap.IsValid():
                    UsdGeom.Imageable(cap).MakeInvisible()
                if tube.IsValid():
                    UsdGeom.Imageable(tube).MakeInvisible()

            # Assign 1 damaged / misaligned vial (damaged_vial)
            damaged_slots = all_slots[7:8]
            for s in damaged_slots:
                slot_labels[s] = "damaged_vial"
                tube = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_tube")
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                if tube.IsValid():
                    xf_t = UsdGeom.Xformable(tube)
                    rot_op_t = next((op for op in xf_t.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                    if rot_op_t:
                        rot_op_t.Set(Gf.Vec3f(random.uniform(10.0, 20.0), random.uniform(-15.0, 15.0), 0.0))
                if cap.IsValid():
                    xf_c = UsdGeom.Xformable(cap)
                    rot_op_c = next((op for op in xf_c.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                    if rot_op_c:
                        rot_op_c.Set(Gf.Vec3f(random.uniform(20.0, 35.0), 0.0, 0.0))

            # Render scene
            for _ in range(6):
                kit.update()
                rep.orchestrator.step(rt_subframes=16, delta_time=0.0)

            data = np.asarray(ann.get_data())
            img_bgr = data[..., :3][..., ::-1].copy()

            # Apply lens distortion via temporary file
            tmp_pinhole = f"/tmp/pinhole_scene_{scene_idx}.png"
            tmp_distorted = f"/tmp/distorted_scene_{scene_idx}.png"
            cv2.imwrite(tmp_pinhole, img_bgr)

            subprocess.run([
                sys.executable,
                str(CSI_DIR / "tools" / "apply_lens.py"),
                tmp_pinhole,
                str(cam_json_path),
                tmp_distorted
            ], check=True)

            final_img = cv2.imread(tmp_distorted)
            if os.path.exists(tmp_pinhole):
                os.remove(tmp_pinhole)
            if os.path.exists(tmp_distorted):
                os.remove(tmp_distorted)

            # Cut crops for each aperture
            for s_id, info in apertures.items():
                slot_num = info["slot"]
                label = slot_labels[slot_num]
                cx, cy = int(info["x"]), int(info["y"])

                x1, x2 = max(0, cx - crop_r), min(final_img.shape[1], cx + crop_r)
                y1, y2 = max(0, cy - crop_r), min(final_img.shape[0], cy + crop_r)
                crop = final_img[y1:y2, x1:x2]

                crop_filename = f"scene{scene_idx:03d}_{s_id}_{label}.png"
                save_path = DATASET_DIR / split / label / crop_filename
                cv2.imwrite(str(save_path), crop)
                total_crops[label] += 1

        print("\n[Synthetic Generator] Generation Complete!")
        print("Total crops generated per class:")
        for k, v in total_crops.items():
            print(f"  {k}: {v}")

    finally:
        kit.close()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic crop dataset for YOLOv11")
    parser.add_argument("--scenes", type=int, default=12, help="Number of scene variations to render")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_batch(num_scenes=args.scenes, seed=args.seed)


if __name__ == "__main__":
    main()
