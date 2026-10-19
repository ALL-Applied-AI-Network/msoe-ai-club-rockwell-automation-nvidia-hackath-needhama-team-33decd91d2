#!/usr/bin/env python3
"""
Generate Full-Frame YOLO Object Detection Dataset (YOLO Format) from Isaac Sim Twin.
Generates full 640x480 images with bounding boxes [class_id, x_center, y_center, width, height]
Classes:
  0: pass (normal seated vial + cap)
  1: crooked_cap (tilted / cross-threaded / unseated cap)
  2: missing_cap (open vial body without cap)
  3: damaged_vial (fractured or severely damaged vial)
  4: no_vial (empty aperture hole)
"""

import argparse
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
YOLO_DET_DIR = REPO_ROOT / "shared" / "inspection_models" / "yolo_detection_dataset"

CLASS_MAP = {
    "pass": 0,
    "crooked_cap": 1,
    "missing_cap": 2,
    "damaged_vial": 3,
    "no_vial": 4
}


def setup_dataset_dirs():
    for split in ["train", "val"]:
        (YOLO_DET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def generate_fullframe_dataset(num_scenes=30, seed=42):
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
    img_w, img_h = cam_cfg["final_image_px"]  # 640, 480

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
        stage = ctx.get_stage()

        prim_path = "/TrayInspection/CapSealCam/CapSealCam"
        rp = rep.create.render_product(prim_path, (rw, rh))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        rep.orchestrator.set_capture_on_play(False)

        box_w = 34.0 / img_w  # Normalized width ~ 34px
        box_h = 34.0 / img_h  # Normalized height ~ 34px

        for scene_idx in range(num_scenes):
            split = "val" if (scene_idx % 5 == 0) else "train"
            print(f"\n--- [YOLO Detection Generator] Scene {scene_idx + 1}/{num_scenes} ({split}) ---")

            # 1. Reset all components
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

            # 2. Defect assignments
            slot_labels = {i: "pass" for i in range(1, 31)}
            all_slots = list(range(1, 31))
            random.shuffle(all_slots)

            # Missing caps
            for s in all_slots[:2]:
                slot_labels[s] = "missing_cap"
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                if cap.IsValid():
                    UsdGeom.Imageable(cap).MakeInvisible()

            # Crooked caps with multi-angle variance
            for idx_c, s in enumerate(all_slots[2:6]):
                slot_labels[s] = "crooked_cap"
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                if cap.IsValid():
                    xf = UsdGeom.Xformable(cap)
                    rot_op = next((op for op in xf.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                    if rot_op:
                        if idx_c == 0:
                            tilt_x = random.choice([-1, 1]) * random.uniform(8.0, 16.0)
                            tilt_y = random.choice([-1, 1]) * random.uniform(8.0, 16.0)
                        elif idx_c == 1:
                            tilt_x = random.choice([-1, 1]) * random.uniform(25.0, 45.0)
                            tilt_y = random.choice([-1, 1]) * random.uniform(25.0, 45.0)
                        elif idx_c == 2:
                            tilt_x = random.choice([-1, 1]) * random.uniform(65.0, 88.0)
                            tilt_y = random.uniform(-15.0, 15.0)
                        else:
                            tilt_x = random.uniform(-35.0, 35.0)
                            tilt_y = random.uniform(-35.0, 35.0)
                        rot_op.Set(Gf.Vec3f(tilt_x, tilt_y, random.uniform(0.0, 360.0)))

            # Empty holes (no_vial)
            for s in all_slots[6:9]:
                slot_labels[s] = "no_vial"
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                tube = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_tube")
                if cap.IsValid():
                    UsdGeom.Imageable(cap).MakeInvisible()
                if tube.IsValid():
                    UsdGeom.Imageable(tube).MakeInvisible()

            # Damaged / skewed vials
            for s in all_slots[9:11]:
                slot_labels[s] = "damaged_vial"
                tube = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_tube")
                cap = stage.GetPrimAtPath(f"/TrayInspection/G{s:02d}_cap")
                if tube.IsValid():
                    xf_t = UsdGeom.Xformable(tube)
                    rot_op_t = next((op for op in xf_t.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                    if rot_op_t:
                        rot_op_t.Set(Gf.Vec3f(random.uniform(8.0, 22.0), random.uniform(-20.0, 20.0), random.uniform(0.0, 360.0)))
                if cap.IsValid():
                    xf_c = UsdGeom.Xformable(cap)
                    rot_op_c = next((op for op in xf_c.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                    if rot_op_c:
                        rot_op_c.Set(Gf.Vec3f(random.uniform(15.0, 40.0), random.uniform(-25.0, 25.0), random.uniform(0.0, 360.0)))

            # 3. Step & render
            for _ in range(6):
                kit.update()
                rep.orchestrator.step(rt_subframes=16, delta_time=0.0)

            data = np.asarray(ann.get_data())
            img_bgr = data[..., :3][..., ::-1].copy()

            # Apply camera lens distortion
            tmp_pinhole = f"/tmp/pinhole_det_{scene_idx}.png"
            tmp_distorted = f"/tmp/distorted_det_{scene_idx}.png"
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

            # 4. Save image and bounding box label file
            img_name = f"scene_{scene_idx:03d}.jpg"
            img_out = YOLO_DET_DIR / "images" / split / img_name
            cv2.imwrite(str(img_out), final_img)

            txt_name = f"scene_{scene_idx:03d}.txt"
            txt_out = YOLO_DET_DIR / "labels" / split / txt_name

            lines = []
            for s_id, info in apertures.items():
                slot_num = info["slot"]
                cls_name = slot_labels[slot_num]
                cls_id = CLASS_MAP[cls_name]

                # Normalized coordinates
                cx_norm = info["x"] / float(img_w)
                cy_norm = info["y"] / float(img_h)

                lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {box_w:.6f} {box_h:.6f}")

            with open(txt_out, "w") as f:
                f.write("\n".join(lines) + "\n")

        print(f"[YOLO Detection Generator] Complete! Dataset saved to: {YOLO_DET_DIR}")

        # Write data.yaml configuration file for Ultralytics YOLO
        data_yaml = YOLO_DET_DIR / "data.yaml"
        with open(data_yaml, "w") as f:
            f.write(f"""path: {YOLO_DET_DIR}
train: images/train
val: images/val
names:
  0: pass
  1: crooked_cap
  2: missing_cap
  3: damaged_vial
  4: no_vial
""")
        print(f"[YOLO Detection Generator] Wrote {data_yaml}")

    finally:
        kit.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=30)
    args = parser.parse_args()
    generate_fullframe_dataset(num_scenes=args.scenes)
