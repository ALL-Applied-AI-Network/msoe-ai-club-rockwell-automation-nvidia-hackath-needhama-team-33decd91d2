#!/usr/bin/env python3
"""
Train YOLOv11 Object Detection Model on Full-Frame Tray Images.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_YAML = REPO_ROOT / "shared" / "inspection_models" / "yolo_detection_dataset" / "data.yaml"
RUNS_DIR = REPO_ROOT / "runs" / "train_yolo_detection"
PERM_WEIGHTS = REPO_ROOT / "shared" / "inspection_models" / "vial_fullframe_det_best.pt"


def train_yolo_detector(epochs=35, batch=16, imgsz=640, model_type="yolo11n.pt"):
    print(f"[YOLOv11-det] Starting training on {DATA_YAML}...")
    model = YOLO(model_type)

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR),
        name="vial_fullframe_det",
        exist_ok=True,
        device=0,
        verbose=True,
        # Full-frame geometric & color augmentations:
        degrees=15.0,
        fliplr=0.5,
        flipud=0.0,
        scale=0.1,
        mosaic=1.0,
        hsv_h=0.02,
        hsv_s=0.3,
        hsv_v=0.3
    )

    best_pt = RUNS_DIR / "vial_fullframe_det" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, PERM_WEIGHTS)
        print(f"[YOLOv11-det] Permanent model weights saved to: {PERM_WEIGHTS}")

    print("[YOLOv11-det] Training complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_yolo_detector(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
