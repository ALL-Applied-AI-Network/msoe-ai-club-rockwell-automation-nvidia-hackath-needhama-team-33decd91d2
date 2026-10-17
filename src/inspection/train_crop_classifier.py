"""
Train YOLOv11 Crop Classifier for Dataset 1 Fixture Inspection.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = REPO_ROOT / "shared" / "inspection_models" / "crop_dataset"


def train_yolo_classifier(epochs=30, batch=32, imgsz=64, model_type="yolo11n-cls.pt"):
    print(f"[YOLOv11-cls] Starting training on {DATASET_DIR}...")
    model = YOLO(model_type)

    results = model.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(REPO_ROOT / "runs" / "train_yolo_classifier"),
        name="vial_aperture_cls",
        exist_ok=True,
        device=0,
        verbose=True
    )
    print("[YOLOv11-cls] Training complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 classifier on vial crops")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=64)
    args = parser.parse_args()

    train_yolo_classifier(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
