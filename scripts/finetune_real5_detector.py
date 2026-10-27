#!/usr/bin/env python3
"""Conservatively adapt the qualified-scene detector to five reviewed real frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    for required in (args.model, args.data):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.out.exists():
        raise FileExistsError(f"Training output already exists: {args.out}")

    model = YOLO(str(args.model.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=4,
        imgsz=args.imgsz,
        device=0,
        workers=4,
        cache="ram",
        project=str(args.out.resolve().parent),
        name=args.out.name,
        exist_ok=False,
        optimizer="AdamW",
        lr0=2e-4,
        lrf=0.1,
        weight_decay=5e-4,
        warmup_epochs=3,
        patience=50,
        cos_lr=True,
        seed=42,
        deterministic=True,
        degrees=2.0,
        translate=0.02,
        scale=0.08,
        shear=0.5,
        perspective=0.0001,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.15,
        save_period=30,
        plots=True,
    )

    best = args.out / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"No best checkpoint produced at {best}")
    (args.out / "real5_training_provenance.json").write_text(json.dumps({
        "base_checkpoint": str(args.model),
        "training_data": str(args.data),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "real_frames": 5,
        "train_equals_validation": True,
        "interpretation": "real metrics are resubstitution/training-set fit",
    }, indent=2) + "\n")
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
