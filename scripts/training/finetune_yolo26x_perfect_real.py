#!/usr/bin/env python3
"""Conservatively continue YOLO26x on corrected real boxes plus synthetic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0,1,2,3")
    args = parser.parse_args()

    model_path = args.model.resolve()
    data_path = args.data.resolve()
    output = args.output.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {output}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(output.parent),
        name=output.name,
        exist_ok=False,
        workers=8,
        cache="ram",
        optimizer="AdamW",
        lr0=2.0e-5,
        lrf=0.1,
        momentum=0.937,
        weight_decay=5.0e-4,
        warmup_epochs=1.0,
        cos_lr=True,
        patience=8,
        box=10.0,
        cls=0.5,
        dfl=1.5,
        seed=42,
        deterministic=True,
        amp=True,
        save=True,
        save_period=2,
        val=True,
        split="val",
        plots=True,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.15,
        degrees=2.0,
        translate=0.02,
        scale=0.08,
        shear=0.5,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        max_det=300,
        verbose=True,
    )

    best = output / "weights" / "best.pt"
    last = output / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"Expected checkpoints were not produced under {output / 'weights'}")
    provenance = {
        "purpose": "Improve native YOLO localization using reviewed classes and VLM-refined real boxes",
        "initial_checkpoint": str(model_path),
        "training_data": str(data_path),
        "output": str(output),
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "epochs_requested": args.epochs,
        "batch_total": args.batch,
        "imgsz": args.imgsz,
        "devices": args.device,
        "optimizer": "AdamW",
        "learning_rate": 2.0e-5,
        "box_loss_weight": 10.0,
        "class_loss_weight": 0.5,
        "selection_note": "best.pt is selected against the untouched synthetic validation split",
        "real_evaluation_warning": "all five real images occur in training and are memorization/localization checks",
    }
    (output / "stage_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
