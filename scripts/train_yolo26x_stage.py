#!/usr/bin/env python3
"""Train one stage of the 1280-pixel YOLO26x remediation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("synthetic", "physical", "mixed"), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.out.exists():
        raise FileExistsError(f"Training output already exists: {args.out}")

    model_path = Path(args.model)
    model_source = str(model_path.resolve()) if model_path.is_file() else args.model
    common = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=8,
        cache="ram",
        project=str(args.out.resolve().parent),
        name=args.out.name,
        exist_ok=False,
        amp=True,
        seed=42,
        deterministic=True,
        fliplr=0.0,
        flipud=0.0,
        plots=True,
        save=True,
        save_period=max(10, args.epochs // 4),
    )
    if args.stage == "synthetic":
        stage_options = dict(
            optimizer="auto",
            degrees=3.0,
            translate=0.04,
            scale=0.10,
            shear=0.5,
            perspective=0.0001,
            mosaic=0.2,
            mixup=0.0,
            hsv_h=0.01,
            hsv_s=0.2,
            hsv_v=0.2,
            patience=20,
        )
    elif args.stage == "physical":
        stage_options = dict(
            optimizer="AdamW",
            lr0=2e-4,
            lrf=0.1,
            weight_decay=5e-4,
            warmup_epochs=3,
            patience=50,
            cos_lr=True,
            degrees=2.0,
            translate=0.02,
            scale=0.08,
            shear=0.5,
            perspective=0.0001,
            mosaic=0.0,
            mixup=0.0,
            hsv_h=0.01,
            hsv_s=0.15,
            hsv_v=0.15,
        )
    else:
        stage_options = dict(
            optimizer="AdamW",
            lr0=1e-4,
            lrf=0.1,
            weight_decay=5e-4,
            warmup_epochs=2,
            patience=15,
            cos_lr=True,
            degrees=2.0,
            translate=0.02,
            scale=0.08,
            shear=0.5,
            perspective=0.0001,
            mosaic=0.0,
            mixup=0.0,
            hsv_h=0.01,
            hsv_s=0.15,
            hsv_v=0.15,
        )

    model = YOLO(model_source)
    model.train(**common, **stage_options)

    best = args.out / "weights" / "best.pt"
    last = args.out / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"Training did not retain expected weights under {args.out / 'weights'}")
    (args.out / "stage_provenance.json").write_text(json.dumps({
        "stage": args.stage,
        "initial_model": model_source,
        "training_data": str(args.data.resolve()),
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "best_weights": str(best),
        "last_weights": str(last),
    }, indent=2) + "\n")
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
