#!/usr/bin/env python3
"""Train a diagnostic YOLO26x seed detector from a tiny real-only dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite training output: {args.output}")
    model = YOLO(args.model)
    augmentation = {
        "hsv_h": 0.0 if args.no_augment else 0.01,
        "hsv_s": 0.0 if args.no_augment else 0.20,
        "hsv_v": 0.0 if args.no_augment else 0.20,
        "degrees": 0.0 if args.no_augment else 2.0,
        "translate": 0.0 if args.no_augment else 0.03,
        "scale": 0.0 if args.no_augment else 0.10,
        "shear": 0.0 if args.no_augment else 0.5,
        "perspective": 0.0 if args.no_augment else 0.0001,
    }
    train_args = dict(
        data=str(args.data.resolve()), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, project=str(args.output.parent.resolve()), name=args.output.name,
        exist_ok=False, workers=4, cache="ram", optimizer="AdamW", lr0=args.learning_rate,
        lrf=0.05, weight_decay=5.0e-4, warmup_epochs=3.0, cos_lr=True, patience=args.patience,
        box=10.0, cls=1.0, dfl=1.5, seed=42, deterministic=True,
        amp=True, save=True, save_period=max(10, args.epochs // 10), val=True, plots=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, max_det=100, verbose=True,
        flipud=0.0, fliplr=0.0, **augmentation,
    )
    if args.freeze > 0:
        train_args["freeze"] = args.freeze
    model.train(**train_args)
    best = args.output / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training did not produce {best}")
    provenance = {
        "purpose": "Two-image real-only seed model for pseudo-label proposal generation",
        "initial_checkpoint": args.model,
        "domain_training_data": str(args.data.resolve()),
        "synthetic_domain_images_used": False,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "freeze": args.freeze,
        "learning_rate": args.learning_rate,
        "augmentation_enabled": not args.no_augment,
        "best_checkpoint": str(best.resolve()),
        "warning": "This seed model is for low-threshold proposals only; its reused-train validation metrics are not generalization evidence.",
    }
    (args.output / "seed_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
