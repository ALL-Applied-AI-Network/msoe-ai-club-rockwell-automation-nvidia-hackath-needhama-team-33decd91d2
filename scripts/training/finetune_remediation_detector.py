#!/usr/bin/env python3
"""Fine-tune a remediation YOLO checkpoint and evaluate it on fixed val data."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device string, e.g. 0 or 0,1,2,3")
    args = parser.parse_args()
    # Preserve Ultralytics model aliases (for example ``yolo11m.pt``) so it
    # can fetch an official pretrained checkpoint when no local file exists.
    model_path = args.model.resolve() if args.model.exists() else str(args.model)
    data_path = args.data.resolve()
    eval_data_path = args.eval_data.resolve()
    out_path = args.out.resolve()
    if out_path.exists():
        raise FileExistsError(f"Training output already exists: {out_path}")
    model = YOLO(str(model_path))
    model.train(
        data=str(data_path), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        project=str(out_path.parent), name=out_path.name, exist_ok=False, device=args.device,
        seed=42, degrees=5.0, fliplr=0.0, scale=0.1, mosaic=0.2, hsv_s=0.2, hsv_v=0.2,
    )
    weights = out_path / "weights" / "best.pt"
    if not weights.is_file():
        raise RuntimeError(f"No best checkpoint produced at {weights}")
    metrics = YOLO(str(weights)).val(
        data=str(eval_data_path), split="val", device=0, imgsz=args.imgsz,
        plots=False, project=str(out_path.parent), name=out_path.name + "_eval", exist_ok=False,
    )
    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    result = {
        "initial_model": str(model_path), "weights": str(weights),
        "training_data": str(data_path), "evaluation_data": str(eval_data_path),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": {
            name: {
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "mAP50": float(metrics.box.ap50[index]),
                "mAP50_95": float(metrics.box.ap[index]),
            }
            for index, name in names.items()
            if index < len(metrics.box.p)
        },
    }
    (out_path / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
