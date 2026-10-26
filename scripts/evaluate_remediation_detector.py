#!/usr/bin/env python3
"""Evaluate a remediation detector on a fixed validation dataset."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    args = parser.parse_args()
    metrics = YOLO(str(args.model)).val(
        data=str(args.data), split=args.split, device=0, imgsz=args.imgsz,
        plots=False, project=str(args.out.parent), name=args.out.stem, exist_ok=False,
    )
    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    result = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
