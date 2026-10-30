#!/usr/bin/env python3
"""Select diverse TableCam raw frames and render low-threshold seed-YOLO predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from ultralytics import YOLO


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    return [paths[round(index * (len(paths) - 1) / (count - 1))] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {args.output}")
    input_dir = args.output / "inputs"
    input_dir.mkdir(parents=True)
    selected = []
    for raw_dir in sorted(args.raw_root.glob("2026-10-26_*_physical_policy/tablecam_raw")):
        frames = sorted(raw_dir.glob("*.png"))
        for frame in evenly_spaced(frames, args.per_sequence):
            destination = input_dir / f"{raw_dir.parent.name}__{frame.name}"
            shutil.copy2(frame, destination)
            selected.append({"source": str(frame.resolve()), "sample": destination.name})
    if not selected:
        raise FileNotFoundError(f"No TableCam frames found below {args.raw_root}")

    model = YOLO(str(args.model.resolve()))
    results = model.predict(
        source=str(input_dir), conf=args.confidence, iou=0.6, imgsz=args.imgsz,
        max_det=100, device=0, save=True, save_txt=True, save_conf=True,
        project=str(args.output), name="predictions", exist_ok=False, verbose=True,
    )
    payload = {
        "model": str(args.model.resolve()), "confidence": args.confidence,
        "imgsz": args.imgsz, "samples": selected,
        "detections_per_image": {
            Path(result.path).name: len(result.boxes) for result in results
        },
    }
    (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
