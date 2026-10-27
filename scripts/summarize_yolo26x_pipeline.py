#!/usr/bin/env python3
"""Create a compact machine-readable and Markdown comparison of all stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric_row(name: str, metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text())
    return {
        "model": name,
        "checkpoint": payload["checkpoint"],
        "imgsz": payload["imgsz"],
        "synthetic": payload["synthetic"]["overall"],
        "real": payload["real"]["overall"],
        "synthetic_per_class": payload["synthetic"]["per_class"],
        "real_per_class": payload["real"]["per_class"],
        "real_evaluation_is_resubstitution": True,
    }


def value(number: float | None) -> str:
    return "N/A" if number is None else f"{number:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    stages = (
        ("Incumbent YOLO26l real-five", args.root / "00_incumbent_yolo26l_1280/evaluation/metrics.json"),
        ("YOLO26x synthetic-only", args.root / "01_synthetic_only_1280/evaluation/metrics.json"),
        ("YOLO26x physical fine-tune", args.root / "02_real_finetuned_1280/evaluation/metrics.json"),
        ("YOLO26x balanced replay", args.root / "03_mixed_replay_1280/evaluation/metrics.json"),
    )
    rows = []
    for name, path in stages:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(metric_row(name, path))

    (args.root / "comparison.json").write_text(json.dumps({
        "imgsz": 1280,
        "real_images": 5,
        "warning": "Physical metrics reuse the five training images and measure fit, not unseen-scene generalization.",
        "models": rows,
    }, indent=2) + "\n")

    lines = [
        "# YOLO26x 1280 pipeline comparison",
        "",
        "All stages use the same 100-image synthetic test set and all five reviewed physical frames. "
        "The physical frames are training data for stages 2 and 3, so those metrics are resubstitution.",
        "",
        "| Model | Synthetic P | Synthetic R | Synthetic mAP50 | Synthetic mAP50-95 | Real P | Real R | Real mAP50 | Real mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        synthetic = row["synthetic"]
        real = row["real"]
        lines.append(
            f"| {row['model']} | {value(synthetic['precision'])} | {value(synthetic['recall'])} | "
            f"{value(synthetic['mAP50'])} | {value(synthetic['mAP50_95'])} | "
            f"{value(real['precision'])} | {value(real['recall'])} | "
            f"{value(real['mAP50'])} | {value(real['mAP50_95'])} |"
        )
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(args.root / "comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
