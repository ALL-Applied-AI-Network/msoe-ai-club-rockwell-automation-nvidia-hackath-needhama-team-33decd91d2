#!/usr/bin/env python3
"""Run a LocateAnything adapter over a labeled YOLO dataset with resume support."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.vlm.evaluate_locateanything_real5 import (  # noqa: E402
    CLASS_NAMES,
    PROMPTS,
    Worker,
    load_ground_truth,
    parse_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    record_dir = args.output / "vlm_records"
    record_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not image_paths:
        raise FileNotFoundError(f"No images in {image_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Loading {args.model} + {args.adapter_root} on {torch.cuda.get_device_name(0)}", flush=True)
    worker = Worker(args.model, args.adapter_root)
    started = time.time()

    for image_index, image_path in enumerate(image_paths, 1):
        record_path = record_dir / f"{image_path.stem}.json"
        if record_path.is_file():
            print(f"[{image_index}/{len(image_paths)}] resume: {image_path.name}", flush=True)
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ground_truth = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        predictions, responses = [], {}
        for class_id, phrase in enumerate(PROMPTS.values()):
            query_started = time.time()
            answer = worker.predict(image, phrase, args.max_new_tokens)
            boxes = parse_boxes(answer, width, height, class_id)
            predictions.extend(boxes)
            responses[CLASS_NAMES[class_id]] = {
                "prompt": phrase,
                "answer": answer,
                "box_count": len(boxes),
            }
            print(
                f"[{image_index}/{len(image_paths)}] {image_path.name} {CLASS_NAMES[class_id]}: "
                f"{len(boxes)} boxes in {time.time() - query_started:.1f}s",
                flush=True,
            )
        record = {
            "image": str(image_path),
            "view": "raw",
            "size": [width, height],
            "ground_truth": ground_truth,
            "predictions": predictions,
            "responses": responses,
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")

    records = [json.loads((record_dir / f"{path.stem}.json").read_text()) for path in image_paths]
    payload = {
        "model": args.model,
        "adapter_root": str(args.adapter_root),
        "dataset": str(args.dataset),
        "split": args.split,
        "elapsed_seconds": time.time() - started,
        "raw": records,
    }
    output_path = args.output / "predictions_and_raw_responses.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(records)} records to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
