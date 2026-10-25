#!/usr/bin/env python3
"""Recover valid per-image records from an interrupted VLM JSON checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = REPO_ROOT / "images" / "pruned_images"
DEFAULT_OUTPUT = REPO_ROOT / "images" / "VLM_samples" / "full_image_labels.json"
RECORD_START = re.compile(r'\{\s*"filename"\s*:')


def recover_records(raw: bytes) -> Dict[str, Dict[str, Any]]:
    """Decode complete image objects even when the outer JSON is damaged."""

    # NUL padding can be left behind when a job is killed during a checkpoint.
    # Whitespace keeps damaged regions separated while allowing JSON decoding.
    text = raw.replace(b"\0", b" ").decode("utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    records: Dict[str, Dict[str, Any]] = {}
    for match in RECORD_START.finditer(text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        filename = value.get("filename")
        if not isinstance(filename, str) or not filename:
            continue
        if not isinstance(value.get("apertures"), dict):
            continue

        # Prefer a successful record when a damaged file contains both a
        # timeout and a later successful copy of the same image.
        previous = records.get(filename)
        if previous is None or (previous.get("error") and not value.get("error")):
            records[filename] = value
    return records


def ordered_records(records: Dict[str, Dict[str, Any]], image_dir: Path) -> list[Dict[str, Any]]:
    paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    return [records[path.name] for path in paths if path.name in records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    image_dir = args.images.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input does not exist: {input_path}")
    if not image_dir.is_dir():
        raise SystemExit(f"image directory does not exist: {image_dir}")

    raw = input_path.read_bytes()
    records = recover_records(raw)
    if not records:
        raise SystemExit("no complete image records could be recovered")
    ordered = ordered_records(records, image_dir)

    if input_path == output_path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = input_path.with_name(f"{input_path.name}.corrupt-{stamp}")
        shutil.copy2(input_path, backup)
        print(f"Backup: {backup}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired = {
        "model": "qwen3.8-27b-ud-iq3-s",
        "images_dir": str(image_dir),
        "crop_fraction": 0.5,
        "images": ordered,
    }
    temp_path = output_path.with_name(f".{output_path.name}.repair.tmp")
    with temp_path.open("w") as handle:
        json.dump(repaired, handle, indent=2)
        handle.write("\n")
    temp_path.replace(output_path)

    failures = sum(1 for record in ordered if record.get("error"))
    print(f"Recovered records: {len(ordered)}")
    print(f"Recovered timeout/error records: {failures}")
    print(f"Images still needing requests: {len(list(image_dir.iterdir())) - len(ordered) + failures}")
    print(f"Repaired output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
