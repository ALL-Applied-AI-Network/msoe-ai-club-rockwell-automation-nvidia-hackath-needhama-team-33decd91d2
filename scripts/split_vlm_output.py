#!/usr/bin/env python3
"""Split a combined VLM result file into one JSON file per image."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "images" / "VLM_samples" / "full_image_labels.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "images" / "VLM_samples"
DEFAULT_TIMEOUT_LIST = DEFAULT_OUTPUT_DIR / "timeout_images.txt"


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-list", type=Path, default=DEFAULT_TIMEOUT_LIST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    timeout_list = args.timeout_list.expanduser().resolve()
    with input_path.open() as handle:
        payload = json.load(handle)

    records = payload.get("images", [])
    successful = []
    timed_out = []
    for record in records:
        if not isinstance(record, dict) or not record.get("filename"):
            continue
        if record.get("error"):
            timed_out.append(record["filename"])
        else:
            successful.append(record)

    for record in successful:
        write_json_atomic(output_dir / f"{record['filename']}.json", record)

    timeout_list.parent.mkdir(parents=True, exist_ok=True)
    timeout_list.write_text(
        "# Reserved for a stronger-model labeling pass.\n"
        + "\n".join(sorted(timed_out))
        + ("\n" if timed_out else "")
    )
    print(f"Wrote successful per-image labels: {len(successful)}")
    print(f"Reserved timeout images: {len(timed_out)}")
    print(f"Output directory: {output_dir}")
    print(f"Timeout list: {timeout_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
