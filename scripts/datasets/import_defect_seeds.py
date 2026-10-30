#!/usr/bin/env python3
"""Import a reviewed YOLO seed set into the generic defect-training schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inspection.defect_training.config import load_project_config  # noqa: E402
from src.inspection.defect_training.seed_import import (  # noqa: E402
    import_seed_dataset,
    load_split_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = import_seed_dataset(
        load_project_config(args.config),
        args.images,
        args.labels,
        load_split_manifest(args.splits),
        args.output,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
