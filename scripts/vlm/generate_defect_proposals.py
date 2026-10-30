#!/usr/bin/env python3
"""Run class-agnostic LocateAnything proposal generation over real images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inspection.defect_training.config import load_project_config  # noqa: E402
from src.inspection.defect_training.locate_anything import generate_proposals  # noqa: E402
from src.inspection.defect_training.seed_import import load_split_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--include-splits",
        default="",
        help="Optional comma-separated subset of train,val,test",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from scripts.vlm.evaluate_locateanything_real5 import Worker

    config = load_project_config(args.config)
    model = args.model or config.localization.model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    worker = Worker(model, args.adapter_root)
    report = generate_proposals(
        config,
        args.images,
        load_split_manifest(args.splits),
        args.output,
        lambda image, prompt: worker.predict(image, prompt, args.max_new_tokens),
        model=model,
        adapter=str(args.adapter_root.resolve()) if args.adapter_root else None,
        include_splits={
            value.strip() for value in re.split(r"[,+]", args.include_splits) if value.strip()
        } or None,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
