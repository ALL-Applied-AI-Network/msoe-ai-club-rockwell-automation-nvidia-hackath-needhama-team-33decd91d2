#!/usr/bin/env python3
"""Convert a run's raw uint16 depth PNGs into outlined color heatmaps."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inspection.depth_visualization import render_depth_heatmap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-dir", type=Path, required=True,
                        help="directory containing raw single-channel depth PNGs")
    parser.add_argument("--raw-dir", type=Path,
                        help="where to preserve raw PNGs; default: sibling depth_raw")
    args = parser.parse_args()

    depth_dir = args.depth_dir.resolve()
    raw_dir = (args.raw_dir.resolve() if args.raw_dir else depth_dir.with_name("depth_raw"))
    paths = sorted(depth_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG files under {depth_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for path in paths:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise ValueError(f"{path} is not a raw single-channel depth image")
        raw_path = raw_dir / path.name
        if raw_path.exists():
            raise FileExistsError(f"refusing to overwrite preserved raw depth: {raw_path}")
        shutil.copy2(path, raw_path)
        if not cv2.imwrite(str(path), render_depth_heatmap(depth)):
            raise OSError(f"failed to write {path}")
        converted += 1
    print(f"converted {converted} depth frame(s); raw depth preserved under {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
