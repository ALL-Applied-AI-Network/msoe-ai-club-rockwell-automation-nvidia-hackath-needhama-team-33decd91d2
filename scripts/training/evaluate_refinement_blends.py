#!/usr/bin/env python3
"""Score interpolations between saved YOLO and VLM-refined boxes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evaluate_yolo_vlm_box_refinement import summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--blends", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    source_records = json.loads(args.predictions.read_text())
    results = {}
    for blend in args.blends:
        records = copy.deepcopy(source_records)
        for record in records:
            blended = []
            for original, full_vlm in zip(record["yolo"], record["hybrid"]):
                item = copy.deepcopy(original)
                if full_vlm.get("refined"):
                    item["box"] = [
                        (1.0 - blend) * yolo_value + blend * vlm_value
                        for yolo_value, vlm_value in zip(original["box"], full_vlm["box"])
                    ]
                blended.append(item)
            record["blend"] = blended
        results[str(blend)] = summarize(records, "blend", args.confidence)
        overall = results[str(blend)]["overall"]
        print(
            f"blend={blend:.2f} AP50={overall['ap50']:.6f} "
            f"mAP50-95={overall['map50_95']:.6f} IoU={overall['mean_iou_matched']:.6f} "
            f"center={overall['mean_center_error_percent_diagonal']:.6f}",
            flush=True,
        )
    args.output.write_text(json.dumps({"confidence": args.confidence, "results": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
