#!/usr/bin/env python3
"""Measure RGB/depth detector locations against Isaac truth for evaluation only.

The truth scene is never fed back into detector or controller commands.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.remediation.geometry import world_to_base


def ground_truth(report):
    result = []
    objects = {item["path"]: item for item in report["objects"]}
    for label in report["labels"]:
        if label["class_name"] == "missing_cap":
            point = report["inventory"]["slots"][label["slot"]]["rim_world_m"]
        else:
            point = objects[label["path"]]["position_world_m"]
        result.append(
            {
                "class_name": label["class_name"],
                "base_pos_m": world_to_base(point).tolist(),
            }
        )
    return result


def evaluate(detections, truth):
    rows = []
    for name in ("good_cap", "crooked_cap", "missing_cap", "table_cap"):
        predicted = [p for p in detections if p["class_name"] == name]
        expected = [g for g in truth if g["class_name"] == name]
        unmatched = set(range(len(expected)))
        for item in sorted(predicted, key=lambda x: -x["confidence"]):
            if not unmatched:
                break
            p = np.asarray(item["base_pos_m"])[:2]
            index = min(
                unmatched,
                key=lambda i: np.linalg.norm(
                    p - np.asarray(expected[i]["base_pos_m"])[:2]
                ),
            )
            error = float(
                np.linalg.norm(p - np.asarray(expected[index]["base_pos_m"])[:2])
            )
            rows.append(
                {
                    "class_name": name,
                    "confidence": item["confidence"],
                    "xy_error_m": error,
                    "depth_median_m": item.get("depth_median_m"),
                }
            )
            unmatched.remove(index)
    summary = {}
    for name in ("good_cap", "crooked_cap", "missing_cap", "table_cap"):
        errors = [r["xy_error_m"] for r in rows if r["class_name"] == name]
        summary[name] = {
            "matched": len(errors),
            "median_xy_error_m": float(np.median(errors)) if errors else None,
            "max_xy_error_m": float(np.max(errors)) if errors else None,
        }
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    detection = json.loads(Path(args.detections).read_text())
    report = json.loads(Path(args.qualification).read_text())
    rows, summary = evaluate(detection["targets"], ground_truth(report))
    result = {
        "schema_version": 1,
        "evaluation_only": True,
        "contact_localization_validated": False,
        "matches": rows,
        "summary": summary,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
