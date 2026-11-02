#!/usr/bin/env python3
"""Validate rendered remediation shards and publish their YOLO data file."""

import argparse
import json
from collections import Counter
from pathlib import Path


NAMES = {0: "good_cap", 1: "crooked_cap", 2: "missing_cap", 3: "table_cap"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    manifests = []
    for shard in range(args.num_shards):
        path = root / "manifests" / f"shard_{shard:02d}.json"
        manifests.append(json.loads(path.read_text()))
    records = sorted((record for manifest in manifests for record in manifest["records"]), key=lambda x: x["index"])
    expected = list(range(args.scenes))
    actual = [record["index"] for record in records]
    if actual != expected:
        raise RuntimeError(f"Rendered indices do not match 0..{args.scenes - 1}")
    split_counts = Counter()
    class_counts = Counter()
    for record in records:
        image = Path(record["image"])
        label = root / "labels" / record["split"] / f"{image.stem}.txt"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Incomplete sample {record['index']}: {image}, {label}")
        lines = label.read_text().splitlines()
        slot_states = record["slots"]
        represented_slots = sum(len(slot_states[state]) for state in
                                ("good", "crooked", "missing", "no_vial"))
        if represented_slots != 30:
            raise RuntimeError(
                f"Sample {record['index']} describes {represented_slots} fixture slots, expected 30"
            )
        expected_labels = 30 - len(slot_states["no_vial"]) + len(record["table_caps"])
        if len(lines) != expected_labels or len(lines) != record["label_count"]:
            raise RuntimeError(
                f"Sample {record['index']} has {len(lines)} labels, expected {expected_labels}"
            )
        split_counts[record["split"]] += 1
        class_counts.update(int(line.split()[0]) for line in lines)
    expected_counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.scenes - args.train_count - args.val_count,
    }
    if dict(split_counts) != {key: value for key, value in expected_counts.items() if value}:
        raise RuntimeError(f"Split counts {dict(split_counts)} do not match {expected_counts}")
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": NAMES,
    }
    (root / "data.yaml").write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "accepted": True,
        "source": "deployment FANUC remediation scene",
        "scenes": args.scenes,
        "split_counts": dict(split_counts),
        "class_counts": {NAMES[index]: count for index, count in sorted(class_counts.items())},
        "shards": args.num_shards,
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
