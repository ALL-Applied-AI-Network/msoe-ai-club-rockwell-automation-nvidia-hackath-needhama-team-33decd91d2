#!/usr/bin/env python3
"""Prepare translated synthetic and conservative real-image YOLO datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")
SOURCE_CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: None, 4: None}
REAL_CLASS_MAP = {"pass": 0, "crooked_cap": 1, "missing_cap": 2, "no_vial": None}
UNSUPPORTED_REAL = {"damaged_vial", "occluded", "unsure"}


def copy_image_and_label(src_image: Path, src_label: Path, dst_image: Path, dst_label: Path, class_map) -> int:
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, dst_image)
    rows = []
    for line in src_label.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        mapped = class_map.get(int(fields[0]))
        if mapped is not None:
            rows.append(" ".join([str(mapped), *fields[1:]]))
    dst_label.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)


def write_yaml(path: Path) -> None:
    path.write_text(
        f"path: {path.parent.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )


def build_synthetic(source: Path, base: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already contains files: {out}")
    report = {"source_images": 0, "source_boxes": 0, "base_images": 0, "base_boxes": 0, "dropped_source_boxes": 0}
    for split in ("train", "val"):
        for image in sorted((source / "images" / split).glob("*")):
            if not image.is_file():
                continue
            label = source / "labels" / split / f"{image.stem}.txt"
            dst_image = out / "images" / split / f"fullframe_{split}_{image.name}"
            dst_label = out / "labels" / split / f"fullframe_{split}_{image.stem}.txt"
            before = sum(1 for line in label.read_text().splitlines() if line.strip())
            after = copy_image_and_label(image, label, dst_image, dst_label, SOURCE_CLASS_MAP)
            report["source_images"] += 1
            report["source_boxes"] += after
            report["dropped_source_boxes"] += before - after

    # Preserve the remediation model's original four-class domain, especially
    # table_cap, while keeping the original val split untouched for evaluation.
    for image in sorted((base / "images" / "train").glob("*")):
        if not image.is_file():
            continue
        label = base / "labels" / "train" / f"{image.stem}.txt"
        dst_image = out / "images" / "train" / f"base_{image.name}"
        dst_label = out / "labels" / "train" / f"base_{image.stem}.txt"
        report["base_images"] += 1
        report["base_boxes"] += copy_image_and_label(image, label, dst_image, dst_label, {i: i for i in range(4)})

    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "data.yaml")
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_per_image_labels(vlm_dir: Path, gpt_dir: Path) -> dict[str, dict]:
    labels = {}
    for path in sorted(vlm_dir.glob("*.jpg.json")):
        value = json.loads(path.read_text())
        if value.get("filename"):
            labels[value["filename"]] = value
    for path in sorted(gpt_dir.glob("*.jpg.json")):
        value = json.loads(path.read_text())
        if value.get("filename"):
            labels[value["filename"]] = value
    return labels


def build_real(images: Path, vlm_dir: Path, gpt_dir: Path, apertures_path: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already contains files: {out}")
    apertures = json.loads(apertures_path.read_text())
    labels = load_per_image_labels(vlm_dir, gpt_dir)
    report = {"candidate_images": len(labels), "accepted_images": 0, "skipped_images": 0,
              "skip_reasons": Counter(), "boxes": Counter(), "gpt_overrides": 0}
    gpt_names = {json.loads(p.read_text()).get("filename") for p in gpt_dir.glob("*.jpg.json")}
    report["gpt_overrides"] = len(gpt_names)

    for filename, record in sorted(labels.items()):
        source = images / filename
        if not source.is_file():
            report["skipped_images"] += 1
            report["skip_reasons"]["missing_image"] += 1
            continue
        decisions = record.get("apertures", {})
        states = [str(value.get("label", "unsure")).lower() for value in decisions.values()]
        unsupported = set(states) & UNSUPPORTED_REAL
        if unsupported:
            report["skipped_images"] += 1
            report["skip_reasons"]["unsupported_or_ambiguous"] += 1
            continue
        if any(state not in REAL_CLASS_MAP for state in states):
            report["skipped_images"] += 1
            report["skip_reasons"]["unknown_label"] += 1
            continue

        dst_image = out / "images" / "train" / f"real_{filename}"
        dst_label = out / "labels" / "train" / f"real_{Path(filename).stem}.txt"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst_image)
        rows = []
        for slot, value in decisions.items():
            label = str(value.get("label", "unsure")).lower()
            mapped = REAL_CLASS_MAP[label]
            if mapped is None:
                continue
            info = apertures[slot]
            # The real images are centered 50% crops of 640x480 captures,
            # yielding 320x240 images. Map source fixture coordinates into it.
            x = (float(info["x"]) - 160.0) / 320.0
            y = (float(info["y"]) - 120.0) / 240.0
            side = 32.0 if label == "crooked_cap" else 26.0
            rows.append(f"{mapped} {x:.6f} {y:.6f} {side / 320.0:.6f} {side / 240.0:.6f}")
            report["boxes"][CLASS_NAMES[mapped]] += 1
        dst_label.write_text("\n".join(rows) + ("\n" if rows else ""))
        report["accepted_images"] += 1

    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "data.yaml")
    report["skip_reasons"] = dict(report["skip_reasons"])
    report["boxes"] = dict(report["boxes"])
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_mixed(synthetic: Path, real: Path, out: Path) -> dict:
    """Copy translated synthetic data and accepted real frames into one dataset."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already contains files: {out}")
    report = {"synthetic_train_images": 0, "synthetic_val_images": 0, "real_train_images": 0}
    for split in ("train", "val"):
        for image in sorted((synthetic / "images" / split).glob("*")):
            if not image.is_file():
                continue
            label = synthetic / "labels" / split / f"{image.stem}.txt"
            copy_image_and_label(
                image, label,
                out / "images" / split / image.name,
                out / "labels" / split / label.name,
                {i: i for i in range(4)},
            )
            report[f"synthetic_{split}_images"] += 1
    for image in sorted((real / "images" / "train").glob("*")):
        if not image.is_file():
            continue
        label = real / "labels" / "train" / f"{image.stem}.txt"
        copy_image_and_label(
            image, label,
            out / "images" / "train" / image.name,
            out / "labels" / "train" / label.name,
            {i: i for i in range(4)},
        )
        report["real_train_images"] += 1
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "data.yaml")
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-source", type=Path, required=True)
    parser.add_argument("--base-remediation", type=Path, required=True)
    parser.add_argument("--synthetic-out", type=Path, required=True)
    parser.add_argument("--real-images", type=Path, required=True)
    parser.add_argument("--vlm-dir", type=Path, required=True)
    parser.add_argument("--gpt-dir", type=Path, required=True)
    parser.add_argument("--apertures", type=Path, required=True)
    parser.add_argument("--real-out", type=Path, required=True)
    parser.add_argument("--mixed-out", type=Path, required=True)
    args = parser.parse_args()
    synthetic_report = build_synthetic(args.synthetic_source, args.base_remediation, args.synthetic_out)
    real_report = build_real(args.real_images, args.vlm_dir, args.gpt_dir, args.apertures, args.real_out)
    mixed_report = build_mixed(args.synthetic_out, args.real_out, args.mixed_out)
    print(json.dumps({"synthetic": synthetic_report, "real": real_report, "mixed": mixed_report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
