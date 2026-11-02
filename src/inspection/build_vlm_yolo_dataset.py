#!/usr/bin/env python3
"""Convert the team's VLM per-aperture labels (cap-detection branch) into a YOLO dataset.

Inputs are the real CapSeal camera photos in images/pruned_images (the centre
crop of 640x480 captures), Qwen labels in images/VLM_samples/full_image_labels.json
and GPT labels in images/GPT_samples/*.json (these replace Qwen for the images
Qwen timed out on).  Each labelled aperture becomes a square box at its fixture
pixel position; classes use the remediation names (geometry.CLASS_NAMES).

Images with any 'unsure' or 'occluded' aperture are skipped: YOLO has no ignore
regions, so an unlabelled cap would be taught as background.  'no_vial' is an
empty hole and gets no box.  An optional existing YOLO dataset (e.g. the
synthetic TableCam remediation renders, the only source of table_cap) is merged.
"""
import argparse
import json
import random
import shutil
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.remediation.geometry import CLASS_NAMES

VLM_TO_CLASS = {'pass': 'good_cap', 'crooked_cap': 'crooked_cap', 'missing_cap': 'missing_cap'}
SKIP_IMAGE = {'unsure', 'occluded'}
NO_BOX = {'no_vial'}


def load_labels(qwen_json, gpt_dir):
    labels = {im['filename']: im['apertures'] for im in json.loads(Path(qwen_json).read_text())['images']}
    for path in sorted(Path(gpt_dir).glob('*.json')):
        im = json.loads(path.read_text())
        labels[im['filename']] = im['apertures']
    return labels


def aperture_pixels(apertures, image_size, source_size=(640, 480), crop_fraction=.5):
    """Same mapping as vlm_full_image_labeler.aperture_centers."""
    sw, sh = source_size
    tw, th = image_size
    cw, ch = round(sw * crop_fraction), round(sh * crop_fraction)
    left, top = (sw - cw) // 2, (sh - ch) // 2
    return {slot: ((info['x'] - left) * tw / cw, (info['y'] - top) * th / ch) for slot, info in apertures.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--qwen', type=Path, required=True)
    p.add_argument('--gpt-dir', type=Path, required=True)
    p.add_argument('--apertures', type=Path, required=True)
    p.add_argument('--merge', type=Path, help='existing YOLO dataset root (images/<split>, labels/<split>)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--box-px', type=float, default=26., help='seated cap / uncapped vial box side, image px')
    p.add_argument('--crooked-box-px', type=float, default=32.)
    p.add_argument('--val', type=float, default=.15)
    p.add_argument('--test', type=float, default=.10)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    apertures = json.loads(a.apertures.read_text())
    labels = load_labels(a.qwen, a.gpt_dir)
    names = sorted(labels)
    random.Random(a.seed).shuffle(names)
    n_test, n_val = round(len(names) * a.test), round(len(names) * a.val)
    split_of = {n: 'test' if i < n_test else 'val' if i < n_test + n_val else 'train' for i, n in enumerate(names)}
    report = {'skipped_ambiguous': [], 'missing_image': [], 'boxes': {c: 0 for c in CLASS_NAMES}, 'images': {}}
    for name in names:
        decisions = {slot: d['label'] for slot, d in labels[name].items()}
        if SKIP_IMAGE & set(decisions.values()):
            report['skipped_ambiguous'].append(name); continue
        src = a.images / name
        image = cv2.imread(str(src))
        if image is None:
            report['missing_image'].append(name); continue
        h, w = image.shape[:2]
        centers = aperture_pixels(apertures, (w, h))
        rows = []
        for slot, label in decisions.items():
            if label in NO_BOX:
                continue
            cls = VLM_TO_CLASS[label]
            side = a.crooked_box_px if cls == 'crooked_cap' else a.box_px
            x, y = centers[slot]
            rows.append(f'{CLASS_NAMES.index(cls)} {x / w:.6f} {y / h:.6f} {side / w:.6f} {side / h:.6f}')
            report['boxes'][cls] += 1
        split = split_of[name]
        for sub in ('images', 'labels'):
            (a.out / sub / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, a.out / 'images' / split / name)
        (a.out / 'labels' / split / (Path(name).stem + '.txt')).write_text('\n'.join(rows) + ('\n' if rows else ''))
        report['images'][split] = report['images'].get(split, 0) + 1
    if a.merge:
        for img in sorted((a.merge / 'images').glob('*/*')):
            split = img.parent.name
            lbl = a.merge / 'labels' / split / (img.stem + '.txt')
            for sub, f in (('images', img), ('labels', lbl)):
                (a.out / sub / split).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, a.out / sub / split / ('synthetic_' + f.name))
            for line in lbl.read_text().split('\n'):
                if line.strip():
                    report['boxes'][CLASS_NAMES[int(line.split()[0])]] += 1
            report['images'][split + '_merged'] = report['images'].get(split + '_merged', 0) + 1
    (a.out / 'data.yaml').write_text(f'path: {a.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                                     'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASS_NAMES)))
    (a.out / 'build_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'skipped_ambiguous'} | {'skipped_ambiguous': len(report['skipped_ambiguous'])}, indent=2))


if __name__ == '__main__':
    main()
