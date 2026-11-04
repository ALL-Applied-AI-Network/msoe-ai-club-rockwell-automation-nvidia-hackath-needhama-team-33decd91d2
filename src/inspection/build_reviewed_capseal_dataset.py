#!/usr/bin/env python3
"""Build a YOLO dataset from the human-reviewed cap-seal tray states.

The 1,181 real cap-seal photos are video frames; they collapse into a few dozen
tray states (frames whose 30 aperture crops all match).  One representative per
state was reviewed by eye and its slot decisions are in --review (codes C / K / M,
everything else an empty hole).  Those decisions are propagated to every frame of
the state, and each frame is audited: a slot labelled as capped must show a yellow
cap, and an empty or uncapped slot must not.  Frames that fail the audit are
dropped and listed in the report.

Boxes come from image evidence, not the fixture grid: 8-inch vials lean and sit
off the plate-hole coordinates by parallax.  Caps are the yellow blob around the
slot (re-centred iteratively so touching neighbours split); a crooked cap box also
covers the exposed vial top; an uncapped vial is placed at the slot's mean cap
offset.  Splits are by tray state so near-duplicate frames never straddle splits.
"""
import argparse
import json
import random
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.remediation.geometry import CLASS_NAMES

CODE_TO_CLASS = {'C': 'good_cap', 'K': 'crooked_cap', 'M': 'missing_cap'}
CAP_R = 12          # px: cap radius window in the 320x240 cap-seal frame (caps are ~20 px across)
VIAL_TOP_PX = 18    # px: uncapped vial-top box side


def yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 18) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 110) & (hsv[..., 2] > 110)


def cap_blob(mask, x, y, iters=4, r=CAP_R + 2):
    """Re-centre on yellow pixels near (x, y); return (cx, cy, box, pixel count)."""
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    for _ in range(iters):
        sel = mask & ((xx - x) ** 2 + (yy - y) ** 2 < r * r)
        if sel.sum() < 15:
            return None
        x, y = xx[sel].mean(), yy[sel].mean()
    sel = mask & ((xx - x) ** 2 + (yy - y) ** 2 < CAP_R * CAP_R)
    if sel.sum() < 40:
        return None
    return x, y, (xx[sel].min(), yy[sel].min(), xx[sel].max() + 1, yy[sel].max() + 1), int(sel.sum())


def yellow_near(mask, x, y, r=9):
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    return int((mask & ((xx - x) ** 2 + (yy - y) ** 2 < r * r)).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--apertures', type=Path, required=True, help='fixture_g_apertures.json (640x480 source px)')
    p.add_argument('--clusters', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--merge', type=Path, help='extra YOLO dataset (images/<split>, labels/<split>) to merge')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-frames-per-state', type=int, default=40)
    p.add_argument('--val-states', default='4,12,20,26')
    p.add_argument('--test-states', default='7,15,24')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    aps = json.loads(a.apertures.read_text())
    centre = {s: (v['x'] - 160, v['y'] - 120) for s, v in aps.items()}   # centre 50 % crop of 640x480
    clusters = [c for c in json.loads(a.clusters.read_text()) if len(c) >= 3]
    states = {st['cluster']: st for st in json.loads(a.review.read_text())['states']}
    val = {int(v) for v in a.val_states.split(',') if v}
    test = {int(v) for v in a.test_states.split(',') if v}
    rng = random.Random(a.seed)

    # Pass 1: per-slot mean cap offset from seated caps (used to place uncapped vial tops).
    offsets = {s: [] for s in aps}
    for k, frames in enumerate(clusters):
        for f in frames[:5]:
            mask = yellow_mask(cv2.imread(str(a.images / f)))
            for s, code in states[k]['slots'].items():
                if code == 'C':
                    blob = cap_blob(mask, *centre[s])
                    if blob:
                        offsets[s].append((blob[0] - centre[s][0], blob[1] - centre[s][1]))
    col_of = {s: v['col'] for s, v in aps.items()}
    def slot_offset(s):
        pts = offsets[s] or [o for t in aps if col_of[t] == col_of[s] for o in offsets[t]] or [(0., 0.)]
        return np.mean(pts, axis=0)

    report = {'boxes': {c: 0 for c in CLASS_NAMES}, 'frames': {}, 'audit_dropped': [], 'states': {}}
    for k, frames in enumerate(clusters):
        split = 'test' if k in test else 'val' if k in val else 'train'
        picks = frames if len(frames) <= a.max_frames_per_state else sorted(rng.sample(frames, a.max_frames_per_state))
        kept = 0
        for f in picks:
            img = cv2.imread(str(a.images / f))
            h, w = img.shape[:2]
            mask = yellow_mask(img)
            rows, bad = [], []
            for s in aps:
                code = states[k]['slots'].get(s, 'E')
                cx, cy = centre[s]
                if code in ('C', 'K'):
                    blob = cap_blob(mask, cx, cy)
                    if blob is None:
                        bad.append(f'{s}:{code} no cap visible'); continue
                    x0, y0, x1, y1 = blob[2]
                    if code == 'K':     # include the exposed vial top beside the displaced cap
                        ox, oy = slot_offset(s)
                        x0, y0 = min(x0, cx + ox - VIAL_TOP_PX / 2), min(y0, cy + oy - VIAL_TOP_PX / 2)
                        x1, y1 = max(x1, cx + ox + VIAL_TOP_PX / 2), max(y1, cy + oy + VIAL_TOP_PX / 2)
                else:
                    ox, oy = slot_offset(s)
                    if yellow_near(mask, cx + ox, cy + oy) > 60:
                        bad.append(f'{s}:{code} yellow present'); continue
                    if code != 'M':
                        continue
                    x0, y0 = cx + ox - VIAL_TOP_PX / 2, cy + oy - VIAL_TOP_PX / 2
                    x1, y1 = x0 + VIAL_TOP_PX, y0 + VIAL_TOP_PX
                cls = CODE_TO_CLASS[code]
                rows.append(f'{CLASS_NAMES.index(cls)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} '
                            f'{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}')
            if bad:
                report['audit_dropped'].append({'frame': f, 'state': k, 'issues': bad}); continue
            for sub in ('images', 'labels'):
                (a.out / sub / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(a.images / f, a.out / 'images' / split / f)
            (a.out / 'labels' / split / (Path(f).stem + '.txt')).write_text('\n'.join(rows) + ('\n' if rows else ''))
            for r in rows:
                report['boxes'][CLASS_NAMES[int(r.split()[0])]] += 1
            kept += 1
        report['frames'][split] = report['frames'].get(split, 0) + kept
        report['states'][k] = {'split': split, 'frames_in_state': len(frames), 'kept': kept}
    if a.merge:
        for img in sorted((a.merge / 'images').glob('*/*')):
            split = img.parent.name
            lbl = a.merge / 'labels' / split / (img.stem + '.txt')
            for sub, f in (('images', img), ('labels', lbl)):
                (a.out / sub / split).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, a.out / sub / split / ('synthetic_' + f.name))
            for line in filter(str.strip, lbl.read_text().split('\n')):
                report['boxes'][CLASS_NAMES[int(line.split()[0])]] += 1
            report['frames'][split + '_synthetic'] = report['frames'].get(split + '_synthetic', 0) + 1
    (a.out / 'data.yaml').write_text(f'path: {a.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                                     'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASS_NAMES)))
    (a.out / 'build_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('audit_dropped', 'states')}
                     | {'audit_dropped': len(report['audit_dropped'])}, indent=2))


if __name__ == '__main__':
    main()
