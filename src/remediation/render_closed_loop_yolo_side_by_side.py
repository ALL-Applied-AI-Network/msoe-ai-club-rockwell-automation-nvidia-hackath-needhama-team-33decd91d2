#!/usr/bin/env python3
"""Pair a remediation render with the exact TableCam YOLO observation used.

The right pane is the detector input image annotated from its saved prediction
JSON.  It is intentionally a fixed pre-action observation: it does not claim
that YOLO tracked objects through contact or verified the outcome.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.remediation.render_perception_shadow_video import draw


def banner(image, title, color):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 36), color, -1)
    cv2.putText(result, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True,
                        help='Rendered overview frame directory')
    parser.add_argument('--detections', type=Path, required=True,
                        help='Saved YOLO/TableCam prediction JSON')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    observations = json.loads(args.detections.read_text())
    source = Path(observations['image'])
    if not source.is_absolute():
        source = ROOT / source
    tablecam = cv2.imread(str(source))
    if tablecam is None:
        raise FileNotFoundError(f'Could not read YOLO input: {source}')
    labelled = draw(tablecam, observations['targets'], source.parent.name)
    args.out.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(args.frames.glob('*.png'))
    if not frame_paths:
        raise ValueError(f'No rendered frames under {args.frames}')
    for index, path in enumerate(frame_paths):
        output = args.out / f'{index:04d}.png'
        if output.exists():
            continue
        overview = cv2.imread(str(path))
        if overview is None:
            raise RuntimeError(f'Could not read {path}')
        height = overview.shape[0]
        width = round(labelled.shape[1] * height / labelled.shape[0])
        right = cv2.resize(labelled, (width, height), interpolation=cv2.INTER_AREA)
        left = banner(overview, 'Isaac Sim remediation camera', (56, 90, 130))
        right = banner(right, 'TableCam: YOLO labels used for target selection', (45, 105, 65))
        combined = cv2.hconcat([left, right])
        if not cv2.imwrite(str(output), combined):
            raise RuntimeError(f'Could not write {output}')
    report = {
        'mode': 'side_by_side_remediation_and_pre_action_yolo_observation',
        'overview_frames': str(args.frames), 'detections': str(args.detections),
        'tablecam_input': str(source), 'frame_count': len(frame_paths),
        'limitation': 'Right pane is the pre-action observation used for selection; it is not post-contact verification.'
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
