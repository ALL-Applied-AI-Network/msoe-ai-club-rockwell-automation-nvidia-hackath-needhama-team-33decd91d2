#!/usr/bin/env python3
"""Render a YOLO perception-shadow video from Isaac TableCam captures.

The video is intentionally observation-only: it visualizes the detector inputs
and proposed classes without commanding the robot or consulting scene truth.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inspection.vision_pipeline import RemediationVisionDetector, camera_from_usd_capture

COLORS = {'good_cap': (72, 190, 72), 'crooked_cap': (45, 115, 240),
          'missing_cap': (50, 65, 220), 'table_cap': (25, 205, 235),
          'pass': (72, 190, 72), 'damaged_vial': (50, 65, 220)}


def draw(frame, targets, scene_id):
    rendered = frame.copy()
    # Good caps are shown as a count, rather than thirty overlapping boxes.
    # The video emphasizes the objects the controller may need to act on.
    for target in (item for item in targets if item['class_name'] != 'good_cap'):
        x0, y0, x1, y1 = np.round(target['bbox_xyxy']).astype(int)
        color = COLORS[target['class_name']]
        cv2.rectangle(rendered, (x0, y0), (x1, y1), color, 2)
        cv2.putText(rendered, f"{target['class_name']} {target['confidence']:.2f}",
                    (x0, max(22, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
    cv2.putText(rendered, f'YOLO perception shadow mode | Isaac scene {scene_id}',
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)
    counts = {name: sum(item['class_name'] == name for item in targets)
              for name in ('crooked_cap', 'missing_cap', 'table_cap', 'good_cap')}
    cv2.putText(rendered, f"crooked {counts['crooked_cap']} | missing {counts['missing_cap']} | spares {counts['table_cap']} | good {counts['good_cap']}",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(rendered, 'Shadow mode: detections are not contact-qualified',
                (18, 90), cv2.FONT_HERSHEY_SIMPLEX, .55, (190, 225, 255), 2, cv2.LINE_AA)
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=16)
    parser.add_argument('--fps', type=float, default=3.0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    scene_dirs = sorted(path.parent for path in args.dataset.glob('scenes/*/table.png'))[:args.scenes]
    if not scene_dirs:
        raise ValueError(f'No Isaac TableCam captures found below {args.dataset}')
    args.out.mkdir(parents=True, exist_ok=True)
    frame_dir = args.out / 'frames'
    frame_dir.mkdir(exist_ok=True)
    frames = []
    for directory in scene_dirs:
        image = cv2.imread(str(directory / 'table.png'))
        metadata = json.loads((directory / 'qualification.json').read_text())
        detector = RemediationVisionDetector(str(args.checkpoint),
                                             camera=camera_from_usd_capture(metadata['camera']),
                                             device=args.device)
        targets = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        rendered = draw(image, targets, directory.name)
        for repeat in range(3):
            number = len(frames) * 3 + repeat
            if not cv2.imwrite(str(frame_dir / f'{number:04d}.png'), rendered):
                raise RuntimeError('Could not write perception frame')
        frames.append({'scene': directory.name, 'detection_count': len(targets), 'targets': targets})
    report = {'schema_version': 1, 'mode': 'isaac_rendered_perception_shadow',
              'motion_enabled': False, 'contact_qualified': False,
              'checkpoint': str(args.checkpoint), 'source_dataset': str(args.dataset),
              'scenes': frames, 'frame_directory': str(frame_dir), 'video_pending_external_encode': True}
    (args.out / 'perception_shadow_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'scenes': len(frames), 'frames': str(frame_dir)}, indent=2))


if __name__ == '__main__':
    main()
