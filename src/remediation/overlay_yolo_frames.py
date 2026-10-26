#!/usr/bin/env python3
"""Run YOLO on every rendered TableCam frame and save labelled copies."""
import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.remediation.render_perception_shadow_video import COLORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='0')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(args.checkpoint))
    paths = sorted(args.frames.glob('*.png'))
    if not paths: raise ValueError(f'No frames found in {args.frames}')
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        result = model.predict(image, conf=.6, imgsz=1280, device=args.device, verbose=False)[0]
        for box, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            name = model.names[int(class_id)]; color = COLORS.get(name, (200, 200, 200))
            x0, y0, x1, y1 = box.round().astype(int)
            cv2.rectangle(image, (x0,y0), (x1,y1), color, 2)
            cv2.putText(image, f'{name} {confidence:.2f}', (x0, max(22,y0-6)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 2, cv2.LINE_AA)
        cv2.putText(image, 'YOLO predictions on this TableCam frame', (20, image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, .58, (255,255,255), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(args.out / f'{i:04d}.png'), image): raise RuntimeError(f'Could not write frame {i}')
        if i % 50 == 0: print(f'[yolo-overlay] {i + 1}/{len(paths)}', flush=True)


if __name__ == '__main__':
    main()
