#!/usr/bin/env python3
"""Run YOLO on every rendered TableCam frame and save labelled copies."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.remediation.render_perception_shadow_video import COLORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--frames', type=Path, help='Directory of PNG frames')
    source.add_argument('--image', type=Path, help='One PNG image to label')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--confidence', type=float, default=.70)
    parser.add_argument('--label-scale', type=float, default=.45,
                        help='detection-label font scale (default: .45)')
    parser.add_argument('--label-thickness', type=int, default=2,
                        help='detection-label text thickness (default: 2)')
    parser.add_argument('--outline-only', action='store_true',
                        help='draw detection outlines without per-object text')
    parser.add_argument('--count-header', action='store_true',
                        help='draw a compact header containing YOLO class counts')
    parser.add_argument('--watch', action='store_true',
                        help='continue annotating new frames until a stop file appears')
    parser.add_argument('--stop-file', action='append', type=Path, default=[],
                        help='with --watch, stop after this file appears and all frames are processed; repeatable')
    parser.add_argument('--done-file', type=Path,
                        help='write this marker after a watched frame directory is complete')
    parser.add_argument('--poll-seconds', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='number of frames per YOLO call (default: 1)')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.confidence < 1.0:
        parser.error('--confidence must be between 0 and 1')
    if args.label_scale <= 0.0:
        parser.error('--label-scale must be positive')
    if args.label_thickness < 1:
        parser.error('--label-thickness must be at least 1')
    if args.watch and args.image:
        parser.error('--watch requires --frames')
    if args.watch and not args.stop_file:
        parser.error('--watch requires at least one --stop-file')
    if args.poll_seconds <= 0.0:
        parser.error('--poll-seconds must be positive')
    if args.batch_size < 1 or args.batch_size > 8:
        parser.error('--batch-size must be between 1 and 8')
    from ultralytics import YOLO
    model = YOLO(str(args.checkpoint))
    processed = 0
    while True:
        paths = [args.image] if args.image else sorted(args.frames.glob('*.png'))
        if not paths and not args.watch:
            raise ValueError(f'No frames found in {args.frames}')
        pending = [path for path in paths
                   if not (args.out / path.name).is_file()]
        for offset in range(0, len(pending), args.batch_size):
            batch = []
            for path in pending[offset:offset + args.batch_size]:
                image = cv2.imread(str(path))
                # The renderer writes directly to the final PNG name. A
                # watcher can see the directory entry before the image footer
                # is visible over NFS, so retry it on the next poll.
                if image is not None:
                    batch.append((path, image))
            if not batch:
                continue
            results = model.predict(
                [image for _path, image in batch],
                conf=args.confidence, imgsz=1280, device=args.device,
                batch=len(batch), verbose=False)
            for (path, image), result in zip(batch, results):
                class_counts = Counter()
                for box, confidence, class_id in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.cls.cpu().numpy()):
                    name = model.names[int(class_id)]
                    class_counts[name] += 1
                    color = COLORS.get(name, (200, 200, 200))
                    x0, y0, x1, y1 = box.round().astype(int)
                    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
                    if not args.outline_only:
                        cv2.putText(image, f'{name} {confidence:.2f}',
                                    (x0, max(22, y0 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, args.label_scale, color,
                                    args.label_thickness,
                                    cv2.LINE_AA)
                if args.count_header:
                    order = ('good_cap', 'crooked_cap', 'missing_cap', 'table_cap')
                    summary = 'YOLO  ' + '  '.join(
                        f'{name} {class_counts[name]}' for name in order
                        if class_counts[name])
                    (text_width, text_height), baseline = cv2.getTextSize(
                        summary, cv2.FONT_HERSHEY_SIMPLEX, .58, 1)
                    cv2.rectangle(
                        image, (12, 12),
                        (min(image.shape[1] - 1, 28 + text_width),
                         22 + text_height + baseline),
                        (0, 0, 0), -1)
                    cv2.putText(image, summary, (20, 18 + text_height),
                                cv2.FONT_HERSHEY_SIMPLEX, .58,
                                (255, 255, 255), 1, cv2.LINE_AA)
                elif not args.outline_only:
                    cv2.putText(image, 'YOLO predictions on this TableCam frame',
                                (20, image.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2,
                                cv2.LINE_AA)
                if not cv2.imwrite(str(args.out / path.name), image):
                    raise RuntimeError(f'Could not write frame {path.name}')
                processed += 1
                if processed == 1 or processed % 50 == 0:
                    print(f'[yolo-overlay] processed {processed}; latest={path.name}',
                          flush=True)
        if not args.watch:
            break
        stopped = any(path.exists() for path in args.stop_file)
        remaining = [path for path in args.frames.glob('*.png')
                     if not (args.out / path.name).is_file()]
        if stopped and not remaining:
            break
        time.sleep(args.poll_seconds)
    if args.done_file is not None:
        args.done_file.parent.mkdir(parents=True, exist_ok=True)
        args.done_file.write_text(
            f'annotated_frames={len(list(args.out.glob("*.png")))}\n')


if __name__ == '__main__':
    main()
