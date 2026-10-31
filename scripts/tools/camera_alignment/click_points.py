#!/usr/bin/env python3
"""Read fingertip/feature pixels off probe frames into the T3 --clicks CSV.

For each frame: click the point in the table image, press any key, then click
the same point in the side image (or press any key without clicking to skip
it). Esc stops. Needs a display; on a headless node, read the pixels in any
image viewer and type the CSV by hand (run,frame,label,table_u,table_v,side_u,side_v).

    python scripts/tools/camera_alignment/click_points.py <probe_run_dir> --label tip --out clicks.csv
"""
import argparse
import csv
from pathlib import Path

import cv2


def click(path, title):
    image = cv2.imread(str(path))
    if image is None:
        return None
    picked = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked[:] = [(x, y)]
            cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.imshow(title, image)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, image)
    key = cv2.waitKey(0)
    cv2.destroyWindow(title)
    return 'esc' if key == 27 else (picked[0] if picked else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    parser.add_argument('--label', default='tip')
    parser.add_argument('--frames', nargs='*', help='default: every tablecam_raw frame')
    parser.add_argument('--out', type=Path, default=Path('clicks.csv'))
    args = parser.parse_args()
    frames = args.frames or sorted(p.name for p in (args.run / 'tablecam_raw').glob('*.png'))
    new = not args.out.exists()
    with open(args.out, 'a', newline='') as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(['run', 'frame', 'label', 'table_u', 'table_v', 'side_u', 'side_v'])
        for frame in frames:
            table = click(args.run / 'tablecam_raw' / frame, f'table {frame}: click {args.label}, then a key')
            if table == 'esc':
                break
            if table is None:
                continue
            side = click(args.run / 'side_raw' / frame, f'side {frame}: click {args.label}, then a key')
            if side == 'esc':
                break
            writer.writerow([str(args.run), frame, args.label, *table, *(side or ('', ''))])
    print('appended to', args.out)


if __name__ == '__main__':
    main()
