#!/usr/bin/env python3
"""T1 (H1): how a height-plane error moves each target, in robot XY and in the image.

Every detection is placed by intersecting its box-centre ray with an assumed
plane. If the true feature is dz higher than that plane, the computed target
lands farther from the camera along the ray. This prints the XY error per
detection and draws it on the frame, with the vertical vanishing point that
decides whether it looks like "left" or "right" in the TableCam image.

    .conda_env/bin/python scripts/tools/camera_alignment/t1_parallax_map.py \
        2026-10-28_15-22-05_physical_full_pick_m2mm 0001.png --dz-mm 10
"""
import argparse

import cv2
import numpy as np

from _common import (load_cameras, out_dir, frame_record, plane_error_xy, project, run_path,
                     vertical_vanishing_pixel)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('frame')
    parser.add_argument('--dz-mm', type=float, default=10.0, help='plane is this many mm too low (negative: too high)')
    parser.add_argument('--out')
    args = parser.parse_args()

    cam = load_cameras()['table']
    record = frame_record(args.run, args.frame)
    image = cv2.imread(str(run_path(args.run) / 'tablecam_raw' / args.frame))
    vp = vertical_vanishing_pixel(cam)
    print(f'vertical vanishing pixel (u, v) = {vp.round(1)}; image-left of u={vp[0]:.0f} '
          'height errors read as "left", right of it as "right"')
    print(f'{"class":12s} {"pixel":>12s} {"base mm":>22s} {"XY error mm for dz=%+.0f mm" % args.dz_mm:>30s}')
    dz = args.dz_mm / 1000.0
    for det in record.get('detections') or []:
        p = np.asarray(det['base_pos_m'], dtype=float)
        err = plane_error_xy(cam, p, dz)
        print(f'{det["class_name"]:12s} {str(np.round(det["pixel"]).astype(int)):>12s} '
              f'{str(np.round(p * 1000).astype(int)):>22s} {str(np.round(err * 1000, 1)):>30s}')
        # Arrow: true feature (dz higher on the same ray) -> computed target, exaggerated 5x.
        start = project(cam, [p - np.r_[err, 0.0]])[0]
        end = project(cam, [p + 4 * np.r_[err, 0.0]])[0]
        colour = (0, 0, 255) if det['class_name'] != 'good_cap' else (0, 200, 0)
        cv2.arrowedLine(image, tuple(start.astype(int)), tuple(end.astype(int)), colour, 2, tipLength=.3)
    cv2.line(image, (int(vp[0]), 0), (int(vp[0]), image.shape[0]), (255, 255, 0), 1)
    cv2.putText(image, f'vanishing column u={vp[0]:.0f}; arrows = computed-target error x5 for a plane '
                f'{args.dz_mm:+.0f} mm too low', (20, 40), 0, .8, (255, 255, 0), 2)
    path = out_dir('t1_parallax', args.out) / f'{run_path(args.run).name}_{args.frame}'
    cv2.imwrite(str(path), image)
    print('wrote', path)


if __name__ == '__main__':
    main()
