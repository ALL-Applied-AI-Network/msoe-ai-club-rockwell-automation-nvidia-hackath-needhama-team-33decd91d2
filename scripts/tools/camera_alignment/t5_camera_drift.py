#!/usr/bin/env python3
"""T5 (H3): has the TableCam moved since the frames you trust?

cameras.json was solved on 2026-10-07 and the table camera was not re-solved
after the testbed rebuild. This registers a current frame against a reference
frame on the static scene (table, tape, fixed fixtures) with ORB + RANSAC and
reports the image shift, rotation and scale. Objects that moved between the
two frames (arm, caps, tray) fall out as RANSAC outliers; mask large ones with
--mask x0,y0,x1,y1 (repeatable).

About 1 px is 0.8-1.0 mm on the work surface. A shift above ~2 px or a
rotation above ~0.1 deg is enough to explain several mm of target error and
calls for a fresh ArUco solve (the distortion model stays as it is).

    .conda_env/bin/python scripts/tools/camera_alignment/t5_camera_drift.py \
        shared/runs/2026-10-26_15-41-05_physical_policy/tablecam_raw/0000.png \
        shared/runs/2026-10-28_15-22-05_physical_full_pick_m2mm/tablecam_raw/0000.png \
        --mask 600,0,1500,560
"""
import argparse

import cv2
import numpy as np

from _common import load_cameras, out_dir, pixel_to_base


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reference')
    parser.add_argument('current')
    parser.add_argument('--mask', action='append', default=[], help='x0,y0,x1,y1 region to ignore')
    parser.add_argument('--out')
    args = parser.parse_args()

    ref, cur = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in (args.reference, args.current))
    keep = np.full(ref.shape, 255, np.uint8)
    for box in args.mask:
        x0, y0, x1, y1 = map(int, box.split(','))
        keep[y0:y1, x0:x1] = 0
    orb = cv2.ORB_create(6000)
    k1, d1 = orb.detectAndCompute(ref, keep)
    k2, d2 = orb.detectAndCompute(cur, keep)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    a = np.float32([k1[m.queryIdx].pt for m in matches])
    b = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    inliers = inliers.ravel().astype(bool)
    shift = b[inliers] - a[inliers]
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.hypot(M[0, 0], M[1, 0])
    print(f'{inliers.sum()} static inliers of {len(matches)} matches')
    print(f'median shift (du, dv) = {np.median(shift, 0).round(2)} px, rotation {angle:+.3f} deg, '
          f'scale {scale:.5f}')
    cam = load_cameras()['table']
    for name, uv in (('tray centre', (920, 540)), ('spare caps', (1330, 650))):
        moved = M @ np.array([*uv, 1.0])
        delta = (pixel_to_base(cam, moved, 0.0) - pixel_to_base(cam, uv, 0.0))[:2]
        print(f'  {name:11s} pixel {uv} moves {np.round(moved - uv, 2)} px = {np.round(delta * 1000, 1)} mm '
              'on the table plane')
    vis = cv2.cvtColor(cv2.addWeighted(ref, .5, cur, .5, 0), cv2.COLOR_GRAY2BGR)
    for p, q in zip(a[inliers][::5], b[inliers][::5]):
        cv2.arrowedLine(vis, tuple(p.astype(int)), tuple((p + 20 * (q - p)).astype(int)), (0, 0, 255), 1)
    path = out_dir('t5_drift', args.out) / 'drift_x20.jpg'
    cv2.imwrite(str(path), vis)
    print('wrote', path, '(arrows are 20x the measured shift)')


if __name__ == '__main__':
    main()
