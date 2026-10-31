#!/usr/bin/env python3
"""T6 (H6): what heights does the depth stream report, and does it land on the RGB objects?

Depth only feeds crooked-cap heights (`surface_top_base_z_m`) and cap axes,
but it is the one sensor that measures height directly. This prints, per
detection class, the 90th-percentile base-frame z of the depth points that
project into each box, and measures how far those depth points sit from the
yellow caps in the RGB image (the depth-to-colour registration).

If depth says one height and the tray pitch (T2) says another, trust neither:
measure with a ruler (step 1 of the workflow) and re-solve table_depth.

    .conda_env/bin/python scripts/tools/camera_alignment/t6_depth_heights.py \
        2026-10-28_15-22-05_physical_full_pick_m2mm 0001.png
"""
import argparse

import cv2
import numpy as np

from _common import depth_to_base, frame_record, load_cameras, out_dir, project, run_path


def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (18, 90, 90), (38, 255, 255))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('frame')
    parser.add_argument('--out')
    args = parser.parse_args()

    cams = load_cameras()
    run = run_path(args.run)
    depth = cv2.imread(str(run / 'depth_raw' / args.frame), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit('this run has no depth_raw/ frames')
    rgb = cv2.imread(str(run / 'tablecam_raw' / args.frame))
    points = depth_to_base(cams['table_depth'], depth)
    uv = project(cams['table'], points)
    table = points[points[:, 2] < 0.03, 2]
    print(f'work surface from depth: median z {np.median(table) * 1000:+.1f} mm (code assumes -5.1)')

    heights = {}
    detections = frame_record(args.run, args.frame).get('detections') or []
    for det in detections:
        x0, y0, x1, y1 = det['bbox_xyxy']
        mx, my = .15 * (x1 - x0), .15 * (y1 - y0)
        inside = (uv[:, 0] > x0 + mx) & (uv[:, 0] < x1 - mx) & (uv[:, 1] > y0 + my) & (uv[:, 1] < y1 - my)
        if inside.sum() >= 8:
            heights.setdefault(det['class_name'], []).append(
                (np.percentile(points[inside, 2], 90), det['plane_base_z_m']))
    for name, values in heights.items():
        z = np.array(values) * 1000
        print(f'  {name:12s} n={len(z):2d} depth top p50 {np.median(z[:, 0]):6.1f} mm '
              f'(p10-p90 {np.percentile(z[:, 0], 10):.1f}-{np.percentile(z[:, 0], 90):.1f}); '
              f'plane used {np.median(z[:, 1]):.1f} mm')

    # Registration: depth points at cap-top height near a tray detection should
    # project onto yellow pixels (the 40 px margin keeps the arm out).
    mask = yellow_mask(rgb)
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    near = np.zeros(len(points), bool)
    for det in detections:
        if det['class_name'] in ('good_cap', 'crooked_cap'):
            x0, y0, x1, y1 = det['bbox_xyxy']
            near |= (uv[:, 0] > x0 - 40) & (uv[:, 0] < x1 + 40) & (uv[:, 1] > y0 - 40) & (uv[:, 1] < y1 + 40)
    caps = near & (points[:, 2] > 0.19) & (points[:, 2] < 0.23)
    u, v = np.round(uv[caps]).astype(int).T
    ok = (u >= 0) & (u < rgb.shape[1]) & (v >= 0) & (v < rgb.shape[0])
    d = distance[v[ok], u[ok]]
    print(f'cap-height depth points: {ok.sum()}; distance to nearest yellow pixel '
          f'median {np.median(d):.1f} px, p90 {np.percentile(d, 90):.1f} px (0-3 px means registered)')
    vis = rgb.copy()
    vis[v[ok], u[ok]] = (255, 0, 255)
    path = out_dir('t6_depth', args.out) / f'{run.name}_{args.frame}'
    cv2.imwrite(str(path), vis)
    print('wrote', path, '(magenta = depth points between 190 and 230 mm)')


if __name__ == '__main__':
    main()
