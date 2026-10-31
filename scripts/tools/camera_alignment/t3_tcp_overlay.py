#!/usr/bin/env python3
"""T3 (H2/H4): where the robot says the fingertip is vs. where the camera sees it.

Draws the commanded UTOOL1 point, the tool axis (10 mm ticks up to 100 mm),
the flange face and the calibrated wrist-camera origin on every trace frame
whose TCP is near a target. If the drawn tool axis lies on the real gripper
but the drawn tip is not at the real fingertips, the TCP (UTOOL1) is wrong.
If the whole axis is displaced, the camera or the robot base frame is wrong.

Optional --clicks CSV (header: run,frame,label,table_u,table_v,side_u,side_v;
side columns may be empty) holds the pixel of the real fingertip centre that a
person read off the image. With side pixels the point is triangulated; with
the table pixel alone two readings are printed: the XY offset if the height
were right, and the height offset if the XY were right.

    .conda_env/bin/python scripts/tools/camera_alignment/t3_tcp_overlay.py \
        2026-10-28_15-22-05_physical_full_pick_m2mm --max-z-mm 30
"""
import argparse
import csv
import json

import cv2
import numpy as np

from _common import (CLIENT, UTOOL1_FROM_J6_M, load_cameras, out_dir, pixel_to_base, project, run_path,
                     trace, triangulate, uf0_mm_to_base, wpr_to_rotation)


def tool_points(pose):
    """Base-frame points of the commanded tool for a UF0 [x, y, z, w, p, r] pose."""
    tcp = uf0_mm_to_base(pose[:3])
    R = wpr_to_rotation(pose[3:6])
    approach = R[:, 2]                       # tool +z points out of the fingertips
    wrist_centre = tcp - approach * UTOOL1_FROM_J6_M
    # J6 frame of the twin: x along the tool, y the finger spread, z = x cross y.
    j6 = np.column_stack((approach, R[:, 1], np.cross(approach, R[:, 1])))
    wrist_cam = np.array(json.load(open(CLIENT / 'calibration/cameras.json'))['wrist']['T_flange_cam'])
    return {'tcp': tcp, 'axis': [tcp - approach * k / 100 for k in range(1, 11)],
            'flange': wrist_centre + approach * 0.070, 'wrist_cam': wrist_centre + j6 @ wrist_cam[:3, 3]}


def overlay(cam, image, pose, label):
    pts = tool_points(pose)
    tcp_px = project(cam, [pts['tcp']])[0]
    axis_px = project(cam, pts['axis'])
    cv2.polylines(image, [np.vstack([tcp_px, axis_px]).astype(np.int32)], False, (0, 165, 255), 2)
    for k, p in enumerate(axis_px, 1):
        cv2.circle(image, tuple(p.astype(int)), 4 if k % 5 else 7, (0, 165, 255), -1)
    for name, colour in (('flange', (255, 0, 255)), ('wrist_cam', (255, 255, 0))):
        p = project(cam, [pts[name]])[0].astype(int)
        cv2.circle(image, tuple(p), 7, colour, -1)
        cv2.putText(image, name, (p[0] + 9, p[1]), 0, .6, colour, 2)
    cv2.drawMarker(image, tuple(tcp_px.astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    cv2.putText(image, label, (20, 40), 0, .9, (255, 255, 255), 2)
    return tcp_px


def read_clicks(path, cams):
    rows = []
    for row in csv.DictReader(open(path)):
        record = next(r for r in trace(row['run']) if r.get('frame') == row['frame'])
        pose = np.asarray(record['tcp_pose_before'], dtype=float)
        tcp = uf0_mm_to_base(pose[:3])
        table_uv = (float(row['table_u']), float(row['table_v']))
        if row.get('side_u'):
            seen = triangulate(cams, {'table': table_uv, 'side': (float(row['side_u']), float(row['side_v']))})
            rows.append((row, tcp, seen - tcp, None))
            continue
        # Table only: (a) height right, XY wrong; (b) XY right, height wrong.
        xy_reading = pixel_to_base(cams['table'], table_uv, tcp[2]) - tcp
        zs = np.linspace(tcp[2] - .15, tcp[2] + .15, 3001)
        candidates = np.array([pixel_to_base(cams['table'], table_uv, z) for z in zs])
        best = int(np.argmin(np.linalg.norm(candidates[:, :2] - tcp[:2], axis=1)))
        rows.append((row, tcp, xy_reading, (zs[best] - tcp[2],
                                            float(np.linalg.norm(candidates[best, :2] - tcp[:2])))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--max-z-mm', type=float, default=230.0,
                        help='only frames whose TCP is below this base-frame height (targets, not transit)')
    parser.add_argument('--clicks', help='CSV of fingertip pixels read off the images')
    parser.add_argument('--out')
    args = parser.parse_args()

    cams = load_cameras()
    target = out_dir('t3_tcp_overlay', args.out) / run_path(args.run).name
    target.mkdir(parents=True, exist_ok=True)
    last = None
    for record in trace(args.run):
        pose = record.get('tcp_pose_before')
        if not record.get('frame') or pose is None:
            continue
        pose = np.asarray(pose, dtype=float)
        if uf0_mm_to_base(pose[:3])[2] * 1000 > args.max_z_mm or (last is not None and np.allclose(pose, last, atol=.5)):
            continue
        last = pose
        image = cv2.imread(str(run_path(args.run) / 'tablecam_raw' / record['frame']))
        overlay(cams['table'], image, pose, f'{record["frame"]} UF0 {np.round(pose[:3], 1)} WPR {np.round(pose[3:6])}')
        cv2.imwrite(str(target / record['frame']), image)
        print('wrote', target / record['frame'])
    if args.clicks:
        print('\nfingertip offsets (seen - commanded), mm')
        for row, tcp, delta, alt in read_clicks(args.clicks, cams):
            text = f'{row["run"]} {row["frame"]} {row["label"]:10s} '
            if alt is None:
                print(text + f'triangulated dXYZ = {np.round(delta * 1000, 1)}')
            else:
                print(text + f'if height right: dXY = {np.round(delta[:2] * 1000, 1)}; '
                      f'if XY right: dZ = {alt[0] * 1000:+.1f} (residual {alt[1] * 1000:.1f})')


if __name__ == '__main__':
    main()
