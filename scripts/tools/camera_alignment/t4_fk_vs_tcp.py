#!/usr/bin/env python3
"""T4 (H2/H4): does the controller's TCP agree with the arm's kinematics?

The camera calibration lives in the robot base frame; commands are UTOOL1 in
UF0. This recomputes the fingertip from the reported joints (the twin's FK in
calibration/geometry.py plus the 293 mm UTOOL1 along J6 +x) and compares it
with the reported tcp_pose. A constant z difference means UF0 is not
base - 330 mm; a difference that changes with the wrist orientation means the
controller's UTOOL1 is not the one in FRAMES.md (for example, it was
re-taught for the custom fingers, or never was).

Needs records that carry both "joints" and "tcp_pose" (probe_policy.py writes
them; the LeRobot dataset from `run.py --download` also has the joint state).

    .conda_env/bin/python scripts/tools/camera_alignment/t4_fk_vs_tcp.py \
        shared/runs/2026-10-29_camera_alignment/probe/<run>
"""
import argparse

import numpy as np

from _common import UTOOL1_FROM_J6_M, flange_pose, trace, uf0_mm_to_base, wpr_to_rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--tool-m', type=float, default=UTOOL1_FROM_J6_M, help='J6 origin -> TCP along J6 +x')
    args = parser.parse_args()

    rows = []
    for record in trace(args.run):
        joints, pose = record.get('joints'), record.get('tcp_pose')
        if joints is None or pose is None:
            continue
        pose = np.asarray(pose, dtype=float)
        T = flange_pose(joints)
        fk_tcp = T[:3, :3] @ np.array([args.tool_m, 0.0, 0.0]) + T[:3, 3]
        reported = uf0_mm_to_base(pose[:3])
        axis_angle = np.degrees(np.arccos(np.clip(T[:3, 0] @ wpr_to_rotation(pose[3:6])[:, 2], -1, 1)))
        rows.append(((fk_tcp - reported) * 1000, axis_angle, pose[3:6]))
    if not rows:
        raise SystemExit('no records with both joints and tcp_pose')
    delta = np.array([r[0] for r in rows])
    print(f'{len(rows)} poses; FK fingertip - reported tcp (mm)')
    print(f'  mean {delta.mean(0).round(2)}  std {delta.std(0).round(2)}  max |.| {abs(delta).max(0).round(2)}')
    print(f'  tool-axis angle FK vs W/P/R: median {np.median([r[1] for r in rows]):.2f} deg')
    for wpr in {tuple(np.round(r[2])) for r in rows}:
        sel = np.array([tuple(np.round(r[2])) == wpr for r in rows])
        print(f'  W/P/R {wpr}: mean {delta[sel].mean(0).round(2)} mm over {sel.sum()} poses')
    if abs(delta.mean(0)).max() < 1.5 and delta.std(0).max() < 1.5:
        print('FK and the controller agree: UF0 and the 293 mm UTOOL1 are what the code assumes.')
    else:
        print('Mismatch: fix UF0/UTOOL1 assumptions before tuning camera planes or offsets.')


if __name__ == '__main__':
    main()
