#!/usr/bin/env python3
"""T8: decide which hypothesis explains the measured misses, and by how much.

Input CSV, one row per measurement, all positions in robot-base metres:

  kind,class,pred_x,pred_y,pred_z,meas_x,meas_y,meas_z,w,p,r
  target,missing_cap,...   pred = camera-computed target (trace base_pos_m),
                           meas = true feature centre (triangulated T3 click,
                                  or the TCP after jogging the real fingertip onto it
                                  once the tool error is known)
  tip,,...                 pred = commanded UTOOL1 position, meas = the real
                           fingertip centre (triangulated); w,p,r = commanded W/P/R

Target rows are fitted with competing localization models; tip rows with
tool models. Each model is reported with its RMS residual and BIC (lower is
better; a model must beat the simpler one by > ~6 to be worth its parameters).

  L0 constant XYZ offset                                   (3)
  L1 constant + the measured heights pushed through the
     policy's ray/plane mapping (H1: plane height error)   (3)
  L2 L1 + a small TableCam rotation (H3)                   (6)
  T0 constant base-frame tip offset                        (3)
  T1 tip offset fixed in the tool frame (H2)               (3)
  T2 both                                                  (6)

    .conda_env/bin/python scripts/tools/camera_alignment/t8_fit_error_model.py measurements.csv
"""
import argparse
import csv

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from _common import load_cameras, pixel_to_base, project, wpr_to_rotation


def bic(residual, k):
    n = residual.size
    return n * np.log(max(np.mean(residual ** 2), 1e-12)) + k * np.log(n)


def rotated_camera(cam, rotvec):
    """The TableCam turned by a small base-frame rotation about its own centre."""
    moved = dict(cam)
    T = cam['T'].copy()
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ T[:3, :3]
    moved['T'] = T
    return moved


def fit_targets(cam, rows):
    pred = np.array([[r['pred_x'], r['pred_y'], r['pred_z']] for r in rows])
    meas = np.array([[r['meas_x'], r['meas_y'], r['meas_z']] for r in rows])

    def through_camera(true_cam):
        # The pixel the (possibly moved) camera saw, mapped back the way the policy did.
        return np.array([pixel_to_base(cam, project(true_cam, [m])[0], z)
                         for m, z in zip(meas, pred[:, 2])])

    models = {
        'L0 constant': (3, lambda q: meas + q[:3]),
        'L1 plane height only': (3, lambda q: through_camera(cam) + q[:3]),
        'L2 plane + camera rot.': (6, lambda q: through_camera(rotated_camera(cam, q[3:6])) + q[:3]),
    }
    for name, (k, model) in models.items():
        if len(rows) * 3 <= k:
            continue
        fit = least_squares(lambda q: (model(q) - pred).ravel(), np.zeros(k))
        line = (f'  {name:24s} k={k} RMS {np.sqrt(np.mean(fit.fun ** 2)) * 1000:5.2f} mm  '
                f'BIC {bic(fit.fun, k):8.1f}  offset {np.round(fit.x[:3] * 1000, 1)} mm')
        if k == 6:
            line += f'  rotation {np.round(np.degrees(fit.x[3:6]), 3)} deg'
        print(line)
    for name in sorted({r['class'] for r in rows}):
        sel = np.array([r['class'] == name for r in rows])
        print(f'  measured height - plane used, {name}: {np.mean(meas[sel, 2] - pred[sel, 2]) * 1000:+.1f} mm '
              f'(n={sel.sum()})')


def fit_tips(rows):
    cmd = np.array([[r['pred_x'], r['pred_y'], r['pred_z']] for r in rows])
    tip = np.array([[r['meas_x'], r['meas_y'], r['meas_z']] for r in rows])
    R = np.array([wpr_to_rotation([r['w'], r['p'], r['r']]) for r in rows])
    for name, base, tool in (('T0 base-frame offset', True, False), ('T1 tool-frame offset', False, True),
                             ('T2 both', True, True)):
        k = 3 * (base + tool)
        if len(rows) * 3 <= k:
            continue

        def residual(q):
            b = q[:3] if base else np.zeros(3)
            t = q[-3:] if tool else np.zeros(3)
            return (cmd + b + np.einsum('nij,j->ni', R, t) - tip).ravel()

        fit = least_squares(residual, np.zeros(k))
        line = f'  {name:22s} k={k:2d} RMS {np.sqrt(np.mean(fit.fun ** 2)) * 1000:5.2f} mm  BIC {bic(fit.fun, k):8.1f}'
        if base:
            line += f'  base {np.round(fit.x[:3] * 1000, 1)} mm'
        if tool:
            line += f'  tool-frame {np.round(fit.x[-3:] * 1000, 1)} mm (tool z = approach)'
        print(line)
    print('  A tool-frame offset (T1) changes direction with the wrist roll; it is why the vertical pick/place '
          'and the 45-degree cone press can miss in different directions with one hardware error.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv')
    args = parser.parse_args()
    rows = []
    for row in csv.DictReader(open(args.csv)):
        for key in ('pred_x', 'pred_y', 'pred_z', 'meas_x', 'meas_y', 'meas_z', 'w', 'p', 'r'):
            row[key] = float(row[key]) if row.get(key) not in (None, '') else 0.0
        rows.append(row)
    targets = [r for r in rows if r['kind'] == 'target']
    tips = [r for r in rows if r['kind'] == 'tip']
    cam = load_cameras()['table']
    if targets:
        print(f'localization ({len(targets)} targets): computed - true')
        fit_targets(cam, targets)
        print('  If L1 beats L0, set the planes from the measured heights: --vial-rim-z-m for the tray '
              'classes, --work-surface-z-m for table_cap. Believe L2 only if it also lowers the RMS by '
              '> 0.5 mm with a rotation > 0.1 deg (0.5 px of click noise alone fits ~0.1-0.2 deg); then '
              're-solve the TableCam pose with ArUco rather than applying the fitted rotation.')
    if tips:
        print(f'tool ({len(tips)} tips): commanded + offset = real fingertip')
        fit_tips(tips)
        print('  For the vertical posture the fitted tool-frame z is the --pick/--place/--press-offset-m to use '
              '(negative: the real tip stops above the TCP). Tool-frame x/y need UTOOL1 re-taught; the cone press '
              'also needs --cone-tip-from-tcp-m re-measured.')


if __name__ == '__main__':
    main()
