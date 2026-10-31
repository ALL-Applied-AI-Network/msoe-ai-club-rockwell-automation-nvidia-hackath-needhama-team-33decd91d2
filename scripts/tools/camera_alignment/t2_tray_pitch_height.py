#!/usr/bin/env python3
"""T2 (H1/H5): infer the true cap-top height from the tray hole spacing.

The ray/plane mapping scales the tray lattice with the assumed height: a plane
that is too low makes the tray look larger. Measure the real centre-to-centre
pitch of the tray with a ruler (rows and columns), then this finds the plane
height that reproduces it. If the row and column answers disagree by more
than a few mm, a single height cannot explain the image: suspect the camera
pose (H3) or a tray that is not the one you measured.

    .conda_env/bin/python scripts/tools/camera_alignment/t2_tray_pitch_height.py \
        2026-10-28_15-22-05_physical_full_pick_m2mm 0001.png --pitch-x-mm 51.3 --pitch-y-mm 61.2
"""
import argparse

import numpy as np

from _common import frame_record, load_cameras, pixel_to_base

TRAY_CLASSES = ('good_cap', 'crooked_cap', 'missing_cap')


def lattice_pitch(values, gap=0.02):
    """Mean spacing between 1-D clusters of tray centres."""
    v = np.sort(values)
    centres = [c.mean() for c in np.split(v, np.where(np.diff(v) > gap)[0] + 1)]
    return (float(np.mean(np.diff(centres))) if len(centres) > 1 else np.nan), len(centres)


def pitch_at(cam, pixels, z):
    xy = np.array([pixel_to_base(cam, p, z)[:2] for p in pixels])
    return lattice_pitch(xy[:, 0]), lattice_pitch(xy[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('frame', help='a frame with the arm at the inspection pose and the tray in view')
    parser.add_argument('--pitch-x-mm', type=float, required=True, help='measured spacing along robot x')
    parser.add_argument('--pitch-y-mm', type=float, required=True, help='measured spacing along robot y')
    parser.add_argument('--assumed-z-m', type=float, default=0.2101, help='plane the policy used for tray caps')
    args = parser.parse_args()

    cam = load_cameras()['table']
    detections = frame_record(args.run, args.frame).get('detections') or []
    pixels = [d['pixel'] for d in detections if d['class_name'] in TRAY_CLASSES]
    if len(pixels) < 8:
        raise SystemExit(f'only {len(pixels)} tray detections; choose a clearer inspection frame')
    zs = np.arange(0.150, 0.280, 0.0005)
    table = np.array([[x[0], y[0]] for x, y in (pitch_at(cam, pixels, z) for z in zs)]) * 1000
    (_, nx), (_, ny) = pitch_at(cam, pixels, args.assumed_z_m)
    print(f'{len(pixels)} tray detections, {nx} x-clusters, {ny} y-clusters')
    for z in (args.assumed_z_m - .01, args.assumed_z_m, args.assumed_z_m + .01):
        (px, _), (py, _) = pitch_at(cam, pixels, z)
        print(f'  plane z={z * 1000:6.1f} mm -> pitch x {px * 1000:5.2f} mm, y {py * 1000:5.2f} mm')
    zx = zs[np.nanargmin(abs(table[:, 0] - args.pitch_x_mm))]
    zy = zs[np.nanargmin(abs(table[:, 1] - args.pitch_y_mm))]
    print(f'height that reproduces the measured x pitch: {zx * 1000:.1f} mm')
    print(f'height that reproduces the measured y pitch: {zy * 1000:.1f} mm')
    print(f'assumed plane: {args.assumed_z_m * 1000:.1f} mm -> implied plane error '
          f'{(zx - args.assumed_z_m) * 1000:+.1f} (x) / {(zy - args.assumed_z_m) * 1000:+.1f} (y) mm')
    if abs(zx - zy) > 0.005:
        print('x and y disagree by more than 5 mm: a pure height error does not explain this frame; '
              'check the ruler pitch and run T5 (camera drift) before changing planes.')


if __name__ == '__main__':
    main()
