"""The real cell's wrist camera, from its calibration.  No simulator imports.

PhysicalAI-Hackathon-2026/calibration/cameras.json (and FRAMES.md) describe two streams:

* colour ("wrist"): captured at 848 x 480 (15 fps, USB 2.0) and served at 1280 x 720
  over the same field of view.  At 1280 x 720: fx = fy = 619.6 px, principal point at
  the image centre, one radial term k1 = -0.0196 (u = fx x (1 + k1 r^2) + W/2).
* depth ("wrist_depth"): a separate rectified pinhole stream, 848 x 480, fx = fy =
  416.9 px, principal point (421.35, 239.34), mounted 21 mm from and 2.9 deg off the
  colour camera.  uint16 mm along its optical axis, 0 = no depth.  Plane-fit RMS
  3.0 mm at the solve pose.

Both poses are T_flange_cam (CV camera: x right, y down, z forward -> J6_link).

The simulator renders ideal, centred pinhole images (`render_spec`); `real_colour` and
`real_depth` turn them into what the cell delivers.  The servo then treats them like the
real camera's: `undistort_px` / `distort_px` for colour pixels and `register_depth` to
put depth onto the colour image.
"""
import json
from pathlib import Path

import cv2
import numpy as np

CALIBRATION = Path(__file__).resolve().parents[2] / 'PhysicalAI-Hackathon-2026/calibration/cameras.json'
_RAW = json.loads(CALIBRATION.read_text())
COLOUR, DEPTH = _RAW['wrist'], _RAW['wrist_depth']
T_FLANGE_COLOUR = np.asarray(COLOUR['T_flange_cam'], float)
T_FLANGE_DEPTH = np.asarray(DEPTH['T_flange_cam'], float)
SERVED_SIZE = tuple(COLOUR['size'])      # (1280, 720)
SENSOR_SIZE = (848, 480)                 # FRAMES.md: colour captured at 848 x 480
DEPTH_SIZE = tuple(DEPTH['size'])        # (848, 480)
FX, K1 = float(COLOUR['fx']), float(COLOUR['k1'])
# Depth noise: the calibration's plane-fit RMS at its solve range, grown with range^2 as
# for stereo depth.  0.363 m is the depth camera's range to the work surface along its
# axis at solved_at_joints (tests/test_wrist_camera_model.py checks it).
DEPTH_RMS_M, DEPTH_RMS_RANGE_M = DEPTH['rms_plane_mm'] / 1000., 0.363


def colour_K():
    """Pinhole intrinsics of the served colour image once undistorted (undistort_px)."""
    W, H = SERVED_SIZE
    return np.array([[FX, 0., W / 2.], [0., FX, H / 2.], [0., 0., 1.]])


def distort_px(uv):
    """(N, 2) pinhole pixels (colour_K) -> served colour pixels, with the lens's k1."""
    W, H = SERVED_SIZE
    uv = np.asarray(uv, float).reshape(-1, 2)
    x, y = (uv[:, 0] - W / 2.) / FX, (uv[:, 1] - H / 2.) / FX
    d = 1. + K1 * (x * x + y * y)
    return np.stack([FX * x * d + W / 2., FX * y * d + H / 2.], -1)


def undistort_px(uv):
    """(N, 2) served colour pixels -> pinhole pixels (colour_K); inverse of distort_px."""
    W, H = SERVED_SIZE
    uv = np.asarray(uv, float).reshape(-1, 2)
    xd, yd = (uv[:, 0] - W / 2.) / FX, (uv[:, 1] - H / 2.) / FX
    x, y = xd, yd
    for _ in range(10):        # as calibration.geometry.pixel_to_base
        d = 1. + K1 * (x * x + y * y)
        x, y = xd / d, yd / d
    return np.stack([FX * x + W / 2., FX * y + H / 2.], -1)


def render_spec():
    """Ideal centred pinhole renders, with margins covering the distorted colour image
    and the depth stream's off-centre principal point: {stream: (W, H, fx)}."""
    W, H = SERVED_SIZE
    centre = np.array([W / 2., H / 2.])
    scale = max((centre - undistort_px([[0., 0.]])[0]) / centre)   # barrel: the corners see further out
    colour = (2 * int(np.ceil(scale * W / 2.)) + 8, 2 * int(np.ceil(scale * H / 2.)) + 8, FX)
    dw, dh = DEPTH_SIZE
    depth = (2 * int(np.ceil(max(DEPTH['cx'], dw - DEPTH['cx']))) + 4,
             2 * int(np.ceil(max(DEPTH['cy'], dh - DEPTH['cy']))) + 4, float(DEPTH['fx']))
    return {'colour': colour, 'depth': depth}


def real_colour(ideal_rgb):
    """Ideal pinhole render (render_spec()['colour']) -> the served 1280 x 720 image:
    lens distortion and the 848 x 480 sensor, upscaled to the served size."""
    Wr, Hr, fr = render_spec()['colour']
    W, H = SERVED_SIZE
    sw, sh = SENSOR_SIZE
    # Each sensor pixel's position in the served image (cv2.resize's centre alignment),
    # undistorted, then located in the ideal render.
    i, j = np.meshgrid(np.arange(sw), np.arange(sh))
    served = np.stack([(i + .5) * W / sw - .5, (j + .5) * H / sh - .5], -1).reshape(-1, 2)
    pin = undistort_px(served)
    mx = ((pin[:, 0] - W / 2.) * fr / FX + Wr / 2.).reshape(sh, sw).astype(np.float32)
    my = ((pin[:, 1] - H / 2.) * fr / FX + Hr / 2.).reshape(sh, sw).astype(np.float32)
    # A sensor pixel is ~1.5 render pixels wide: integrate over it before sampling.
    blurred = cv2.GaussianBlur(np.ascontiguousarray(ideal_rgb), (0, 0), 0.5 * W / sw)
    sensor = cv2.remap(blurred, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(sensor, SERVED_SIZE, interpolation=cv2.INTER_LINEAR)


def real_depth(ideal_depth_m, rng=None):
    """Ideal distance-to-image-plane render (render_spec()['depth'], metres) -> the
    depth stream: 848 x 480 at the calibrated principal point, range noise, uint16 mm
    with 0 where there is no return."""
    Wr, Hr, _ = render_spec()['depth']
    dw, dh = DEPTH_SIZE
    i, j = np.meshgrid(np.arange(dw, dtype=np.float32), np.arange(dh, dtype=np.float32))
    z = cv2.remap(np.asarray(ideal_depth_m, np.float32), i - DEPTH['cx'] + Wr / 2., j - DEPTH['cy'] + Hr / 2.,
                  cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    ok = np.isfinite(z) & (z > 0)
    z = np.where(ok, z, 0.).astype(np.float64)
    if rng is not None:
        z += ok * rng.standard_normal(z.shape) * DEPTH_RMS_M * (z / DEPTH_RMS_RANGE_M) ** 2
    mm = np.rint(z * 1000.)
    return np.where(ok & (mm > 0) & (mm < 65536), mm, 0).astype(np.uint16)


def register_depth(depth_mm):
    """Depth stream (uint16 mm) -> (720, 1280) float32 depth along the colour camera's
    axis (m) on the served colour pixels; NaN where no depth pixel lands."""
    dw, dh = DEPTH_SIZE
    v, u = np.nonzero(np.asarray(depth_mm) > 0)
    z = np.asarray(depth_mm, float)[v, u] / 1000.
    p_d = np.stack([(u - DEPTH['cx']) / DEPTH['fx'] * z, (v - DEPTH['cy']) / DEPTH['fy'] * z, z], -1)
    T = np.linalg.inv(T_FLANGE_COLOUR) @ T_FLANGE_DEPTH
    p_c = p_d @ T[:3, :3].T + T[:3, 3]
    p_c = p_c[p_c[:, 2] > 1e-3]
    W, H = SERVED_SIZE
    uv = np.rint(distort_px(FX * p_c[:, :2] / p_c[:, 2:] + [W / 2., H / 2.])).astype(int)
    out = np.full((H, W), np.inf, np.float32)
    # Depth pixels land ~1.5 colour pixels apart: splat each over 3 x 3, nearest wins, so
    # surfaces behind cannot show through the gaps between a near surface's samples.
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu, vv = uv[:, 0] + du, uv[:, 1] + dv
            inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            np.minimum.at(out, (vv[inside], uu[inside]), p_c[inside, 2].astype(np.float32))
    out[~np.isfinite(out)] = np.nan
    return out
