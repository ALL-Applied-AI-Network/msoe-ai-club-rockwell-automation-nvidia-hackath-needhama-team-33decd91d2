"""Wrist-camera servoing onto a crooked cap's axis.  No simulator imports.

A circle seen by a pinhole camera projects to an ellipse; it projects to a
circle exactly when the image plane is parallel to the circle's plane.  The
cap's open top is a 21.15 mm circle (docs/PHYSICAL_SPECIFICATIONS.md sec. 2),
so the wrist is turned until that opening looks round: the camera axis is then
the cap axis.  Each image also gives a closed-form estimate of the circle's
plane normal (two-fold ambiguous), used to choose the next wrist direction.
"""
import cv2
import numpy as np


def intrinsics(width, height, fx):
    return np.array([[fx, 0., width / 2.], [0., fx, height / 2.], [0., 0., 1.]])


def ellipse_conic(ellipse):
    """3x3 conic matrix (pixels) of an OpenCV ellipse ((cx, cy), (w, h), angle_deg)."""
    (cx, cy), (w, h), ang = ellipse
    a, b, t = w / 2., h / 2., np.radians(ang)
    c, s = np.cos(t), np.sin(t)
    A = c * c / (a * a) + s * s / (b * b)
    B = c * s * (1. / (a * a) - 1. / (b * b))
    Cc = s * s / (a * a) + c * c / (b * b)
    M = np.array([[A, B], [B, Cc]])
    ctr = np.array([cx, cy])
    Q = np.zeros((3, 3))
    Q[:2, :2] = M
    Q[:2, 2] = Q[2, :2] = -M @ ctr
    Q[2, 2] = ctr @ M @ ctr - 1.
    return Q


def circle_normals(ellipse, K):
    """The two candidate unit normals (camera frame: x right, y down, z forward) of
    the plane of a circle whose image is `ellipse`; each points toward the camera."""
    Q = K.T @ ellipse_conic(ellipse) @ K
    Q = Q / np.cbrt(np.linalg.det(Q)) if abs(np.linalg.det(Q)) > 0 else Q
    w, V = np.linalg.eigh(Q)
    if np.sum(w > 0) == 1:          # want two positive, one negative eigenvalue
        w, V = -w, V
    order = np.argsort(-w)          # l1 >= l2 > 0 > l3
    w, V = w[order], V[:, order]
    l1, l2, l3 = w
    g = np.sqrt(max(l1 - l2, 0.) / (l1 - l3))
    k = np.sqrt(max(l2 - l3, 0.) / (l1 - l3))
    out = []
    for sg in (1., -1.):
        n = V @ np.array([sg * g, 0., k])
        n /= np.linalg.norm(n)
        if n[2] > 0:                # face the camera (camera looks along +z)
            n = -n
        out.append(n)
    return out


def roundness(ellipse):
    """Minor / major axis ratio of the fitted ellipse (1.0 = round)."""
    (_, _), (w, h), _ = ellipse
    return float(min(w, h) / max(w, h))


def find_cap_opening(rgb, expected_px, max_dist_px=120, min_value=150, to_pinhole=None):
    """Ellipse fitted to the inner edge of the cap's lit flange top nearest `expected_px`.

    The flange top is bright yellow; the cavity inside is a darker olive and the
    plug sides are shaded, so a brightness threshold isolates the flange ring.  Its
    hole (child contour) is the 21.15 mm opening edge.  `to_pinhole` maps (N, 2) image
    pixels to undistorted pinhole pixels; the ellipse is fitted to the mapped edge, as
    a circle images to an ellipse only through a pinhole.  Returns (ellipse, ring_mask)
    or (None, None); the mask is in image pixels.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ring = cv2.inRange(hsv, (15, 80, min_value), (40, 255, 255))
    ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, hier = cv2.findContours(ring, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return None, None
    best = None
    for i, c in enumerate(contours):
        parent = hier[0][i][3]
        if parent < 0 or len(c) < 30 or cv2.contourArea(c) < 300:
            continue
        # The hole must be well inside its ring (ring area comparable to the hole's).
        if cv2.contourArea(contours[parent]) < 1.3 * cv2.contourArea(c):
            continue
        m = cv2.moments(c)
        ctr = np.array([m['m10'] / m['m00'], m['m01'] / m['m00']])
        d = np.linalg.norm(ctr - expected_px)
        if d < max_dist_px and (best is None or d < best[0]):
            best = (d, i)
    if best is None:
        return None, None
    i = best[1]
    mask = np.zeros(ring.shape, np.uint8)
    cv2.drawContours(mask, contours, hier[0][i][3], 255, -1)
    cv2.drawContours(mask, contours, i, 0, -1)
    edge = contours[i].reshape(-1, 2).astype(np.float32)
    if to_pinhole is not None:
        edge = np.asarray(to_pinhole(edge), np.float32)
    return cv2.fitEllipse(edge), mask


def depth_rise_direction(depth, mask):
    """Image direction (unit 2-vector) in which depth increases across the masked
    flange ring, from a least-squares plane fit depth ~ a u + b v + c."""
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(float)
    ok = np.isfinite(z) & (z > 0)
    if ok.sum() < 30:
        return None
    A = np.c_[u[ok], v[ok], np.ones(ok.sum())]
    (a, b, _), *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    g = np.array([a, b])
    return g / np.linalg.norm(g) if np.linalg.norm(g) > 0 else None


def choose_normal(candidates, rise):
    """Pick the candidate normal (camera frame) consistent with the measured depth rise:
    depth increases in the image direction of the normal's x-y components."""
    if rise is None:
        return None
    return max(candidates, key=lambda n: float(np.asarray(n[:2]) @ rise))


def look_rotation(forward, up_hint=(0., 0., 1.)):
    """Rotation whose third column is `forward` (CV camera z) and whose -y column
    (image up) is as close as possible to `up_hint`."""
    f = np.asarray(forward, float) / np.linalg.norm(forward)
    up = np.asarray(up_hint, float)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(f, [1., 0., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    return np.column_stack((x, y, f))
