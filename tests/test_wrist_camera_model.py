import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'PhysicalAI-Hackathon-2026')]

from calibration.geometry import SURFACE_Z, depth_to_base, flange_pose, load, project  # noqa: E402
from src.remediation import wrist_camera_model as M  # noqa: E402

CAMS = load()
SOLVED_AT = json.loads(M.CALIBRATION.read_text())['wrist_depth']['solved_at_joints']


def flange_point(T_cam, uvz, fx, cx, cy):
    """Point (flange frame) at pinhole pixel (u, v) and axial depth z of a camera."""
    u, v, z = uvz
    return T_cam[:3, :3] @ np.array([(u - cx) / fx * z, (v - cy) / fx * z, z]) + T_cam[:3, 3]


class WristCameraModelTest(unittest.TestCase):
    def test_distortion_matches_calibration_projection(self):
        pts = [flange_point(M.T_FLANGE_COLOUR, (u, v, .2), M.FX, 640, 360) for u, v in ((50, 40), (640, 360), (1200, 650))]
        pin = M.colour_K() @ (np.linalg.inv(M.T_FLANGE_COLOUR)[:3, :3] @ np.array(pts).T
                              + np.linalg.inv(M.T_FLANGE_COLOUR)[:3, 3:])
        np.testing.assert_allclose(M.distort_px((pin[:2] / pin[2]).T), project(CAMS['wrist'], pts), atol=1e-6)
        uv = np.array([[0., 0.], [1279., 719.], [333., 120.]])
        np.testing.assert_allclose(M.distort_px(M.undistort_px(uv)), uv, atol=1e-6)

    def test_deployment_size_preserves_field_of_view(self):
        deployment = (1920, 1080)
        self.assertAlmostEqual(M.served_fx(deployment), M.FX * 1.5)
        calibrated = np.array([[100., 80.], [640., 360.], [1150., 660.]])
        scaled = calibrated * 1.5
        np.testing.assert_allclose(
            M.distort_px(M.undistort_px(scaled, deployment), deployment),
            scaled, atol=1e-6)
        K0, K1 = M.colour_K(), M.colour_K(deployment)
        np.testing.assert_allclose(K1, np.diag([1.5, 1.5, 1.0]) @ K0)

    def test_depth_noise_reference_range(self):
        T = flange_pose(SOLVED_AT) @ M.T_FLANGE_DEPTH
        self.assertAlmostEqual((SURFACE_Z - T[2, 3]) / T[2, 2], M.DEPTH_RMS_RANGE_M, places=3)

    def test_real_colour_puts_a_point_where_the_calibration_does(self):
        Wr, Hr, fr = M.render_spec()['colour']
        p = flange_point(M.T_FLANGE_COLOUR, (1100, 600, .25), M.FX, 640, 360)   # near a corner
        Tinv = np.linalg.inv(M.T_FLANGE_COLOUR)
        pc = Tinv[:3, :3] @ p + Tinv[:3, 3]
        ideal = np.zeros((Hr, Wr, 3), np.uint8)
        cv2.circle(ideal, (int(round(fr * pc[0] / pc[2] + Wr / 2)), int(round(fr * pc[1] / pc[2] + Hr / 2))), 6,
                   (255, 255, 255), -1)
        served = M.real_colour(ideal)
        self.assertEqual(served.shape, (720, 1280, 3))
        v, u = np.nonzero(served[:, :, 0] > 128)
        np.testing.assert_allclose([u.mean(), v.mean()], project(CAMS['wrist'], [p])[0], atol=1.0)

    def test_real_depth_and_registration_on_a_plane(self):
        q = SOLVED_AT
        Wr, Hr, fr = M.render_spec()['depth']
        T_base_depth = flange_pose(q) @ M.T_FLANGE_DEPTH
        # Ideal centred render of the work surface seen by the depth camera.
        j, i = np.mgrid[0:Hr, 0:Wr].astype(float)
        rays = np.stack([(i - Wr / 2) / fr, (j - Hr / 2) / fr, np.ones_like(i)], -1) @ T_base_depth[:3, :3].T
        ideal = ((SURFACE_Z - T_base_depth[2, 3]) / rays[..., 2]).astype(np.float32)
        depth = M.real_depth(ideal)
        self.assertEqual((depth.shape, depth.dtype), ((480, 848), np.uint16))
        pts = depth_to_base(CAMS['wrist_depth'], depth, joints=q)
        self.assertLess(np.abs(pts[:, 2] - SURFACE_Z).max(), 0.002)
        # Registered onto colour pixels: depth along the colour axis to the same plane.
        reg = M.register_depth(depth)
        self.assertGreater(np.isfinite(reg).mean(), 0.3)
        T_base_colour = flange_pose(q) @ M.T_FLANGE_COLOUR
        v, u = np.nonzero(np.isfinite(reg))
        pin = M.undistort_px(np.stack([u, v], -1))
        rays = np.stack([(pin[:, 0] - 640) / M.FX, (pin[:, 1] - 360) / M.FX, np.ones(len(u))], -1) @ T_base_colour[:3, :3].T
        z_true = (SURFACE_Z - T_base_colour[2, 3]) / rays[:, 2]
        self.assertLess(np.percentile(np.abs(reg[v, u] - z_true), 99), 0.003)

    def test_registration_keeps_the_far_surface_out_of_a_near_one(self):
        # A near patch (0.18 m) over a far background (0.40 m), as a cap over the table.
        depth = np.full((480, 848), 400, np.uint16)
        depth[200:280, 380:460] = 180
        reg = M.register_depth(depth)
        near = reg < 0.25
        v, u = np.nonzero(near)
        inner = reg[v.min() + 3:v.max() - 2, u.min() + 3:u.max() - 2]
        self.assertTrue(np.all(inner < 0.25))

    def test_depth_noise_scale(self):
        z = np.full(M.render_spec()['depth'][1::-1], M.DEPTH_RMS_RANGE_M, np.float32)
        noisy = M.real_depth(z, np.random.default_rng(0)).astype(float) / 1000.
        self.assertAlmostEqual(noisy.std(), M.DEPTH_RMS_M, delta=0.0005)


if __name__ == '__main__':
    unittest.main()
