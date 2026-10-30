import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "PhysicalAI-Hackathon-2026")]

from src.remediation import wrist_camera_model as M  # noqa: E402
from src.remediation.controller import command_rotation_to_j6, wpr_to_rotation  # noqa: E402
from src.remediation.wrist_refinement import (FINGERTIP_TCP_J6_M,  # noqa: E402
                                               calibrated_home_joints_deg,
                                               fit_local_support_plane,
                                               project_base_to_wrist,
                                               refine_from_vlm_box,
                                               render_wrist_refinement_annotation,
                                               wrist_pixel_to_plane,
                                               wrist_view_tcp)


class WristRefinementTests(unittest.TestCase):
    def setUp(self):
        calibration = json.loads(M.CALIBRATION.read_text())
        self.joints = calibration["wrist_depth"]["solved_at_joints"]
        self.size = (1920, 1080)

    def test_calibrated_home_keeps_physical_near_zero_j6(self):
        home = calibrated_home_joints_deg()
        self.assertLess(abs(float(home[5])), 0.1)

    def test_fast_view_keeps_tcp_over_coarse_target(self):
        target = np.array([0.31, -0.04, 0.01])
        command = wpr_to_rotation((-180.0, 0.0, 0.0))
        tcp = wrist_view_tcp(target, command, 0.300, centre_camera=False)
        np.testing.assert_allclose(tcp[:2], target[:2], atol=1e-12)

    def test_projection_and_plane_backprojection_round_trip(self):
        point = np.array([0.31, -0.08, -0.0051])
        pixel = project_base_to_wrist(point, self.joints, self.size)
        reconstructed = wrist_pixel_to_plane(pixel, point[2], self.joints, self.size)
        np.testing.assert_allclose(reconstructed, point, atol=1e-9)

    def test_vlm_box_center_produces_bounded_base_correction(self):
        expected = np.array([0.31, -0.08, -0.0051])
        pixel = project_base_to_wrist(expected, self.joints, self.size)
        box = [pixel[0] - 20, pixel[1] - 15, pixel[0] + 20, pixel[1] + 15]
        result = refine_from_vlm_box(
            box, expected, expected[2], self.joints, self.size)
        np.testing.assert_allclose(result["base_pos_m"], expected, atol=1e-9)
        self.assertLess(result["correction_norm_m"], 1e-9)

    def test_view_tcp_places_target_on_optical_axis(self):
        target = np.array([0.34, -0.20, 0.006])
        command = wpr_to_rotation((-180.0, 0.0, 0.0))
        R_j6 = command_rotation_to_j6(command)
        tcp = wrist_view_tcp(target, command, 0.18)
        flange = tcp - R_j6 @ FINGERTIP_TCP_J6_M
        eye = flange + R_j6 @ M.T_FLANGE_COLOUR[:3, 3]
        axis = R_j6 @ M.T_FLANGE_COLOUR[:3, 2]
        np.testing.assert_allclose(target - eye, axis * 0.18, atol=1e-12)

    def test_local_support_plane_uses_annulus_not_cap_depth(self):
        rng = np.random.default_rng(4)
        xy = rng.uniform([0.25, -0.10], [0.35, 0.02], size=(80, 2))
        expected_z = -0.003 + 0.01 * (xy[:, 0] - 0.30)
        points = np.column_stack((xy, expected_z + rng.normal(0, 0.0004, len(xy))))
        # Support pixels lie outside the cap but inside its expanded window.
        pixels = np.column_stack((rng.uniform(800, 870, len(xy)),
                                  rng.uniform(400, 700, len(xy))))
        cap_xy = rng.uniform([0.29, -0.03], [0.31, -0.01], size=(10, 2))
        points = np.vstack((points, np.column_stack((cap_xy, np.full(10, 0.010)))))
        pixels = np.vstack((pixels, rng.uniform([910, 510], [1010, 610], size=(10, 2))))
        result = fit_local_support_plane(
            points, pixels, [900, 500, 1020, 620], [0.30, -0.02], -0.003)
        self.assertAlmostEqual(result["support_z_base_m"], -0.003, delta=0.001)
        self.assertGreaterEqual(result["point_count"], 70)
        self.assertLess(result["rms_m"], 0.001)

    def test_local_support_plane_rejects_sparse_depth(self):
        with self.assertRaisesRegex(ValueError, "valid wrist-depth support points"):
            fit_local_support_plane(
                [[0.3, 0.0, -0.005]] * 4,
                [[800, 500], [810, 510], [820, 520], [830, 530]],
                [900, 500, 1020, 620], [0.30, -0.02], -0.005)

    def test_wrist_annotation_marks_vlm_and_coarse_centres(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        result = {
            "bbox_xyxy": [130, 90, 170, 130],
            "all_boxes_xyxy": [[130, 90, 170, 130]],
            "latency_s": 2.5,
        }
        annotated = render_wrist_refinement_annotation(
            image, "table_cap", result, [100, 110],
            {"correction_norm_m": 0.004})
        np.testing.assert_array_equal(annotated[90, 150], [0, 255, 0])
        np.testing.assert_array_equal(annotated[110, 100], [255, 0, 0])
        self.assertFalse(np.any(image))


if __name__ == "__main__":
    unittest.main()
