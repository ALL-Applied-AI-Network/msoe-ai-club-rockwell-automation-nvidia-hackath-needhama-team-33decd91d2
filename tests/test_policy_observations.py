import unittest

import numpy as np

from src.remediation.policy_observations import find_free_table_spot, upright_cap_near


class PolicyObservationTests(unittest.TestCase):
    def test_free_spot_comes_from_observed_support(self):
        xs, ys = np.meshgrid(np.linspace(0.25, 0.38, 80), np.linspace(-0.28, -0.08, 80))
        points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.0051)])
        spot = find_free_table_spot(points, [[0.30, -0.20, 0.01]])
        self.assertAlmostEqual(spot[2], -0.003)
        self.assertGreaterEqual(np.linalg.norm(spot[:2] - [0.30, -0.20]), 0.045)

    def test_obstacle_rejects_candidate(self):
        xs, ys = np.meshgrid(np.linspace(0.18, 0.42, 100), np.linspace(-0.30, 0.25, 100))
        support = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.0051)])
        obstacle = np.array([[0.36, -0.20, 0.03]])
        spot = find_free_table_spot(np.vstack([support, obstacle]), [[0.30, -0.20, 0.01]])
        self.assertGreater(np.linalg.norm(spot[:2] - [0.36, -0.20]), 0.025)

    def test_free_spot_without_loose_cap_anchor_comes_from_depth(self):
        xs, ys = np.meshgrid(np.linspace(0.20, 0.40, 100), np.linspace(-0.28, -0.10, 100))
        points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.0051)])
        spot = find_free_table_spot(points, [], exclusions=[[0.30, -0.10, 0.2]])
        self.assertGreaterEqual(spot[0], 0.18)
        self.assertLessEqual(spot[0], 0.42)
        self.assertGreaterEqual(spot[1], -0.30)
        self.assertAlmostEqual(spot[2], -0.003)

    def test_upright_cap_requires_depth_and_proximity(self):
        detections = [
            {"class_name": "table_cap", "base_pos_m": [0.341, -0.201, 0.01], "depth_available": True},
            {"class_name": "table_cap", "base_pos_m": [0.20, 0.20, 0.01], "depth_available": True},
        ]
        result = upright_cap_near(detections, [0.34, -0.20])
        np.testing.assert_allclose(result, [0.341, -0.201, 0.01])
        self.assertIsNone(upright_cap_near(detections, [0.40, 0.20]))


if __name__ == "__main__":
    unittest.main()
