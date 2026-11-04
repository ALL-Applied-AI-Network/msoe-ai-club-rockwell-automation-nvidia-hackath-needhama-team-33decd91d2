import unittest

import numpy as np

from src.inspection.height_plane import fit_parallel_work_surface


class HeightPlaneTests(unittest.TestCase):
    def test_parallel_fit_rejects_fixture_and_depth_outliers(self):
        rng = np.random.default_rng(4)
        surface = np.column_stack((
            rng.uniform(0.0, 0.7, 3000),
            rng.uniform(-0.4, 0.4, 3000),
            rng.normal(-0.0024, 0.0007, 3000),
        ))
        fixture = np.column_stack((
            rng.uniform(0.1, 0.5, 800),
            rng.uniform(-0.2, 0.2, 800),
            rng.normal(0.088, 0.002, 800),
        ))
        outliers = np.column_stack((
            rng.uniform(0.0, 0.7, 300),
            rng.uniform(-0.4, 0.4, 300),
            rng.uniform(-0.020, 0.015, 300),
        ))
        result = fit_parallel_work_surface(
            np.vstack((surface, fixture, outliers)), -0.0051)
        self.assertAlmostEqual(result["fitted_z_base_m"], -0.0024, delta=0.00015)
        self.assertGreater(result["inlier_count"], 2500)
        self.assertLess(result["rms_m"], 0.001)

    def test_parallel_fit_rejects_too_few_points(self):
        with self.assertRaisesRegex(ValueError, "only 20 depth points"):
            fit_parallel_work_surface(np.zeros((20, 3)), 0.0)


if __name__ == "__main__":
    unittest.main()
