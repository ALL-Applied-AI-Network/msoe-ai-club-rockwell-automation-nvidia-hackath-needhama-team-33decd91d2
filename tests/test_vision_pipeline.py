import unittest

import numpy as np

from src.inspection.vision_pipeline import cap_axis_from_rgbd_points


class CapAxisFromRgbdTests(unittest.TestCase):
    def test_recovers_upward_normal_of_sloped_cap_surface(self):
        axis = np.array([0.30, -0.20, 0.9327379])
        axis /= np.linalg.norm(axis)
        u = np.array([axis[2], 0.0, -axis[0]])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        grid = np.linspace(-0.012, 0.012, 9)
        points = np.array([np.array([0.31, 0.02, 0.22]) + x * u + y * v
                           for x in grid for y in grid])
        measured = cap_axis_from_rgbd_points(points)
        np.testing.assert_allclose(measured, axis, atol=1e-6)

    def test_rejects_too_few_points(self):
        self.assertIsNone(cap_axis_from_rgbd_points(np.zeros((5, 3))))


if __name__ == "__main__":
    unittest.main()
