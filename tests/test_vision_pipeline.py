import unittest

import numpy as np

from src.inspection.vision_pipeline import (cap_axis_from_rgbd_points,
                                            refine_defect_centres_from_grid)


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

    def test_rim_filter_rejects_cavity_points(self):
        axis = np.array([-0.25, -0.50, 1.0])
        axis /= np.linalg.norm(axis)
        gradient = -axis[:2] / axis[2]
        xy = np.array([(x, y) for x in np.linspace(0.29, 0.33, 9)
                       for y in np.linspace(0.0, 0.04, 9)])
        surface = np.c_[xy, 0.22 + (xy - [0.31, 0.02]) @ gradient]
        cavity = np.c_[xy[:20], np.full(20, 0.19)]
        measured = cap_axis_from_rgbd_points(
            np.vstack([surface, cavity]), min_surface_z=0.2086)
        np.testing.assert_allclose(measured, axis, atol=1e-6)


class TrayGridRefinementTests(unittest.TestCase):
    def test_defect_centres_snap_to_camera_fitted_rotated_grid(self):
        angle = np.radians(13.0)
        first = 0.052 * np.array([np.cos(angle), np.sin(angle)])
        second = 0.061 * np.array([-np.sin(angle), np.cos(angle)])
        origin = np.array([0.18, -0.11])
        targets = []
        crooked_truth = origin + 4 * first + 2 * second
        missing_truth = origin + 2 * first + second
        for i in range(6):
            for j in range(5):
                xy = origin + i * first + j * second
                name = 'good_cap'
                observed = xy.copy()
                if (i, j) == (4, 2):
                    name = 'crooked_cap'
                    observed += np.array([0.007, -0.003])
                elif (i, j) == (2, 1):
                    name = 'missing_cap'
                    observed += np.array([-0.005, 0.002])
                targets.append({'class_name': name,
                                'base_pos_m': [*observed, 0.21],
                                'plane_base_z_m': 0.2081,
                                'surface_top_base_z_m': 0.22,
                                'localization_note': ''})
        targets.append({'class_name': 'crooked_cap',
                        'base_pos_m': [0.02, 0.31, 0.24],
                        'surface_top_base_z_m': 0.61,
                        'localization_note': ''})
        refine_defect_centres_from_grid(targets)
        crooked = next(t for t in targets if t['class_name'] == 'crooked_cap'
                       and t['surface_top_base_z_m'] < 0.3)
        missing = next(t for t in targets if t['class_name'] == 'missing_cap')
        np.testing.assert_allclose(crooked['base_pos_m'][:2], crooked_truth, atol=0.001)
        np.testing.assert_allclose(crooked['feature_base_pos_m'][:2],
                                   crooked_truth + [0.007, -0.003], atol=1e-10)
        np.testing.assert_allclose(missing['base_pos_m'][:2], missing_truth, atol=0.001)
        self.assertAlmostEqual(missing['base_pos_m'][2], 0.2081)
        self.assertIn('tray-grid fit', crooked['localization_note'])


if __name__ == "__main__":
    unittest.main()
