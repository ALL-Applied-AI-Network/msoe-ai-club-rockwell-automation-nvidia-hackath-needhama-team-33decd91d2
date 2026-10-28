import unittest

import numpy as np

from src.inspection.depth_visualization import render_depth_heatmap


class DepthVisualizationTests(unittest.TestCase):
    def test_invalid_depth_stays_black_and_valid_depth_is_colored(self):
        depth = np.zeros((40, 60), dtype=np.uint16)
        depth[5:35, 5:30] = 900
        depth[5:35, 30:55] = 1100
        rendered = render_depth_heatmap(depth)
        self.assertEqual(rendered.shape, (40, 60, 3))
        np.testing.assert_array_equal(rendered[0, 0], [0, 0, 0])
        self.assertGreater(int(rendered[15, 15].sum()), 0)

    def test_depth_jump_gets_white_outline(self):
        depth = np.full((40, 60), 1000, dtype=np.uint16)
        depth[:, 30:] = 1040
        rendered = render_depth_heatmap(depth, edge_threshold_mm=8)
        white = np.all(rendered == 255, axis=2)
        self.assertGreater(int(white[:, 28:33].sum()), 30)

    def test_uniform_and_empty_depth_are_supported(self):
        uniform = render_depth_heatmap(np.full((8, 9), 700, dtype=np.uint16))
        empty = render_depth_heatmap(np.zeros((8, 9), dtype=np.uint16))
        self.assertEqual(uniform.dtype, np.uint8)
        self.assertFalse(empty.any())

    def test_non_image_input_is_rejected(self):
        with self.assertRaises(ValueError):
            render_depth_heatmap(np.zeros((2, 3, 1), dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
