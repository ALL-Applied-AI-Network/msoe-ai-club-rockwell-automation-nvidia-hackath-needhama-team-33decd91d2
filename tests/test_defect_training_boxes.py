import unittest

from src.inspection.defect_training.boxes import (
    Box,
    expand_box,
    pixel_box_to_yolo,
    rasterize_box,
    yolo_box_to_pixel,
)


class DefectTrainingBoxTests(unittest.TestCase):
    def test_centered_box_expands_exactly_25_percent(self):
        expanded, clipped = expand_box(Box(40, 30, 80, 50), (200, 100), 1.25)
        self.assertEqual(expanded, Box(35, 27.5, 85, 52.5))
        self.assertFalse(clipped)
        self.assertEqual(expanded.width, 50)
        self.assertEqual(expanded.height, 25)

    def test_edge_and_corner_expansion_clips_without_shifting(self):
        expanded, clipped = expand_box(Box(0, 0, 20, 20), (100, 80), 1.25)
        self.assertEqual(expanded, Box(0, 0, 22.5, 22.5))
        self.assertTrue(clipped)

    def test_square_mode_uses_longest_expanded_dimension(self):
        expanded, clipped = expand_box(Box(40, 40, 80, 60), (200, 100), 1.25, square=True)
        self.assertEqual(expanded, Box(35, 25, 85, 75))
        self.assertFalse(clipped)

    def test_yolo_pixel_round_trip(self):
        original = Box(123.25, 44.5, 456.75, 200.25)
        values = pixel_box_to_yolo(original, (640, 480))
        converted = yolo_box_to_pixel(values, (640, 480))
        for expected, actual in zip(original.as_list(), converted.as_list()):
            self.assertAlmostEqual(expected, actual, places=10)

    def test_crop_rasterization_rounds_outward(self):
        self.assertEqual(rasterize_box(Box(1.2, 2.8, 9.1, 10.01), (20, 20)), Box(1, 2, 10, 11))

    def test_invalid_boxes_and_scale_are_rejected(self):
        with self.assertRaises(ValueError):
            Box(2, 2, 2, 3)
        with self.assertRaises(ValueError):
            expand_box(Box(1, 1, 3, 3), (10, 10), 0.9)
        with self.assertRaises(ValueError):
            yolo_box_to_pixel((0.5, 0.5, -0.1, 0.2), (10, 10))


if __name__ == "__main__":
    unittest.main()
