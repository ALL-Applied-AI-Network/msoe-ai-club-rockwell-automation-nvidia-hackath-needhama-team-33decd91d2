import unittest

from src.inspection.locateanything_runtime import (LocateAnythingRuntime,
                                                   WRIST_ROI_SIZE_BY_CLASS,
                                                   box_from_roi, target_roi)


class LocateAnythingRuntimeTests(unittest.TestCase):
    def test_missing_vial_uses_tight_target_isolation_crop(self):
        self.assertEqual(WRIST_ROI_SIZE_BY_CLASS["missing_cap"], 208)
        self.assertEqual(WRIST_ROI_SIZE_BY_CLASS["table_cap"], 640)

    def test_target_roi_centres_interior_target(self):
        self.assertEqual(target_roi((1920, 1080), (960, 540)),
                         (640, 220, 1280, 860))

    def test_target_roi_clamps_at_image_edge(self):
        self.assertEqual(target_roi((1920, 1080), (20, 30)),
                         (0, 0, 640, 640))
        self.assertEqual(target_roi((1920, 1080), (1900, 1060)),
                         (1280, 440, 1920, 1080))

    def test_roi_box_maps_back_to_calibrated_full_image(self):
        self.assertEqual(box_from_roi([100, 120, 180, 200], [640, 220, 1280, 860]),
                         [740.0, 340.0, 820.0, 420.0])

    def test_outside_expected_pixel_is_rejected(self):
        with self.assertRaises(ValueError):
            target_roi((1920, 1080), (1920, 540))

    def test_missing_vial_prefers_full_opening_over_partial_rim_near_bad_prior(self):
        full_opening = [17.0, 118.0, 106.0, 208.0]
        partial_rim = [56.0, 102.0, 122.0, 167.0]
        selected = LocateAnythingRuntime._select_box(
            [full_opening, partial_rim], [104.0, 104.0], "missing_cap")
        self.assertEqual(selected, full_opening)

    def test_other_classes_still_choose_box_nearest_expected_pixel(self):
        near = [90.0, 90.0, 110.0, 110.0]
        large_far = [0.0, 0.0, 80.0, 80.0]
        selected = LocateAnythingRuntime._select_box(
            [large_far, near], [100.0, 100.0], "table_cap")
        self.assertEqual(selected, near)


if __name__ == "__main__":
    unittest.main()
