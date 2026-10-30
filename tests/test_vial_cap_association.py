import unittest

from src.inspection.defect_training.boxes import Box
from src.inspection.defect_training.vial_cap_association import (
    AssociationConfig,
    associate_vials_and_caps,
    inferred_top_box,
)


class VialCapAssociationTests(unittest.TestCase):
    def test_matches_caps_one_to_one_and_infers_unmatched_vial(self):
        vials = [Box(10, 20, 30, 80), Box(40, 20, 60, 80)]
        caps = [Box(11, 17, 29, 31)]
        result = associate_vials_and_caps(vials, caps, (100, 100))
        self.assertEqual([(item["vial_index"], item["cap_index"]) for item in result["matches"]], [(0, 0)])
        self.assertEqual(result["unmatched_vial_indices"], [1])
        self.assertEqual(result["unmatched_cap_indices"], [])
        self.assertEqual(result["inferred_missing"][0]["vial_index"], 1)

    def test_unmatched_cap_is_not_forced_to_a_vial(self):
        result = associate_vials_and_caps(
            [Box(10, 20, 30, 80)], [Box(70, 10, 90, 30)], (100, 100)
        )
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["unmatched_cap_indices"], [0])
        self.assertEqual(len(result["inferred_missing"]), 1)

    def test_inferred_top_box_clamps_at_image_boundary(self):
        box = inferred_top_box(Box(0, 0, 20, 80), (100, 100), 1.2)
        self.assertEqual(box.x1, 0)
        self.assertGreater(box.width, 20)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            AssociationConfig(upper_fraction=0)


if __name__ == "__main__":
    unittest.main()
