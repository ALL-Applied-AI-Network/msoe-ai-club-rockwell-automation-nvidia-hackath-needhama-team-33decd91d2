import unittest

import numpy as np

from src.remediation.cone_reach import ConeReachSelector
from src.remediation.wrist_refinement import FINGERTIP_TCP_J6_M


class ConeReachSelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = ConeReachSelector(
            FINGERTIP_TCP_J6_M,
            (0.0, -0.0369851212, -0.0098332793),
            vial_rim_z_m=0.2081,
            cone_press_below_rim_m=0.003,
        )

    def test_nominal_45_degree_lean_is_used_when_reachable(self):
        self.assertEqual(self.selector((0.16, 0.0, 0.21)),
                         (135.0, 0.0, -90.0))

    def test_heading_changes_before_lean_for_continuous_far_target(self):
        # This is an ordinary base-frame XYZ, not a tray/G-slot identifier.
        # Preserve the 45-degree cone geometry and search heading instead.
        result = self.selector((0.38435, 0.03822, 0.21))
        self.assertEqual(result[0], 135.0)
        self.assertEqual(result[1], 0.0)

    def test_measured_joint_seed_is_optional(self):
        result = self.selector(
            np.array((0.16, 0.0, 0.21)),
            joints_deg=(46.2, 12.7, 3.3, 0.0, -80.6, -136.2, 0.0),
        )
        self.assertEqual(result, (135.0, 0.0, -90.0))


if __name__ == "__main__":
    unittest.main()
