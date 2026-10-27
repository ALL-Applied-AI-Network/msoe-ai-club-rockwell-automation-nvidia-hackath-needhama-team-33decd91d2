import unittest

import numpy as np

from src.remediation.controller import ContactTool, RemediationController
from src.remediation.geometry import base_to_uf0


class RemediationControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = RemediationController(ContactTool(0.0, 0.0, 0.0005, validated=True))
        self.tcp = [*base_to_uf0([0.30, 0.0, 0.30]), -180.0, 0.0, 0.0]
        self.obs = {"tcp_pose": self.tcp, "joints": [0.0] * 7}

    @staticmethod
    def detection(class_name, xyz, *, qualified=True, depth=True, confidence=0.95):
        return {"class_name": class_name, "base_pos_m": list(xyz),
                "confidence": confidence, "contact_qualified": qualified,
                "depth_available": depth}

    def test_unqualified_detection_holds(self):
        action = self.controller.act(
            self.obs, [self.detection("crooked_cap", [0.31, 0.02, 0.01], qualified=False)])
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertEqual(self.controller.state, "inspect")

    def test_depthless_detection_holds(self):
        action = self.controller.act(
            self.obs, [self.detection("crooked_cap", [0.31, 0.02, 0.01], depth=False)])
        np.testing.assert_allclose(action[:6], self.tcp)

    def test_qualified_crooked_cap_builds_bounded_route(self):
        action = self.controller.act(
            self.obs, [self.detection("crooked_cap", [0.31, 0.02, 0.01])])
        self.assertEqual(self.controller.state, "repair")
        self.assertEqual(len(action), 7)
        self.assertTrue(all(np.isfinite(action)))
        self.assertGreaterEqual(action[0], 150.0)
        self.assertLessEqual(action[0], 650.0)

    def test_missing_cap_requires_spare(self):
        action = self.controller.act(
            self.obs, [self.detection("missing_cap", [0.31, 0.02, 0.01])])
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertEqual(self.controller.state, "inspect")

    def test_missing_cap_with_spare_starts_pickup(self):
        detections = [self.detection("missing_cap", [0.31, 0.02, 0.01]),
                      self.detection("table_cap", [0.34, -0.20, 0.02])]
        action = self.controller.act(self.obs, detections)
        self.assertEqual(self.controller.state, "pickup")
        self.assertEqual(len(action), 7)

    def test_unaccepted_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            RemediationController(ContactTool(0.0, 0.0, 0.0, validated=False))


if __name__ == "__main__":
    unittest.main()
