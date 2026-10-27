import unittest

import numpy as np

from src.remediation.controller import (ContactTool, RemediationController, axis_to_wpr,
                                        policy_observation, wpr_to_rotation)
from src.remediation.geometry import base_to_uf0


class RemediationControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = RemediationController(ContactTool(0.0, 0.0, 0.0005, validated=True))
        self.tcp = [*base_to_uf0([0.30, 0.0, 0.30]), -180.0, 0.0, 0.0]
        self.obs = {"tcp_pose": self.tcp, "joints": [0.0] * 7}

    @staticmethod
    def detection(class_name, xyz, *, qualified=True, depth=True, confidence=0.95):
        return {"class_name": class_name, "base_pos_m": list(xyz),
                "surface_top_base_z_m": float(xyz[2]),
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
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.21])
        actions = [self.controller.act(self.obs, [detection])]
        while self.controller.queue:
            actions.append(self.controller.act(self.obs, [detection]))
        self.assertEqual(self.controller.state, "repair")
        self.assertTrue(all(len(action) == 7 for action in actions))
        self.assertTrue(all(np.isfinite(action).all() for action in map(np.asarray, actions)))
        self.assertTrue(all(150.0 <= action[0] <= 650.0 for action in actions))
        self.assertTrue(all(np.allclose(wpr_to_rotation(action[3:6])[:, 2], [0, 0, -1])
                            for action in actions))

    def test_vertical_axis_has_standard_down_orientation(self):
        np.testing.assert_allclose(axis_to_wpr([0, 0, 1]), [-180, 0, 0], atol=1e-8)
        np.testing.assert_allclose(wpr_to_rotation(axis_to_wpr([0, 0, 1]))[:, 2],
                                   [0, 0, -1], atol=1e-8)

    def test_axis_orientation_round_trip(self):
        axis = np.array([0.4, -0.2, 0.89442719])
        axis /= np.linalg.norm(axis)
        np.testing.assert_allclose(wpr_to_rotation(axis_to_wpr(axis))[:, 2], -axis, atol=1e-8)

    def test_crooked_open_claw_descends_to_internal_shoulder_and_retreats(self):
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.21])
        actions = [self.controller.act(self.obs, [detection])]
        while self.controller.queue:
            actions.append(self.controller.act(self.obs, [detection]))
        self.assertEqual(self.controller.state, "repair")
        self.assertTrue(all(action[6] == 0.0 for action in actions))
        self.assertTrue(all(action[3:6] == [-180.0, 0.0, 0.0] for action in actions))
        target_uf0_z = base_to_uf0([0.31, 0.02, 0.21])[2]
        self.assertAlmostEqual(min(action[2] for action in actions),
                               target_uf0_z - 32.6838 - 5.0 + 0.5, places=4)

    def test_crooked_shoulder_depth_limits_are_rejected(self):
        tool = ContactTool(0.0, 0.0, 0.0, validated=True)
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_shoulder_depth_m=0.019)
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_shoulder_depth_m=0.041)
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_tap_depth_m=0.0)
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_tap_depth_m=0.011)

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

    def test_verified_pickup_starts_place_and_press_route(self):
        missing = self.detection("missing_cap", [0.31, 0.02, 0.01])
        spare = self.detection("table_cap", [0.34, -0.20, 0.02])
        self.controller.act(self.obs, [missing, spare])
        while self.controller.queue:
            self.controller.act(self.obs, [missing, spare])
        action = self.controller.act({**self.obs, "grasp_verified": True}, [missing])
        self.assertEqual(self.controller.state, "repair")
        self.assertEqual(len(action), 7)

    def test_unaccepted_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            RemediationController(ContactTool(0.0, 0.0, 0.0, validated=False))

    def test_oracle_observation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "oracle"):
            self.controller.act({**self.obs, "true_axis": [0, 0, 1]}, [])

    def test_oracle_detection_is_rejected(self):
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.01])
        detection["prim"] = "/World/secret_cap"
        with self.assertRaisesRegex(ValueError, "oracle"):
            self.controller.act(self.obs, [detection])

    def test_adapter_metadata_is_stripped_and_audited(self):
        obs, detections = policy_observation(
            {**self.obs, "adapter_debug": "not policy data"},
            [{**self.detection("crooked_cap", [0.31, 0.02, 0.01]), "adapter_debug": 3}],
        )
        self.assertNotIn("adapter_debug", obs)
        self.assertNotIn("adapter_debug", detections[0])
        self.controller.act(obs, detections)
        record = self.controller.decision_records[-1]
        self.assertEqual(record["oracle_inputs_used"], [])
        self.assertNotIn("adapter_debug", record["observation_fields"])


if __name__ == "__main__":
    unittest.main()
