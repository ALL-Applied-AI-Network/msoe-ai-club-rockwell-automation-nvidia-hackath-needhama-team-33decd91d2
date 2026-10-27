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
        self.assertEqual(self.controller.state, "crooked_observe_axis")
        self.assertEqual(len(action), 7)
        self.assertTrue(all(np.isfinite(action)))
        self.assertGreaterEqual(action[0], 150.0)
        self.assertLessEqual(action[0], 650.0)

    def test_vertical_axis_has_standard_down_orientation(self):
        np.testing.assert_allclose(axis_to_wpr([0, 0, 1]), [-180, 0, 0], atol=1e-8)
        np.testing.assert_allclose(wpr_to_rotation(axis_to_wpr([0, 0, 1]))[:, 2],
                                   [0, 0, -1], atol=1e-8)

    def test_axis_orientation_round_trip(self):
        axis = np.array([0.4, -0.2, 0.89442719])
        axis /= np.linalg.norm(axis)
        np.testing.assert_allclose(wpr_to_rotation(axis_to_wpr(axis))[:, 2], -axis, atol=1e-8)

    def test_crooked_requires_camera_axis_and_table_spot(self):
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.01])
        self.controller.act(self.obs, [detection])
        while self.controller.queue:
            self.controller.act(self.obs, [detection])
        held = self.controller.act(self.obs, [detection])
        np.testing.assert_allclose(held[:6], self.tcp)
        self.assertEqual(self.controller.state, "crooked_observe_axis")

    def test_crooked_camera_observation_starts_remove_sequence(self):
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.01])
        self.controller.act(self.obs, [detection])
        while self.controller.queue:
            self.controller.act(self.obs, [detection])
        observed = {**self.obs, "crooked_axis_base": [0.4, 0.0, 0.9165],
                    "table_regrasp_base_pos": [0.34, -0.20, 0.02]}
        action = self.controller.act(observed, [detection])
        self.assertEqual(self.controller.state, "crooked_verify_table")
        self.assertEqual(len(action), 7)

    def test_crooked_full_sequence_regrips_replaces_and_presses(self):
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.01])
        self.controller.act(self.obs, [detection])
        while self.controller.queue:
            self.controller.act(self.obs, [detection])
        camera_obs = {**self.obs, "crooked_axis_base": [0.4, 0.0, 0.9165],
                      "table_regrasp_base_pos": [0.34, -0.20, 0.02]}
        remove_actions = [self.controller.act(camera_obs, [detection])]
        while self.controller.queue:
            remove_actions.append(self.controller.act(camera_obs, [detection]))
        self.assertEqual(self.controller.state, "crooked_verify_table")
        self.assertTrue(any(abs(action[4]) > 1.0 or abs(action[5]) > 1.0
                            for action in remove_actions))
        replace_obs = {**self.obs, "upright_cap_base_pos": [0.341, -0.199, 0.02]}
        replace_actions = [self.controller.act(replace_obs, [detection])]
        while self.controller.queue:
            replace_actions.append(self.controller.act(replace_obs, [detection]))
        self.assertEqual(self.controller.state, "repair")
        gripper = [action[6] for action in replace_actions]
        self.assertIn(1.0, gripper)
        self.assertIn(0.0, gripper)
        # Regrasp, release and closed-fingertip press require a substantial route.
        self.assertGreaterEqual(len(replace_actions), 10)

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

    def test_verified_pickup_records_camera_observed_free_spot(self):
        missing = self.detection("missing_cap", [0.31, 0.02, 0.01])
        spare = self.detection("table_cap", [0.34, -0.20, 0.02])
        self.controller.act(self.obs, [missing, spare])
        while self.controller.queue:
            self.controller.act(self.obs, [missing, spare])
        self.controller.act({**self.obs, "grasp_verified": True}, [missing])
        self.assertEqual(self.controller.camera_verified_free_table_spots,
                         [[0.34, -0.20, 0.02]])

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
