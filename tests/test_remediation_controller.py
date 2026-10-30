import unittest

import numpy as np

from src.remediation.controller import (DEFAULT_CONE_PRESS_WPR_DEG,
                                        DEFAULT_CONE_TIP_FROM_TCP_M, ContactTool,
                                        DEFAULT_CONE_WRIST_WPR_DEG,
                                        RemediationController, axis_to_wpr,
                                        command_rotation_to_j6,
                                        j6_rotation_to_command, policy_observation,
                                        wpr_to_rotation)
from src.remediation.geometry import (base_to_uf0, inspection_base_position,
                                      rim_base_z, uf0_to_base)


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
                "depth_available": depth,
                "localization_note": "RGB box-center ray intersected with the calibrated object plane"}

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

    def test_command_and_j6_rotation_mapping_round_trip(self):
        command = wpr_to_rotation((-147.0, 31.0, -83.0))
        j6 = command_rotation_to_j6(command)
        np.testing.assert_allclose(j6_rotation_to_command(j6), command, atol=1e-12)
        np.testing.assert_allclose(j6.T @ j6, np.eye(3), atol=1e-12)

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
        low = min(action[2] for action in actions)
        low_actions = [action for action in actions if abs(action[2] - low) < 1e-6]
        self.assertGreater(max(action[0] for action in low_actions)
                           - min(action[0] for action in low_actions), 1.49)
        np.testing.assert_allclose(actions[-1][:3], base_to_uf0(inspection_base_position()))
        self.assertEqual(actions[-1][6], 0.0)

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
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_action="unknown")
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_wiggle_radius_m=0.0021)
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_wiggle_cycles=4)
        with self.assertRaises(ValueError):
            RemediationController(tool, pickup_wiggle_radius_m=0.0051)
        with self.assertRaises(ValueError):
            RemediationController(tool, pickup_wiggle_cycles=3)
        with self.assertRaises(ValueError):
            RemediationController(tool, table_cap_xy_offset_m=(0.026, 0.0))
        with self.assertRaises(ValueError):
            RemediationController(tool, missing_cap_xy_offset_m=(0.0,))
        with self.assertRaises(ValueError):
            RemediationController(tool, crooked_cap_xy_offset_m=(0.0, np.nan))
        with self.assertRaises(ValueError):
            RemediationController(tool, wrist_view_distance_m=0.10)
        with self.assertRaises(ValueError):
            RemediationController(tool, wrist_crooked_view_distance_m=0.10)
        with self.assertRaises(ValueError):
            RemediationController(tool, missing_release_side_step_m=-0.001)
        with self.assertRaises(ValueError):
            RemediationController(tool, missing_release_side_step_m=0.011)
        with self.assertRaises(ValueError):
            RemediationController(tool, missing_release_exit_step_m=-0.001)
        with self.assertRaises(ValueError):
            RemediationController(tool, missing_release_exit_step_m=0.016)

    def test_wrist_refinement_stages_spare_then_corrects_before_pickup(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            wrist_refinement=True)
        missing = self.detection("missing_cap", [0.31, 0.02, rim_base_z()])
        spare = self.detection("table_cap", [0.34, -0.20, 0.006], depth=False)
        controller.act(self.obs, [missing, spare])
        while controller.queue:
            controller.act(self.obs, [])
        self.assertEqual(controller.state, "refine_spare")

        coarse_actions = [controller.act(self.obs, [])]
        while controller.queue:
            coarse_actions.append(controller.act(self.obs, []))
        self.assertEqual(controller.state, "await_refine_spare")
        np.testing.assert_allclose(
            coarse_actions[-1][:3], base_to_uf0([0.34, -0.20, 0.031]), atol=1e-9)
        self.assertGreater(coarse_actions[0][2], coarse_actions[-1][2])

        refined = [0.321, -0.184, 0.006]
        correction = controller.act({**self.obs, "wrist_refinement": {
            "class_name": "table_cap", "base_pos_m": refined}}, [])
        self.assertEqual(controller.state, "pickup")
        np.testing.assert_allclose(
            correction[:3], base_to_uf0([refined[0], refined[1], 0.031]), atol=1e-9)
        # No old TableCam empirical offset is added after wrist localization.
        queued = list(controller.queue)
        pick_actions = [action for action in queued if action[6] == 0.0]
        self.assertTrue(any(np.allclose(action[:2], np.array(refined[:2]) * 1000)
                            for action in pick_actions))

    def test_centred_wrist_view_clamps_only_camera_staging_pose_at_workspace_edge(self):
        from src.remediation.wrist_refinement import fingertip_tcp_from_joints

        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            wrist_refinement=True)
        # The physical camera hand-eye translation puts the ideal centred
        # fingertip TCP about 18 mm past the near-X workspace boundary for
        # this otherwise reachable vial.  The wrist view may be off-centre;
        # the subsequent perception-derived contact target may not be clipped.
        target = np.array([0.228, 0.157, rim_base_z()])
        # Measured inspection-pose joints from the two-missing/three-crooked
        # regression run which exposed the edge case.  Use their matching FK
        # TCP so the controller accepts the measured-J6 camera transform.
        joints = np.array([0.80588, 0.22214, 0.05845, -0.00011,
                           -1.40647, -0.80590, 0.0])
        tcp_base = fingertip_tcp_from_joints(joints)
        obs = {"joints": joints,
               "tcp_pose": [*base_to_uf0(tcp_base), -180.0, 0.0, 0.0]}
        view = controller._wrist_view(
            target, obs, centre_camera=True, use_measured_j6=True)
        self.assertAlmostEqual(view[0], 0.155)
        self.assertGreaterEqual(view[1], -0.445)
        self.assertLessEqual(view[1], 0.445)

        # The original target is unchanged and remains the point refined by
        # the image-space correction rather than a fabricated slot location.
        np.testing.assert_allclose(target, [0.228, 0.157, rim_base_z()])

    def test_missing_wrist_workflow_stays_straight_until_crooked_phase(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", wrist_refinement=True)
        missing = self.detection("missing_cap", [0.31, 0.02, rim_base_z()])
        spare = self.detection("table_cap", [0.34, -0.20, 0.006], depth=False)
        crooked = self.detection("crooked_cap", [0.38, 0.04, rim_base_z() + 0.01])
        actions = [controller.act(self.obs, [missing, spare, crooked])]
        while controller.queue:
            actions.append(controller.act(self.obs, []))
        actions.append(controller.act(self.obs, []))
        while controller.queue:
            actions.append(controller.act(self.obs, []))
        actions.append(controller.act({**self.obs, "wrist_refinement": {
            "class_name": "table_cap", "base_pos_m": spare["base_pos_m"]}}, []))
        while controller.queue:
            actions.append(controller.act(self.obs, []))
        actions.append(controller.act({**self.obs, "grasp_verified": True}, []))
        while controller.queue:
            actions.append(controller.act(self.obs, []))
        actions.append(controller.act(self.obs, []))
        while controller.queue:
            actions.append(controller.act(self.obs, []))
        actions.append(controller.act({**self.obs, "wrist_refinement": {
            "class_name": "missing_cap", "base_pos_m": missing["base_pos_m"]}}, []))
        while controller.queue:
            actions.append(controller.act(self.obs, []))

        self.assertFalse(controller.cone_oriented)
        self.assertTrue(all(action[3:6] == [-180.0, 0.0, 0.0]
                            for action in actions))
        self.assertFalse(any(np.allclose(action[3:6], DEFAULT_CONE_WRIST_WPR_DEG)
                             for action in actions))
        self.assertFalse(any(np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
                             for action in actions))

    def test_wrist_refinement_accepts_correction_above_legacy_40_mm_limit(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            wrist_refinement=True)
        controller.pending = {
            "target": self.detection("missing_cap", [0.31, 0.02, rim_base_z()]),
            "spare": self.detection("table_cap", [0.34, -0.20, 0.006], depth=False),
            "height": 0.30,
        }
        controller.state = "await_refine_spare"
        action = controller.act({**self.obs, "wrist_refinement": {
            "class_name": "table_cap", "base_pos_m": [0.40, -0.20, 0.006]}}, [])
        self.assertEqual(len(action), 7)

    def test_wrist_crooked_refinement_orients_cone_once_while_waiting(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", wrist_refinement=True)
        crooked = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        crooked["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        last_camera_action = controller.act(self.obs, [crooked])
        while controller.queue:
            last_camera_action = controller.act(self.obs, [])
        self.assertEqual(controller.state, "refine_crooked")
        camera_obs = {**self.obs, "tcp_pose": last_camera_action[:6]}
        first_orientation = controller.act(camera_obs, [])
        self.assertTrue(controller.cone_oriented)
        self.assertEqual(controller.state, "await_refine_crooked")
        self.assertEqual(first_orientation[6], 1.0)
        staged = [first_orientation, *list(controller.queue)]
        current = uf0_to_base(last_camera_action[:3])
        np.testing.assert_allclose(
            uf0_to_base(first_orientation[:3]),
            [current[0], current[1], controller.pending["height"]])
        inspection = inspection_base_position()
        np.testing.assert_allclose(
            uf0_to_base(staged[1][:3]),
            [inspection[0], inspection[1], controller.pending["height"]])
        self.assertEqual(sum(np.allclose(action[3:6], DEFAULT_CONE_WRIST_WPR_DEG)
                             for action in staged), 1)
        self.assertEqual(sum(np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
                             for action in staged), 1)

    def test_closed_tip_action_closes_only_above_cap_and_reopens_above_cap(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="closed-tip")
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.21])
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))
        closed = [action for action in actions if action[6] == 1.0]
        self.assertGreaterEqual(len(closed), 3)
        self.assertEqual(actions[-1][6], 0.0)
        self.assertAlmostEqual(min(action[2] for action in closed),
                               base_to_uf0([0.31, 0.02, 0.21])[2] - 5.0 + 0.5,
                               places=4)

    def test_shoulder_close_closes_clear_presses_opens_then_retreats(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="shoulder-close", crooked_wiggle_cycles=0)
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.21])
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))
        jaw = [action[6] for action in actions]
        first_closed = jaw.index(1.0)
        last_closed = len(jaw) - 1 - jaw[::-1].index(1.0)
        self.assertEqual(jaw[first_closed - 1], 0.0)
        self.assertEqual(jaw[last_closed + 1], 0.0)
        self.assertTrue(all(value == 1.0 for value in jaw[first_closed:last_closed + 1]))
        contact_z_mm = base_to_uf0([0.31, 0.02, 0.21 - 0.0326838])[2]
        self.assertAlmostEqual(actions[first_closed][2], contact_z_mm + 3.5, places=4)
        self.assertLess(min(action[2] for action in actions[first_closed:last_closed + 1]),
                        contact_z_mm)
        self.assertGreater(actions[last_closed + 2][2], actions[last_closed + 1][2])

    def test_regrip_action_grips_lifts_releases_and_taps(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="regrip")
        axis = np.array([0.30, -0.10, 0.9486833])
        detection = {**self.detection("crooked_cap", [0.31, 0.02, 0.21]),
                     "cap_axis_base": axis.tolist()}
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))
        jaw = [action[6] for action in actions]
        self.assertIn([0.0, 1.0, 1.0], [jaw[i:i + 3] for i in range(len(jaw) - 2)])
        self.assertGreaterEqual(sum(value == 1.0 for value in jaw), 5)
        self.assertEqual(jaw[-1], 0.0)
        tilted = [action for action in actions if not np.allclose(action[3:6], [-180, 0, 0])]
        self.assertGreaterEqual(len(tilted), 4)
        np.testing.assert_allclose(
            wpr_to_rotation(tilted[0][3:6])[:, 2], -axis / np.linalg.norm(axis), atol=1e-7)
        first_tilted = next(i for i, action in enumerate(actions)
                            if not np.allclose(action[3:6], [-180, 0, 0]))
        self.assertAlmostEqual(actions[first_tilted][2], actions[first_tilted - 1][2])
        self.assertAlmostEqual(actions[first_tilted][0], actions[first_tilted - 1][0])
        self.assertAlmostEqual(actions[first_tilted][1], actions[first_tilted - 1][1])

    def test_regrip_rejects_missing_camera_axis(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True), crooked_action="regrip")
        action = controller.act(self.obs, [self.detection("crooked_cap", [0.31, 0.02, 0.21])])
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertTrue(any("RGB-D cap axis" in event.get("reason", "")
                            for event in controller.events))

    def test_cone_press_uses_cad_nose_at_45_degree_tilt_and_90_degree_roll(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press")
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.21])
        detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))

        tilted = [action for action in actions
                  if np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)]
        self.assertGreaterEqual(len(tilted), 6)
        self.assertTrue(all(action[6] == 1.0 for action in tilted))

        # The selected outside cone is 45 degrees from J6 +X. The reachable
        # backward W lean lowers the cone-side jaw without pitching the wrist
        # camera/cables forward through the work area, and keeps the negative-Y
        # (TableCam-left) jaw low during a press.
        command_rotation = wpr_to_rotation(DEFAULT_CONE_PRESS_WPR_DEG)
        self.assertEqual(DEFAULT_CONE_PRESS_WPR_DEG, (135.0, 0.0, -90.0))
        j6_rotation = command_rotation_to_j6(command_rotation)
        cone_axis_j6 = np.array([-np.sqrt(0.5), np.sqrt(0.5), 0.0])
        cone_axis_base = j6_rotation @ cone_axis_j6
        self.assertGreater(cone_axis_base[2], np.cos(np.radians(0.1)))
        self.assertLess(np.linalg.norm(cone_axis_base[:2]), np.sin(np.radians(0.1)))
        self.assertGreater(j6_rotation[2, 1], 0.0)

        nose_positions = []
        for action in tilted:
            tcp_base = uf0_to_base(action[:3])
            nose_positions.append(tcp_base + command_rotation @
                                  np.asarray(DEFAULT_CONE_TIP_FROM_TCP_M))
        nose_positions = np.asarray(nose_positions)
        expected_xy = np.asarray(detection["base_pos_m"][:2])
        np.testing.assert_allclose(nose_positions[np.argmin(nose_positions[:, 2]), :2],
                                   expected_xy, atol=1e-10)
        expected_contact_z = rim_base_z() - 0.003 + 0.0005
        self.assertAlmostEqual(nose_positions[:, 2].min(), expected_contact_z,
                               places=10)
        self.assertEqual(actions[-1][6], 1.0)
        np.testing.assert_allclose(actions[-1][3:6], DEFAULT_CONE_PRESS_WPR_DEG,
                                   atol=1e-10)

    def test_cone_press_centres_broad_cone_on_observed_cap(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press")
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        detection["feature_base_pos_m"] = [0.317, 0.017, 0.23]
        detection["cap_axis_base"] = [-np.sin(np.radians(35.0)), 0.0,
                                      np.cos(np.radians(35.0))]
        detection["cap_high_point_base_m"] = [0.318, 0.024, 0.235]
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))
        rotation = wpr_to_rotation(DEFAULT_CONE_PRESS_WPR_DEG)
        noses = np.asarray([
            uf0_to_base(action[:3]) + rotation @ np.asarray(DEFAULT_CONE_TIP_FROM_TCP_M)
            for action in actions if np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
        ])
        expected_xy = np.asarray(detection["feature_base_pos_m"][:2])
        self.assertTrue(any(np.allclose(nose[:2], expected_xy)
                            and abs(nose[2] - 0.240) < 1e-9 for nose in noses))
        lowest = noses[np.argmin(noses[:, 2])]
        np.testing.assert_allclose(lowest[:2], expected_xy, atol=1e-10)
        self.assertAlmostEqual(lowest[2], rim_base_z() - 0.003 + 0.0005, places=10)

    def test_cone_press_applies_base_frame_xy_calibration(self):
        offset = np.array([0.004, -0.003])
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", crooked_cap_xy_offset_m=offset)
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        detection["feature_base_pos_m"] = [0.317, 0.017, 0.23]
        detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))
        rotation = wpr_to_rotation(DEFAULT_CONE_PRESS_WPR_DEG)
        noses = np.asarray([
            uf0_to_base(action[:3]) + rotation @ np.asarray(DEFAULT_CONE_TIP_FROM_TCP_M)
            for action in actions if np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
        ])
        lowest = noses[np.argmin(noses[:, 2])]
        np.testing.assert_allclose(
            lowest[:2], np.asarray(detection["feature_base_pos_m"][:2]) + offset,
            atol=1e-10)

    def test_cone_lean_selector_receives_continuous_feature_coordinate(self):
        requested = []

        def select_wpr(target_base_m, joints_deg):
            requested.append((np.asarray(target_base_m, dtype=float), joints_deg))
            return (135.0, 0.0, -90.0)

        offset = np.array([0.002, -0.001])
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", crooked_cap_xy_offset_m=offset,
            cone_wpr_selector=select_wpr)
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        detection["feature_base_pos_m"] = [0.317, 0.017, 0.23]
        detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))

        self.assertEqual(len(requested), 1)
        np.testing.assert_allclose(
            requested[0][0], [0.319, 0.016, 0.23], atol=1e-12)
        self.assertIs(requested[0][1], self.obs["joints"])
        self.assertTrue(any(np.allclose(action[3:6], [135.0, 0.0, -90.0])
                            for action in actions))
        self.assertEqual(controller.active_cone_press_wpr_deg,
                         (135.0, 0.0, -90.0))

    def test_cone_press_xy_calibration_is_not_doubled_without_feature_center(self):
        offset = np.array([0.004, -0.003])
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", crooked_cap_xy_offset_m=offset)
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))
        rotation = wpr_to_rotation(DEFAULT_CONE_PRESS_WPR_DEG)
        noses = np.asarray([
            uf0_to_base(action[:3]) + rotation @ np.asarray(DEFAULT_CONE_TIP_FROM_TCP_M)
            for action in actions if np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
        ])
        lowest = noses[np.argmin(noses[:, 2])]
        np.testing.assert_allclose(
            lowest[:2], np.asarray(detection["base_pos_m"][:2]) + offset,
            atol=1e-10)

    def test_cone_press_depth_is_tunable_above_or_below_rim(self):
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        for depth in (-0.005, 0.006):
            controller = RemediationController(
                ContactTool(0.0, 0.0, 0.0, validated=True),
                crooked_action="cone-press", cone_press_below_rim_m=depth)
            actions = [controller.act(self.obs, [detection])]
            while controller.queue:
                actions.append(controller.act(self.obs, [detection]))
            rotation = wpr_to_rotation(DEFAULT_CONE_PRESS_WPR_DEG)
            noses = np.asarray([
                uf0_to_base(action[:3]) + rotation @ np.asarray(DEFAULT_CONE_TIP_FROM_TCP_M)
                for action in actions if np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
            ])
            self.assertAlmostEqual(noses[:, 2].min(), rim_base_z() - depth, places=10)

    def test_cone_press_hands_off_to_cap_center_below_32_degrees(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press")
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        detection["feature_base_pos_m"] = [0.317, 0.017, 0.23]
        detection["cap_axis_base"] = [np.sin(np.radians(20.0)), 0.0,
                                      np.cos(np.radians(20.0))]
        detection["cap_high_point_base_m"] = [0.318, 0.024, 0.235]
        actions = [controller.act(self.obs, [detection])]
        while controller.queue:
            actions.append(controller.act(self.obs, [detection]))
        rotation = wpr_to_rotation(DEFAULT_CONE_PRESS_WPR_DEG)
        noses = np.asarray([
            uf0_to_base(action[:3]) + rotation @ np.asarray(DEFAULT_CONE_TIP_FROM_TCP_M)
            for action in actions if np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
        ])
        expected_xy = np.asarray(detection["feature_base_pos_m"][:2])
        lowest = noses[np.argmin(noses[:, 2])]
        np.testing.assert_allclose(lowest[:2], expected_xy, atol=1e-10)
        self.assertAlmostEqual(lowest[2], rim_base_z() - 0.003 + 0.0005, places=10)

    def test_cone_batch_orients_once_and_keeps_posture_between_targets(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press")
        first = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        first["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        controller._route(self.obs, first)
        first_route = list(controller.queue)
        controller.queue.clear()

        second = self.detection("crooked_cap", [0.35, 0.02, 0.23])
        second["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        controller._route(self.obs, second)
        second_route = list(controller.queue)

        self.assertTrue(controller.cone_oriented)
        home_down = [-180.0, 0.0, 0.0]
        self.assertEqual(sum(np.allclose(action[3:6], home_down)
                             for action in first_route), 1)
        self.assertEqual(sum(np.allclose(action[3:6], DEFAULT_CONE_WRIST_WPR_DEG)
                             for action in first_route), 1)
        np.testing.assert_allclose(first_route[0][3:6], home_down)
        np.testing.assert_allclose(first_route[1][3:6], DEFAULT_CONE_WRIST_WPR_DEG)
        self.assertTrue(all(np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
                            for action in first_route[2:]))
        self.assertTrue(all(np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
                            for action in second_route))
        # Both routes clear TableCam at the inspection point but never change
        # the fixed cone posture while doing so.
        np.testing.assert_allclose(
            uf0_to_base(first_route[-1][:3]), inspection_base_position())
        np.testing.assert_allclose(
            uf0_to_base(second_route[-1][:3]), inspection_base_position())

    def test_cone_batch_retains_unattempted_camera_target_when_occluded(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press")
        first = self.detection("crooked_cap", [0.38, 0.04, 0.23])
        second = self.detection("crooked_cap", [0.14, 0.44, 0.23])
        for detection in (first, second):
            detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]

        initial = controller._tracked_defects([first, second])
        self.assertEqual(len(initial), 2)
        initial[0][0]["attempts"] = 1
        remembered = controller._tracked_defects([])

        self.assertEqual(len(remembered), 1)
        np.testing.assert_allclose(
            remembered[0][1]["base_pos_m"], second["base_pos_m"])
        self.assertNotIn("slot", remembered[0][1])

    def test_three_cone_presses_transfer_directly_then_verify_once(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press")
        crooked = [
            self.detection("crooked_cap", [0.28, 0.05, 0.23]),
            self.detection("crooked_cap", [0.33, 0.00, 0.23]),
            self.detection("crooked_cap", [0.38, -0.05, 0.23]),
        ]
        for detection in crooked:
            detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]

        routes = []
        current_obs = dict(self.obs)
        for index in range(3):
            action = controller.act(current_obs, crooked if index == 0 else [])
            route = [action]
            current_obs = {**current_obs, "tcp_pose": action[:6]}
            while controller.queue:
                action = controller.act(current_obs, [])
                route.append(action)
                current_obs = {**current_obs, "tcp_pose": action[:6]}
            routes.append(route)

        inspection = inspection_base_position()
        for route in routes[:2]:
            self.assertFalse(any(np.allclose(
                uf0_to_base(action[:3]), inspection) for action in route))
            self.assertTrue(all(np.allclose(
                action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
                for action in route[3:] if len(routes[0]) > 3))
        np.testing.assert_allclose(
            uf0_to_base(routes[-1][-1][:3]), inspection)
        self.assertEqual(controller.state, "repair_cone_batch")
        self.assertEqual(len(controller.cone_batch_targets), 3)
        self.assertEqual(sum(np.allclose(
            action[3:6], DEFAULT_CONE_WRIST_WPR_DEG)
            for route in routes for action in route), 1)

        good = [self.detection("good_cap", target, depth=True)
                for target in controller.cone_batch_targets]
        controller.act(current_obs, good)
        self.assertEqual(controller.state, "inspect")
        self.assertFalse(controller.cone_batch_targets)
        verification = [event for event in controller.events
                        if event["event"] == "cone_batch_verification"]
        self.assertEqual(len(verification), 1)
        self.assertTrue(verification[0]["all_fixed"])
        self.assertEqual(len(verification[0]["targets"]), 3)

    def test_next_cone_wrist_view_keeps_roll_and_removes_only_lean(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", wrist_refinement=True)
        controller.cone_oriented = True
        controller.active_cone_press_wpr_deg = DEFAULT_CONE_PRESS_WPR_DEG
        target = self.detection("crooked_cap", [0.14, 0.44, 0.23])
        target["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]

        controller._route(self.obs, target)
        route = list(controller.queue)

        self.assertEqual(len(route), 3)
        self.assertTrue(all(np.allclose(action[3:6], DEFAULT_CONE_WRIST_WPR_DEG)
                            for action in route))
        self.assertTrue(all(action[6] == 1.0 for action in route))
        self.assertEqual(controller.active_cone_press_wpr_deg,
                         DEFAULT_CONE_WRIST_WPR_DEG)

    def test_cached_crooked_batch_keeps_one_cone_orientation_between_targets(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", wrist_refinement=True)
        first = self.detection("crooked_cap", [0.31, 0.02, 0.23])
        second = self.detection("crooked_cap", [0.35, -0.04, 0.23])
        for detection in (first, second):
            detection["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]

        batch = [
            {"class_name": "crooked_cap", "base_pos_m": [0.312, 0.019, 0.23]},
            {"class_name": "crooked_cap", "base_pos_m": [0.347, -0.038, 0.23]},
        ]
        refined_first = controller._refinement(
            {"wrist_refinement_batch": batch}, "crooked_cap", first["base_pos_m"])
        np.testing.assert_allclose(refined_first, batch[0]["base_pos_m"])
        self.assertEqual(len(controller.crooked_refinement_cache), 1)

        controller.cone_oriented = True
        controller.active_cone_press_wpr_deg = DEFAULT_CONE_PRESS_WPR_DEG
        controller._route(self.obs, second)
        route = list(controller.queue)

        self.assertFalse(controller.crooked_refinement_cache)
        self.assertTrue(route)
        self.assertTrue(all(np.allclose(action[3:6], DEFAULT_CONE_PRESS_WPR_DEG)
                            for action in route))
        self.assertFalse(any(np.allclose(action[3:6], DEFAULT_CONE_WRIST_WPR_DEG)
                             for action in route))
        self.assertIn("cached wrist LocateAnything batch refinement",
                      controller.pending["target"]["localization_note"])

    def test_first_crooked_wrist_view_centres_perception_derived_batch(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", wrist_refinement=True)
        targets = [
            self.detection("crooked_cap", [0.273, 0.036, 0.229]),
            self.detection("crooked_cap", [0.328, -0.090, 0.230]),
            self.detection("crooked_cap", [0.226, -0.087, 0.228]),
        ]
        for target in targets:
            target["cap_axis_base"] = [0.0, -0.5, np.sqrt(0.75)]
        controller._tracked_defects(targets)

        controller._route(self.obs, targets[0])
        route = list(controller.queue)
        centroid = np.mean([target["base_pos_m"] for target in targets], axis=0)

        self.assertEqual(len(route), 3)
        np.testing.assert_allclose(
            uf0_to_base(route[1][:3])[:2], centroid[:2], atol=1e-12)
        np.testing.assert_allclose(
            uf0_to_base(route[2][:3])[:2], centroid[:2], atol=1e-12)

    def test_fixed_cone_press_does_not_require_unused_depth_axis(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press")
        action = controller.act(
            self.obs, [self.detection("crooked_cap", [0.31, 0.02, 0.21])])
        self.assertTrue(controller.queue)
        self.assertEqual(action[6], 1.0)
        self.assertFalse(any("cap axis" in event.get("reason", "")
                             for event in controller.events))

    def test_wrist_cone_route_accepts_depthless_coarse_candidate(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", wrist_refinement=True)
        detection = self.detection(
            "crooked_cap", [0.31, 0.02, 0.21], depth=False)
        detection["surface_top_base_z_m"] = None

        action = controller.act(self.obs, [detection])

        self.assertEqual(controller.state, "refine_crooked")
        self.assertTrue(controller.queue)
        self.assertEqual(action[6], 0.0)

    def test_wrist_cone_route_rejects_finite_robot_height_false_positive(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            crooked_action="cone-press", wrist_refinement=True)
        detection = self.detection(
            "crooked_cap", [0.31, 0.02, 0.21], depth=True)
        detection["surface_top_base_z_m"] = 0.425

        action = controller.act(self.obs, [detection])

        self.assertEqual(controller.state, "inspect")
        self.assertFalse(controller.queue)
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertTrue(any(
            "surface height outside physical range" in event.get("reason", "")
            for event in controller.events))

    def test_impossible_crooked_surface_height_is_rejected(self):
        detection = self.detection("crooked_cap", [0.31, 0.02, 0.35])
        action = self.controller.act(self.obs, [detection])
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertTrue(any(event.get("event") == "target_rejected"
                            for event in self.controller.events))

    def test_shifted_attempt_is_not_retried_as_a_new_cap(self):
        first = self.detection("crooked_cap", [0.31, 0.02, 0.21])
        self.controller.act(self.obs, [first])
        while self.controller.queue:
            self.controller.act(self.obs, [first])
        for _ in range(5):
            self.controller.act(self.obs, [first])
        self.assertEqual(self.controller.state, "inspect")
        shifted = self.detection("crooked_cap", [0.38, 0.08, 0.21])
        action = self.controller.act(self.obs, [shifted])
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertEqual(self.controller.state, "inspect")
        self.assertEqual(self.controller.defect_tracks[0]["attempts"], 1)

    def test_missing_cap_requires_spare(self):
        action = self.controller.act(
            self.obs, [self.detection("missing_cap", [0.31, 0.02, 0.01])])
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertEqual(self.controller.state, "inspect")

    def test_missing_cap_with_spare_starts_pickup(self):
        detections = [self.detection("missing_cap", [0.31, 0.02, 0.01]),
                      self.detection("table_cap", [0.34, -0.20, 0.02]),
                      self.detection("good_cap", [0.28, 0.01, 0.210132])]
        action = self.controller.act(self.obs, detections)
        self.assertEqual(self.controller.state, "pickup")
        self.assertEqual(len(action), 7)

    def test_missing_cap_with_depthless_spare_starts_rgb_plane_pickup(self):
        detections = [self.detection("missing_cap", [0.31, 0.02, 0.01]),
                      self.detection("table_cap", [0.34, -0.20, 0.02], depth=False)]
        action = self.controller.act(self.obs, detections)
        self.assertEqual(self.controller.state, "pickup")
        self.assertEqual(len(action), 7)
        self.assertIs(self.controller.pending["spare"]["depth_available"], False)

    def test_depthless_missing_cap_remains_ineligible(self):
        detections = [self.detection("missing_cap", [0.31, 0.02, 0.01], depth=False),
                      self.detection("table_cap", [0.34, -0.20, 0.02], depth=False)]
        action = self.controller.act(self.obs, detections)
        np.testing.assert_allclose(action[:6], self.tcp)
        self.assertEqual(self.controller.state, "inspect")

    def test_missing_is_first_and_replacement_induced_crooked_is_added(self):
        missing = self.detection("missing_cap", [0.31, 0.02, 0.21])
        crooked_elsewhere = self.detection("crooked_cap", [0.37, 0.08, 0.21])
        spare = self.detection("table_cap", [0.34, -0.20, 0.02])
        # Deliberately present crooked first; the controller must still choose
        # the missing vial before any crooked-cap repair.
        self.controller.act(self.obs, [crooked_elsewhere, missing, spare])
        self.assertEqual(self.controller.pending["target"]["class_name"], "missing_cap")
        while self.controller.queue:
            self.controller.act(self.obs, [crooked_elsewhere, missing, spare])
        self.controller.act({**self.obs, "grasp_verified": True},
                            [crooked_elsewhere, missing])
        while self.controller.queue:
            self.controller.act(self.obs, [crooked_elsewhere, missing])

        # Simulate the installed cap now appearing crooked at the aperture.
        replacement_crooked = self.detection("crooked_cap", [0.31, 0.02, 0.21])
        for _ in range(5):
            self.controller.act(self.obs, [crooked_elsewhere, replacement_crooked])
        self.assertEqual(self.controller.state, "inspect")
        self.controller.act(self.obs, [crooked_elsewhere, replacement_crooked])
        # The originally observed crooked cap remains ahead of the newly
        # created task, but both have now been tracked.
        np.testing.assert_allclose(self.controller.pending["target"]["base_pos_m"],
                                   crooked_elsewhere["base_pos_m"])
        while self.controller.queue:
            self.controller.act(self.obs, [crooked_elsewhere, replacement_crooked])
        for _ in range(5):
            self.controller.act(self.obs, [crooked_elsewhere, replacement_crooked])
        self.assertEqual(self.controller.state, "inspect")
        self.controller.act(self.obs, [crooked_elsewhere, replacement_crooked])
        self.assertEqual(self.controller.pending["target"]["class_name"], "crooked_cap")
        np.testing.assert_allclose(self.controller.pending["target"]["base_pos_m"],
                                   replacement_crooked["base_pos_m"])
        self.assertTrue(any(track["class_name"] == "crooked_cap" and
                            np.linalg.norm(track["anchor_xy"] - [0.31, 0.02]) < 1e-12
                            for track in self.controller.defect_tracks))

    def test_pickup_returns_to_inspection_while_holding_cap(self):
        missing = self.detection("missing_cap", [0.31, 0.02, 0.01])
        spare = self.detection("table_cap", [0.34, -0.20, 0.02])
        actions = [self.controller.act(self.obs, [missing, spare])]
        while self.controller.queue:
            actions.append(self.controller.act(self.obs, []))
        np.testing.assert_allclose(actions[-1][:3], base_to_uf0(inspection_base_position()))
        self.assertEqual(actions[-1][6], 1.0)

    def test_table_pick_and_missing_place_apply_independent_xy_calibration(self):
        table_offset = np.array([0.004, -0.003])
        missing_offset = np.array([-0.002, 0.005])
        controller = RemediationController(
            ContactTool(-0.006, 0.0, 0.0, validated=True),
            table_cap_xy_offset_m=table_offset,
            missing_cap_xy_offset_m=missing_offset)
        missing = self.detection("missing_cap", [0.31, 0.02, 0.01])
        spare = self.detection("table_cap", [0.34, -0.20, 0.02])
        pickup_actions = [controller.act(self.obs, [missing, spare])]
        while controller.queue:
            pickup_actions.append(controller.act(self.obs, []))
        pickup_base = np.asarray([uf0_to_base(action[:3]) for action in pickup_actions])
        expected_pick_xy = np.asarray(spare["base_pos_m"][:2]) + table_offset
        self.assertTrue(any(np.allclose(point[:2], expected_pick_xy)
                            for point in pickup_base))

        place_actions = [controller.act({**self.obs, "grasp_verified": True}, [missing])]
        while controller.queue:
            place_actions.append(controller.act(self.obs, []))
        place_base = np.asarray([uf0_to_base(action[:3]) for action in place_actions])
        expected_place_xy = np.asarray(missing["base_pos_m"][:2]) + missing_offset
        self.assertTrue(any(np.allclose(point[:2], expected_place_xy)
                            for point in place_base))

    def test_pickup_wiggles_open_then_centers_before_closing(self):
        controller = RemediationController(
            ContactTool(-0.002, 0.0, 0.0, validated=True),
            pickup_wiggle_radius_m=0.0005, pickup_wiggle_cycles=1)
        missing = self.detection("missing_cap", [0.31, 0.02, 0.01])
        spare = self.detection("table_cap", [0.34, -0.20, 0.02])
        actions = [controller.act(self.obs, [missing, spare])]
        while controller.queue:
            actions.append(controller.act(self.obs, []))
        first_closed = next(i for i, action in enumerate(actions) if action[6] == 1.0)
        wiggle = actions[first_closed - 6:first_closed]
        self.assertTrue(all(action[6] == 0.0 for action in wiggle))
        self.assertAlmostEqual(max(action[0] for action in wiggle)
                               - min(action[0] for action in wiggle), 1.0, places=5)
        np.testing.assert_allclose(wiggle[0][:3], wiggle[-1][:3])

    def test_missing_cap_uses_calibrated_rim_without_good_cap(self):
        detections = [self.detection("missing_cap", [0.31, 0.02, 0.01]),
                      self.detection("table_cap", [0.34, -0.20, 0.02])]
        self.controller.act(self.obs, detections)
        self.assertEqual(self.controller.state, "pickup")
        self.assertAlmostEqual(self.controller.pending["target"]["base_pos_m"][2],
                               0.2081, places=4)

    def test_verified_pickup_starts_single_release_and_side_exit_route(self):
        missing = self.detection("missing_cap", [0.31, 0.02, 0.01])
        spare = self.detection("table_cap", [0.34, -0.20, 0.02])
        good = self.detection("good_cap", [0.28, 0.01, 0.210132])
        self.controller.act(self.obs, [missing, spare, good])
        while self.controller.queue:
            self.controller.act(self.obs, [missing, spare])
        action = self.controller.act({**self.obs, "grasp_verified": True}, [missing])
        self.assertEqual(self.controller.state, "repair")
        self.assertEqual(len(action), 7)
        self.assertAlmostEqual(self.controller.pending["target"]["base_pos_m"][2], 0.2081, places=4)
        actions = [action]
        while self.controller.queue:
            actions.append(self.controller.act(self.obs, []))
        jaw = [item[6] for item in actions]
        release = next(index for index in range(1, len(jaw))
                       if jaw[index - 1] == 1.0 and jaw[index] == 0.0)
        self.assertTrue(all(value == 0.0 for value in jaw[release:]))
        base = np.asarray([uf0_to_base(item[:3]) for item in actions])
        # The conservative defaults use one 1 mm jaw-axis clearing step, then
        # a 3 mm crossing toward the opposite side while rising 25 mm.
        self.assertAlmostEqual(np.linalg.norm(base[release + 1, :2] - base[release, :2]),
                               0.001, places=9)
        self.assertAlmostEqual(np.linalg.norm(base[release + 2, :2] - base[release + 1, :2]),
                               0.003, places=9)
        self.assertAlmostEqual(base[release + 2, 2] - base[release, 2], 0.025, places=9)

    def test_missing_release_motion_is_tunable(self):
        controller = RemediationController(
            ContactTool(0.0, 0.0, 0.0005, validated=True),
            missing_release_side_step_m=0.0005,
            missing_release_exit_step_m=0.0015)
        controller.pending = {
            "target": self.detection("missing_cap", [0.31, 0.02, rim_base_z()]),
            "spare": None,
            "height": 0.285,
        }
        controller._add_missing_release([0.31, 0.02, rim_base_z()], 0.285)
        route = list(controller.queue)
        base = np.asarray([uf0_to_base(item[:3]) for item in route])
        release = 2
        self.assertAlmostEqual(
            np.linalg.norm(base[release + 1, :2] - base[release, :2]),
            0.0005, places=9)
        self.assertAlmostEqual(
            np.linalg.norm(base[release + 2, :2] - base[release + 1, :2]),
            0.0020, places=9)

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
