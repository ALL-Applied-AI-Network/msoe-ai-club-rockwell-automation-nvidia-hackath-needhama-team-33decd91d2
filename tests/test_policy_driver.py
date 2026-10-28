import unittest

from src.remediation.controller import ContactTool, RemediationController
from src.remediation.geometry import base_to_uf0
from src.remediation.policy_driver import PolicyDriver


class PolicyDriverTests(unittest.TestCase):
    def test_no_target_terminates_on_hold(self):
        controller = RemediationController(ContactTool(0, 0, 0, validated=True))
        tcp = [*base_to_uf0([0.3, 0, 0.3]), -180, 0, 0]
        observations = lambda _: ({"tcp_pose": tcp, "joints": [0] * 7}, [])
        actions = []
        records = PolicyDriver(controller, observations, actions.append).run()
        self.assertEqual(actions, [])
        self.assertTrue(records[-1]["hold"])
        self.assertEqual(records[-1]["state_before"], "inspect")

    def test_missing_cap_route_executes_shared_actions(self):
        controller = RemediationController(ContactTool(0.005, 0.005, 0.005, validated=True))
        tcp = [*base_to_uf0([0.3, 0, 0.3]), -180, 0, 0]
        detections = [
            {"class_name": "missing_cap", "confidence": .9, "base_pos_m": [.31, .02, .01],
             "depth_available": True, "contact_qualified": True},
            {"class_name": "table_cap", "confidence": .9, "base_pos_m": [.34, -.20, .02],
             "depth_available": True, "contact_qualified": True},
        ]
        state = {"tcp": tcp, "jaw": 0.0, "grasp": False}

        def observe(ctrl):
            obs = {"tcp_pose": state["tcp"], "joints": [0] * 6 + [state["jaw"]]}
            if ctrl.state == "pickup" and not ctrl.queue:
                obs["grasp_verified"] = True
            return obs, detections

        def execute(action, _state):
            state["tcp"] = action[:6]
            state["jaw"] = action[6]

        # Verification cannot succeed against this deliberately static detection;
        # the route still executes and returns to inspect after bounded verification.
        records = PolicyDriver(controller, observe, execute, max_decisions=40).run()
        self.assertGreater(sum(not row["hold"] for row in records), 5)
        self.assertEqual(controller.state, "inspect")
        # Returning from verify to inspect is a transition hold, not completion.
        # The final record is the subsequent inspection that finds no new work.
        self.assertEqual(records[-2]["state_before"], "verify")
        self.assertEqual(records[-2]["state"], "inspect")
        self.assertEqual(records[-1]["state_before"], "inspect")
        self.assertEqual(records[-1]["state"], "inspect")


if __name__ == "__main__":
    unittest.main()
