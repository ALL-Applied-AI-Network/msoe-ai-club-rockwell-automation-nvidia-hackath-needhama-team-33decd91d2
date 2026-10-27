import ast
from pathlib import Path
import unittest

from src.remediation.controller import ContactTool, RemediationController
from src.remediation.geometry import base_to_uf0


ROOT = Path(__file__).resolve().parents[1]


class PolicyParityContractTests(unittest.TestCase):
    def test_shared_controller_has_no_simulator_imports(self):
        path = ROOT / "src/remediation/controller.py"
        tree = ast.parse(path.read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        forbidden = {name for name in modules
                     if name.startswith(("isaacsim", "omni", "pxr", "carb"))}
        self.assertEqual(forbidden, set())

    def test_both_adapters_import_same_controller(self):
        physical = (ROOT / "PhysicalAI-Hackathon-2026/policy.py").read_text()
        simulator = (ROOT / "src/remediation/run_compliant_breakable_remediation.py").read_text()
        needle = "from src.remediation.controller import"
        self.assertIn(needle, physical)
        self.assertIn(needle, simulator)
        self.assertIn("RemediationController", physical)
        self.assertIn("RemediationController", simulator)

    def test_supported_pipeline_selects_shared_policy(self):
        pipeline = (ROOT / "scripts/run_compliant_breakable_pipeline.slurm").read_text()
        self.assertIn("--controller-policy shared", pipeline)
        self.assertIn('--checkpoint "$checkpoint"', pipeline)
        self.assertIn("verify_policy_parity_run.py", pipeline)
        self.assertIn("oracle_evaluation_trace.json", pipeline)
        self.assertIn("-framerate 60", pipeline)

    def test_active_policy_uses_tablecam_and_fixed_wrist_press(self):
        physical = (ROOT / "PhysicalAI-Hackathon-2026/policy.py").read_text()
        controller = (ROOT / "src/remediation/controller.py").read_text()
        self.assertIn('CAMERAS = ["table"]', physical)
        self.assertIn('DEPTH = ["table"]', physical)
        self.assertNotIn('state == "crooked_observe_axis"', physical)
        self.assertNotIn("ctrl.state == 'crooked_observe_axis'", (
            ROOT / "src/remediation/run_compliant_breakable_remediation.py").read_text())
        self.assertNotIn('self.state = "crooked_observe_axis"', controller)

    def test_live_true_axis_is_not_passed_to_servo(self):
        runner = (ROOT / "src/remediation/run_compliant_breakable_remediation.py").read_text()
        self.assertNotIn("truth=true_axis", runner)

    def test_isaac_python_paths_do_not_leak_into_yolo(self):
        runner = (ROOT / "src/remediation/run_compliant_breakable_remediation.py").read_text()
        self.assertIn("inference_env.pop('PYTHONHOME', None)", runner)
        self.assertIn("inference_env.pop('PYTHONPATH', None)", runner)
        self.assertIn("env=inference_env", runner)

    def test_isaac_failure_cannot_be_masked_by_app_close(self):
        runner = (ROOT / "src/remediation/run_compliant_breakable_remediation.py").read_text()
        self.assertIn("os._exit(1)", runner)
        self.assertIn("failure_report.json", runner)

    def test_shared_isaac_adapter_uses_closed_loop_joint_tracking(self):
        runner = (ROOT / "src/remediation/run_compliant_breakable_remediation.py").read_text()
        self.assertIn("from src.remediation.geometry import SPEC, base_to_uf0", runner)
        self.assertIn("closed_loop_tracking=True", runner)
        self.assertIn("POLICY_MAX_FOLLOWING_ERROR_DEG", runner)
        self.assertIn("POLICY_RESUME_ERROR_DEG", runner)
        self.assertIn("EOAT detached from flange", runner)

    def test_identical_observation_replay_produces_identical_action(self):
        tool = ContactTool(.005, .005, .005, validated=True)
        controllers = [RemediationController(tool), RemediationController(tool)]
        obs = {"tcp_pose": [*base_to_uf0([.3, 0, .3]), -180, 0, 0], "joints": [0] * 7}
        detections = [{"class_name": "crooked_cap", "confidence": .91,
                       "base_pos_m": [.31, .02, .01], "depth_available": True,
                       "contact_qualified": True}]
        actions = [controller.act(obs, detections) for controller in controllers]
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(controllers[0].decision_records, controllers[1].decision_records)


if __name__ == "__main__":
    unittest.main()
