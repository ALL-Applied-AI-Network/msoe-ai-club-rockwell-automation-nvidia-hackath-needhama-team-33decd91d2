import json
from pathlib import Path
import tempfile
import unittest

from src.remediation.verify_policy_parity_run import verify


class VerifyPolicyParityRunTests(unittest.TestCase):
    def make_run(self, root):
        root = Path(root)
        report = {"mode": "shared_physical_policy_in_isaac",
                  "controller_module": "src.remediation.controller.RemediationController",
                  "force_available_to_controller": False,
                  "press_control": "position-limited",
                  "decision_provenance": [{"oracle_inputs_used": [], "action_tcp": [1] * 7}]}
        (root / "attempt_report.json").write_text(json.dumps(report))
        (root / "trace.json").write_text(json.dumps(report["decision_provenance"]))
        (root / "oracle_evaluation_trace.json").write_text("[]")

    def test_valid_run_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_run(directory)
            self.assertTrue(verify(directory)["passed"])

    def test_oracle_in_deployable_trace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_run(directory)
            (Path(directory) / "trace.json").write_text(json.dumps([{"true_axis": [0, 0, 1]}]))
            with self.assertRaises(ValueError):
                verify(directory)


if __name__ == "__main__":
    unittest.main()
