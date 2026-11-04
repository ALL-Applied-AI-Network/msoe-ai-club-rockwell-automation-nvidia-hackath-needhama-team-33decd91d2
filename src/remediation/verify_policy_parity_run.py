#!/usr/bin/env python3
"""Verify that a completed Isaac run used only the shared physical-policy contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.remediation.controller import FORBIDDEN_ORACLE_FIELDS


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def verify(action_dir):
    action_dir = Path(action_dir)
    report = json.loads((action_dir / "attempt_report.json").read_text())
    trace = json.loads((action_dir / "trace.json").read_text())
    failures = []
    if report.get("mode") != "shared_physical_policy_in_isaac":
        failures.append("attempt report is not a shared physical-policy run")
    if report.get("controller_module") != "src.remediation.controller.RemediationController":
        failures.append("unexpected controller module")
    if report.get("force_available_to_controller") is not False:
        failures.append("simulated force was available to the controller")
    if report.get("press_control") != "position-limited":
        failures.append("press control was not position-limited")
    decisions = report.get("decision_provenance", [])
    if not decisions:
        failures.append("no policy decision provenance was recorded")
    for index, decision in enumerate(decisions):
        if decision.get("oracle_inputs_used") != []:
            failures.append(f"decision {index} reports oracle input use")
    forbidden = sorted(FORBIDDEN_ORACLE_FIELDS.intersection(_keys(trace)))
    if forbidden:
        failures.append(f"deployable trace contains oracle fields: {forbidden}")
    oracle_path = action_dir / "oracle_evaluation_trace.json"
    if not oracle_path.is_file():
        failures.append("segregated oracle evaluator trace is missing")
    result = {"passed": not failures, "action_dir": str(action_dir),
              "decision_count": len(decisions), "failures": failures}
    if failures:
        raise ValueError(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action_dir")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.action_dir)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
