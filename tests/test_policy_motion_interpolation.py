import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.remediation.run_compliant_breakable_remediation import (
    POLICY_MAX_ANGULAR_STEP_DEG,
    POLICY_MAX_JOINT_STEP_DEG,
    POLICY_MAX_LINEAR_STEP_M,
    Planner,
    joint_step_deg,
    pose_interpolation,
    speed_limited_joints,
)


class PolicyMotionInterpolationTests(unittest.TestCase):
    def test_robot_local_x_down_converts_to_logical_tcp_z_down(self):
        planner = Planner.__new__(Planner)
        planner.tool_axis = np.array([1.0, 0.0, 0.0])
        planner.spread_axis = np.array([0.0, 1.0, 0.0])
        robot_rotation = np.column_stack(([0.0, 0.0, -1.0],
                                          [0.0, -1.0, 0.0],
                                          [-1.0, 0.0, 0.0]))
        logical = planner.logical_tcp_rotation(robot_rotation)
        expected = Rotation.from_euler("xyz", [-180, 0, 0], degrees=True).as_matrix()
        np.testing.assert_allclose(logical, expected, atol=1e-12)

    def test_joint_step_uses_shortest_angle(self):
        q0 = np.radians([179, 0, 0, 0, 0, 0])
        q1 = np.radians([-179, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(joint_step_deg(q0, q1), 2.0)

    def test_large_wrist_change_is_speed_limited_and_reaches_target(self):
        p0 = np.array([0.30, 0.10, 0.28])
        p1 = np.array([0.31, 0.09, 0.22])
        R0 = Rotation.from_euler("xyz", [-180, 0, 0], degrees=True).as_matrix()
        R1 = Rotation.from_euler("xyz", [150, 41, 38], degrees=True).as_matrix()
        samples = pose_interpolation(p0, R0, p1, R1)
        positions = np.vstack([p0, *(p for p, _ in samples)])
        rotations = [R0, *(R for _, R in samples)]
        linear_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        angular_steps = [np.degrees((Rotation.from_matrix(a).inv() *
                                     Rotation.from_matrix(b)).magnitude())
                         for a, b in zip(rotations, rotations[1:])]
        self.assertLessEqual(float(linear_steps.max()), POLICY_MAX_LINEAR_STEP_M + 1e-9)
        self.assertLessEqual(float(max(angular_steps)), POLICY_MAX_ANGULAR_STEP_DEG + 1e-9)
        np.testing.assert_allclose(samples[-1][0], p1, atol=1e-12)
        np.testing.assert_allclose(samples[-1][1], R1, atol=1e-12)

    def test_joint_fallback_is_speed_limited_and_reaches_target(self):
        q0 = np.radians([0, -20, 35, 0, 15, 0])
        q1 = np.radians([35, 10, -15, 150, -40, 80])
        samples = speed_limited_joints(q0, q1)
        steps_deg = np.degrees(np.abs(np.diff(np.vstack([q0, *samples]), axis=0)))
        self.assertLessEqual(float(steps_deg.max()), POLICY_MAX_JOINT_STEP_DEG + 1e-9)
        np.testing.assert_allclose(samples[-1], q1, atol=1e-12)
        self.assertLessEqual(max(joint_step_deg(a, b)
                                 for a, b in zip([q0, *samples[:-1]], samples)),
                             POLICY_MAX_JOINT_STEP_DEG + 1e-9)


if __name__ == "__main__":
    unittest.main()
