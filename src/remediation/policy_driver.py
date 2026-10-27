"""Environment-neutral loop for driving the shared remediation controller."""
from __future__ import annotations

import numpy as np


class PolicyDriver:
    """Repeatedly observe, decide and execute without exposing adapter internals."""

    def __init__(self, controller, observe, execute, max_decisions=200, max_idle=8):
        self.controller = controller
        self.observe = observe
        self.execute = execute
        self.max_decisions = int(max_decisions)
        self.max_idle = int(max_idle)
        self.records = []

    def run(self):
        idle = 0
        for index in range(self.max_decisions):
            obs, detections = self.observe(self.controller)
            state_before = self.controller.state
            action = self.controller.act(obs, detections)
            tcp = np.asarray(obs.get("tcp_pose"), dtype=float)
            is_hold = tcp.shape == (6,) and np.allclose(action[:6], tcp, atol=1e-8)
            record = {"decision": index, "state_before": state_before,
                      "state": self.controller.state,
                      "action_tcp": list(map(float, action)), "hold": bool(is_hold)}
            self.records.append(record)
            if not is_hold:
                self.execute(action, self.controller.state)
                idle = 0
            else:
                idle += 1
            if self.controller.state == "failed":
                return self.records
            # A verify decision deliberately returns a hold while transitioning
            # back to inspect.  That is not completion: the next observation
            # must actually be processed in inspect so another defect can be
            # selected.  Stop only when a decision that began in inspect also
            # ends there without producing motion.
            if (state_before == "inspect" and self.controller.state == "inspect"
                    and not self.controller.queue and is_hold):
                return self.records
            if idle >= self.max_idle:
                raise RuntimeError(f"shared policy made no progress for {idle} observations in state "
                                   f"{self.controller.state}")
        raise RuntimeError(f"shared policy exceeded {self.max_decisions} decisions")
