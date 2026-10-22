"""Image-driven remediation policy. Defaults to observation and joint hold.

Contact motion remains disabled until the custom fingers, localization and model
have passed the documented release gates. Without weights the policy holds
position, allowing the client contract to be exercised safely.
"""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inspection.vision_pipeline import RemediationVisionDetector
from src.remediation.controller import RemediationController


class _HoldDetector:
    """Safe no-model implementation used only for contract/mock checks."""
    def detect(self, image, timestamp=None):
        return []


class Policy:
    CAMERAS=['table']
    IMAGE_SIZE=(1920,1080)
    IMAGE_FORMAT='png'
    ACTION='joints'  # observation-only until a calibrated controller is provided
    PREFETCH=False
    DEPTH=[]

    def __init__(self,checkpoint=None,*,detector=None,contact_tool=None):
        self.detector=(detector if detector is not None else
                       RemediationVisionDetector(checkpoint) if checkpoint else _HoldDetector())
        self.controller=RemediationController(contact_tool) if contact_tool is not None else None
        if self.controller is not None: self.ACTION='tcp'
        self.reset()

    def reset(self):
        self.last_detections=[]; self.last_observation_time=None
        if self.controller is not None: self.controller.reset()

    def act(self,obs):
        joints=np.asarray(obs.get('joints',[]),dtype=float)
        if joints.shape!=(7,) or not np.isfinite(joints).all():
            raise ValueError('Seven finite current joint/gripper values are required')
        image=obs.get('images',{}).get('table')
        stamp=obs.get('t')
        fresh=stamp is None or self.last_observation_time is None or stamp>self.last_observation_time
        if image is None or not fresh:
            if self.controller is not None:
                # Do not advance a queued motion on a missing or repeated observation.
                return self.controller.hold(obs)
            return joints.tolist()
        self.last_observation_time=stamp
        self.last_detections=self.detector.detect(image,timestamp=stamp)
        if self.controller is None: return joints.tolist()
        return self.controller.act(obs,self.last_detections)
