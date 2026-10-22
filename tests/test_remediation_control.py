import importlib.util
from pathlib import Path
import unittest

import numpy as np

from src.remediation.controller import ContactTool, RemediationController, checked_action, load_validated_contact_tool
from src.remediation.geometry import base_to_uf0, transit_base_z

spec=importlib.util.spec_from_file_location('physical_policy',Path(__file__).resolve().parents[1]/'PhysicalAI-Hackathon-2026/policy.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Detector:
    def __init__(self): self.calls=0
    def detect(self,image,timestamp=None): self.calls+=1; return []


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.obs={'joints':[1,2,3,4,5,6,0],'tcp_pose':[350,0,20,-180,0,0],
                  'images':{'table':np.zeros((36,64,3),dtype=np.uint8)},'t':1.}

    def test_policy_does_not_move_without_images(self):
        detector=Detector(); policy=module.Policy(detector=detector)
        self.obs['images']={}
        self.assertEqual(policy.act(self.obs),self.obs['joints'])
        self.assertEqual(detector.calls,0)

    def test_policy_rejects_stale_frame_and_resets(self):
        detector=Detector(); policy=module.Policy(detector=detector)
        policy.act(self.obs); policy.act(self.obs)
        self.assertEqual(detector.calls,1)
        policy.reset(); policy.act(self.obs)
        self.assertEqual(detector.calls,2)

    def test_unvalidated_tool_blocks_contact(self):
        with self.assertRaises(ValueError): RemediationController(ContactTool(0,0,0))

    def test_placeholder_custom_finger_contract_cannot_enable_contact(self):
        with self.assertRaises(ValueError): load_validated_contact_tool()

    def test_policy_without_weights_holds_position(self):
        policy=module.Policy()
        self.assertEqual(policy.act(self.obs),self.obs['joints'])

    def test_workspace_does_not_clip_targets(self):
        with self.assertRaises(ValueError): checked_action([.8,0,.3],False)

    def test_coarse_detection_does_not_trigger_motion(self):
        controller=RemediationController(ContactTool(0,0,0,True))
        target={'class_name':'crooked_cap','confidence':.99,'base_pos_m':[.4,.1,.27],'contact_qualified':False}
        self.assertEqual(controller.act(self.obs,[target]),self.obs['tcp_pose']+[0.])
        self.assertEqual(len(controller.queue),0)

    def test_raise_before_lateral_transit_and_verify_repair(self):
        controller=RemediationController(ContactTool(0,0,0,True))
        target={'class_name':'crooked_cap','confidence':.99,'base_pos_m':[.4,.1,.27],'contact_qualified':True}
        first=controller.act(self.obs,[target])
        np.testing.assert_allclose(first[:2],self.obs['tcp_pose'][:2])
        self.assertGreaterEqual(first[2],base_to_uf0([0,0,transit_base_z()])[2])
        while controller.queue: controller.act(self.obs,[target])
        # Elapsed actions do not prove a repair.
        controller.act(self.obs,[target]); self.assertEqual(controller.state,'verify')
        good={**target,'class_name':'good_cap'}
        controller.act(self.obs,[good]); self.assertEqual(controller.state,'verify')
        controller.act(self.obs,[good]); self.assertEqual(controller.state,'inspect')

    def test_unverified_pickup_does_not_continue_to_tray(self):
        controller=RemediationController(ContactTool(0,0,0,True))
        target={'class_name':'missing_cap','confidence':.99,'base_pos_m':[.4,.1,.27],'contact_qualified':True}
        spare={**target,'class_name':'table_cap','base_pos_m':[.35,-.2,.01]}
        controller.act(self.obs,[target,spare])
        while controller.queue: controller.act(self.obs,[target,spare])
        controller.act(self.obs,[target,spare])
        self.assertEqual(controller.state,'failed')
        self.assertEqual(len(controller.queue),0)


if __name__=='__main__': unittest.main()
