import json
from pathlib import Path
import tempfile
import unittest

from src.remediation.geometry import CLASS_NAMES, configuration_hash
from src.remediation.release_check import check


class ReleaseCheckTests(unittest.TestCase):
    def test_pilot_evidence_is_not_promoted(self):
        root=Path(__file__).resolve().parents[1]
        report=check(root/'runs/archive/2026-10-21_remediation/pilot_training/evaluation.json',
                     root/'src/remediation/config/custom_finger_contract.json')
        self.assertFalse(report['approved_for_contact'])
        self.assertTrue(report['reasons'])

    def test_complete_evidence_can_pass(self):
        evaluation={'classes':list(CLASS_NAMES),'config_sha256':configuration_hash(),
                    'per_class_synthetic_test':{'crooked_cap':{'recall':.96},'missing_cap':{'recall':.96},
                                                'table_cap':{'precision':.995}},
                    'real_validation_passed':True,'contact_localization_validated':True}
        tool={'status':'validated','validation':{'cad_collision_checked':True,'isaac_contact_checked':True,
              'physical_grasp_checked':True,'tablecam_tcp_checked':True,'approved_for_contact':True}}
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); ep=d/'evaluation.json'; tp=d/'tool.json'
            ep.write_text(json.dumps(evaluation)); tp.write_text(json.dumps(tool))
            self.assertTrue(check(ep,tp)['approved_for_contact'])


if __name__=='__main__':
    unittest.main()
