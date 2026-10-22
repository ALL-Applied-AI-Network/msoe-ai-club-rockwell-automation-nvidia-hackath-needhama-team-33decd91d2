#!/usr/bin/env python3
"""Report whether a detector/tool combination may be used for contact motion.

This checker never promotes a checkpoint.  It makes the missing evidence visible
before an Isaac or physical-cell launch is attempted.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.remediation.geometry import CLASS_NAMES, configuration_hash


def read_json(path):
    return json.loads(Path(path).read_text())


def check(evaluation, tool_contract):
    evidence=read_json(evaluation)
    tool=read_json(tool_contract)
    reasons=[]
    if tuple(evidence.get('classes',())) != CLASS_NAMES:
        reasons.append('checkpoint class mapping does not match the remediation contract')
    if evidence.get('config_sha256') != configuration_hash():
        reasons.append('checkpoint was trained against a different physical configuration')
    per_class=evidence.get('per_class_synthetic_test',{})
    for name in ('crooked_cap','missing_cap'):
        metric=per_class.get(name,{})
        if float(metric.get('recall',0.)) < .95:
            reasons.append(f'{name} synthetic recall is below 95%')
    table=per_class.get('table_cap',{})
    if float(table.get('precision',0.)) < .99:
        reasons.append('table_cap synthetic precision is below 99%')
    if not evidence.get('real_validation_passed'):
        reasons.append('no independently reviewed real-image validation is recorded')
    if not evidence.get('contact_localization_validated'):
        reasons.append('contact localization error budget is not validated')
    required_validation=tool.get('validation',{})
    if tool.get('status') != 'validated' or not required_validation or not all(required_validation.values()):
        reasons.append('custom finger CAD/measurements/contact validation are incomplete')
    return {'schema_version':1,'approved_for_contact':not reasons,'reasons':reasons,
            'evaluation':str(Path(evaluation).resolve()),'tool_contract':str(Path(tool_contract).resolve())}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--tool-contract',default=ROOT/'src/remediation/config/custom_finger_contract.json')
    parser.add_argument('--out')
    args=parser.parse_args()
    result=check(args.evaluation,args.tool_contract)
    text=json.dumps(result,indent=2)+'\n'
    if args.out: Path(args.out).write_text(text)
    print(text,end='')
    if not result['approved_for_contact']: raise SystemExit(2)
