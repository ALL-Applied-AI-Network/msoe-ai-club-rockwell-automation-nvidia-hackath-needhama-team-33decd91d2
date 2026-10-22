"""Validate the custom-finger jaw-gap contract before contact trials.

This is a conservative kinematic gate. It checks the documented mesh envelope
and jaw travel; it does not approve contact or replace a PhysX swept collision
test against tray contents.
"""
import argparse
import json
from pathlib import Path

MIN_GAP_M = 0.003
MAX_GAP_M = 0.015
JAW_STROKE_M = 0.006


def validate():
    samples = []
    for opening in (0.0, JAW_STROKE_M / 2.0, JAW_STROKE_M):
        gap = MIN_GAP_M + 2.0 * opening
        samples.append({'jaw_extension_per_side_m': opening,
                        'minimum_mesh_gap_m': gap,
                        'within_contract': MIN_GAP_M <= gap <= MAX_GAP_M})
    return {
        'schema_version': 1,
        'mode': 'kinematic_proxy_gate',
        'accepted': all(x['within_contract'] for x in samples),
        'contact_enabled': False,
        'contract': {'closed_gap_m': MIN_GAP_M, 'open_gap_m': MAX_GAP_M,
                     'jaw_stroke_per_side_m': JAW_STROKE_M},
        'samples': samples,
        'limitations': [
            'does not certify swept PhysX collision against tray, vials, or caps',
            'does not validate TCP or contact forces',
        ],
        'next_gate': 'Isaac swept collision test with tray inventory and custom proxies',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = validate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
