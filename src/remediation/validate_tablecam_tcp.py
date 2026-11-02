"""Validate TableCam backprojection against measured TCP reference captures.

This accepts only externally measured correspondences.  It never reads scene
truth and does not create robot targets.  A passing report is a prerequisite
for contact-qualified perception, not evidence of grasp success.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inspection.vision_pipeline import load_cameras, pixel_to_base


def validate(captures, camera, max_xy_error_m=.003, max_z_error_m=.005, min_xy_span_m=.12):
    rows = []
    for index, item in enumerate(captures):
        for key in ('pixel_xy', 'feature_plane_base_z_m', 'measured_feature_base_pos_m'):
            if key not in item:
                raise ValueError(f'correspondence {index} is missing {key}')
        predicted = pixel_to_base(camera, item['pixel_xy'], item['feature_plane_base_z_m'],
                                  tuple(item['image_size']) if 'image_size' in item else None)
        measured = np.asarray(item['measured_feature_base_pos_m'], dtype=float)
        if measured.shape != (3,) or not np.isfinite(measured).all():
            raise ValueError(f'correspondence {index} has invalid measured feature position')
        delta = predicted - measured
        rows.append({
            'id': item.get('id', str(index)),
            'predicted_feature_base_pos_m': predicted.tolist(),
            'measured_feature_base_pos_m': measured.tolist(),
            'xy_error_m': float(np.linalg.norm(delta[:2])),
            'z_error_m': float(abs(delta[2])),
        })
    if len(rows) < 6:
        raise ValueError('At least six measured references spanning the tray are required')
    measured_xy = np.array([row['measured_feature_base_pos_m'][:2] for row in rows])
    span_xy = measured_xy.max(axis=0) - measured_xy.min(axis=0)
    if np.any(span_xy < min_xy_span_m):
        raise ValueError(f'References must span at least {min_xy_span_m:.3f} m in each table axis; got {span_xy.tolist()}')
    xy = np.array([row['xy_error_m'] for row in rows])
    z = np.array([row['z_error_m'] for row in rows])
    return {
        'schema_version': 1,
        'mode': 'measured_tablecam_to_tcp_reference_validation',
        'evaluation_only': True,
        'contact_localization_validated': bool(xy.max() <= max_xy_error_m and z.max() <= max_z_error_m),
        'thresholds_m': {'max_xy_error_m': max_xy_error_m, 'max_z_error_m': max_z_error_m,
                         'min_xy_span_m': min_xy_span_m},
        'summary': {'reference_count': len(rows), 'median_xy_error_m': float(np.median(xy)),
                    'max_xy_error_m': float(xy.max()), 'median_z_error_m': float(np.median(z)),
                    'max_z_error_m': float(z.max()), 'reference_xy_span_m': span_xy.tolist()},
        'references': rows,
        'limitations': ['Measured correspondence positions must be recorded with the physical TCP/reference fixture.',
                        'A passing result validates localization only; it does not validate contact force or grasp retention.'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', required=True, help='JSON object containing a correspondences list')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-xy-error-m', type=float, default=.003)
    parser.add_argument('--max-z-error-m', type=float, default=.005)
    parser.add_argument('--min-xy-span-m', type=float, default=.12)
    args = parser.parse_args()
    raw = json.loads(Path(args.captures).read_text())
    report = validate(raw.get('correspondences', []), load_cameras()['table'],
                      args.max_xy_error_m, args.max_z_error_m, args.min_xy_span_m)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))
