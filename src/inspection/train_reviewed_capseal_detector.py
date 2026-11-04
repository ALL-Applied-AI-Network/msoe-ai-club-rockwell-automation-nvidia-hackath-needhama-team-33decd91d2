#!/usr/bin/env python3
"""Train the remediation detector on the human-reviewed cap-seal dataset.

train_remediation_detector.py gates on a physics-qualified synthetic manifest;
this dataset is real photos with reviewed labels (build_reviewed_capseal_dataset.py),
so it records that provenance instead of a geometry hash.  Hyperparameters and
the held-out test evaluation match train_remediation_detector.py.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.remediation.geometry import CLASS_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run on a GPU node via Slurm')
    from ultralytics import YOLO
    data, out = a.data.resolve(), a.out.resolve()
    if out.exists():
        raise FileExistsError('Use a new training output directory')
    build = json.loads((data.parent / 'build_report.json').read_text())
    if build['audit_dropped']:
        raise ValueError('Dataset build reported audit failures')
    YOLO(a.model).train(data=str(data), epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, project=str(out.parent),
                        name=out.name, exist_ok=False, device=0, seed=42, patience=25, degrees=5., fliplr=0.,
                        scale=.1, mosaic=.2, hsv_s=.2, hsv_v=.2)
    weights = out / 'weights/best.pt'
    if not weights.is_file():
        raise RuntimeError('Training did not create a best checkpoint')
    metrics = YOLO(str(weights)).val(data=str(data), split='test', device=0, imgsz=a.imgsz,
                                     project=str(out.parent), name=out.name + '_test', exist_ok=False)
    result = {'schema_version': 1, 'checkpoint': str(weights), 'data': str(data), 'classes': list(CLASS_NAMES),
              'provenance': 'real cap-seal photos (csi_vial_fill_dataset) labelled by human review of tray '
                            'states; 10 synthetic TableCam renders merged for table_cap',
              'dataset_build': {k: build[k] for k in ('boxes', 'frames')},
              'held_out_test': metrics.results_dict,
              'per_class_test': {CLASS_NAMES[c]: {'precision': float(metrics.box.p[i]), 'recall': float(metrics.box.r[i]),
                                                  'mAP50': float(metrics.box.ap50[i]), 'mAP50_95': float(metrics.box.ap[i])}
                                 for i, c in enumerate(metrics.box.ap_class_index)},
              'real_validation_passed': False, 'contact_localization_validated': False, 'robot_release_approved': False}
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
