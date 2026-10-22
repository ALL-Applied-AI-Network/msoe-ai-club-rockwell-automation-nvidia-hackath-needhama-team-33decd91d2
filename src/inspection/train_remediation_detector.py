#!/usr/bin/env python3
"""Train a versioned detector only from a physics-qualified dataset manifest."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.remediation.geometry import CLASS_NAMES, configuration_hash


def train_detector(data, out, epochs=50, batch=8, imgsz=1280, model_type='yolo11n.pt'):
    from ultralytics import YOLO
    data=Path(data).resolve(); out=Path(out).resolve()
    manifest=json.loads((data.parent/'manifest.json').read_text())
    if not manifest.get('accepted') or manifest.get('failed_samples'):
        raise ValueError('Dataset has not passed physics/label qualification')
    if tuple(manifest['classes'])!=CLASS_NAMES or manifest['config_sha256']!=configuration_hash():
        raise ValueError('Dataset class mapping or physical configuration is stale')
    if out.exists(): raise FileExistsError('Use a new training output directory')
    model=YOLO(model_type)
    model.train(data=str(data),epochs=epochs,batch=batch,imgsz=imgsz,project=str(out.parent),name=out.name,
                exist_ok=False,device=0,seed=42,degrees=5.,fliplr=0.,scale=.1,mosaic=.2,hsv_s=.2,hsv_v=.2)
    weights=out/'weights/best.pt'
    if not weights.is_file(): raise RuntimeError('Training did not create a best checkpoint')
    best=YOLO(str(weights))
    # Keep the held-out evaluation artifacts with this run, never in Ultralytics'
    # process-default directory (which may belong to another workspace).
    metrics=best.val(data=str(data),split='test',device=0,imgsz=imgsz,
                     project=str(out.parent),name=out.name+'_test',exist_ok=False)
    result={'schema_version':1,'checkpoint':str(weights),'data':str(data),'config_sha256':configuration_hash(),
            'classes':list(CLASS_NAMES),'synthetic_test':metrics.results_dict,
            'per_class_synthetic_test':{name:{'precision':float(metrics.box.p[i]),'recall':float(metrics.box.r[i]),
                                            'mAP50_95':float(metrics.box.maps[i])} for i,name in enumerate(CLASS_NAMES)},
            'real_validation_passed':False,'contact_localization_validated':False,'robot_release_approved':False}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    # Checkpoints remain experiments until class-specific and real-image gates pass.
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch',type=int,default=8)
    p.add_argument('--imgsz',type=int,default=1280); p.add_argument('--model',default='yolo11n.pt')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use scripts/submit_training_job.slurm')
    train_detector(a.data,a.out,a.epochs,a.batch,a.imgsz,a.model)
