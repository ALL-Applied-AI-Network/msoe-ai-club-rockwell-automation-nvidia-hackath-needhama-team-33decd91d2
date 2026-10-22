#!/usr/bin/env python3
"""Generate isolated, physics-qualified remediation samples inside a Slurm job.

Every child must produce an accepted report and valid instance-derived labels.
Failed samples are retained for diagnosis and prevent the dataset being released.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.remediation.geometry import CLASS_NAMES, configuration_hash


def generate_dataset(out, scenes=100, seed=1000, crooked=5, missing=5, table_caps=5, fixed_inventory=False, max_attempts=5):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit generation with scripts/submit_isaac_job.slurm')
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Use a fresh dataset directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'schema_version':1,'accepted':False,'config_sha256':configuration_hash(),'classes':list(CLASS_NAMES),
              'created_utc':datetime.utcnow().isoformat()+'Z','samples':[],'rejected_attempts':[],'failed_samples':[]}
    for split in ['train','val','test']:
        (out/'images'/split).mkdir(parents=True); (out/'labels'/split).mkdir(parents=True)
    try:
        for index in range(scenes):
            split='test' if index%10==0 else 'val' if index%10==1 else 'train'
            accepted=False
            for attempt in range(max_attempts):
                # A qualified but physically rejected candidate is preserved for
                # diagnosis and replaced; a crashed child still fails the run.
                sample_seed=seed+index+attempt*100000
                sample=out/'scenes'/f'{sample_seed:06d}'
                if fixed_inventory:
                    counts=(crooked,missing,table_caps)
                else:
                    rng=np.random.default_rng(sample_seed)
                    counts=(int(rng.integers(0,crooked+1)), int(rng.integers(0,missing+1)),
                            int(rng.integers(0,table_caps+1)))
                    if counts==(0,0,0): counts=(0,0,max(1,table_caps))
                command=[sys.executable,str(ROOT/'src/remediation/qualify_scene.py'),'--out',str(sample),'--seed',str(sample_seed),
                         '--crooked',str(counts[0]),'--missing',str(counts[1]),'--table-caps',str(counts[2])]
                result=subprocess.run(command,check=False)
                report_path=sample/'qualification.json'
                report=json.loads(report_path.read_text()) if report_path.exists() else {}
                accepted=bool(result.returncode==0 and report.get('accepted') and report.get('config_sha256')==manifest['config_sha256']
                              and (sample/'table.png').exists() and (sample/'table.txt').exists())
                record={'sample_index':index,'attempt':attempt+1,'seed':sample_seed,'split':split,
                        'inventory':dict(zip(('crooked','missing','table_caps'),counts)),
                        'report':str(report_path.relative_to(out)),'returncode':result.returncode,'accepted':accepted}
                if accepted:
                    image=out/'images'/split/f'{index:06d}.png'
                    shutil.copyfile(sample/'table.png',image)
                    shutil.copyfile(sample/'table.txt',out/'labels'/split/f'{index:06d}.txt')
                    record['image_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
                    record['label_count']=len(report['labels'])
                    manifest['samples'].append(record)
                    break
                if result.returncode != 0 or not report_path.exists():
                    manifest['failed_samples'].append(record)
                    break
                manifest['rejected_attempts'].append(record)
            if not accepted and (not manifest['failed_samples'] or manifest['failed_samples'][-1].get('sample_index')!=index):
                manifest['failed_samples'].append(record)
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        manifest['accepted']=len(manifest['samples'])==scenes and not manifest['failed_samples']
        if manifest['accepted']:
            # JSON is valid YAML; avoids fragile quoting of absolute paths.
            (out/'data.yaml').write_text(json.dumps({'path':str(out),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(CLASS_NAMES))},indent=2)+'\n')
        return manifest
    finally:
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--scenes',type=int,default=100)
    p.add_argument('--seed',type=int,default=1000); p.add_argument('--crooked',type=int,default=5)
    p.add_argument('--missing',type=int,default=5); p.add_argument('--table-caps',type=int,default=5)
    p.add_argument('--fixed-inventory',action='store_true',help='Use the same inventory in each scene for a plumbing pilot')
    p.add_argument('--max-attempts',type=int,default=5)
    a=p.parse_args()
    if a.scenes<10: p.error('At least 10 independent scenes are required for train/val/test splits')
    r=generate_dataset(a.out,a.scenes,a.seed,a.crooked,a.missing,a.table_caps,a.fixed_inventory,a.max_attempts)
    print('DATASET_ACCEPTED',r['accepted'],'accepted samples',len(r['samples']),'failed',len(r['failed_samples']))
    if not r['accepted']: raise SystemExit(2)
