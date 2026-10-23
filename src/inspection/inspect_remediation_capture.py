"""Run the deployment detector against a captured TableCam image, without motion."""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.inspection.vision_pipeline import RemediationVisionDetector,camera_from_usd_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--image',required=True)
    p.add_argument('--depth',help='Optional aligned Isaac TableCam depth .npy in metres')
    p.add_argument('--capture-report',help='Simulation camera metadata; omit for real calibrated TableCam')
    p.add_argument('--out',required=True); p.add_argument('--device',default='cpu')
    args=p.parse_args()
    camera=None; capture=None
    if args.capture_report:
        capture=json.loads(Path(args.capture_report).read_text())
        camera=camera_from_usd_capture(capture['camera'])
    image=cv2.imread(args.image)
    if image is None: raise ValueError('Cannot read input image')
    detector=RemediationVisionDetector(args.checkpoint,camera=camera,device=args.device)
    depth=None
    if args.depth:
        import numpy as np
        depth=np.load(args.depth)
    targets=detector.detect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),depth_m=depth)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'checkpoint':args.checkpoint,'image':args.image,'depth':args.depth,
                               'camera':capture.get('camera') if capture else None,
                               'source':capture.get('source') if capture else 'external_tablecam_capture',
                               'targets':targets, 'motion_enabled':False,
                               'contact_localization_validated':False},indent=2)+'\n')
    print(f'{len(targets)} detections; positions saved to {out}; no robot commands issued')


if __name__=='__main__': main()
