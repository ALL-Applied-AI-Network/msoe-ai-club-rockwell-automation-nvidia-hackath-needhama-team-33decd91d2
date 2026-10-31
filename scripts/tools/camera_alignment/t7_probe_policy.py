#!/usr/bin/env python3
"""T7 (all hypotheses): real-cell, non-contact alignment probe.

From the inspection pose this detects the scene once, then hovers the open
fingertips over a spread of camera-derived targets (tray vials on the left,
centre and right, and the spare caps) at each --hover-mm height above the
plane the policy used, and at each --r-deg wrist roll. At every hover it holds
still and records the table, side and wrist images, the joints and the
reported tcp_pose. It never descends below hover height and never closes the
gripper unless --grasp-check is given (supervised contact: pick one spare cap,
lift it 60 mm, record, put it back).

What the frames answer:
  * table + side images of the fingertips -> real tip position (T3 --clicks,
    triangulated) vs. commanded TCP: the tool error (H2), per wrist roll.
  * the same at two hover heights -> does the tip-target offset grow with
    height? (camera/plane error, H1/H3) or stay constant? (tool error, H2).
  * left/centre/right targets -> position-dependent error (H1/H3).
  * joints + tcp_pose -> T4 (UF0/UTOOL1 consistency, H4).
  * --grasp-check -> where the cap sits in the fingers (H7).

The cell only serves the cameras listed in the approved submission. Submit
the calibration slot with Policy.CAMERAS = ["table", "side", "wrist"] in
PhysicalAI-Hackathon-2026/policy.py.

    # local plumbing check (no robot)
    .conda_env/bin/python scripts/tools/camera_alignment/t7_probe_policy.py --mock --device cpu
    # during an approved slot
    .conda_env/bin/python scripts/tools/camera_alignment/t7_probe_policy.py --hover-mm 40 80 --r-deg 0 -90
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import CLIENT, out_dir

from cell_client import CellClient, run_all  # noqa: E402
from policy import (ARRIVAL_ORIENTATION_DEG, ARRIVAL_TRANSLATION_MM, DEFAULT_CHECKPOINT,  # noqa: E402
                    DOWN_WPR_DEG, GRIPPER_OPEN, INSPECTION_UF0_MM, MAX_ROW_TRANSLATION_MM,
                    MAX_ROW_WPR_DEG, MOCK_START_TCP, TRANSIT_UF0_Z_MM)
from src.inspection.vision_pipeline import RemediationVisionDetector  # noqa: E402
from src.remediation.controller import wpr_to_rotation  # noqa: E402
from src.remediation.geometry import base_to_uf0  # noqa: E402

TRAY = ('missing_cap', 'crooked_cap', 'good_cap')
UF0_LIMITS_MM = np.array([[150, 650], [-450, 450], [-334, 300]])
MIN_CLEARANCE_MM = 20.0
GRIPPER_CLOSED = 1.0
GRIP_SETTLE_OBS = 6   # observations to hold after a jaw command before moving on


def pick_targets(detections, per_side):
    """Tray targets spread from image-left to image-right, then every spare cap."""
    tray = sorted((d for d in detections if d['class_name'] in TRAY), key=lambda d: d['pixel'][0])
    if tray:
        index = np.unique(np.linspace(0, len(tray) - 1, max(1, 2 * per_side + 1)).round().astype(int))
        tray = [tray[i] for i in index]
    spares = [d for d in detections if d['class_name'] == 'table_cap']
    return tray + spares


class ProbePolicy:
    CAMERAS = ['table', 'side', 'wrist']
    DEPTH = ['table']
    IMAGE_SIZE = (1920, 1080)
    IMAGE_FORMAT = 'png'
    ACTION = 'tcp'
    PREFETCH = False

    def __init__(self, args):
        self.args = args
        self.detector = RemediationVisionDetector(
            args.checkpoint, confidence=args.confidence, device=args.device, imgsz=args.imgsz,
            work_surface_z_m=args.work_surface_z_m, rim_z=args.vial_rim_z_m)
        self.trace_dir = out_dir('probe', args.trace_dir) / f'{dt.datetime.now():%Y-%m-%d_%H-%M-%S}'
        for name in ('tablecam_raw', 'side_raw', 'wrist_raw', 'depth_raw'):
            (self.trace_dir / name).mkdir(parents=True, exist_ok=True)
        (self.trace_dir / 'probe_config.json').write_text(json.dumps(vars(args), indent=2, default=str))
        self.trace = open(self.trace_dir / 'trace.jsonl', 'a')
        self.frames = 0
        print(f'[probe] trace: {self.trace_dir}', flush=True)
        self.reset()

    def reset(self):
        self.plan, self.goal, self.dwell = None, None, 0

    # ---- motion ---------------------------------------------------------
    @staticmethod
    def _tcp(obs):
        tcp = obs.get('tcp_pose')
        if tcp is None and str((obs.get('trial') or {}).get('trial_id', '')).startswith('mock-'):
            return np.asarray(MOCK_START_TCP[:6], dtype=float)
        if tcp is None:
            raise ValueError('the probe needs the current tcp_pose')
        return np.asarray(tcp, dtype=float)

    def _waypoint(self, xyz_mm, r_deg, grip, record=None, settle=0):
        wpr = (DOWN_WPR_DEG[0], DOWN_WPR_DEG[1], float(r_deg))
        p = np.asarray(xyz_mm, dtype=float)
        if np.any(p < UF0_LIMITS_MM[:, 0]) or np.any(p > UF0_LIMITS_MM[:, 1]):
            raise ValueError(f'probe waypoint {p.round(1)} outside the UF0 workspace')
        return {'goal': [*p.tolist(), *wpr, float(grip)], 'record': record, 'settle': settle}

    def _build_plan(self, detections):
        plan = []
        for target in pick_targets(detections, self.args.per_side):
            try:
                plan += self._target_steps(target)
            except ValueError as error:
                print(f'[probe] skipping {target["class_name"]} at {np.round(target["pixel"])}: {error}', flush=True)
        for step in plan:
            record = step['record']
            if record and record['hover_mm'] < MIN_CLEARANCE_MM and not record.get('holding'):
                raise ValueError(f'hover {record["hover_mm"]} mm is below the {MIN_CLEARANCE_MM} mm clearance')
        return plan

    def _target_steps(self, target):
        """Hover (and optionally grasp) waypoints for one target; ValueError if any is unreachable."""
        args, plan = self.args, []
        home = [*INSPECTION_UF0_MM.tolist()]
        uf0 = base_to_uf0(np.asarray(target['base_pos_m'], dtype=float))
        for r_deg in args.r_deg:
            plan.append(self._waypoint([uf0[0], uf0[1], TRANSIT_UF0_Z_MM], r_deg, GRIPPER_OPEN))
            for hover in sorted(args.hover_mm, reverse=True):
                plan.append(self._waypoint([uf0[0], uf0[1], uf0[2] + hover], r_deg, GRIPPER_OPEN,
                                           {'target': target, 'hover_mm': hover, 'r_deg': r_deg}))
            plan.append(self._waypoint([uf0[0], uf0[1], TRANSIT_UF0_Z_MM], r_deg, GRIPPER_OPEN))
        if args.grasp_check and target['class_name'] == 'table_cap':
            pick = uf0[2] + args.pick_offset_mm
            plan += [self._waypoint([uf0[0], uf0[1], pick], 0, GRIPPER_OPEN),
                     self._waypoint([uf0[0], uf0[1], pick], 0, GRIPPER_CLOSED, settle=GRIP_SETTLE_OBS),
                     self._waypoint([uf0[0], uf0[1], pick + 60], 0, GRIPPER_CLOSED,
                                    {'target': target, 'hover_mm': 60, 'r_deg': 0, 'holding': True}),
                     self._waypoint([uf0[0], uf0[1], pick], 0, GRIPPER_CLOSED, settle=GRIP_SETTLE_OBS),
                     self._waypoint([uf0[0], uf0[1], pick], 0, GRIPPER_OPEN, settle=GRIP_SETTLE_OBS),
                     self._waypoint([uf0[0], uf0[1], TRANSIT_UF0_Z_MM], 0, GRIPPER_OPEN)]
            args.grasp_check = False  # one supervised grasp per run
        plan.append(self._waypoint(home, 0, GRIPPER_OPEN))
        return plan

    def _reached(self, obs, goal):
        current = self._tcp(obs)
        relative = wpr_to_rotation(current[3:6]).T @ wpr_to_rotation(goal[3:6])
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        return np.linalg.norm(current[:3] - goal[:3]) <= ARRIVAL_TRANSLATION_MM and angle <= ARRIVAL_ORIENTATION_DEG

    def _chunk(self, obs, goal):
        current = self._tcp(obs)
        delta = np.asarray(goal[:3]) - current[:3]
        turn = (np.asarray(goal[3:6]) - current[3:6] + 180) % 360 - 180
        rows = max(1, int(np.ceil(np.linalg.norm(delta) / MAX_ROW_TRANSLATION_MM)),
                   int(np.ceil(np.max(abs(turn)) / MAX_ROW_WPR_DEG)))
        return [[*(current[:3] + f * delta), *(current[3:6] + f * turn), goal[6]]
                for f in np.linspace(1 / rows, 1, rows)]

    # ---- recording ------------------------------------------------------
    def _save(self, obs, record, phase):
        name = f'{self.frames:04d}.png'
        self.frames += 1
        for camera, folder in (('table', 'tablecam_raw'), ('side', 'side_raw'), ('wrist', 'wrist_raw')):
            image = (obs.get('images') or {}).get(camera)
            if image is not None:
                cv2.imwrite(str(self.trace_dir / folder / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        depth = (obs.get('depth') or {}).get('table')
        if depth is not None:
            cv2.imwrite(str(self.trace_dir / 'depth_raw' / name), np.asarray(depth))
        target = (record or {}).get('target') or {}
        entry = {'t': obs.get('t'), 'phase': phase, 'frame': name,
                 'tcp_pose': None if obs.get('tcp_pose') is None else list(map(float, obs['tcp_pose'])),
                 'tcp_pose_before': None if obs.get('tcp_pose') is None else list(map(float, obs['tcp_pose'])),
                 'joints': None if obs.get('joints') is None else list(map(float, obs['joints'])),
                 'commanded': None if self.goal is None else self.goal['goal'],
                 'hover_mm': (record or {}).get('hover_mm'), 'r_deg': (record or {}).get('r_deg'),
                 'holding': bool((record or {}).get('holding')),
                 'target': {k: target.get(k) for k in ('class_name', 'pixel', 'bbox_xyxy', 'base_pos_m',
                                                       'plane_base_z_m', 'confidence')} if target else None,
                 'detections': record.get('detections') if record and 'detections' in record else None}
        self.trace.write(json.dumps(entry) + '\n')
        self.trace.flush()

    # ---- policy ---------------------------------------------------------
    def act(self, obs):
        current = self._tcp(obs)
        if self.plan is None:
            home = [*INSPECTION_UF0_MM.tolist(), *DOWN_WPR_DEG, GRIPPER_OPEN]
            if np.linalg.norm(current[:3] - INSPECTION_UF0_MM) > 2.0:
                return self._chunk(obs, home)
            image = (obs.get('images') or {}).get('table')
            detections = self.detector.detect(image, timestamp=obs.get('t')) if image is not None else []
            self._save(obs, {'detections': detections}, 'inspect')
            self.plan = self._build_plan(detections)
            print(f'[probe] {len(detections)} detections -> {len(self.plan)} waypoints', flush=True)
        if self.goal is not None and not self._reached(obs, self.goal['goal']):
            return self._chunk(obs, self.goal['goal'])
        if self.goal is not None:
            # Hold still: jaw commands settle; hovers settle 2 frames, then record --dwell frames.
            record = self.goal['record']
            settle = max(self.goal['settle'], 2 if record else 0)
            hold = settle + (self.args.dwell if record else 0)
            self.dwell += 1
            if record and self.dwell > settle:
                self._save(obs, record, 'hover')
            if self.dwell < hold:
                return [[*current, self.goal['goal'][6]]]
        self.dwell = 0
        if not self.plan:
            self.goal = None
            return [[*current, GRIPPER_OPEN]]
        self.goal = self.plan.pop(0)
        return self._chunk(obs, self.goal['goal'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--confidence', type=float, default=0.70)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--device', default='0')
    parser.add_argument('--work-surface-z-m', type=float, default=-0.0051)
    parser.add_argument('--vial-rim-z-m', type=float)
    parser.add_argument('--hover-mm', type=float, nargs='+', default=[40.0, 80.0],
                        help='heights above the plane the detector used; >= 20 mm')
    parser.add_argument('--r-deg', type=float, nargs='+', default=[0.0],
                        help='wrist rolls to test; 0 and -90 are the policy\'s two postures')
    parser.add_argument('--per-side', type=int, default=1, help='tray targets on each side of the centre one')
    parser.add_argument('--dwell', type=int, default=3, help='frames recorded at each hover')
    parser.add_argument('--grasp-check', action='store_true', help='SUPERVISED CONTACT: pick, lift, return one spare cap')
    parser.add_argument('--pick-offset-mm', type=float, default=-2.0)
    parser.add_argument('--trace-dir')
    parser.add_argument('--url', default='https://hackathon.sidclab.org')
    parser.add_argument('--ca')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not Path(args.checkpoint).is_file():
        sys.exit(f'checkpoint not found: {args.checkpoint}')
    from run import start_mock, status
    if args.mock:
        url, _ = start_mock(1, 30.0, 'alignment probe')
        token = 'mock'
    elif os.environ.get('PHYSICAL_CELL_TOKEN'):
        url, token = args.url, os.environ['PHYSICAL_CELL_TOKEN']
    else:
        st = status(args.url, args.ca, wait=True)
        if st['status'] != 'approved':
            sys.exit(f'request {st["request_id"]} was {st["status"]}')
        url, token = json.loads((CLIENT / '.request.json').read_text()).get('url', args.url), st['token']
    run_all(CellClient(url, token, args.ca), ProbePolicy(args))


if __name__ == '__main__':
    main()
