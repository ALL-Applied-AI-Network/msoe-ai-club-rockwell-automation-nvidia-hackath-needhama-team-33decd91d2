"""Observation-driven remediation state machine, independent of Isaac and HTTP.

Contact actions require measured custom-tool offsets and qualified localization.
The current image front end intentionally does not assert contact qualification.
"""
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.remediation.geometry import base_to_uf0, transit_base_z, uf0_to_base


@dataclass(frozen=True)
class ContactTool:
    """Signed fingertip offsets relative to the observed feature, from tool CAD/calibration."""
    pick_offset_m: float
    place_offset_m: float
    press_offset_m: float
    validated: bool = False


def load_validated_contact_tool(path=Path(__file__).with_name('config') / 'custom_finger_contract.json'):
    """Load measured cap-contact offsets only after the custom tool passes every gate."""
    data=json.loads(Path(path).read_text())
    measurements=data.get('required_measurements_m',{})
    validation=data.get('validation',{})
    needed=('cap_pick_offset','cap_place_offset','cap_press_offset','finger_opening_min',
            'finger_opening_max','neighbor_clearance')
    values=[measurements.get(key) for key in needed]
    approved=bool(validation.get('approved_for_contact')) and all(bool(v) for v in validation.values())
    if (data.get('status')!='validated' or not approved or
            not all(isinstance(value,(int,float)) and np.isfinite(value) for value in values)):
        raise ValueError('Custom finger contract is incomplete or has not passed contact validation')
    if measurements['finger_opening_min']>=measurements['finger_opening_max']:
        raise ValueError('Custom finger opening range is invalid')
    return ContactTool(measurements['cap_pick_offset'],measurements['cap_place_offset'],
                       measurements['cap_press_offset'],validated=True)


def checked_action(base_position, closed):
    p=base_to_uf0(base_position)
    limits=np.array([[150,650],[-450,450],[-334,300]])
    if not np.isfinite(p).all() or np.any(p<limits[:,0]) or np.any(p>limits[:,1]):
        raise ValueError('Target outside calibrated workspace; refusing to clip it')
    return [*map(float,p),-180.,0.,0.,float(closed)]


class RemediationController:
    def __init__(self, tool, max_attempts=2):
        if not tool.validated: raise ValueError('Custom finger geometry/contact offsets have not been validated')
        self.tool=tool; self.max_attempts=max_attempts; self.reset()

    def reset(self):
        self.queue=deque(); self.pending=None; self.attempts={}; self.events=[]
        self.state='inspect'; self.confirmations=0; self.verify_frames=0

    def hold(self,obs):
        tcp=obs.get('tcp_pose')
        if tcp is None or len(tcp)!=6 or not np.isfinite(tcp).all():
            raise ValueError('Current TCP pose required to hold safely')
        closed=float(obs['joints'][6])
        return [*map(float,tcp),closed]

    def _add(self,p,closed):
        self.queue.append(checked_action(p,closed))

    def _route(self,obs,target,spare=None):
        # Build all waypoints through ``_add`` before returning a command.  If
        # one is out of workspace, remove the earlier waypoints too: a target
        # rejection must never leave a partial route that can move the arm.
        queue_start=len(self.queue)
        try:
            self._route_checked(obs,target,spare)
        except ValueError:
            while len(self.queue)>queue_start:
                self.queue.pop()
            raise

    def _route_checked(self,obs,target,spare=None):
        current=uf0_to_base(np.array(obs['tcp_pose'][:3]))
        height=max(transit_base_z(),current[2])
        p=np.array(target['base_pos_m'],dtype=float)
        # First raise vertically, then travel laterally, then descend at the target.
        self._add([current[0],current[1],height],False)
        if spare is None:
            self._add([p[0],p[1],height],True)
            self._add(p+[0,0,.025],True)
            self._add(p+[0,0,self.tool.press_offset_m],True)
            self._add(p+[0,0,self.tool.press_offset_m],False)
            self._add([p[0],p[1],height],False)
            self.state='repair'
        else:
            s=np.array(spare['base_pos_m'],dtype=float)
            self._add([s[0],s[1],height],False)
            self._add(s+[0,0,.025],False)
            self._add(s+[0,0,self.tool.pick_offset_m],False)
            self._add(s+[0,0,self.tool.pick_offset_m],True)
            self._add([s[0],s[1],height],True)
            self.state='pickup'
        self.pending={'target':target,'spare':spare,'height':height}

    def act(self,obs,detections):
        if self.queue: return self.queue.popleft()
        if self.state=='failed': return self.hold(obs)
        if self.state=='pickup':
            # Supplied by image-based grasp verification, never a simulator object pose.
            if obs.get('grasp_verified') is not True:
                self.state='failed'; self.events.append({'event':'pickup_unverified'})
                return self.hold(obs)
            p=np.array(self.pending['target']['base_pos_m'])
            self._add([p[0],p[1],self.pending['height']],True)
            self._add(p+[0,0,.025],True)
            self._add(p+[0,0,self.tool.place_offset_m],True)
            self._add(p+[0,0,self.tool.place_offset_m],False)
            self._add([p[0],p[1],self.pending['height']],False)
            self.state='repair'
            return self.queue.popleft()
        if self.state=='repair':
            self.state='verify'; self.confirmations=0; self.verify_frames=0
        if self.state=='verify':
            target=np.array(self.pending['target']['base_pos_m'])
            good=any(d['class_name']=='good_cap' and d['confidence']>=.9
                     and np.linalg.norm(np.array(d['base_pos_m'])[:2]-target[:2])<.003 for d in detections)
            self.confirmations=self.confirmations+1 if good else 0
            self.verify_frames+=1
            if self.confirmations>=2:
                self.events.append({'event':'visually_verified_repair','target':target.tolist()})
                self.state='inspect'; self.pending=None
            elif self.verify_frames>=5:
                self.events.append({'event':'repair_not_verified','target':target.tolist()})
                self.state='inspect'; self.pending=None
            return self.hold(obs)
        # Full-contact motion is only allowed from the calibrated RGB-D point.
        # A plane-projected RGB box can be useful for inspection, but the real
        # camera and depth image have different optics and it must never drive
        # a physical press or grasp when that correspondence was unavailable.
        eligible=[d for d in detections if (d.get('contact_qualified') and
                  d.get('depth_available') is True and d['confidence']>=.9)]
        defects=[d for d in eligible if d['class_name'] in ['crooked_cap','missing_cap']]
        spares=[d for d in eligible if d['class_name']=='table_cap']
        for target in defects:
            key=tuple(np.round(np.array(target['base_pos_m'])[:2]/.005).astype(int))
            if self.attempts.get(key,0)>=self.max_attempts: continue
            spare=None
            if target['class_name']=='missing_cap':
                if not spares: continue
                spare=min(spares,key=lambda s:np.linalg.norm(np.array(s['base_pos_m'])-target['base_pos_m']))
            try:
                self._route(obs,target,spare)
            except ValueError as exc:
                # A false positive can project outside the physical robot's
                # calibrated workspace.  Reject that candidate and keep the
                # client alive; do not turn a safety check into a crashed run.
                self.attempts[key]=self.max_attempts
                self.events.append({'event':'target_rejected',
                                    'reason':str(exc),
                                    'target':np.asarray(target['base_pos_m'],dtype=float).tolist()})
                continue
            self.attempts[key]=self.attempts.get(key,0)+1
            return self.queue.popleft()
        # No detection is never proof that the tray is complete.
        self.events.append({'event':'hold','reason':'no_qualified_repair_target'})
        return self.hold(obs)
