"""Build, settle, measure and capture a scene; never accept a requested label as truth."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.remediation.geometry import SPEC, cap_state, cap_surface_points, configuration_hash, load_layout


def pose(stage,path):
    from pxr import UsdGeom
    matrix=np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0.))
    return matrix[3,:3],matrix[:3,:3].T


def evaluate(stage,inventory,history):
    reports=[]
    local_points=cap_surface_points()
    for item in inventory['caps']:
        p,R=pose(stage,item['path'])
        rim=inventory['slots'][item['slot']]['rim_world_m'] if item['slot'] else None
        state=cap_state(p,R,rim)
        # Record actual rest-pose drift over the last half second.
        past=history[item['path']]
        drift=max(float(np.linalg.norm(p-q[0])) for q in past)
        angular=max(float(np.degrees(np.arccos(np.clip((np.trace(R@q[1].T)-1)/2,-1,1)))) for q in past)
        vertices=local_points@R.T+p
        penetration=max(0.,SPEC.surface_world_z-float(vertices[:,2].min()))
        supported=False
        if rim is not None:
            # Conservative sampled surface check; retain metric explicitly in the report.
            rim=np.array(rim)
            radial=np.linalg.norm(vertices[:,:2]-rim[:2],axis=1)
            depths=np.minimum.reduce([radial-SPEC.vial_inner_radius,SPEC.vial_outer_radius-radial,rim[2]-vertices[:,2],vertices[:,2]-(rim[2]-SPEC.vial_height)])
            penetration=max(penetration,float(np.maximum(depths,0).max()))
            rim_distance=np.hypot(np.maximum(np.maximum(SPEC.vial_inner_radius-radial,radial-SPEC.vial_outer_radius),0),vertices[:,2]-rim[2])
            supported=bool(rim_distance.min()<.0015)
        else:
            supported=abs(float(vertices[:,2].min())-SPEC.surface_world_z)<.001
        valid=bool(state['class_name']==item['requested'] and drift<.0002 and angular<.25 and penetration<=SPEC.penetration_tolerance and supported)
        reports.append({**item,**state,'position_world_m':p.tolist(),'orientation':R.tolist(),'drift_m':drift,'angular_drift_deg':angular,'sampled_penetration_m':penetration,'supported_near_rim_or_table':supported,'accepted':valid})
    return reports


def run(args):
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'RayTracedLighting'})
    report={'schema_version':1,'accepted':False,'config_sha256':configuration_hash(),'seed':args.seed}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    try:
        import omni.usd
        from isaacsim.core.api import World
        from src.remediation.scene import build_scene
        world=World(stage_units_in_meters=1.,physics_dt=1/120.,rendering_dt=1/30.)
        stage=omni.usd.get_context().get_stage()
        paper_color=tuple(float(x) for x in args.paper_color.split(',')) if args.paper_color else None
        if paper_color is not None and (len(paper_color) != 3 or any(x < 0 or x > 1 for x in paper_color)):
            raise ValueError('--paper-color must be r,g,b with values in [0,1]')
        inventory=build_scene(stage,args.seed,args.crooked,args.missing,args.table_caps,not args.fixed,paper_color)
        import cell
        cell.enable_ccd(world)
        world.reset()
        # Initialize outside the tray view; this is not a remediation target.
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        import scene_config as C
        import ik
        arm=SingleArticulation(C.ROBOT_PRIM); arm.initialize()
        names=list(arm.dof_names)
        indices=[names.index(f'J{i}') for i in range(1,7)]
        limits=np.deg2rad([C.JOINT_LIMITS_DEG[f'J{i}'] for i in range(1,7)])
        initial=np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1,7)])
        q,ok=ik.solve(initial,[.24,.25,.38],[0,0,-1],limits,tool_offset=tuple(np.array(C.TOOL_AXIS)*SPEC.tool_length))
        if not ok: raise RuntimeError('Inspection parking pose is unreachable')
        full=arm.get_joint_positions().copy(); full[indices]=q
        arm.set_joint_positions(full)
        arm.apply_action(ArticulationAction(joint_positions=full))
        history={item['path']:[] for item in inventory['caps']}
        for step in range(args.steps):
            world.step(render=False)
            if step>=args.steps-60:
                for item in inventory['caps']: history[item['path']].append(pose(stage,item['path']))
        measurements=evaluate(stage,inventory,history)
        report.update({'inventory':inventory,'objects':measurements,'specification':asdict(SPEC),'layout':load_layout(),
                       'accepted':all(x['accepted'] for x in measurements),'physics_steps':args.steps,
                       'physics_dt_s':1/120.,'penetration_check':'dense sampled cap surfaces against analytic tube wall/table; not a full swept collision certificate',
                       'hardware':'stock starter fingers during scene qualification; custom grooved fingers require CAD dimensions'})
        from pxr import UsdGeom
        from isaacsim.sensors.camera import Camera
        import cv2
        for name,path in [('table',cell.TABLE_CAM_PATH),('side',cell.SIDE_CAM_PATH)]:
            camera=Camera(path,resolution=(1920,1080)); camera.initialize()
            if name=='table':
                # The Camera depth accessor requires this render product annotator.
                camera.add_distance_to_image_plane_to_frame()
                import omni.replicator.core as rep
                instance=rep.AnnotatorRegistry.get_annotator('instance_id_segmentation',init_params={'colorize':False})
                instance.attach([camera.get_render_product_path()])
            for _ in range(12): world.render()
            rgba=camera.get_rgba()
            if rgba is None or rgba.size==0: raise RuntimeError('Empty '+name+' camera capture')
            if not cv2.imwrite(str(out/(name+'.png')),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Failed image write')
            if name=='table':
                depth=np.asarray(camera.get_depth(device='cpu'),dtype=np.float32)
                if depth.shape[:2] != rgba.shape[:2] or not np.isfinite(depth).any():
                    raise RuntimeError('TableCam depth is missing or misaligned with RGB')
                np.save(out/'table_depth_m.npy',depth)
                cam=UsdGeom.Camera(stage.GetPrimAtPath(path))
                report['camera']={'resolution':[1920,1080],'focal_length_mm':float(cam.GetFocalLengthAttr().Get()),
                                  'horizontal_aperture_mm':float(cam.GetHorizontalApertureAttr().Get()),'vertical_aperture_mm':float(cam.GetVerticalApertureAttr().Get()),
                                  'camera_to_world':np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0.)).tolist(),'table_top_z':SPEC.surface_world_z}
                from backproject import CameraModel
                from src.remediation.labels import visible_labels
                data=instance.get_data()
                labels,rejected,text=visible_labels(report,data,CameraModel(report['camera']))
                report['labels']=labels; report['rejected_labels']=rejected
                report['label_quality_passed']=not rejected
                report['accepted']=report['accepted'] and not rejected
                if report['accepted']: (out/'table.txt').write_text(text)
                # Keep raw masks for independent annotation review; no image editing needed.
                np.savez_compressed(out/'instances.npz',ids=data['data'])
                (out/'instance_paths.json').write_text(json.dumps(data['info']['idToLabels'],indent=2)+'\n')
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        if args.scene_out:
            dest=Path(args.scene_out)
            if not report['accepted']: raise RuntimeError('Refusing to publish an unqualified canonical scene')
            dest.parent.mkdir(parents=True,exist_ok=True)
            stage.GetRootLayer().Export(str(dest))
        print('QUALIFICATION',report['accepted'],[(x['path'],x['class_name'],round(x['tilt_deg'],2),x['accepted']) for x in measurements if x['dynamic']])
    except Exception as exc:
        report['accepted']=False; report['error']=str(exc)
        raise
    finally:
        (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
        app.close()
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--crooked',type=int,default=1); p.add_argument('--missing',type=int,default=1)
    p.add_argument('--table-caps',type=int,default=1); p.add_argument('--steps',type=int,default=360)
    p.add_argument('--paper-color', help='Optional colored paper under the table-cap area, as r,g,b in [0,1]')
    p.add_argument('--fixed',action='store_true'); p.add_argument('--scene-out')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.steps<120: raise SystemExit('At least 120 physics steps required')
    result=run(args)
    if not result['accepted']: raise SystemExit(2)
