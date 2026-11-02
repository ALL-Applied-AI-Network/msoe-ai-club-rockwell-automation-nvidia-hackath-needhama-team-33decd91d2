"""Labels from visible instances and measured post-settling state (evaluation only)."""
import numpy as np

from src.remediation.geometry import CLASS_NAMES, SPEC


def visible_labels(report, instance_data, projector):
    image=np.asarray(instance_data['data']).squeeze()
    paths=instance_data['info']['idToLabels']
    height,width=image.shape
    labels=[]; rejected=[]

    def object_mask(path):
        ids=[int(key) for key,value in paths.items() if isinstance(value,str) and (value==path or value.startswith(path+'/'))]
        return np.isin(image,ids)

    for obj in report['objects']:
        mask=object_mask(obj['path'])
        yy,xx=np.where(mask)
        if len(xx)<20:
            rejected.append({'path':obj['path'],'reason':'invisible_or_too_small'})
            continue
        bounds=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        labels.append({'class_name':obj['class_name'],'bbox_xyxy':bounds,'visible_pixels':len(xx),'path':obj['path']})

    angles=np.arange(96)*2*np.pi/96
    for slot,obj in report['inventory']['slots'].items():
        if obj['requested']!='missing_cap': continue
        rim=np.array(obj['rim_world_m'])
        points=rim+np.column_stack([SPEC.vial_outer_radius*np.cos(angles),SPEC.vial_outer_radius*np.sin(angles),np.zeros(96)])
        uv=np.array([projector.project(p) for p in points])
        x0,y0=np.floor(uv.min(axis=0)).astype(int); x1,y1=np.ceil(uv.max(axis=0)).astype(int)
        if not (0<=x0<x1<=width and 0<=y0<y1<=height):
            rejected.append({'slot':slot,'reason':'mouth_out_of_frame'}); continue
        mask=object_mask(obj['tube_path'])
        # Rim samples must be visible on the vial's instance, not just the projected mouth.
        visible=0
        for u,v in uv:
            x,y=int(round(u)),int(round(v))
            visible+=bool(mask[max(0,y-2):min(height,y+3),max(0,x-2):min(width,x+3)].any())
        if visible/len(uv)<.65:
            rejected.append({'slot':slot,'reason':'mouth_occluded','visible_rim_fraction':visible/len(uv)}); continue
        labels.append({'class_name':'missing_cap','bbox_xyxy':[int(x0),int(y0),int(x1),int(y1)],'visible_rim_fraction':visible/len(uv),'slot':slot})
    lines=[]
    for label in labels:
        x0,y0,x1,y1=label['bbox_xyxy']
        lines.append(f"{CLASS_NAMES.index(label['class_name'])} {(x0+x1)/2/width:.8f} {(y0+y1)/2/height:.8f} {(x1-x0)/width:.8f} {(y1-y0)/height:.8f}")
    return labels,rejected,'\n'.join(lines)+'\n'
