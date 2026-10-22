"""Canonical remediation cell. Import only after SimulationApp has started.

The supplied robot starter is read-only. All task geometry is authored here.
Scene coordinates are permitted for construction/evaluation, never motion targets.
"""
import math
from pathlib import Path
import sys

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

from src.remediation.geometry import SPEC, cap_vertices, load_layout, rotation, spawn_height

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'shared/data/fanuc_er4ia_sim'))


def randomize_table_camera(stage, rng):
    """Perturb the synthetic camera within a measured-mount-sized envelope."""
    import cell
    import scene_config as C
    eye=np.asarray(C.TABLE_CAM['eye'],dtype=float)+rng.normal(0,[.008,.008,.012])
    target=np.asarray(C.TABLE_CAM['look_at'],dtype=float)+rng.normal(0,[.005,.005,.002])
    up=np.asarray(C.TABLE_CAM['up'],dtype=float)
    view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(*up))
    ops=UsdGeom.Xformable(stage.GetPrimAtPath(cell.TABLE_CAM_PATH)).GetOrderedXformOps()
    if len(ops)!=1: raise RuntimeError('Unexpected TableCam transform stack')
    ops[0].Set(view.GetInverse())
    return {'eye_world_m':eye.tolist(),'look_at_world_m':target.tolist(),'translation_sigma_m':[.008,.008,.012]}


def material(stage, name, color, opacity=1., roughness=.35):
    mat = UsdShade.Material.Define(stage, '/World/RemediationLooks/' + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def collision(prim, convex=False):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull' if convex else 'none')
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(SPEC.contact_offset)
    api.CreateRestOffsetAttr(0.)


def mesh(stage, path, points, faces, mat, convex=False, with_collision=True):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr([Gf.Vec3f(*map(float,p)) for p in points])
    geom.CreateFaceVertexCountsAttr([len(f) for f in faces])
    geom.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    geom.CreateSubdivisionSchemeAttr('none')
    geom.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    if with_collision:
        collision(geom.GetPrim(), convex)
    return geom.GetPrim()


def annulus(stage, path, outer_bottom, outer_top, inner, z0, z1, mat, segments=96):
    a = np.arange(segments)*2*np.pi/segments
    rings = [(outer_bottom,z0),(outer_top,z1),(inner,z1),(inner,z0)]
    points = np.concatenate([np.column_stack([r*np.cos(a),r*np.sin(a),np.full(segments,z)]) for r,z in rings])
    faces=[]
    for i in range(segments):
        j=(i+1)%segments
        for k in range(4):
            nxt=(k+1)%4
            faces.append([k*segments+i,k*segments+j,nxt*segments+j,nxt*segments+i])
    return mesh(stage,path,points,faces,mat)


def box(stage, path, center, size, mat):
    b = UsdGeom.Cube.Define(stage,path)
    b.CreateSizeAttr(1.)
    b.AddTranslateOp().Set(Gf.Vec3d(*map(float,center)))
    b.AddScaleOp().Set(Gf.Vec3f(*map(float,size)))
    UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(mat)
    collision(b.GetPrim())
    return b.GetPrim()


def perforated_cell(stage,path,center,bounds,z0,z1,mat):
    """Closed plate tile with a real circular aperture, no solid-box proxy."""
    cx,cy=center; xmin,xmax,ymin,ymax=bounds
    angles=list(np.arange(64)*2*np.pi/64)
    angles.extend(math.atan2(y-cy,x-cx)%(2*np.pi) for x in [xmin,xmax] for y in [ymin,ymax])
    angles=sorted(set(angles)); n=len(angles)
    outer=[]; inner=[]
    for a in angles:
        dx,dy=math.cos(a),math.sin(a)
        tx=((xmax if dx>0 else xmin)-cx)/dx if abs(dx)>1e-9 else float('inf')
        ty=((ymax if dy>0 else ymin)-cy)/dy if abs(dy)>1e-9 else float('inf')
        d=min(tx,ty)
        outer.append((cx+d*dx,cy+d*dy)); inner.append((cx+SPEC.aperture_radius*dx,cy+SPEC.aperture_radius*dy))
    points=[(x,y,z) for ring,z in [(outer,z0),(outer,z1),(inner,z1),(inner,z0)] for x,y in ring]
    faces=[]
    for i in range(n):
        j=(i+1)%n
        for k in range(4): faces.append([k*n+i,k*n+j,((k+1)%4)*n+j,((k+1)%4)*n+i])
    mesh(stage,path,points,faces,mat)


def make_cap(stage,path,position,R,mat,cavity_mat,physics_mat,dynamic=True):
    prim=UsdGeom.Xform.Define(stage,path).GetPrim()
    xf=UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)))
    # USD matrices use row vectors.
    quat=Gf.Matrix3d(*np.asarray(R).T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    xf.AddOrientOp().Set(Gf.Quatf(quat))
    # Hollow tapered plug and flange.  The cavity is deliberately open through
    # the top flange: it is a uniform-wall sleeve, not a solid-topped stopper.
    # Convex sectors preserve that void for dynamic PhysX contact.
    # The darkened, non-colliding liner lies 0.03 mm inside the physical inner
    # wall.  It makes the open underside legible in side views without changing
    # the cap's measured wall thickness or its collision geometry.
    n=48
    faces=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    for i in range(n):
        a,b=np.array([i,i+1])*2*np.pi/n
        flange_points=[]
        for z in (0,SPEC.flange_thickness):
            flange_points.extend([(r*math.cos(t),r*math.sin(t),z) for r,t in [
                (SPEC.flange_radius,a),(SPEC.flange_radius,b),
                (SPEC.flange_inner_radius,b),(SPEC.flange_inner_radius,a)]])
        mesh(stage,path+f'/Flange_{i:02d}',flange_points,faces,mat,convex=True)
        points=[]
        for z,outer in [(-SPEC.plug_depth,SPEC.plug_bottom_radius),(0,SPEC.plug_top_radius)]:
            inner=outer-SPEC.plug_wall_thickness
            points.extend([(r*math.cos(t),r*math.sin(t),z) for r,t in [(outer,a),(outer,b),(inner,b),(inner,a)]])
        mesh(stage,path+f'/Plug_{i:02d}',points,faces,mat,convex=True)
        inset=[]
        for z,outer in [(-SPEC.plug_depth+SPEC.plug_bottom_face_thickness,SPEC.plug_bottom_radius),(0,SPEC.plug_top_radius)]:
            inner=outer-SPEC.plug_wall_thickness-0.00003
            inset.extend([(inner*math.cos(t),inner*math.sin(t),z) for t in (a,b)])
        mesh(stage,path+f'/CavityLiner_{i:02d}',inset,[[0,1,3,2]],cavity_mat,with_collision=False)
        flange_inset=SPEC.flange_inner_radius-0.00003
        liner=[(flange_inset*math.cos(t),flange_inset*math.sin(t),h)
               for h in (0,SPEC.flange_thickness) for t in (a,b)]
        mesh(stage,path+f'/FlangeCavityLiner_{i:02d}',liner,[[0,1,3,2]],cavity_mat,with_collision=False)
    # The plug cavity terminates in a flat 2 mm base rather than passing through.
    bottom=UsdGeom.Cylinder.Define(stage,path+'/PlugBottom')
    bottom.CreateRadiusAttr(SPEC.plug_bottom_radius-SPEC.plug_wall_thickness)
    bottom.CreateHeightAttr(SPEC.plug_bottom_face_thickness)
    bottom.CreateAxisAttr('Z')
    bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,-SPEC.plug_depth+SPEC.plug_bottom_face_thickness/2))
    collision(bottom.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(bottom.GetPrim()).Bind(mat)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_mat,materialPurpose='physics')
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.005)
        phys=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        phys.CreateLinearDampingAttr(1.2); phys.CreateAngularDampingAttr(2.)
        phys.CreateSolverPositionIterationCountAttr(32); phys.CreateSolverVelocityIterationCountAttr(8)
        phys.CreateEnableCCDAttr(True)
    return prim


def build_scene(stage, seed=42, crooked=1, missing=1, table_caps=1, randomize=True):
    import cell
    if min(crooked,missing,table_caps)<0 or crooked+missing>30 or table_caps>10:
        raise ValueError('Invalid scene inventory (30 vials, at most 10 separated spares)')
    cell.build(stage,with_objects=False)
    rng=np.random.default_rng(seed)
    camera_variation=randomize_table_camera(stage,rng) if randomize else None
    cap_color=np.clip(np.array((.96,.76,.04))+rng.normal(0,.035,3),0,1)
    yellow=material(stage,'Cap',cap_color,roughness=float(rng.uniform(.22,.48)))
    cavity=material(stage,'CapCavity',np.clip(cap_color*np.array((.52,.48,.22)),0,1),roughness=.58)
    glass=material(stage,'Glass',(.92,.96,1.),.45,roughness=float(rng.uniform(.12,.35)))
    metal=material(stage,'Fixture',np.clip(np.array((.65,.67,.69))+rng.normal(0,.04,3),0,1),roughness=float(rng.uniform(.25,.65)))
    physics_mat=UsdShade.Material.Define(stage,'/World/RemediationLooks/Contact')
    pm=UsdPhysics.MaterialAPI.Apply(physics_mat.GetPrim())
    pm.CreateStaticFrictionAttr(SPEC.friction_static); pm.CreateDynamicFrictionAttr(SPEC.friction_dynamic); pm.CreateRestitutionAttr(0.)
    layout=load_layout()
    center=np.array([.385,.035,SPEC.surface_world_z])
    yaw=float(rng.uniform(-2,2)) if randomize else 0.
    if randomize: center[:2]+=rng.uniform(-.008,.008,2)
    R=rotation(0,yaw)
    root=UsdGeom.Xform.Define(stage,'/World/RemediationTray')
    root.AddTranslateOp().Set(Gf.Vec3d(*center)); root.AddRotateZOp().Set(yaw)
    for name,part in layout['parts'].items():
        if name!='tray_good_top': box(stage,str(root.GetPath())+'/'+name,part['center_m'],part['size_m'],metal)
    top=layout['parts']['tray_good_top']; top_z=top['center_m'][2]
    slots=np.array(list(layout['slots_m'].values())).reshape(6,5,3)
    xs=slots[:,:,0].mean(axis=1); ys=slots[:,:,1].mean(axis=0)
    xe=np.r_[-top['size_m'][0]/2,(xs[:-1]+xs[1:])/2,top['size_m'][0]/2]
    ye=np.r_[-top['size_m'][1]/2,(ys[:-1]+ys[1:])/2,top['size_m'][1]/2]
    shuffled=rng.permutation(30)+1
    crooked_ids=set(shuffled[:crooked]); missing_ids=set(shuffled[crooked:crooked+missing])
    inventory={'seed':seed,'tray_center_world_m':center.tolist(),'tray_yaw_deg':yaw,
               'camera_variation':camera_variation,'slots':{},'caps':[]}
    for index in range(1,31):
        slot=f'G{index:02d}'; p=np.array(layout['slots_m'][slot]); row,col=divmod(index-1,5)
        perforated_cell(stage,str(root.GetPath())+'/'+slot+'_plate',p[:2],(xe[row],xe[row+1],ye[col],ye[col+1]),top_z-top['size_m'][2]/2,top_z+top['size_m'][2]/2,metal)
        tube=UsdGeom.Xform.Define(stage,str(root.GetPath())+'/'+slot+'_tube')
        tube.AddTranslateOp().Set(Gf.Vec3d(*p))
        annulus(stage,str(tube.GetPath())+'/Wall',SPEC.vial_outer_radius,SPEC.vial_outer_radius,SPEC.vial_inner_radius,0,SPEC.vial_height,glass)
        bottom=UsdGeom.Cylinder.Define(stage,str(tube.GetPath())+'/Bottom')
        bottom.CreateRadiusAttr(SPEC.vial_outer_radius); bottom.CreateHeightAttr(SPEC.vial_bottom)
        bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,SPEC.vial_bottom/2)); collision(bottom.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(bottom.GetPrim()).Bind(glass)
        rim=R@p+center+np.array([0,0,SPEC.vial_height])
        inventory['slots'][slot]={'rim_world_m':rim.tolist(),'tube_path':str(tube.GetPath()),'requested':'missing_cap' if index in missing_ids else 'crooked_cap' if index in crooked_ids else 'good_cap'}
        if index in missing_ids: continue
        tilted=index in crooked_ids
        cap_R=rotation(26.,float(rng.uniform(0,360))) if tilted else np.eye(3)
        pos=rim.copy()
        if tilted: pos+=cap_R@np.array([0,-.001,0])
        pos[2]=spawn_height(rim[2],cap_R) if tilted else rim[2]+.0002
        path='/World/RemediationCaps/'+slot
        make_cap(stage,path,pos,cap_R,yellow,cavity,physics_mat,dynamic=tilted)
        inventory['caps'].append({'path':path,'slot':slot,'dynamic':tilted,'requested':'crooked_cap' if tilted else 'good_cap'})
    for i in range(table_caps):
        row,col=divmod(i,5)
        pos=np.array([.28+col*.045,-.255+row*.043,0.])
        if randomize: pos[:2]+=rng.uniform(-.001,.001,2)
        cap_R=rotation(0,float(rng.uniform(0,360))); pos[2]=spawn_height(SPEC.surface_world_z,cap_R)
        path=f'/World/RemediationCaps/Table_{i:02d}'
        make_cap(stage,path,pos,cap_R,yellow,cavity,physics_mat)
        inventory['caps'].append({'path':path,'slot':None,'dynamic':True,'requested':'table_cap'})
    return inventory
