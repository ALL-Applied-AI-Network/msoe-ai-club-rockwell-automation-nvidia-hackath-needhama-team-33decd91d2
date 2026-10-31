"""Canonical remediation cell. Import only after SimulationApp has started.

The supplied robot starter is read-only. All task geometry is authored here.
Scene coordinates are permitted for construction/evaluation, never motion targets.
"""
import math
from pathlib import Path
import sys

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

from src.remediation.geometry import CROOKED_TILT_RANGE_DEG, SPEC, cap_vertices, load_layout, leaning_cap_pose, rotation, spawn_height

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


def material(stage, name, color, opacity=1., roughness=.35, metallic=0., ior=None):
    mat = UsdShade.Material.Define(stage, '/World/RemediationLooks/' + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(metallic))
    if ior is not None:
        shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(float(ior))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def clear_vial_material(stage):
    """Use the same physically based clear-polymer appearance as YOLO training."""
    mat = UsdShade.Material(stage.GetPrimAtPath('/World/Looks/vial_glass'))
    if not mat.GetPrim().IsValid():
        raise RuntimeError('Starter scene is missing /World/Looks/vial_glass')
    shader = stage.GetPrimAtPath('/World/Looks/vial_glass/Shader')
    shader.GetAttribute('inputs:glass_ior').Set(1.57)
    shader.GetAttribute('inputs:frosting_roughness').Set(0.035)
    shader.GetAttribute('inputs:glass_color').Set(Gf.Vec3f(.985, .995, 1.0))
    return mat


def textured_tray_aluminum(stage):
    """Portable satin aluminum that cannot fall back to a mirror finish.

    The former OmniPBR material was fully metallic at roughness 0.30 and gave
    its roughness texture 100% influence.  Under the bright TableCam lighting,
    dark texels (or a failed texture lookup) produced mirror-like plate cells.
    A fixed high roughness keeps the plate recognisably aluminum and consistent
    across cluster nodes without changing any fixture geometry or collision.
    """
    return material(stage, 'Fixture', (.68, .70, .72),
                    roughness=.55, metallic=.90)


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
    cap_geometry(stage,path,mat,cavity_mat)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_mat,materialPurpose='physics')
    _cap_dynamics(prim,dynamic)
    return prim


def cap_geometry(stage,path,mat,cavity_mat):
    """Child geometry of a cap rooted at its CAD origin (flange underside on the axis)."""
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


def _cap_dynamics(prim,dynamic):
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.005)
        phys=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        phys.CreateLinearDampingAttr(1.2); phys.CreateAngularDampingAttr(2.)
        phys.CreateSolverPositionIterationCountAttr(64); phys.CreateSolverVelocityIterationCountAttr(16)
        phys.CreateEnableCCDAttr(True)
        phys.CreateMaxDepenetrationVelocityAttr().Set(0.5)
        # Never sleep: a 5 g cap landing gently can otherwise freeze in a pose that
        # cannot actually support it (e.g. resting on only two rim points).
        phys.CreateSleepThresholdAttr(0.); phys.CreateStabilizationThresholdAttr(0.)


def build_scene(stage, seed=42, crooked=1, missing=1, table_caps=1, randomize=True,
                paper_color=None, tray_center_xy=None, crooked_tilts=None, vials=30,
                spec=None, pattern=None, missing_slots=None, crooked_slots=None, empty_slots=None):
    import cell
    from src.remediation.vial_patterns import ALL_SLOTS, resolve_tray_pattern
    current_spec = spec if spec is not None else SPEC
    pattern_spec = None
    if pattern is not None or missing_slots is not None or crooked_slots is not None or empty_slots is not None:
        pattern_spec = resolve_tray_pattern(
            pattern_name=pattern or "custom",
            missing_slots=missing_slots,
            crooked_slots=crooked_slots,
            empty_slots=empty_slots,
            table_caps=table_caps if table_caps != 1 else None,
        )
    elif min(crooked,missing,table_caps)<0 or not 0<crooked+missing<=vials<=30 or table_caps>10:
        raise ValueError('Invalid scene inventory (at most 30 vials, at most 10 separated spares)')
    cell.build(stage,with_objects=False)
    if paper_color is not None:
        # A thin, non-authoritative colored paper sheet under the spare-cap
        # area creates background variation without changing the tray geometry.
        # Its top is just above the nominal table surface so caps settle on it.
        paper = material(stage, 'TableCapPaper', paper_color, roughness=.72)
        # Positive table Y is the right-hand side in the calibrated TableCam
        # image.  Match the physical spare-cap staging area there.
        # Lift the rendered sheet 0.5 mm beyond its physical half-thickness.
        # The starter cell has a separate white mat coplanar with the nominal
        # work surface; the extra visual clearance prevents depth fighting.
        box(stage, '/World/TableCapPaper', (.37, .315, current_spec.surface_world_z + .0006),
            (.27, .12, .0002), paper)
    rng=np.random.default_rng(seed)
    camera_variation=randomize_table_camera(stage,rng) if randomize else None
    cap_color=np.clip(np.array((.96,.76,.04))+rng.normal(0,.035,3),0,1)
    # Stipulated materials: semi-gloss plastic caps, clear plastic vials,
    # a black base/standoffs, and a silver perforated plate.
    yellow=material(stage,'Cap',cap_color,roughness=.30)
    cavity=material(stage,'CapCavity',np.clip(cap_color*np.array((.52,.48,.22)),0,1),roughness=.42)
    glass=clear_vial_material(stage)
    metal=textured_tray_aluminum(stage)
    black=material(stage,'TrayBase',(.025,.028,.032),roughness=.42)
    physics_mat=UsdShade.Material.Define(stage,'/World/RemediationLooks/Contact')
    pm=UsdPhysics.MaterialAPI.Apply(physics_mat.GetPrim())
    pm.CreateStaticFrictionAttr(current_spec.friction_static); pm.CreateDynamicFrictionAttr(current_spec.friction_dynamic); pm.CreateRestitutionAttr(0.)
    layout=load_layout()
    # The reachable fixture pose is 80 mm closer to the robot than the legacy
    # scene. This is scene-construction data only, never a robot target.
    if tray_center_xy is None:
        tray_center_xy=(.305,.035)
    center=np.array([float(tray_center_xy[0]),float(tray_center_xy[1]),current_spec.surface_world_z])
    yaw=float(rng.uniform(-2,2)) if randomize else 0.
    if randomize: center[:2]+=rng.uniform(-.008,.008,2)
    R=rotation(0,yaw)
    root=UsdGeom.Xform.Define(stage,'/World/RemediationTray')
    root.AddTranslateOp().Set(Gf.Vec3d(*center)); root.AddRotateZOp().Set(yaw)
    # Fabricated remediation tray: a 10 mm black bottom plate, four black
    # 80 mm standoffs, and a 3 mm silver plate with the vial apertures.
    # Dataset 1 contributes the XY slot layout only; its old support geometry
    # is not reused as a dimensional authority.
    top_source=layout['parts']['tray_good_top']
    tray_xy=top_source['size_m'][:2]
    base_t, clearance, top_t = current_spec.tray_base_thickness, current_spec.tray_top_clearance, current_spec.tray_top_thickness
    top_z=base_t + clearance + top_t/2
    box(stage,str(root.GetPath())+'/BlackBottomPlate',(0,0,base_t/2),(tray_xy[0],tray_xy[1],base_t),black)
    for ix, x in enumerate((-tray_xy[0]/2+.012, tray_xy[0]/2-.012)):
        for iy, y in enumerate((-tray_xy[1]/2+.012, tray_xy[1]/2-.012)):
            # USD prim names cannot contain a leading sign in an identifier.
            box(stage,str(root.GetPath())+f'/BlackStandoff_{ix}_{iy}',
                (x,y,base_t+clearance/2),(.010,.010,clearance),black)
    slots=np.array(list(layout['slots_m'].values())).reshape(6,5,3)
    xs=slots[:,:,0].mean(axis=1); ys=slots[:,:,1].mean(axis=0)
    xe=np.r_[-tray_xy[0]/2,(xs[:-1]+xs[1:])/2,tray_xy[0]/2]
    ye=np.r_[-tray_xy[1]/2,(ys[:-1]+ys[1:])/2,tray_xy[1]/2]
    if pattern_spec is not None:
        empty_slot_names = set(pattern_spec.empty_slots)
        loaded_ids = {int(s.replace('G', '')) for s in ALL_SLOTS if s not in empty_slot_names}
        crooked_ids = {int(s.replace('G', '')) for s in pattern_spec.crooked_slots}
        missing_ids = {int(s.replace('G', '')) for s in pattern_spec.missing_slots}
        effective_table_caps = pattern_spec.table_caps
        effective_vials = len(loaded_ids)
    else:
        # With fewer than 30 vials the remaining apertures are left empty (plate only).
        shuffled=rng.permutation(30)+1
        loaded_ids=set(shuffled[:vials])
        crooked_ids=set(shuffled[:crooked]); missing_ids=set(shuffled[crooked:crooked+missing])
        effective_table_caps = table_caps
        effective_vials = vials
    inventory={'seed':seed,'tray_center_world_m':center.tolist(),'tray_yaw_deg':yaw,
               'camera_variation':camera_variation,'vials':effective_vials,'slots':{},'caps':[],
               'pattern': pattern_spec.to_dict() if pattern_spec is not None else None}
    for index in range(1,31):
        slot=f'G{index:02d}'; p=np.array(layout['slots_m'][slot]); p[2]=base_t; row,col=divmod(index-1,5)
        perforated_cell(stage,str(root.GetPath())+'/'+slot+'_plate',p[:2],(xe[row],xe[row+1],ye[col],ye[col+1]),top_z-top_t/2,top_z+top_t/2,metal)
        if index not in loaded_ids: continue
        tube=UsdGeom.Xform.Define(stage,str(root.GetPath())+'/'+slot+'_tube')
        tube.AddTranslateOp().Set(Gf.Vec3d(*p))
        annulus(stage,str(tube.GetPath())+'/Wall',current_spec.vial_outer_radius,current_spec.vial_outer_radius,current_spec.vial_inner_radius,0,current_spec.vial_height,glass)
        bottom=UsdGeom.Cylinder.Define(stage,str(tube.GetPath())+'/Bottom')
        bottom.CreateRadiusAttr(current_spec.vial_outer_radius); bottom.CreateHeightAttr(current_spec.vial_bottom)
        bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,current_spec.vial_bottom/2)); collision(bottom.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(bottom.GetPrim()).Bind(glass)
        rim=R@p+center+np.array([0,0,current_spec.vial_height])
        inventory['slots'][slot]={'rim_world_m':rim.tolist(),'tube_path':str(tube.GetPath()),'requested':'missing_cap' if index in missing_ids else 'crooked_cap' if index in crooked_ids else 'good_cap'}
        if index in missing_ids: continue
        tilted=index in crooked_ids
        if tilted:
            # Leaning on the rim at a random tilt from the range that holds in PhysX,
            # or at an explicit tilt (one per crooked cap, in slot order) for sweeps.
            k=sum(1 for c in inventory['caps'] if c['dynamic'])
            tilt=float(crooked_tilts[k]) if crooked_tilts is not None else float(rng.uniform(*CROOKED_TILT_RANGE_DEG))
            pos,cap_R=leaning_cap_pose(rim,tilt,float(rng.uniform(0,360)))
        else:
            cap_R=np.eye(3); pos=rim.copy(); pos[2]=rim[2]+.0002
        path='/World/RemediationCaps/'+slot
        make_cap(stage,path,pos,cap_R,yellow,cavity,physics_mat,dynamic=tilted)
        inventory['caps'].append({'path':path,'slot':slot,'dynamic':tilted,'requested':'crooked_cap' if tilted else 'good_cap',
                                  **({'spawn_tilt_deg':tilt} if tilted else {})})
    for i in range(effective_table_caps):
        # Three per row: a fourth column (x 0.415 m and beyond) is at the edge of the arm's reach.
        row,col=divmod(i,3)
        # Stage loose replacement caps on the right from TableCam's viewpoint.
        pos=np.array([.28+col*.045,.325-row*.043,0.])
        if randomize: pos[:2]+=rng.uniform(-.001,.001,2)
        cap_R=rotation(0,float(rng.uniform(0,360))); pos[2]=spawn_height(current_spec.surface_world_z,cap_R)
        path=f'/World/RemediationCaps/Table_{i:02d}'
        make_cap(stage,path,pos,cap_R,yellow,cavity,physics_mat)
        inventory['caps'].append({'path':path,'slot':None,'dynamic':True,'requested':'table_cap'})
    return inventory
