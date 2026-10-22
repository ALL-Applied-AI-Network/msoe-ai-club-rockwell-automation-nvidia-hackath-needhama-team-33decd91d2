"""Visual-only first-pass mount for the two custom gripper halves.

The user confirmed that the supplied STEP is one half: local -Z is the gripping
direction and the top, +Y inset face mates to the pneumatic jaw's inner face.
The transform below is therefore an inspectable mounting hypothesis, not a
contact-qualified TCP or collision model.
"""
from pathlib import Path

from pxr import Gf, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / 'data/gripper.usd'
SOURCE_Y_MAX = 0.00175355771760241
SOURCE_Y_MIN = -0.0205384609807311
SOURCE_Y_CENTER = (SOURCE_Y_MIN + SOURCE_Y_MAX) / 2.0
SOURCE_Z_MAX = 0.0437506223631226
SOURCE_Z_MIN = -0.0477762395223592
SOURCE_Z_TOP = 0.0437506223631226
SOURCE_Z_BOTTOM = -0.0477762395223592
# First visual alignment adjustment from the rendered inspection image.
# +X is farther along the tool, away from the robot base.
FORWARD_EXTENSION = 0.020
DOWNWARD_SHIFT = 0.006
MIN_FINGER_GAP = 0.003
STARTING_FINGER_GAP = MIN_FINGER_GAP + 2 * 0.006
# The cell places each jaw root laterally before applying our CAD reference.
# Compensate that existing root offset when positioning the asymmetric STEP.
# The articulated Isaac jaws add 12 mm of total separation over their travel.
# The initial visual-only pose was centered at the wide end, yielding a
# measured 15--27 mm CAD gap.  Shift each half 6 mm inward so the commanded
# mechanism spans the required 3--15 mm mesh separation.
JAW_ROOT_LATERAL_COMP = 0.00425


def attach_collision_proxies(stage):
    """Attach conservative invisible box proxies for contact-gate testing.

    These proxies follow the jaw rigid bodies but intentionally do not approve
    contact actions; the release checker still requires swept-path and grasp
    validation.
    """
    import cell
    import scene_config as C
    carriage_x, finger_x = cell.gripper_layout()
    length = SOURCE_Z_MAX - SOURCE_Z_MIN
    width = SOURCE_Y_MAX - SOURCE_Y_MIN
    height = 0.0356293480873969
    paths = []
    for sign, tag in ((1, 'L'), (-1, 'R')):
        path = cell.JAW_PRIMS[tag] + '/CustomFingerCollisionProxy'
        proxy = UsdGeom.Cube.Define(stage, path)
        proxy.CreateSizeAttr(1.0)
        UsdGeom.Xformable(proxy).AddTranslateOp().Set(
            Gf.Vec3d(finger_x - carriage_x + C.FINGER_LENGTH / 2 + FORWARD_EXTENSION,
                     -sign * (width / 2 + STARTING_FINGER_GAP / 2 + JAW_ROOT_LATERAL_COMP),
                     -0.0178 - DOWNWARD_SHIFT)
        )
        UsdGeom.Xformable(proxy).AddScaleOp().Set(Gf.Vec3f(length / 2, width / 2, height / 2))
        UsdGeom.Imageable(proxy).CreateVisibilityAttr().Set('invisible')
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim()).CreateApproximationAttr().Set('convexHull')
        paths.append(path)
    return paths


def attach_visual_pair(stage):
    if not ASSET.is_file():
        raise FileNotFoundError(f'Convert custom STEP before mounting: {ASSET}')
    import cell
    import scene_config as C
    carriage_x, finger_x = cell.gripper_layout()
    mount_x = finger_x - carriage_x + C.FINGER_LENGTH + SOURCE_Z_BOTTOM + FORWARD_EXTENSION
    target_center = (SOURCE_Y_MAX - SOURCE_Y_MIN) / 2.0 + STARTING_FINGER_GAP / 2.0
    paths = []
    for sign, tag in ((1, 'L'), (-1, 'R')):
        jaw = cell.JAW_PRIMS[tag]
        if not stage.GetPrimAtPath(jaw).IsValid():
            raise RuntimeError(f'Custom gripper requires active jaw {jaw}')
        path = jaw + '/CustomFingerCad'
        root = UsdGeom.Xform.Define(stage, path)
        root.GetPrim().GetReferences().AddReference(str(ASSET))
        xf = UsdGeom.Xformable(root)
        # -Z source (finger bottom) maps to +X tool direction.  The right half
        # mirrors across the jaw-spread axis, putting each +Y inset face inward.
        # Center the asymmetric STEP envelope first, then set the inner
        # surfaces to the 3 mm closed gap. Mirroring applies to the source
        # center as well as the mesh geometry.
        y_translation = -sign * (target_center + SOURCE_Y_CENTER + JAW_ROOT_LATERAL_COMP)
        xf.AddTranslateOp().Set(Gf.Vec3d(mount_x, y_translation, -0.0178 - DOWNWARD_SHIFT))
        xf.AddRotateYOp().Set(-90.0)
        if sign < 0:
            xf.AddScaleOp().Set(Gf.Vec3f(1.0, -1.0, 1.0))
        paths.append(path)
    # Hide only the stock long blade, soft tip, and detailed extender. Keep the
    # SMC jaw body visible because it is the moving left/right assembly.
    for tag in ('L', 'R'):
        jaw_root = cell.JAW_PRIMS[tag]
        for name in ('Finger', 'SoftTip', 'Detail', 'SensorCarrier', 'SensorRedNose', 'OuterCrossBolt'):
            visual = stage.GetPrimAtPath(jaw_root + '/' + name)
            if visual.IsValid():
                UsdGeom.Imageable(visual).CreateVisibilityAttr().Set('invisible')
            # Detail is authored as meshes directly below an implicit scope in
            # some cell revisions, so also hide each concrete child prim.
            prefix = jaw_root + '/' + name + '/'
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(prefix) and prim.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set('invisible')
    # The stock cell adds a fixed laser carrier with a red nose near the
    # fingertips. It is unrelated to the replacement gripper and is removed
    # from the visual assembly (there is no collision body on this detail).
    hidden_wrist_details = [
        cell.FLANGE + '/Gripper/LaserBody',
        cell.FLANGE + '/Gripper/PhotoBlackSensorLead',
    ]
    for detail_root in hidden_wrist_details:
        detail = stage.GetPrimAtPath(detail_root)
        if detail.IsValid():
            UsdGeom.Imageable(detail).CreateVisibilityAttr().Set('invisible')
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(detail_root + '/') and prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set('invisible')
    return {
        'asset': str(ASSET.relative_to(ROOT)),
        'paths': paths,
        'source_down_axis': '-Z',
        'mounting_face': '+Y inset face',
        'forward_extension_m': FORWARD_EXTENSION,
        'downward_shift_m': DOWNWARD_SHIFT,
        'closed_min_mesh_gap_m': MIN_FINGER_GAP,
        'open_min_mesh_gap_m': STARTING_FINGER_GAP,
        'jaw_root_lateral_compensation_m': JAW_ROOT_LATERAL_COMP,
        'stock_finger_visuals_hidden': ['Finger', 'SoftTip', 'Detail', 'SensorCarrier', 'SensorRedNose', 'OuterCrossBolt'],
        'moving_jaw_visual_retained': 'SMCJaw',
        'starting_jaw_state': 'fully_extended',
        'starting_extension_per_jaw_m': float(C.JAW_STROKE),
        'hidden_wrist_visuals': hidden_wrist_details,
        'mode': 'visual_only_no_collision_no_tcp',
    }
