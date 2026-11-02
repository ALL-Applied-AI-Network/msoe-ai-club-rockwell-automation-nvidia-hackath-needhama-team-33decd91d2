"""CAD-derived mount, collision geometry and TCP for the two custom finger halves.

Sources of truth (see runs/archive/2026-10-26_cad_audit for the extraction):
- data/gripper.step: one finger half, metres.  Dimensions below are B-rep
  vertex / face values, NOT the raw CARTESIAN_POINT envelope (which includes
  axis placements and once produced a phantom 91.5 x 22.3 x 35.6 mm finger).
  The solid is 50.8 x 15.875 x 22.225 mm.  Its inner face carries two coaxial
  pockets sized to the CSI CAD parts: r 15.875 mm (cap flange, 1.250 in) and
  r 13.589 mm (vial OD, 1.070 in).
- jeremiah_given_context/SMC-MHZL2-16D2.step: SMC body and prongs, mm.  Each
  prong is a carriage plus a 15 x 5 x 8 mm blade with two 3.4 mm holes 7.0 mm
  apart, matching the finger's two 2.8 mm holes 7.0 mm apart.  The STEP pose is
  the open pose (jaw centres +/-13.9 mm, as cell.py assumes).

Finger frame (STEP): length along Z with the tip at -Z, grip side +Y, notch
for the blade cut into the -Y back.  The blade slides into the notch, its inner
face against the inset face, its tip against the notch floor.
"""
from pathlib import Path

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / 'data/gripper.usd'
CONE_ASSET = ROOT / 'data/cone_gripper.usd'

# data/gripper.step, finger frame, metres.
FINGER_X_RANGE = (0.0134043, 0.0356293)
FINGER_BACK_Y = -0.015875
FINGER_INSET_FACE_Y = -0.00635          # mates to the SMC blade inner face
FINGER_NOTCH_X_RANGE = (0.020389, 0.028644)
FINGER_NOTCH_FLOOR_Z = 0.0297806        # SMC blade tip seats here
FINGER_TOP_Z = 0.0437506
FINGER_TIP_Z = -0.0070494
FINGER_POCKET_AXIS_XY = (0.0245168, 0.0017536)
FINGER_FLANGE_LEDGE_Z = -0.004304       # cap flange underside rests here
# (z0, z1, radius) of the coaxial pocket from the tip upward.
FINGER_POCKET_BANDS = (
    (FINGER_TIP_Z, FINGER_FLANGE_LEDGE_Z, 0.013589),
    (FINGER_FLANGE_LEDGE_Z, 0.0004585, 0.015875),
    (0.0004585, 0.0208719, 0.013589),
    (0.0208719, 0.0256344, 0.015875),
)
FINGER_MOUNT_Z = 0.0256344              # solid mount block above the pocket

# SMC-MHZL2-16D2.step (prong 'hole_v' placed by the assembly transforms).
SMC_BLADE_TIP_X = 0.0173                # beyond the body jaw face, along the tool axis
SMC_BLADE_INNER_FACE_OPEN = 0.01345     # from the gripper centreline, jaws open

POCKET_FROM_INSET_FACE = FINGER_POCKET_AXIS_XY[1] - FINGER_INSET_FACE_Y
# Radial clearance added to the pocket collision surfaces only.  The CAD fit is
# line-to-line (lip r = vial OD r = 13.589 mm); with zero clearance PhysX caught
# the lip on the rim top, held the wrist off the seat and friction-locked the
# jaws shut.  0.2 mm (print tolerance scale) seats contact-free.  Visual geometry
# and the TCP stay exactly at CAD.
FINGER_POCKET_SIM_CLEARANCE = 0.0002

# data/cone_gripper.step, finger frame, metres.  STEP face #1286 is bounded by
# its small circle (#121, centre #2199) and the four 12.7 mm-radius arcs at the
# broad end (#101/#102/#103/#119).  Its axis points from the small contact end
# back into the TPU finger body.
CONE_TIP_FINGER_M = np.array((0.024516842, -0.0347125091901217,
                              0.00370293361826785))
CONE_AXIS_FINGER = np.array((0.0, np.sqrt(0.5), np.sqrt(0.5)))
CONE_NOSE_LENGTH_M = 0.013940260977997894
CONE_REFERENCE_RADIUS_M = 0.00293892417859514
CONE_SEMIANGLE_DEG = 35.0
CONE_BASE_RADIUS_M = 0.0127


def cap_grip_jaw(jaw_open):
    """Per-jaw opening at which both half-pockets are coaxial (the flange grip pose)."""
    return jaw_open - (SMC_BLADE_INNER_FACE_OPEN - POCKET_FROM_INSET_FACE)


def cap_seat_tcp(body_face_x):
    """J6-frame point on the pocket axis at the flange ledge.

    This is the CAD origin of the held cap (flange underside on its axis), so at
    seating it coincides with the vial rim plane.
    """
    return (body_face_x + SMC_BLADE_TIP_X + FINGER_NOTCH_FLOOR_Z - FINGER_FLANGE_LEDGE_Z, 0.0, 0.0)


def finger_tip_tcp(cap_seat_tcp_j6):
    """J6-frame centre of the closed fingertip face (on the pocket axis)."""
    x = cap_seat_tcp_j6[0] + (FINGER_FLANGE_LEDGE_Z - FINGER_TIP_Z)
    return (x, 0.0, 0.0)


def finger_back_face_tcp(cap_seat_tcp_j6, jaw, jaw_grip, along=0.0, up=0.0):
    """J6-frame point on the outer back face of the -Y finger.

    The back face is solid over the finger's full length (the cap pockets,
    r <= 15.875 mm about an axis 1.75 mm inside the grip side, stop short of it).
    At the coaxial grip pose it sits FINGER_POCKET_AXIS_XY[1] - FINGER_BACK_Y from
    the centreline and moves with the jaw.  `up` is the distance back from the
    fingertip along the tool axis (0 = the tip edge, next to the pocket mouth) and
    `along` an offset along J6 Z across the face.
    """
    back = (FINGER_POCKET_AXIS_XY[1] - FINGER_BACK_Y) + (jaw - jaw_grip)
    return (finger_tip_tcp(cap_seat_tcp_j6)[0] - float(up), -back, float(along))


def cone_tip_from_fingertip_tcp(jaw, jaw_grip, side=-1):
    """Cone-nose offset from the fingertip TCP, expressed in the command frame.

    ``side=-1`` selects the negative-J6-Y finger and ``side=+1`` its mirrored
    mate.  Closing each jaw moves the outside nose toward the centreline.  The
    command frame used by the shared controller maps a J6 vector (x, y, z) to
    (-z, y, x).
    """
    if side not in (-1, 1):
        raise ValueError("cone side must be -1 or +1")
    j6_x = -(CONE_TIP_FINGER_M[2] - FINGER_TIP_Z)
    negative_y = ((CONE_TIP_FINGER_M[1] - FINGER_POCKET_AXIS_XY[1]) -
                  (jaw - jaw_grip))
    j6_y = negative_y if side == -1 else -negative_y
    return (0.0, float(j6_y), float(j6_x))


def _jaw_mount(stage, jaw_root):
    """(finger->jaw Gf.Matrix4d, sign) from the jaw's prismatic joint and the SMC blade."""
    import cell
    import scene_config as C
    joint = next((p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)
                  and [str(t) for t in UsdPhysics.PrismaticJoint(p).GetBody1Rel().GetTargets()] == [jaw_root]), None)
    if joint is None:
        raise RuntimeError(f'No prismatic joint drives {jaw_root}')
    j = UsdPhysics.PrismaticJoint(joint)
    if j.GetAxisAttr().Get() != 'Y':
        raise RuntimeError(f'{joint.GetPath()} axis is not the finger spread axis')
    # Joint zero is the open pose (cell.jaw_targets), so the centreline sits at -localPos0.y.
    origin = np.asarray(j.GetLocalPos0Attr().Get(), float)
    sign = 1.0 if origin[1] > 0 else -1.0
    d = -sign                                   # jaw-frame +Y direction toward the centreline
    y_face = -origin[1] + sign * SMC_BLADE_INNER_FACE_OPEN
    _check_blade(stage, jaw_root, y_face, sign)
    # Row-vector matrix: rows are the images of the finger x, y, z axes, then the origin.
    m = Gf.Matrix4d(0, 0, d, 0,
                    0, d, 0, 0,
                    -1, 0, 0, 0,
                    SMC_BLADE_TIP_X + FINGER_NOTCH_FLOOR_Z, y_face - d * FINGER_INSET_FACE_Y,
                    -d * FINGER_POCKET_AXIS_XY[0], 1)
    return m, sign, float(origin[0])


def _check_blade(stage, jaw_root, y_face, sign):
    """Fail loudly if the mounted SMC blade disagrees with the STEP-derived face/tip."""
    blade = stage.GetPrimAtPath(jaw_root + '/SMCJaw')
    if not blade.IsValid():
        return
    to_jaw = UsdGeom.Xformable(blade).ComputeLocalToWorldTransform(0.) * \
        UsdGeom.Xformable(stage.GetPrimAtPath(jaw_root)).ComputeLocalToWorldTransform(0.).GetInverse()
    m = np.asarray(to_jaw, float)
    v = np.asarray(UsdGeom.Mesh(blade).GetPointsAttr().Get(), float) @ m[:3, :3] + m[3, :3]
    v = v[v[:, 0] > 0.006]                      # blade beyond the carriage
    inner = v[:, 1].min() if sign > 0 else v[:, 1].max()
    if abs(inner - y_face) > 2e-4 or abs(v[:, 0].max() - SMC_BLADE_TIP_X) > 2e-4:
        raise RuntimeError(f'{jaw_root}/SMCJaw blade (face {inner:.5f}, tip {v[:, 0].max():.5f}) '
                           f'disagrees with SMC STEP (face {y_face:.5f}, tip {SMC_BLADE_TIP_X:.5f})')


def _clip(poly, n, c):
    """Keep the part of convex polygon `poly` with n.p >= c."""
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        da, db = n @ a - c, n @ b - c
        if da >= 0:
            out.append(a)
        if da * db < 0:
            out.append(a + (b - a) * (da / (da - db)))
    return out


def finger_collision_pieces(sectors=12):
    """Convex prisms (finger frame) covering the solid but never entering the pockets.

    Each pocket band is the finger rectangle minus a disk; it is split into angular
    sectors about the pocket axis and each sector is clipped by the tangent line at
    its mid-angle, so collision surfaces sit on or just outside the CAD pocket.
    """
    x0, x1 = FINGER_X_RANGE
    rect = [np.array(p) for p in ((x0, FINGER_BACK_Y), (x1, FINGER_BACK_Y), (x1, 0.), (x0, 0.))]
    c = np.array(FINGER_POCKET_AXIS_XY)
    ta, tb = (np.arctan2(-c[1], x - c[0]) for x in (x0, x1))
    pieces = []
    for z0, z1, r in FINGER_POCKET_BANDS:
        for t0, t1 in zip(*(lambda t: (t[:-1], t[1:]))(np.linspace(ta, tb, sectors + 1))):
            u = lambda t: np.array([np.cos(t), np.sin(t)])
            poly = _clip(rect, np.array([-u(t0)[1], u(t0)[0]]), np.array([-u(t0)[1], u(t0)[0]]) @ c)
            poly = _clip(poly, -np.array([-u(t1)[1], u(t1)[0]]), -np.array([-u(t1)[1], u(t1)[0]]) @ c)
            poly = _clip(poly, u((t0 + t1) / 2), u((t0 + t1) / 2) @ c + r + FINGER_POCKET_SIM_CLEARANCE)
            if len(poly) >= 3:
                pieces.append(([tuple(p) for p in poly], z0, z1))
    nx0, nx1 = FINGER_NOTCH_X_RANGE
    for (a, b), (ya, yb), (za, zb) in (
            ((x0, x1), (FINGER_BACK_Y, 0.), (FINGER_MOUNT_Z, FINGER_NOTCH_FLOOR_Z)),
            ((x0, x1), (FINGER_INSET_FACE_Y, 0.), (FINGER_NOTCH_FLOOR_Z, FINGER_TOP_Z)),
            ((x0, nx0), (FINGER_BACK_Y, FINGER_INSET_FACE_Y), (FINGER_NOTCH_FLOOR_Z, FINGER_TOP_Z)),
            ((nx1, x1), (FINGER_BACK_Y, FINGER_INSET_FACE_Y), (FINGER_NOTCH_FLOOR_Z, FINGER_TOP_Z))):
        pieces.append(([(a, ya), (b, ya), (b, yb), (a, yb)], za, zb))
    return pieces


def _cone_proxy_pieces(segments=24, shell_thickness=0.0005):
    """Open, segmented STEP-measured cone contact shell in the finger frame.

    A single convex-hull proxy closes both ends of the frustum.  During a deep
    press that invisible disk can overlap the cap flange and carry it upward on
    retraction.  Convex annular sectors retain the measured sloping contact
    face while leaving the nose and base open, so there is no capture disk.
    """
    axis = CONE_AXIS_FINGER / np.linalg.norm(CONE_AXIS_FINGER)
    reference = np.array([1.0, 0.0, 0.0])
    u = reference - axis * np.dot(reference, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    base_centre = CONE_TIP_FINGER_M + axis * CONE_NOSE_LENGTH_M
    pieces = []
    for i in range(segments):
        a = 2.0 * np.pi * i / segments
        b = 2.0 * np.pi * (i + 1) / segments
        ring = lambda centre, radius, angle: (
            centre + radius * (np.cos(angle) * u + np.sin(angle) * v))
        inner_small = max(0.0005, CONE_REFERENCE_RADIUS_M - shell_thickness)
        inner_base = CONE_BASE_RADIUS_M - shell_thickness
        points = [
            ring(CONE_TIP_FINGER_M, CONE_REFERENCE_RADIUS_M, a),
            ring(CONE_TIP_FINGER_M, CONE_REFERENCE_RADIUS_M, b),
            ring(base_centre, CONE_BASE_RADIUS_M, b),
            ring(base_centre, CONE_BASE_RADIUS_M, a),
            ring(CONE_TIP_FINGER_M, inner_small, a),
            ring(CONE_TIP_FINGER_M, inner_small, b),
            ring(base_centre, inner_base, b),
            ring(base_centre, inner_base, a),
        ]
        pieces.append(points)
    return pieces


def attach_collision_proxies(stage, cone_press=False):
    """Attach invisible convex collision pieces that follow the CAD finger, pockets open.

    These proxies follow the jaw rigid bodies but intentionally do not approve
    contact actions; the release checker still requires swept-path and grasp
    validation.
    """
    import cell
    from pxr import PhysxSchema
    paths = []
    for tag in ('L', 'R'):
        jaw_root = cell.JAW_PRIMS[tag]
        m, _, _ = _jaw_mount(stage, jaw_root)
        path = jaw_root + '/CustomFingerCollisionProxy'
        root = UsdGeom.Xform.Define(stage, path)
        UsdGeom.Xformable(root).AddTransformOp().Set(m)
        UsdGeom.Imageable(root).CreateVisibilityAttr().Set('invisible')
        for k, (poly, z0, z1) in enumerate(finger_collision_pieces()):
            n = len(poly)
            piece = UsdGeom.Mesh.Define(stage, f'{path}/Piece_{k:02d}')
            piece.CreatePointsAttr([Gf.Vec3f(x, y, z) for z in (z0, z1) for x, y in poly])
            piece.CreateFaceVertexCountsAttr([n, n] + [4] * n)
            piece.CreateFaceVertexIndicesAttr(list(range(n))[::-1] + list(range(n, 2 * n)) +
                                              [i for a in range(n) for i in (a, (a + 1) % n, n + (a + 1) % n, n + a)])
            UsdPhysics.CollisionAPI.Apply(piece.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(piece.GetPrim()).CreateApproximationAttr('convexHull')
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(piece.GetPrim())
            physx_col.CreateContactOffsetAttr().Set(0.0005)
            physx_col.CreateRestOffsetAttr().Set(0.0)
        if cone_press:
            faces = [4, 4, 4, 4, 4, 4]
            indices = [0, 1, 2, 3, 4, 7, 6, 5,
                       0, 4, 5, 1, 3, 2, 6, 7,
                       0, 3, 7, 4, 1, 5, 6, 2]
            for k, points in enumerate(_cone_proxy_pieces()):
                cone = UsdGeom.Mesh.Define(
                    stage, f'{path}/ConePressSurface_{k:02d}')
                cone.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
                cone.CreateFaceVertexCountsAttr(faces)
                cone.CreateFaceVertexIndicesAttr(indices)
                UsdPhysics.CollisionAPI.Apply(cone.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(
                    cone.GetPrim()).CreateApproximationAttr('convexHull')
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(cone.GetPrim())
                physx_col.CreateContactOffsetAttr().Set(0.0005)
                physx_col.CreateRestOffsetAttr().Set(0.0)
        paths.append(path)
    return paths


def attach_visual_pair(stage, asset=ASSET):
    asset = Path(asset).resolve()
    if not asset.is_file():
        raise FileNotFoundError(f'Convert custom STEP before mounting: {asset}')
    import cell
    import scene_config as C
    paths = []
    for tag in ('L', 'R'):
        jaw = cell.JAW_PRIMS[tag]
        if not stage.GetPrimAtPath(jaw).IsValid():
            raise RuntimeError(f'Custom gripper requires active jaw {jaw}')
        m, _, body_face_x = _jaw_mount(stage, jaw)
        path = jaw + '/CustomFingerCad'
        root = UsdGeom.Xform.Define(stage, path)
        root.GetPrim().GetReferences().AddReference(str(asset))
        UsdGeom.Xformable(root).AddTransformOp().Set(m)
        paths.append(path)
    # Hide only the stock long blade, soft tip, and detailed extender. Keep the
    # SMC jaw body visible because it is the moving left/right assembly.
    for tag in ('L', 'R'):
        jaw_root = cell.JAW_PRIMS[tag]
        for name in ('Finger', 'SoftTip', 'Detail', 'SensorCarrier', 'SensorRedNose', 'OuterCrossBolt',
                     'MountScrew', 'ScrewSocket', 'Tape', 'Collision'):
            visual = stage.GetPrimAtPath(jaw_root + '/' + name)
            if visual.IsValid():
                UsdGeom.Imageable(visual).CreateVisibilityAttr().Set('invisible')
            # Detail is authored as meshes directly below an implicit scope in
            # some cell revisions, so also hide each concrete child prim.
            prefix = jaw_root + '/' + name + '/'
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(prefix) and prim.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set('invisible')
        # The starter scene names some retired extender fasteners with numbered
        # leaves (for example MountScrew0_front), rather than a common scope.
        # Hide those visual-only remnants while retaining SMCJaw, the part that
        # actually translates left/right with the pneumatic mechanism.
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(jaw_root + '/') and prim.IsA(UsdGeom.Gprim):
                leaf = path.rsplit('/', 1)[-1].lower()
                if any(token in leaf for token in ('mountscrew', 'screwsocket', 'tape')):
                    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set('invisible')
        # Retain the supplied black SMC carrier mesh and its mounting-hole
        # detail. It is the visual reference for the pneumatic jaw assembly.
        starter_carrier = stage.GetPrimAtPath(jaw_root + '/SMCJaw')
        if starter_carrier.IsValid():
            UsdGeom.Imageable(starter_carrier).CreateVisibilityAttr().Set('inherited')
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
        'asset': str(asset.relative_to(ROOT)) if asset.is_relative_to(ROOT) else str(asset),
        'paths': paths,
        'mount': 'SMC blade in finger notch: inset face on blade inner face, blade tip on notch floor',
        'cap_seat_tcp_j6_m': list(cap_seat_tcp(body_face_x)),
        'cap_grip_jaw_m': cap_grip_jaw(C.JAW_OPEN),
        'stock_finger_visuals_hidden': ['Finger', 'SoftTip', 'Detail', 'SensorCarrier', 'SensorRedNose', 'OuterCrossBolt',
                                        'MountScrew*', 'ScrewSocket*', 'Tape', 'Collision'],
        'moving_jaw_visual_retained': 'SMCJaw carrier with mounting holes',
        'hidden_wrist_visuals': hidden_wrist_details,
        'mode': 'cone_cad_mounted' if asset == CONE_ASSET.resolve() else 'cad_mounted',
    }
