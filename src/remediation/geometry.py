"""Metre-based physical geometry. Contains no simulator or robot commands."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ('good_cap', 'crooked_cap', 'missing_cap', 'table_cap')
LAYOUT_PATH = Path(__file__).with_name('config') / 'dataset1_layout.json'


@dataclass(frozen=True)
class PhysicalSpec:
    # docs/PHYSICAL_SPECIFICATIONS.md is authoritative. The CAD models in cad_models/
    # are used only where the document is silent (vial_bottom floor, appearance).
    # Vial: 1.00 in ID, 1.05 in OD, 8.00 in tall (CAD 0890-01 is 1.038/1.070/7.992 in).
    vial_outer_radius: float = 0.02667 / 2    # 1.05 in OD = 26.67 mm
    vial_inner_radius: float = 0.02540 / 2    # 1.00 in ID = 25.40 mm
    vial_height: float = 0.20320              # 8.00 in = 203.20 mm
    vial_bottom: float = 0.003327             # not in doc; CAD inner_floor = 3.327 mm

    # Cap: MOCAP tapered flanged insert plug, 2.00 mm walls and 2.00 mm closed base.
    flange_radius: float = 0.03150 / 2        # 1.24 in = 31.50 mm
    flange_thickness: float = 0.002032        # 0.08 in = 2.032 mm
    plug_bottom_radius: float = 0.02286 / 2   # 0.90 in = 22.86 mm
    plug_top_radius: float = 0.025146 / 2     # 0.99 in = 25.146 mm
    plug_wall_thickness: float = 0.002        # 2.00 mm
    flange_inner_radius: float = 0.02115 / 2  # top opening 21.15 mm
    plug_bottom_face_thickness: float = 0.002 # 2.00 mm
    plug_depth: float = 0.01346               # 0.53 in = 13.46 mm
    tray_base_thickness: float = 0.010
    tray_top_clearance: float = 0.080
    tray_top_thickness: float = 0.003
    aperture_radius: float = 0.03175 / 2
    world_base_z: float = 0.90
    surface_base_z: float = -0.0051
    tool_length: float = 0.293
    jaw_travel: float = 0.006
    pad_width: float = 0.034
    spawn_clearance: float = 0.003
    contact_offset: float = 0.0005
    penetration_tolerance: float = 0.0001
    friction_static: float = 1.2
    friction_dynamic: float = 1.0

    @property
    def surface_world_z(self):
        return self.world_base_z + self.surface_base_z


SPEC = PhysicalSpec()


def load_layout():
    return json.loads(LAYOUT_PATH.read_text())


def configuration_hash():
    return hashlib.sha256((json.dumps(asdict(SPEC), sort_keys=True) + LAYOUT_PATH.read_text()).encode()).hexdigest()


def rotation(tilt_deg=0., yaw_deg=0.):
    t, y = np.radians([tilt_deg, yaw_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ rx


def cap_vertices(segments=96, spec=SPEC):
    angle = np.arange(segments) * (2 * np.pi / segments)
    rings = [(spec.flange_radius, 0), (spec.flange_radius, spec.flange_thickness),
             (spec.flange_inner_radius, 0), (spec.flange_inner_radius, spec.flange_thickness),
             (spec.plug_top_radius, 0), (spec.plug_bottom_radius, -spec.plug_depth),
             (spec.plug_top_radius-spec.plug_wall_thickness, 0),
             (spec.plug_bottom_radius-spec.plug_wall_thickness, -spec.plug_depth),
             (0, -spec.plug_depth),
             (spec.plug_bottom_radius-spec.plug_wall_thickness, -spec.plug_depth + spec.plug_bottom_face_thickness)]
    return np.concatenate([np.column_stack((r*np.cos(angle), r*np.sin(angle), np.full(segments,z))) for r,z in rings])


def _vial_wall_penetration(points, rim, spec=SPEC):
    """Deepest sampled point inside the glass wall below the rim plane (metres)."""
    q = points - rim
    r = np.linalg.norm(q[:, :2], axis=1)
    inside = (q[:, 2] < 0) & (r > spec.vial_inner_radius) & (r < spec.vial_outer_radius)
    return float(np.max(np.minimum(r - spec.vial_inner_radius, spec.vial_outer_radius - r)[inside])) if inside.any() else 0.


def perched_cap_pose(rim, yaw_deg=0., lift=0.0002, spec=SPEC):
    """Crooked cap balanced on the vial edge (docs/PHYSICAL_SPECIFICATIONS.md sec. 5).

    The plug's bottom edge rests on the inner edge of the glass rim while the
    opposite side of the plug leans against the inner wall.  The tilt is not a
    target: it is the largest tilt at which the perched cap clears the glass,
    found by bisection on the documented dimensions.  `lift` leaves a small gap
    so PhysX settles the cap into contact instead of starting interpenetrated.
    """
    rim = np.asarray(rim, float)
    pts = cap_surface_points(spec)

    def place(tilt):
        R = rotation(tilt, yaw_deg)
        edge = np.array([0., spec.plug_bottom_radius, -spec.plug_depth])
        d = R @ np.array([0., 1., 0.]); d[2] = 0; d /= np.linalg.norm(d)
        p = rim + spec.vial_inner_radius * d + np.array([0., 0., lift]) - R @ edge
        return p, R

    lo, hi = 1., 60.
    for _ in range(40):
        mid = (lo + hi) / 2
        p, R = place(mid)
        if _vial_wall_penetration(pts @ R.T + p, rim, spec) > 0:
            hi = mid
        else:
            lo = mid
    p, R = place(lo)
    return p, R, lo


def spawn_height(support_z, orientation, spec=SPEC):
    lowest = np.min((cap_vertices(spec=spec) @ np.asarray(orientation).T)[:, 2])
    return float(support_z - lowest + spec.spawn_clearance)


def base_to_uf0(position):
    result = np.asarray(position, dtype=float).copy() * 1000
    result[..., 2] -= 330
    return result


def cap_surface_points(spec=SPEC):
    """Dense surface samples for independent post-settling penetration checks."""
    a=np.arange(192)*2*np.pi/192
    rings=[]
    for z in np.linspace(-spec.plug_depth + spec.plug_bottom_face_thickness,0,65):
        r=spec.plug_bottom_radius+(z+spec.plug_depth)/spec.plug_depth*(spec.plug_top_radius-spec.plug_bottom_radius)
        rings.extend([(r,z),(r-spec.plug_wall_thickness,z)])
    inner_bottom=spec.plug_bottom_radius-spec.plug_wall_thickness
    for r in np.linspace(0,inner_bottom,32):
        rings.extend([(r,-spec.plug_depth),(r,-spec.plug_depth+spec.plug_bottom_face_thickness)])
    for r in np.linspace(spec.flange_inner_radius,spec.flange_radius,32):
        rings.extend([(r,0),(r,spec.flange_thickness)])
    return np.concatenate([np.column_stack([r*np.cos(a),r*np.sin(a),np.full(len(a),z)]) for r,z in rings])


def uf0_to_base(position):
    result = np.asarray(position, dtype=float).copy()
    result[..., 2] += 330
    return result / 1000


def world_to_base(position):
    result = np.asarray(position, dtype=float).copy()
    result[..., 2] -= SPEC.world_base_z
    return result


def rim_base_z(spec=SPEC):
    # Vials stand on the 10 mm black base plate (docs/PHYSICAL_SPECIFICATIONS.md sec. 3),
    # so the rim is base thickness + vial height above the work surface (213.2 mm).
    return spec.surface_base_z + spec.tray_base_thickness + spec.vial_height


def transit_base_z():
    # Clearance height across the tray must be > 0.203 m (minimum 0.26 m) to clear all 8-inch vials
    return max(0.285, rim_base_z() + 0.03 + SPEC.plug_depth)


def cap_state(position, orientation, rim_position=None, spec=SPEC):
    """Evaluation only: derive state from a settled pose, never a requested label."""
    p, R = np.asarray(position), np.asarray(orientation)
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    points = cap_vertices(spec=spec) @ R.T + p
    if rim_position is None:
        low = float(points[:,2].min())
        return {'tilt_deg': tilt, 'lowest_z': low,
                'class_name': 'table_cap' if tilt < 5 and abs(low-spec.surface_world_z) < .001 else 'invalid'}
    rim = np.asarray(rim_position)
    offset = float(np.linalg.norm(p[:2]-rim[:2]))
    elevation = float(p[2]-rim[2])
    # A crooked cap is defined by how it rests, not by a tilt angle
    # (docs/PHYSICAL_SPECIFICATIONS.md sec. 5): the plug's bottom edge is balanced
    # on the glass rim while the opposite side of the plug is down inside the mouth.
    a = np.arange(96) * 2 * np.pi / 96
    plug = np.column_stack((spec.plug_bottom_radius*np.cos(a), spec.plug_bottom_radius*np.sin(a),
                            np.full(96, -spec.plug_depth))) @ R.T + p
    radial = np.linalg.norm(plug[:, :2] - rim[:2], axis=1)
    in_mouth = (plug[:, 2] < rim[2] - .001) & (radial < spec.vial_inner_radius)
    # The plug's bottom edge must touch the top of the glass rim, and the whole
    # flange must stay above the rim plane; a flange edge below the rim means the
    # cap has tipped past the balanced state onto the plug side.
    radial_plug = np.linalg.norm(plug[:, :2] - rim[:2], axis=1)
    edge_on_rim = bool(np.any((np.abs(plug[:, 2] - rim[2]) < .0006)
                              & (radial_plug > spec.vial_inner_radius - .0003)
                              & (radial_plug < spec.vial_outer_radius + .0003)))
    flange = np.column_stack((spec.flange_radius*np.cos(a), spec.flange_radius*np.sin(a), np.zeros(96))) @ R.T + p
    edge_on_rim = edge_on_rim and bool(flange[:, 2].min() > rim[2] - .0003)
    plug_in_mouth = bool(in_mouth.any())
    cls = 'invalid'
    if offset < .002 and tilt < 5 and abs(elevation) < .001:
        cls = 'good_cap'
    elif tilt >= 5 and edge_on_rim and plug_in_mouth:
        cls = 'crooked_cap'
    return {'tilt_deg': tilt, 'rim_offset_m': offset, 'rim_elevation_m': elevation, 'class_name': cls,
            'plug_edge_on_rim': edge_on_rim, 'plug_in_mouth': plug_in_mouth}
