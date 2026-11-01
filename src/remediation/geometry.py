"""Metre-based physical geometry. Contains no simulator or robot commands."""
from dataclasses import asdict, dataclass
from functools import lru_cache
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
    contact_offset: float = 0.00015
    # 0.15 mm: PhysX settles a leaning cap ~0.13 mm into the rim at its pivot contact.
    penetration_tolerance: float = 0.00015
    friction_static: float = 0.35
    friction_dynamic: float = 0.25

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


# Crooked-cap spawn tilts that hold in PhysX in the observed leaning pose (flange
# dipped into the mouth, plug bottom corner on the opposite rim).  Below 48 deg the
# pose is geometrically impossible; 79 deg failed in every seed and higher tilts
# slid down to ~80 deg.  Sweep: shared/runs/2026-10-26_crooked_tilt_sweep_v2_seed*,
# recorded in docs/PHYSICAL_SPECIFICATIONS.md sec. 5.
CROOKED_TILT_RANGE_DEG = (48., 77.)


@lru_cache(maxsize=None)
def _leaning_offset(tilt_tenths, spec=SPEC):
    """(cap-origin shift along the high side, height above rim) for a leaning cap at yaw 0."""
    tilt = tilt_tenths / 10.
    ri, ro = spec.vial_inner_radius, spec.vial_outer_radius
    local = cap_surface_points(spec)
    sub = local[::3]
    flange_pts = sub[:, 2] >= 0
    bottom_pts = sub[:, 2] < -spec.plug_depth + spec.plug_bottom_face_thickness + 1e-4
    L = sub @ rotation(tilt, 0.).T

    def penetrates(W):
        r = np.linalg.norm(W[:, :2], axis=1)
        return bool(np.any((W[:, 2] < 0) & (r > ri) & (r < ro)))

    best = None
    for py in np.linspace(-0.008, 0.008, 81):
        lo, hi = -0.03, 0.03
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if penetrates(L + [0, py, mid]) else (lo, mid)
        W = L + [0, py, hi]
        r = np.linalg.norm(W[:, :2], axis=1)
        near = np.hypot(np.maximum(np.maximum(ri - r, r - ro), 0), np.minimum(W[:, 2], 0)) < 1.5e-4
        dipped = flange_pts & (W[:, 2] < -5e-4)
        if (near & flange_pts).any() and (near & bottom_pts).any() and dipped.any() and r[dipped].max() < ri:
            if best is None or hi < best[1]:
                best = (float(py), float(hi))
    return best


def leaning_cap_pose(rim, tilt_deg, yaw_deg=0., lift=0.0003, spec=SPEC):
    """Crooked cap as observed on the real cell (docs/PHYSICAL_SPECIFICATIONS.md sec. 5).

    The wide flange dips into the vial mouth on the low side, resting on the inner
    rim edge, while the plug's bottom face rests on the rim on the opposite (high)
    side with the plug corner overhanging the outside of the vial.  Of the
    resting positions at this tilt, the lowest one is returned, lifted by `lift`
    so PhysX settles it into contact.  Raises ValueError if the tilt cannot rest
    this way.
    """
    off = _leaning_offset(int(round(tilt_deg * 10)), spec)
    if off is None:
        raise ValueError(f'a cap cannot lean on the rim at {tilt_deg:.1f} deg')
    R = rotation(tilt_deg, yaw_deg)
    p = np.asarray(rim, float) + rotation(0., yaw_deg) @ np.array([0., off[0], off[1] + lift])
    return p, R


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
    taper=lambda z: spec.plug_bottom_radius+(z+spec.plug_depth)/spec.plug_depth*(spec.plug_top_radius-spec.plug_bottom_radius)
    # Outer plug wall over its full depth (including the bottom outer corner).
    for z in np.linspace(-spec.plug_depth,0,75):
        rings.append((taper(z),z))
    for z in np.linspace(-spec.plug_depth + spec.plug_bottom_face_thickness,0,65):
        rings.append((taper(z)-spec.plug_wall_thickness,z))
    inner_bottom=spec.plug_bottom_radius-spec.plug_wall_thickness
    # Bottom face out to the outer radius; cavity floor out to the inner wall.
    for r in np.linspace(0,spec.plug_bottom_radius,38):
        rings.append((r,-spec.plug_depth))
    for r in np.linspace(0,inner_bottom,32):
        rings.append((r,-spec.plug_depth+spec.plug_bottom_face_thickness))
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


def resolve_height_planes(work_surface_z_m=None, vial_rim_z_m=None, spec=SPEC):
    """Return effective base-frame work-surface and vial-rim Z planes."""
    surface = spec.surface_base_z if work_surface_z_m is None else float(work_surface_z_m)
    derived_rim = surface + spec.tray_base_thickness + spec.vial_height
    rim = derived_rim if vial_rim_z_m is None else float(vial_rim_z_m)
    if not (-0.20 <= surface <= 0.20):
        raise ValueError("work-surface Z must be between -0.20 and 0.20 m in robot-base coordinates")
    if not (0.05 <= rim <= 0.45):
        raise ValueError("vial-rim Z must be between 0.05 and 0.45 m in robot-base coordinates")
    return surface, rim, derived_rim


def transit_base_z():
    # Clearance height across the tray must be > 0.203 m (minimum 0.26 m) to clear all 8-inch vials
    return max(0.285, rim_base_z() + 0.03 + SPEC.plug_depth)


def inspection_base_position():
    """Cable-safe, TableCam-clear inspection TCP used by sim and physical policy."""
    return np.array([0.240, 0.250, transit_base_z()], dtype=float)


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
    # (docs/PHYSICAL_SPECIFICATIONS.md sec. 5): the flange dips inside the vial mouth
    # on the low side while the plug's bottom face rests on the rim on the high side.
    local = cap_surface_points(spec)[::3]
    world = local @ R.T + p
    r = np.linalg.norm(world[:, :2] - rim[:2], axis=1)
    dz = world[:, 2] - rim[2]
    flange = local[:, 2] >= 0
    bottom = local[:, 2] < -spec.plug_depth + spec.plug_bottom_face_thickness + 1e-4
    dipped = flange & (dz < -.0003)
    flange_in_mouth = bool(dipped.any() and r[dipped].max() < spec.vial_inner_radius + .0003)
    bottom_on_rim = bool(np.any(bottom & (np.abs(dz) < .0006) & (r > spec.vial_inner_radius - .0003)
                                & (r < spec.vial_outer_radius + .0003)))
    cls = 'invalid'
    if offset < .002 and tilt < 5 and abs(elevation) < .001:
        cls = 'good_cap'
    elif tilt >= 5 and flange_in_mouth and bottom_on_rim:
        cls = 'crooked_cap'
    return {'tilt_deg': tilt, 'rim_offset_m': offset, 'rim_elevation_m': elevation, 'class_name': cls,
            'flange_in_mouth': flange_in_mouth, 'plug_bottom_on_rim': bottom_on_rim}
