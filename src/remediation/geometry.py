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
    vial_outer_radius: float = 0.02667 / 2
    vial_inner_radius: float = 0.02540 / 2
    vial_height: float = 0.20320
    vial_bottom: float = 0.001  # unmeasured; only affects the tube's closed bottom
    flange_radius: float = 0.031496 / 2
    flange_thickness: float = 0.002032
    plug_bottom_radius: float = 0.02286 / 2
    plug_top_radius: float = 0.025146 / 2  # 0.99 in, estimate in physical specification
    plug_wall_thickness: float = 0.002  # team measurement/update; use a tapered inner wall
    flange_inner_radius: float = 0.025146 / 2 - 0.002  # open-through cavity continues through flange
    plug_bottom_face_thickness: float = 0.002  # flat closed end under the hollow plug
    plug_depth: float = 0.013462
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


def rim_base_z():
    # Vials rest on the 10 mm black base; their mouths extend through the
    # silver perforated plate, which is supported 80 mm above that base.
    return SPEC.surface_base_z + SPEC.tray_base_thickness + SPEC.vial_height


def transit_base_z():
    # Includes the carried plug below the fingertips and 30 mm obstacle margin.
    return max(SPEC.surface_base_z + .26, rim_base_z() + .03 + SPEC.plug_depth)


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
    cls = 'invalid'
    if offset < .002 and tilt < 5 and abs(elevation) < .001:
        cls = 'good_cap'
    elif offset < spec.vial_inner_radius and 8 <= tilt <= 45 and -.001 < elevation < .018:
        cls = 'crooked_cap'
    return {'tilt_deg': tilt, 'rim_offset_m': offset, 'rim_elevation_m': elevation, 'class_name': cls,
            'documented_tilt_band': 31.5 <= tilt <= 34.0}
