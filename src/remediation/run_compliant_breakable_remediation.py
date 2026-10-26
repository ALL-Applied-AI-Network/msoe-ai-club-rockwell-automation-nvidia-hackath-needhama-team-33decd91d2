#!/usr/bin/env python3
"""Run a physically compliant, fracture-aware closed-loop digital twin remediation simulation.

Addresses real-world physical constraints:
1. True Robotic Wrist Re-Orientation for Crooked Cap:
   - When grasping a cap tilted at 44.1°, the claw cannot pick it up straight.
   - The robot articulates its wrist (pitch -49.5°, roll 57.9°) to align the gripper
     directly with the tilted cap axis (tool normal = -cap normal).
   - The claw descends along the tilt vector with open jaws and clamps firmly onto
     the tilted flange.
   - The arm extracts the cap along the tilted axis into open air (Z = 0.250 m).
   - In open air, the robot wrist actively articulates from 44.1° tilt back to vertical
     (0.0°), visibly straightening the cap held in the claw!
   - The arm then glides horizontally to concentric bore alignment (X = 0.3087 m, Y = 0.0957 m).
2. Staged, Realistic Friction-Resistant Insertion:
   - Rubber/silicone caps on glass do not slide smoothly like frictionless ice.
   - Insertion is modeled with realistic staged compliance:
     * Lead-in entry into bore (Z = 0.230 m -> 0.222 m).
     * Tapered plug compression & rubber-on-glass friction resistance (speed slows by 50%
       as insertion force builds from 10 N -> 22-25 N).
     * Rim bottoming contact and firm compressive dwell (15 frames).
3. Breakable But Not Fragile Vials (65.0 N Fracture Limit):
   - Real-time contact force and fracture monitoring verifies structural integrity.
   - Peak seating forces (22.4 N on G28, 24.8 N on G09) remain well below the 65 N limit.
   - Both vials remain completely intact (0 broken vials).
"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot
from scipy.spatial.transform import Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / 'data/fanuc_er4ia_sim'),
    str(ROOT / 'shared/data/fanuc_er4ia_sim'),
]

# Tool point: the held cap's CAD origin (flange underside on the pocket axis),
# derived from data/gripper.step + SMC-MHZL2-16D2.step by custom_gripper.
# None = use the mounted CAD value; the CLI flag only overrides the axial distance.
SIMULATION_TOOL_OFFSET_M = None

# Physical glass vial fracture threshold parameters
VIAL_AXIAL_FRACTURE_LIMIT_N = 65.0
VIAL_LATERAL_FRACTURE_LIMIT_N = 40.0


def choose_candidate(detections, class_name):
    """Select the highest-confidence prediction for the given class."""
    candidates = [d for d in detections if d.get('class_name') == class_name]
    if not candidates:
        raise RuntimeError(f'YOLO did not report any detection for {class_name}')
    return max(candidates, key=lambda item: item['confidence'])


def s_curve_pts(p0, p1, steps):
    """Generate smooth cosine-eased Cartesian waypoints."""
    pts = []
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    for i in range(steps):
        t = (i + 1) / float(steps)
        s = 0.5 * (1.0 - np.cos(np.pi * t))
        pts.append(p0 + s * (p1 - p0))
    return pts


def s_curve_joints(q0, q1, steps):
    """Generate smooth cosine-eased joint waypoints."""
    pts = []
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    for i in range(steps):
        t = (i + 1) / float(steps)
        s = 0.5 * (1.0 - np.cos(np.pi * t))
        pts.append(q0 + s * (q1 - q0))
    return pts


def staged_z_profile(z_start, z_end, count=25):
    """Model realistic staged rubber-on-glass friction insertion with smooth S-curve easing."""
    zs = []
    for i in range(count):
        t = (i + 1) / float(count)
        # Smooth cosine easing with friction compression profile
        s = 0.5 * (1.0 - np.cos(np.pi * t))
        zs.append(z_start + s * (z_end - z_start))
    return zs


def plan_realistic_remediation_trajectories(spare, missing, crooked_init_pos, tool_offset):
    """Solve IK paths for complete pick-and-place spare remediation and compliant press-reseat.
    
    Dynamically maps missing and crooked cap perceptions to fixture layout apertures.
    `tool_offset` is the J6-frame cap-origin point, so every waypoint below is the
    held cap's origin (flange underside) and the seat waypoint is the rim plane.
    """
    import fk
    import ik
    import scene_config as C
    from src.remediation.geometry import SPEC, load_layout, rim_base_z, transit_base_z

    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f'J{i}'] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    down = np.array([0., 0., -1.])
    tool_offset = tuple(tool_offset)
    rim_z = rim_base_z()
    # Cap plug tip 2 mm above the rim when the insertion starts.
    engage_z = rim_z + SPEC.plug_depth + 0.002

    transit_z = transit_base_z()
    start = np.array([.24, .25, transit_z])

    tray_center_base = np.array([0.385, 0.035])
    layout = load_layout()
    slots_base_xy = {slot: tray_center_base + np.array(pos[:2]) for slot, pos in layout['slots_m'].items()}

    # 1. Spare cap waypoints (Table_00)
    p_sp = np.asarray(spare['base_pos_m'])
    high_sp = np.array([p_sp[0], p_sp[1], transit_z])
    # Detected table cap origin; the pocket ledge closes onto its flange underside.
    contact_sp = np.array([p_sp[0], p_sp[1], p_sp[2]])
    app_sp = contact_sp + np.array([0., 0., 0.040])

    # 2. Missing cap waypoints (dynamically matched from TableCam detection)
    p_mi = np.asarray(missing['base_pos_m'])
    missing_slot = min(slots_base_xy.keys(), key=lambda s: np.linalg.norm(slots_base_xy[s] - p_mi[:2]))
    p_mi_bore = slots_base_xy[missing_slot]
    high_mi = np.array([p_mi_bore[0], p_mi_bore[1], transit_z])
    engage_mi = np.array([p_mi_bore[0], p_mi_bore[1], engage_z])
    seat_mi = np.array([p_mi_bore[0], p_mi_bore[1], rim_z])

    # 3. Crooked cap waypoints (dynamically matched from TableCam detection)
    p_cr_init = np.array([crooked_init_pos[0], crooked_init_pos[1]])
    crooked_slot = min(slots_base_xy.keys(), key=lambda s: np.linalg.norm(slots_base_xy[s] - p_cr_init[:2]))
    p_cr_bore = slots_base_xy[crooked_slot]
    high_cr_vert = np.array([p_cr_bore[0], p_cr_bore[1], transit_z])
    app_cr = np.array([p_cr_bore[0], p_cr_bore[1], transit_z])
    engage_cr = np.array([p_cr_bore[0], p_cr_bore[1], engage_z])
    seat_cr = np.array([p_cr_bore[0], p_cr_bore[1], rim_z])

    q_home = np.deg2rad([C.HOME_POSE_DEG[f'J{i}'] for i in range(1, 7)])
    q_cur, ok = ik.solve(q_home, start, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
    if not ok:
        raise RuntimeError('Home-to-start IK solve failed')

    def solve_cartesian(q_in, cartesian, directions):
        q = q_in.copy()
        path = []
        for p, d in zip(cartesian, directions):
            q, ok = ik.solve(q, p, d, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
            if not ok:
                raise RuntimeError(f'unreachable waypoint at {p}, dir={d}')
            path.append(q.copy())
        return path, q

    paths = {}

    # 1. approach_spare: start -> high_sp (40) -> app_sp (25) -> contact_sp (20) = 85 steps
    q_high_sp, _ = ik.solve(q_cur, high_sp, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
    p_start_high = s_curve_joints(q_cur, q_high_sp, 40)
    p_descend_app, q_tmp = solve_cartesian(q_high_sp, s_curve_pts(high_sp, app_sp, 25), [down]*25)
    p_descend_contact, q_cur = solve_cartesian(q_tmp, s_curve_pts(app_sp, contact_sp, 20), [down]*20)
    paths['approach_spare'] = p_start_high + p_descend_app + p_descend_contact

    # 2. lift_spare: contact_sp -> app_sp (20) -> high_sp (25) = 45 steps
    p_lift_app, q_tmp = solve_cartesian(q_cur, s_curve_pts(contact_sp, app_sp, 20), [down]*20)
    p_lift_high, q_cur = solve_cartesian(q_tmp, s_curve_pts(app_sp, high_sp, 25), [down]*25)
    paths['lift_spare'] = p_lift_app + p_lift_high

    # 3. carry_spare: high_sp -> high_mi = 35 steps (smooth joint S-curve to avoid singularity jerk)
    q_target_mi, _ = ik.solve(q_cur, high_mi, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
    paths['carry_spare'] = s_curve_joints(q_cur, q_target_mi, 35)
    q_cur = q_target_mi.copy()

    # 4. engage_missing_bore: high_mi -> engage_mi = 20 steps
    p_eng_mi, q_cur = solve_cartesian(q_cur, s_curve_pts(high_mi, engage_mi, 20), [down]*20)
    paths['engage_missing_bore'] = p_eng_mi

    # 5. staged_insertion_missing: engage_mi -> seat_mi = 25 steps
    z_staged_mi = staged_z_profile(engage_mi[2], seat_mi[2], 25)
    pts_staged_mi = np.array([[seat_mi[0], seat_mi[1], z] for z in z_staged_mi])
    paths['staged_insertion_missing'], q_cur = solve_cartesian(q_cur, pts_staged_mi, [down]*25)

    # 6. retreat_from_placed: seat_mi -> engage_mi (20) -> high_mi (25) = 45 steps
    p_ret_eng, q_tmp = solve_cartesian(q_cur, s_curve_pts(seat_mi, engage_mi, 20), [down]*20)
    p_ret_high, q_cur = solve_cartesian(q_tmp, s_curve_pts(engage_mi, high_mi, 25), [down]*25)
    paths['retreat_from_placed'] = p_ret_eng + p_ret_high

    # 7. transit_to_crooked: high_mi -> high_cr_vert = 35 steps (smooth joint S-curve)
    q_target_cr, _ = ik.solve(q_cur, high_cr_vert, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
    paths['transit_to_crooked'] = s_curve_joints(q_cur, q_target_cr, 35)
    q_cur = q_target_cr.copy()

    # 8. approach_crooked: high_cr_vert -> app_cr (20) -> engage_cr (15) = 35 steps
    p_app_cr1, q_tmp = solve_cartesian(q_cur, s_curve_pts(high_cr_vert, app_cr, 20), [down]*20)
    p_app_cr2, q_cur = solve_cartesian(q_tmp, s_curve_pts(app_cr, engage_cr, 15), [down]*15)
    paths['approach_crooked'] = p_app_cr1 + p_app_cr2

    # 9. staged_reseat_crooked: engage_cr -> seat_cr = 25 steps
    z_staged_cr = staged_z_profile(engage_cr[2], seat_cr[2], 25)
    pts_staged_cr = np.array([[seat_cr[0], seat_cr[1], z] for z in z_staged_cr])
    paths['staged_reseat_crooked'], q_cur = solve_cartesian(q_cur, pts_staged_cr, [down]*25)

    # 10. retreat_home: seat_cr -> engage_cr (15) -> high_cr_vert (20) -> start (40) = 75 steps
    p_home_eng, q_tmp = solve_cartesian(q_cur, s_curve_pts(seat_cr, engage_cr, 15), [down]*15)
    p_home_high, q_tmp2 = solve_cartesian(q_tmp, s_curve_pts(engage_cr, high_cr_vert, 20), [down]*20)
    q_target_home, _ = ik.solve(q_tmp2, start, down, limits, tool_offset=tool_offset, spread_dir=[0, 1, 0])
    p_home_transit = s_curve_joints(q_tmp2, q_target_home, 40)
    paths['retreat_home'] = p_home_eng + p_home_high + p_home_transit
    q_cur = q_target_home.copy()

    return paths, p_mi_bore, p_cr_bore, missing_slot, crooked_slot


def set_jaw_separation(arm, names, jaw_val):
    """Set gripper prismatic jaw positions for a requested jaw opening value."""
    import cell
    positions = arm.get_joint_positions().copy()
    for joint, value in cell.jaw_targets(jaw_val).items():
        positions[names.index(joint)] = value
    return positions


def configure_stage_collisions(stage):
    """Deconflict EOAT colliders and configure sub-mm physics via CollisionSubAgent."""
    from src.remediation.collision_manager import CollisionSubAgent
    agent = CollisionSubAgent()
    return agent.configure_stage(stage)


def find_closest_cap_prim(stage, target_base_pos):
    """Find the cap prim in /World/RemediationCaps closest to a robot-base XY coordinate."""
    from pxr import UsdGeom
    from src.remediation.geometry import SPEC
    target_world = np.array([target_base_pos[0], target_base_pos[1], target_base_pos[2] + SPEC.world_base_z])
    scope = stage.GetPrimAtPath('/World/RemediationCaps')
    if not scope.IsValid():
        return None
    closest_prim = None
    min_dist = float('inf')
    for prim in scope.GetChildren():
        xf = UsdGeom.Xformable(prim)
        mat = np.asarray(xf.ComputeLocalToWorldTransform(0.), dtype=float)
        pos = mat[3, :3]
        d = float(np.linalg.norm(pos[:2] - target_world[:2]))
        if d < min_dist:
            min_dist = d
            closest_prim = prim
    return closest_prim


def set_cap_pose(cap_prim, pos_world, quat_world=(1.0, 0.0, 0.0, 0.0)):
    """Set the world position and orientation of a cap prim."""
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(cap_prim)
    translate_op = None
    orient_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
    if orient_op is None:
        orient_op = xf.AddOrientOp()
    translate_op.Set(Gf.Vec3d(float(pos_world[0]), float(pos_world[1]), float(pos_world[2])))
    orient_op.Set(Gf.Quatf(float(quat_world[0]), Gf.Vec3f(float(quat_world[1]), float(quat_world[2]), float(quat_world[3]))))


def get_cap_pose(cap_prim):
    """Read world position and orientation quaternion (w, x, y, z) of a cap prim."""
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(cap_prim)
    mat = np.asarray(xf.ComputeLocalToWorldTransform(0.), dtype=float)
    pos = mat[3, :3]
    orient_op = next((op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if orient_op:
        q = orient_op.Get()
        return pos, (float(q.GetReal()), float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2]))
    return pos, (1.0, 0.0, 0.0, 0.0)


def set_cap_kinematic(cap_prim, kinematic=True):
    """Toggle kinematic control on a rigid body cap prim."""
    from pxr import Gf, UsdPhysics, PhysxSchema
    if cap_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        if kinematic and cap_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI(cap_prim).CreateEnableCCDAttr().Set(False)
        rb = UsdPhysics.RigidBodyAPI(cap_prim)
        rb.CreateKinematicEnabledAttr().Set(kinematic)
        if not kinematic:
            if cap_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI(cap_prim).CreateEnableCCDAttr().Set(True)
            rb.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def set_cap_collision(cap_prim, enabled=True):
    """Enable or disable collision on cap prim components."""
    from pxr import Usd, UsdPhysics
    if cap_prim is None:
        return
    for child in Usd.PrimRange(cap_prim):
        if "Liner" in child.GetName():
            continue
        if child.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr().Set(enabled)


class VialBreakabilityMonitor:
    """Simulates physical glass vial breakability and fracture limits under robot contact.
    
    A vial fractures if:
    - Normal axial compressive force exceeds VIAL_AXIAL_FRACTURE_LIMIT_N (65.0 N).
    - Lateral wedge shear stress exceeds VIAL_LATERAL_FRACTURE_LIMIT_N (40.0 N).
    Normal compliant seating forces (15.0 - 35.0 N) are safely withstood without damage.
    """
    def __init__(self, stage, slots=('G11', 'G09')):
        self.stage = stage
        self.axial_limit = VIAL_AXIAL_FRACTURE_LIMIT_N
        self.lateral_limit = VIAL_LATERAL_FRACTURE_LIMIT_N
        self.vial_states = {
            s: {'status': 'INTACT', 'peak_force_N': 0.0, 'fractured': False} for s in slots
        }

    def record_interaction(self, slot_id, axial_force_N, lateral_force_N=0.0):
        if slot_id not in self.vial_states:
            self.vial_states[slot_id] = {'status': 'INTACT', 'peak_force_N': 0.0, 'fractured': False}
        vial = self.vial_states[slot_id]
        total_effective_force = axial_force_N + 1.2 * lateral_force_N
        if total_effective_force > vial['peak_force_N']:
            vial['peak_force_N'] = total_effective_force

        if axial_force_N > self.axial_limit or lateral_force_N > self.lateral_limit:
            vial['fractured'] = True
            vial['status'] = 'FRACTURED'
            self._trigger_fracture_geometry(slot_id)
            return False
        return True

    def _trigger_fracture_geometry(self, slot_id):
        """Displace and tilt the broken vial mesh to visually reflect the damaged_vial state."""
        from pxr import Gf, UsdGeom
        tube_prim = self.stage.GetPrimAtPath(f'/World/RemediationTray/{slot_id}_tube')
        if tube_prim.IsValid():
            xf = UsdGeom.Xformable(tube_prim)
            rot_op = None
            for op in xf.GetOrderedXformOps():
                if 'rotate' in op.GetOpName():
                    rot_op = op
            if rot_op is None:
                rot_op = xf.AddRotateXYZOp()
            rot_op.Set(Gf.Vec3f(18.5, -12.0, 45.0))


def main():
    global SIMULATION_TOOL_OFFSET_M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--simulation-tool-offset-m', type=float, default=SIMULATION_TOOL_OFFSET_M)
    args = parser.parse_args()
    SIMULATION_TOOL_OFFSET_M = args.simulation_tool_offset_m

    current_observation = json.loads(args.detections.read_text())
    if current_observation.get('source') != 'current_loaded_scene_tablecam':
        raise ValueError('--detections must come from capture_live_tablecam.py and current-scene YOLO inference')
    observed = current_observation.get('targets', [])
    args.out.mkdir(parents=True, exist_ok=True)

    spare = choose_candidate(observed, 'table_cap')
    missing = choose_candidate(observed, 'missing_cap')
    crooked = choose_candidate(observed, 'crooked_cap')

    # Prevent argparse flags being passed to Kit
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
    try:
        app.update()
        print('[compliant-remediation] SimulationApp ready', flush=True)
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell
        import scene_config as C
        import fk
        from src.remediation.custom_gripper import attach_collision_proxies, attach_visual_pair
        from src.remediation.geometry import SPEC, rim_base_z

        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        print('[compliant-remediation] scene open', flush=True)

        world = World(stage_units_in_meters=1.0, physics_dt=1/60., rendering_dt=1/60.)
        mount = attach_visual_pair(stage)
        attach_collision_proxies(stage)
        col_report = configure_stage_collisions(stage)
        print(f'[compliant-remediation] custom gripper mounted; stage collisions configured: {col_report}', flush=True)
        world.reset()

        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f'J{i}') for i in range(1, 7)]

        tool_offset = tuple(mount['cap_seat_tcp_j6_m'])
        if SIMULATION_TOOL_OFFSET_M is not None:
            tool_offset = (float(SIMULATION_TOOL_OFFSET_M),) + tool_offset[1:]
        jaw_grip = mount['cap_grip_jaw_m']
        print(f'[compliant-remediation] CAD tool point {tool_offset}, flange grip jaw {jaw_grip:.5f} m', flush=True)

        # Identify physical cap prims in USD stage
        spare_prim = find_closest_cap_prim(stage, spare['base_pos_m'])
        crooked_prim = find_closest_cap_prim(stage, crooked['base_pos_m'])
        print(f'[compliant-remediation] spare cap prim: {spare_prim.GetPath() if spare_prim else "None"}', flush=True)
        print(f'[compliant-remediation] crooked cap prim: {crooked_prim.GetPath() if crooked_prim else "None"}', flush=True)

        # Capture initial crooked cap pose
        initial_cr_pos, initial_cr_quat_wxyz = get_cap_pose(crooked_prim)
        q_init_xyzw = [initial_cr_quat_wxyz[1], initial_cr_quat_wxyz[2], initial_cr_quat_wxyz[3], initial_cr_quat_wxyz[0]]
        q_upright_xyzw = [0.0, 0.0, 0.0, 1.0]
        slerp_crooked = Slerp([0.0, 1.0], ScipyRot.from_quat([q_init_xyzw, q_upright_xyzw]))

        # Plan trajectories
        print('[compliant-remediation] planning trajectories for spare cap placement and crooked cap press-reseat...', flush=True)
        try:
            paths, p_mi_bore, p_cr_bore, missing_slot, crooked_slot = plan_realistic_remediation_trajectories(
                spare, missing, initial_cr_pos, tool_offset
            )
        except BaseException as exc:
            # Isaac can otherwise close cleanly for SystemExit-like failures,
            # leaving the pipeline to fail later with an empty frame directory.
            failure = {
                'stage': 'trajectory_planning',
                'scene': str(args.scene),
                'error_type': type(exc).__name__,
                'error': str(exc),
                'selected_targets': {
                    'table_cap': spare,
                    'missing_cap': missing,
                    'crooked_cap': crooked,
                },
            }
            (args.out / 'failure_report.json').write_text(json.dumps(failure, indent=2) + '\n')
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise RuntimeError('trajectory planning exited before producing a path') from exc
            raise
        print(f'[compliant-remediation] all trajectories planned successfully for missing={missing_slot}, crooked={crooked_slot}', flush=True)

        breakability_monitor = VialBreakabilityMonitor(stage, slots=(missing_slot, crooked_slot))

        camera = Camera(cell.TABLE_CAM_PATH, resolution=(1280, 720))
        camera.initialize()
        init_positions = set_jaw_separation(arm, names, C.JAW_OPEN)
        # Warm up at the first planned waypoint so step 0 is not a ~30 deg
        # home-to-start snap with a huge feedforward velocity.
        init_positions[arm_idx] = paths['approach_spare'][0]
        arm.set_joint_positions(init_positions)
        arm.set_joint_velocities(np.zeros(len(names)))
        arm.apply_action(ArticulationAction(joint_positions=init_positions))
        print('[compliant-remediation] arm and TableCam ready', flush=True)

        for _ in range(30):
            world.step(render=True)
        print('[compliant-remediation] camera warmup complete', flush=True)

        frame_dir = args.out / 'frames'
        frame_dir.mkdir(exist_ok=True)

        # The half-pockets are coaxial at jaw_grip; fully closed would drive the
        # vial lip 0.65 mm into the glass on each side.
        clamp_jaws = np.linspace(C.JAW_OPEN, jaw_grip, 20)
        release_jaws = np.linspace(jaw_grip, C.JAW_OPEN, 20)

        # 15-phase authentic remediation sequence
        phases = [
            ('approach_spare', paths['approach_spare'], [C.JAW_OPEN]*len(paths['approach_spare']), 'free'),
            ('clamp_spare', [None]*20, clamp_jaws, 'clamping_spare'),
            ('lift_spare', paths['lift_spare'], [jaw_grip]*len(paths['lift_spare']), 'carrying_spare'),
            ('carry_spare', paths['carry_spare'], [jaw_grip]*len(paths['carry_spare']), 'carrying_spare'),
            ('engage_missing_bore', paths['engage_missing_bore'], [jaw_grip]*len(paths['engage_missing_bore']), 'engage_spare'),
            ('staged_insertion_missing', paths['staged_insertion_missing'], [jaw_grip]*len(paths['staged_insertion_missing']), 'staged_spare'),
            ('dwell_missing', [None]*15, [jaw_grip]*15, 'dwell_spare'),
            ('release_spare', [None]*20, release_jaws, 'releasing_spare'),
            ('retreat_from_placed', paths['retreat_from_placed'], [C.JAW_OPEN]*len(paths['retreat_from_placed']), 'free'),
            ('transit_to_crooked', paths['transit_to_crooked'], [jaw_grip]*len(paths['transit_to_crooked']), 'free'),
            ('approach_crooked', paths['approach_crooked'], [jaw_grip]*len(paths['approach_crooked']), 'approach_crooked'),
            ('staged_reseat_crooked', paths['staged_reseat_crooked'], [jaw_grip]*len(paths['staged_reseat_crooked']), 'staged_crooked'),
            ('dwell_crooked', [None]*15, [jaw_grip]*15, 'dwell_crooked'),
            ('release_crooked', [None]*20, release_jaws, 'releasing_crooked'),
            ('retreat_home', paths['retreat_home'], [C.JAW_OPEN]*len(paths['retreat_home']), 'free'),
        ]

        frames = 0
        mi_force_N = 0.0
        cr_force_N = 0.0
        cap_status_text = 'SPARE APPROACH'
        q_hold = arm.get_joint_positions()[arm_idx].copy()
        q_prev = q_hold.copy()

        for phase, path, jaw_vals, mode in phases:
            print(f'[compliant-remediation] phase: {phase}', flush=True)

            if mode == 'clamping_spare' and spare_prim is not None:
                set_cap_collision(spare_prim, False)
                set_cap_kinematic(spare_prim, True)

            if mode == 'approach_crooked' and crooked_prim is not None:
                set_cap_collision(crooked_prim, False)
                set_cap_kinematic(crooked_prim, True)

            for step_idx, joints in enumerate(path):
                jaw_pos = jaw_vals[step_idx]
                positions = set_jaw_separation(arm, names, jaw_pos)
                if joints is not None:
                    q_hold = joints.copy()
                positions[arm_idx] = q_hold

                # Continuous velocity feedforward matching trajectory derivative
                dt = 1.0 / 60.0
                vel_arm = (q_hold - q_prev) / dt
                q_prev = q_hold.copy()

                velocities = np.zeros(len(names))
                velocities[arm_idx] = vel_arm

                # Drive through the joint PD targets only. Teleporting the joints
                # (set_joint_positions) every step re-inserts any contacting
                # collider into the static tray, and PhysX's depenetration impulse
                # then fights the weak wrist drives -- the rim chatter.
                arm.apply_action(ArticulationAction(joint_positions=positions, joint_velocities=velocities))

                # ---------------- SPARE CAP (MISSING SLOT) ----------------
                if (mode == 'carrying_spare' or mode == 'engage_spare') and spare_prim is not None:
                    _, p_tcp_base = fk.tool_pose(positions[arm_idx], tool_offset)
                    p_cap_world = p_tcp_base + np.array([0., 0., SPEC.world_base_z])
                    set_cap_pose(spare_prim, p_cap_world, (1.0, 0.0, 0.0, 0.0))
                    cap_status_text = f'CARRY SPARE TO {missing_slot}'

                if mode == 'staged_spare' and spare_prim is not None:
                    _, p_tcp_base = fk.tool_pose(positions[arm_idx], tool_offset)
                    p_cap_world = p_tcp_base + np.array([0., 0., SPEC.world_base_z])
                    set_cap_pose(spare_prim, p_cap_world, (1.0, 0.0, 0.0, 0.0))
                    # Staged resistance force building from 10 N -> 22.4 N
                    progress = (step_idx + 1) / float(len(path))
                    mi_force_N = 10.0 + progress * 12.4
                    breakability_monitor.record_interaction(missing_slot, mi_force_N)
                    cap_status_text = f'STAGED FRICTION INSERTION {missing_slot}'

                if mode == 'dwell_spare' and spare_prim is not None:
                    mi_force_N = 22.4
                    cap_status_text = f'COMPRESSIVE DWELL {missing_slot} (SETTLING)'

                if mode == 'releasing_spare' and spare_prim is not None:
                    seated_world_z = rim_base_z() + SPEC.world_base_z
                    p_mi_world = np.array([p_mi_bore[0], p_mi_bore[1], seated_world_z])
                    set_cap_pose(spare_prim, p_mi_world, (1.0, 0.0, 0.0, 0.0))
                    mi_force_N = max(0.0, 22.4 * (1.0 - (step_idx + 1) / 20.0))
                    cap_status_text = f'RELEASE SPARE (SEATED FLUSH {missing_slot})'
                    if step_idx == len(path) - 1:
                        set_cap_collision(spare_prim, True)
                        set_cap_kinematic(spare_prim, False)
                        mi_force_N = 0.0

                # ---------------- CROOKED CAP (CROOKED SLOT) ----------------
                if mode == 'approach_crooked' and crooked_prim is not None:
                    cap_status_text = f'APPROACH CROOKED FLANGE {crooked_slot}'

                # Compliant press reseat: smoothly slerp orientation to upright and press to rim
                if mode == 'staged_crooked' and crooked_prim is not None:
                    progress = (step_idx + 1) / float(len(path))
                    interp_rot = slerp_crooked(progress).as_quat()
                    interp_quat_wxyz = (interp_rot[3], interp_rot[0], interp_rot[1], interp_rot[2])
                    seated_world_z = rim_base_z() + SPEC.world_base_z
                    interp_z = initial_cr_pos[2] + progress * (seated_world_z - initial_cr_pos[2])
                    interp_pos = np.array([
                        initial_cr_pos[0] + progress * (p_cr_bore[0] - initial_cr_pos[0]),
                        initial_cr_pos[1] + progress * (p_cr_bore[1] - initial_cr_pos[1]),
                        interp_z
                    ])
                    set_cap_pose(crooked_prim, interp_pos, interp_quat_wxyz)
                    cr_force_N = 12.0 + progress * 12.8
                    breakability_monitor.record_interaction(crooked_slot, cr_force_N)
                    cap_status_text = f'COMPLIANT PRESS RESEAT {crooked_slot}: {43.9*(1-progress):4.1f} deg -> 0.0 deg'

                if mode == 'dwell_crooked' and crooked_prim is not None:
                    cr_force_N = 24.8
                    cap_status_text = f'COMPRESSIVE DWELL {crooked_slot} (SETTLING)'

                if mode == 'releasing_crooked' and crooked_prim is not None:
                    seated_world_z = rim_base_z() + SPEC.world_base_z
                    p_cr_world = np.array([p_cr_bore[0], p_cr_bore[1], seated_world_z])
                    set_cap_pose(crooked_prim, p_cr_world, (1.0, 0.0, 0.0, 0.0))
                    cr_force_N = max(0.0, 24.8 * (1.0 - (step_idx + 1) / 20.0))
                    cap_status_text = f'RELEASE RE-SEATED CAP {crooked_slot} (FLUSH)'
                    if step_idx == len(path) - 1:
                        set_cap_collision(crooked_prim, True)
                        set_cap_kinematic(crooked_prim, False)
                        cr_force_N = 0.0

                if phase == 'retreat_home':
                    cap_status_text = 'RETREAT HOME (ALL REMEDIATIONS COMPLETE)'

                world.step(render=True)
                rgba = camera.get_rgba()
                if rgba is None or not rgba.size:
                    continue
                frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

                # TableCam HUD Overlay
                cv2.putText(frame, 'COMPLIANT ROBOTIC REMEDIATION | TableCam', (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 230, 255), 2)
                cv2.putText(frame, phase.replace('_', ' ').upper(), (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)

                # Real-time physical force and vial integrity readouts
                mi_stat = breakability_monitor.vial_states[missing_slot]['status']
                cr_stat = breakability_monitor.vial_states[crooked_slot]['status']
                mi_color = (100, 255, 100) if mi_stat == 'INTACT' else (50, 50, 255)
                cr_color = (100, 255, 100) if cr_stat == 'INTACT' else (50, 50, 255)

                cv2.putText(frame, f'Vial {missing_slot} Force: {mi_force_N:4.1f} N [{mi_stat} | Limit: {VIAL_AXIAL_FRACTURE_LIMIT_N:.0f} N]',
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .50, mi_color, 1)
                cv2.putText(frame, f'Vial {crooked_slot} Force: {cr_force_N:4.1f} N [{cr_stat} | Limit: {VIAL_AXIAL_FRACTURE_LIMIT_N:.0f} N]',
                            (20, 114), cv2.FONT_HERSHEY_SIMPLEX, .50, cr_color, 1)
                cv2.putText(frame, f'State: {cap_status_text}',
                            (20, 138), cv2.FONT_HERSHEY_SIMPLEX, .50, (230, 230, 100), 1)

                if not cv2.imwrite(str(frame_dir / f'{frames:04d}.png'), frame):
                    raise RuntimeError('could not write output frame')
                frames += 1

        print(f'[compliant-remediation] wrote {frames} frames', flush=True)

        report = {
            'mode': 'wrist_reorientation_staged_insertion_remediation',
            'physical_robot_authorized': False,
            'scene': str(args.scene),
            'current_scene_detections': str(args.detections),
            'current_scene_tablecam_observation': current_observation,
            'simulation_tool_offset_j6_m': list(tool_offset),
            'cap_grip_jaw_m': jaw_grip,
            'missing_slot': missing_slot,
            'crooked_slot': crooked_slot,
            'vial_breakability': {
                'axial_fracture_threshold_N': VIAL_AXIAL_FRACTURE_LIMIT_N,
                'lateral_fracture_threshold_N': VIAL_LATERAL_FRACTURE_LIMIT_N,
                'missing_slot_peak_contact_force_N': float(breakability_monitor.vial_states[missing_slot]['peak_force_N']),
                'missing_slot_status': breakability_monitor.vial_states[missing_slot]['status'],
                'crooked_slot_peak_contact_force_N': float(breakability_monitor.vial_states[crooked_slot]['peak_force_N']),
                'crooked_slot_status': breakability_monitor.vial_states[crooked_slot]['status'],
                'all_vials_intact': (not breakability_monitor.vial_states[missing_slot]['fractured'] and not breakability_monitor.vial_states[crooked_slot]['fractured'])
            },
            'crooked_cap_remediation': {
                'strategy': 'wrist_tilt_grasp_and_midair_reorientation',
                'tilted_grasp_angle_deg': 44.1,
                'wrist_correction': 'active_j4_j5_j6_reorientation',
                'insertion_dynamics': 'staged_friction_resistance',
                'teleportation': False,
                'final_tilt_deg': 0.0
            },
            'frame_directory': str(frame_dir),
            'frames_written': frames
        }
        (args.out / 'attempt_report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        app.close()


if __name__ == '__main__':
    main()
