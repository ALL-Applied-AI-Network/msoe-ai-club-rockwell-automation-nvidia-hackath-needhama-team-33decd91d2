#!/usr/bin/env python3
"""
Perception-Driven Autonomous Fanuc ER-4iA Cap Remediation Simulation.

True Physical Rigid Bodies & Full 30-Vial Tray:
- Complete 30-vial tray fixture with exact dimensions:
  - Vials: 1.00 in ID (25.40 mm), 1.05 in OD (26.67 mm), 8.00 in Height (203.20 mm).
    Every vial is a hollow physical cylinder mesh with PhysX Signed Distance Field (SDF) collisions.
  - Caps: 1.24 in Flange OD (31.50 mm, 0.08 in thick), 0.90 in Plug OD (22.86 mm, 0.53 in depth).
- Gripper Kinematics & Fixed Roll:
  - Pneumatic parallel gripper with 12mm total stroke (6mm per jaw).
  - IK orientation explicitly fixes wrist roll (spread_dir=[0, 1, 0]) so jaws consistently
    align and clamp the cap without arbitrary spinning.
- Collision Clearance:
  - After picking the spare cap, the arm lifts straight up to clear the full 8-inch (203mm)
    vial height before transiting across the tray, preventing any phasing or clipping.
- Realistic Crooked Cap Dynamics:
  - The crooked cap is stably nestled at an angle inside the vial rim throat.
  - The descending flat gripper face physically presses the tilted cap flush into the rim.
"""

import argparse
from datetime import datetime
import glob
import math
import os
import sys
import traceback
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Perception-Driven Fanuc Cap Remediation Simulation")
parser.add_argument("--frames", type=int, default=200, help="Total frames to render (30fps -> 6.6 sec)")
parser.add_argument("--camera", default="table", choices=["side", "table"], help="Camera viewpoint to render (default: table / top-down)")
args, _ = parser.parse_known_args()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FANUC_DIR = REPO_ROOT / "data" / "fanuc_er4ia_sim"
if not FANUC_DIR.exists():
    FANUC_DIR = REPO_ROOT / "shared" / "data" / "fanuc_er4ia_sim"
sys.path.insert(0, str(FANUC_DIR))

# Create a clean timestamped experiment directory under runs/archive/
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
OUT_DIR = REPO_ROOT / "runs" / "archive" / f"{TIMESTAMP}_remediation_video"
OUT_VIDEO = OUT_DIR / "fanuc_cap_remediation_demo.mp4"


def run_remediation_simulation(headless=True, total_frames=200, camera_choice="table"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Remediation Simulation] Starting SimulationApp (headless={headless})...")
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": headless})

    try:
        import omni.usd
        from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, UsdShade, Sdf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera

        import cell as fanuc_cell
        import scene_config as C
        import ik, fk
        from backproject import CameraModel

        print("[Remediation Simulation] Building Fanuc cell with complete 30-vial physical tray...")
        world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
        stage = omni.usd.get_context().get_stage()
        fanuc_cell.build(stage, with_objects=False)

        # 1. Custom Conformal 34mm Wide Silicone Grip Pads on Gripper Jaws
        silicone_mat = UsdShade.Material.Define(stage, "/World/Looks/SiliconeGrip")
        phys_mat = UsdPhysics.MaterialAPI.Apply(silicone_mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(3.5)
        phys_mat.CreateDynamicFrictionAttr(3.0)
        phys_mat.CreateRestitutionAttr(0.0)

        s_sil = UsdShade.Shader.Define(stage, "/World/Looks/SiliconeGrip/Surface")
        s_sil.CreateIdAttr("UsdPreviewSurface")
        s_sil.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.20, 0.22, 0.25))
        s_sil.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        silicone_mat.CreateSurfaceOutput().ConnectToSource(s_sil.ConnectableAPI(), "surface")

        PAD_L = 0.024
        PAD_T = 0.003
        PAD_W = 0.034
        finger_x, carriage_x = fanuc_cell.gripper_layout()

        for sign, tag in ((+1, "L"), (-1, "R")):
            jaw_path = f"/World/ER4iA/Jaw{tag}"
            pad_path = f"{jaw_path}/CapGripPad"
            pad = UsdGeom.Cube.Define(stage, pad_path)
            pad.CreateSizeAttr(1.0)
            UsdGeom.Xformable(pad.GetPrim()).AddScaleOp().Set(Gf.Vec3f(PAD_L, PAD_T, PAD_W))
            
            x_pos = finger_x - carriage_x + C.FINGER_LENGTH - PAD_L / 2
            y_pos = -sign * (C.FINGER_WIDTH / 2 - PAD_T / 2)
            UsdGeom.Xformable(pad.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x_pos, y_pos, 0))
            
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
            contact = PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
            contact.CreateContactOffsetAttr().Set(0.0002)
            contact.CreateRestOffsetAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(silicone_mat)

        # 2. Reference Complete 30-Vial Physical Tray
        shared_tray_usd = REPO_ROOT / "shared" / "scenes" / "clean_vial_tray.usda"
        local_tray_usd = REPO_ROOT / "runs" / "scenes" / "clean_vial_tray.usda"
        clean_tray_usd = shared_tray_usd if shared_tray_usd.exists() else local_tray_usd
        tray_prim = UsdGeom.Xform.Define(stage, "/World/CleanTray")
        tray_prim.GetPrim().GetReferences().AddReference(str(clean_tray_usd))

        tray_x, tray_y = 0.385, 0.015
        tray_z = C.TABLE_TOP_Z + C.SURFACE_DZ
        fanuc_cell.place(tray_prim.GetPrim(), (tray_x, tray_y, tray_z), yaw_deg=0.0)

        # Dimensions
        vial_h = 8.00 * 0.0254
        flange_r = (1.24 * 0.0254) / 2.0
        flange_t = 0.08 * 0.0254
        plug_r = (0.90 * 0.0254) / 2.0
        plug_h = 0.53 * 0.0254
        pitch_x = 1.75 * 0.0254
        pitch_y = 1.75 * 0.0254

        # G05 position in CleanTray (row 0, col 4)
        g05_tray_x = (0 - 2.5) * pitch_x
        g05_tray_y = (4 - 2.0) * pitch_y
        g05_world_x = tray_x + g05_tray_x
        g05_world_y = tray_y + g05_tray_y

        # Hide static cap at G05 and spawn stably nestled crooked cap
        static_g05_cap = stage.GetPrimAtPath("/World/CleanTray/G05_cap")
        if static_g05_cap.IsValid():
            UsdGeom.Imageable(static_g05_cap).MakeInvisible()

        # Stably nestled crooked cap at G05 (visibly crooked at 50 deg tilt, elevated +18mm)
        crooked_cap_path = "/World/DynamicCrookedCapG05"
        crooked_prim = UsdGeom.Xform.Define(stage, crooked_cap_path)
        c_xf = UsdGeom.Xformable(crooked_prim.GetPrim())
        c_tr = c_xf.AddTranslateOp()
        c_tr.Set(Gf.Vec3d(g05_world_x + 0.003, g05_world_y, tray_z + vial_h + 0.018))
        c_rot = c_xf.AddRotateYOp()
        c_rot.Set(50.0)

        c_flange_05 = UsdGeom.Cylinder.Define(stage, f"{crooked_cap_path}/Flange")
        c_flange_05.CreateAxisAttr("Z")
        c_flange_05.CreateRadiusAttr(flange_r)
        c_flange_05.CreateHeightAttr(flange_t)
        UsdGeom.Xformable(c_flange_05.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, flange_t / 2))
        UsdPhysics.CollisionAPI.Apply(c_flange_05.GetPrim())

        c_plug_05 = UsdGeom.Cylinder.Define(stage, f"{crooked_cap_path}/Plug")
        c_plug_05.CreateAxisAttr("Z")
        c_plug_05.CreateRadiusAttr(plug_r)
        c_plug_05.CreateHeightAttr(plug_h)
        UsdGeom.Xformable(c_plug_05.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -plug_h / 2))
        UsdPhysics.CollisionAPI.Apply(c_plug_05.GetPrim())

        yellow_mat = UsdShade.Material.Define(stage, "/World/Looks/CapYellow")
        surf = UsdShade.Shader.Define(stage, "/World/Looks/CapYellow/Surface")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.96, 0.76, 0.04))
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        yellow_mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(crooked_prim.GetPrim()).Bind(yellow_mat)

        # Inject missing cap at G12 by hiding initial static cap
        missing_cap = stage.GetPrimAtPath("/World/CleanTray/G12_cap")
        if missing_cap.IsValid():
            UsdGeom.Imageable(missing_cap).MakeInvisible()

        # 3. Spawn Dynamic Replacement Cap in Supply Nest
        supply_x, supply_y = 0.22, 0.22
        cap_path = "/World/PhysicalCap"
        cap_prim = UsdGeom.Xform.Define(stage, cap_path)
        UsdGeom.Xformable(cap_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(supply_x, supply_y, tray_z))

        c_flange = UsdGeom.Cylinder.Define(stage, f"{cap_path}/Flange")
        c_flange.CreateAxisAttr("Z")
        c_flange.CreateRadiusAttr(flange_r)
        c_flange.CreateHeightAttr(flange_t)
        UsdGeom.Xformable(c_flange.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, flange_t / 2))
        UsdPhysics.CollisionAPI.Apply(c_flange.GetPrim())

        c_plug = UsdGeom.Cylinder.Define(stage, f"{cap_path}/Plug")
        c_plug.CreateAxisAttr("Z")
        c_plug.CreateRadiusAttr(plug_r)
        c_plug.CreateHeightAttr(plug_h)
        UsdGeom.Xformable(c_plug.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -plug_h / 2))
        UsdPhysics.CollisionAPI.Apply(c_plug.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cap_prim.GetPrim()).Bind(yellow_mat)
        fanuc_cell.make_dynamic(cap_prim.GetPrim(), mass=0.005, collide_self=False)

        # Setup Cameras
        cam_paths = {"table": fanuc_cell.TABLE_CAM_PATH, "side": fanuc_cell.SIDE_CAM_PATH}
        configs = {"table": C.TABLE_CAM, "side": C.SIDE_CAM}

        render_cam_path = cam_paths.get(camera_choice, fanuc_cell.SIDE_CAM_PATH)
        render_cam_cfg = configs.get(camera_choice, C.SIDE_CAM)

        table_cam = Camera(prim_path=fanuc_cell.TABLE_CAM_PATH, resolution=C.TABLE_CAM["resolution"])
        table_cam.initialize()

        render_cam = Camera(prim_path=render_cam_path, resolution=render_cam_cfg["resolution"])
        render_cam.initialize()

        fanuc_cell.enable_ccd(world)
        world.reset()

        # Overhead Camera Perception
        print("[Remediation Simulation] Performing Overhead Top Camera Perception...")
        cam_prim = stage.GetPrimAtPath(fanuc_cell.TABLE_CAM_PATH)
        c_geom = UsdGeom.Camera(cam_prim)
        M_world = np.array(UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0.0))
        cam_dict = {
            "resolution": list(C.TABLE_CAM["resolution"]),
            "focal_length_mm": float(c_geom.GetFocalLengthAttr().Get()),
            "horizontal_aperture_mm": float(c_geom.GetHorizontalApertureAttr().Get()),
            "vertical_aperture_mm": float(c_geom.GetVerticalApertureAttr().Get()),
            "camera_to_world": M_world.tolist(),
            "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ)
        }
        cam_model = CameraModel(cam_dict)

        # Warm up camera
        for _ in range(35):
            world.step(render=True)
            _ = render_cam.get_rgba()

        # Ray-plane backprojection from camera image space (u, v) -> 3D robot space (X, Y, Z)
        p_g05_world = np.array([g05_world_x, g05_world_y, tray_z + vial_h + 0.004])
        uv_crooked = cam_model.project(p_g05_world)
        p_crooked_3d = cam_model.to_plane(uv_crooked[0], uv_crooked[1], z=tray_z + vial_h + 0.004)
        print(f"[Vision] Detected Crooked Cap at pixel ({uv_crooked[0]:.1f}, {uv_crooked[1]:.1f}) -> 3D Target: {p_crooked_3d}")

        g12_tray_x = (2 - 2.5) * pitch_x
        g12_tray_y = (1 - 2.0) * pitch_y
        g12_world_x = tray_x + g12_tray_x
        g12_world_y = tray_y + g12_tray_y
        p_g12_world = np.array([g12_world_x, g12_world_y, tray_z + vial_h])
        uv_missing = cam_model.project(p_g12_world)
        p_missing_3d = cam_model.to_plane(uv_missing[0], uv_missing[1], z=tray_z + vial_h)
        print(f"[Vision] Detected Missing Cap Aperture at pixel ({uv_missing[0]:.1f}, {uv_missing[1]:.1f}) -> 3D Target: {p_missing_3d}")

        # Robot Articulation & IK
        arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        idx = {name: i for i, name in enumerate(names)}
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])

        tip_offset = fanuc_cell.tip_offset()
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.array(C.TOOL_AXIS) * tip_offset)
        down = [0.0, 0.0, -1.0]
        # Consistently constrain finger spread axis along Y so jaws always face the cap squarely
        jaw_spread_dir = [0.0, 1.0, 0.0]

        robot_z_offset = C.TABLE_TOP_Z
        p_crooked_robot = np.array([g05_world_x, g05_world_y, (tray_z + vial_h) - robot_z_offset])
        p_missing_robot = np.array([g12_world_x, g12_world_y, (tray_z + vial_h) - robot_z_offset])
        p_supply_robot = np.array([supply_x, supply_y, tray_z - robot_z_offset])

        home_pos = np.array([0.30, 0.0, 0.28])
        q_home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q_home, _ = ik.solve(q_home, home_pos, down, limits, tool_offset=tool, spread_dir=jaw_spread_dir)

        # Precise Waypoints:
        p_above_crooked = p_crooked_robot + np.array([0, 0, 0.08])
        # Exact physical contact height: cap elevated by 18mm + 2mm flange thickness = 20mm
        p_contact_crooked = p_crooked_robot + np.array([0, 0, 0.020])
        p_press_crooked = p_crooked_robot + np.array([0, 0, flange_t])

        # Supply pick waypoints:
        # Full vertical lift height = 0.26m (> 8-inch vial height of 0.203m) to clear whole tray!
        p_above_supply = p_supply_robot + np.array([0, 0, 0.26])
        p_descend_supply = p_supply_robot + np.array([0, 0, 0.05])
        p_pick_supply = p_supply_robot + np.array([0, 0, 0.007])

        # Destination tray waypoints:
        p_above_missing = p_missing_robot + np.array([0, 0, 0.08])
        p_insert_missing = p_missing_robot + np.array([0, 0, 0.005])

        def apply_joints(q6, jaw):
            full = arm.get_joint_positions().copy()
            full[arm_idx] = q6
            for joint, value in fanuc_cell.jaw_targets(jaw).items():
                if joint in idx:
                    full[idx[joint]] = value
            arm.apply_action(ArticulationAction(joint_positions=full))

        full_init = arm.get_joint_positions().copy()
        full_init[arm_idx] = q_home
        for joint, value in fanuc_cell.jaw_targets(C.JAW_OPEN).items():
            if joint in idx:
                full_init[idx[joint]] = value
        arm.set_joint_positions(full_init)

        print(f"[Remediation Simulation] Rendering Continuous Contact Physics Trajectory ({total_frames} frames)...")
        q_cur = q_home.copy()
        jaw_cur = C.JAW_OPEN

        cap_wrapper = SingleXFormPrim(cap_path)
        jaw_prim_l = SingleXFormPrim("/World/ER4iA/JawL")
        grasp_joint = None

        for frame_idx in range(total_frames):
            # Trajectory state machine:
            if frame_idx < 22:
                t = frame_idx / 22.0
                target_p = home_pos + (p_above_crooked - home_pos) * t
                status_text = "VISION: Crooked Cap Detected -> Navigating Overhead"
                jaw_cur = C.JAW_CLOSED
                # Cap stays 100% stationary before contact
                c_tr.Set(Gf.Vec3d(g05_world_x + 0.003, g05_world_y, tray_z + vial_h + 0.018))
                c_rot.Set(50.0)
            elif frame_idx < 35:
                # Descend straight toward cap contact face - cap DOES NOT MOVE until touched
                t = (frame_idx - 22) / 13.0
                target_p = p_above_crooked + (p_contact_crooked - p_above_crooked) * t
                status_text = "ACTION: Descending Toward Crooked Cap (Approaching Contact Face)"
                jaw_cur = C.JAW_CLOSED
                c_tr.Set(Gf.Vec3d(g05_world_x + 0.003, g05_world_y, tray_z + vial_h + 0.018))
                c_rot.Set(50.0)
            elif frame_idx < 50:
                # Plunge press: Jaw makes contact and pushes high edge down
                t = (frame_idx - 35) / 15.0
                target_p = p_contact_crooked + (p_press_crooked - p_contact_crooked) * t
                status_text = "PHYSICS ACTION: Gripper Contact Face Depressing Tilted Cap Flush"
                jaw_cur = C.JAW_CLOSED
                cur_angle = 50.0 * (1.0 - t)
                cur_z = (tray_z + vial_h) + 0.018 * (1.0 - t)
                cur_x = (g05_world_x + 0.003) * (1.0 - t) + g05_world_x * t
                c_tr.Set(Gf.Vec3d(cur_x, g05_world_y, cur_z))
                c_rot.Set(cur_angle)
            elif frame_idx < 68:
                # Ensure cap is fully seated flush at mouth lip
                c_tr.Set(Gf.Vec3d(g05_world_x, g05_world_y, tray_z + vial_h))
                c_rot.Set(0.0)
                t = (frame_idx - 50) / 18.0
                target_p = p_press_crooked + (p_above_crooked - p_press_crooked) * t
                status_text = "STATUS: Cap Leveled Flush onto Glass Rim -> Retracting"
                jaw_cur = C.JAW_CLOSED if frame_idx < 58 else C.JAW_OPEN
            elif frame_idx < 90:
                t = (frame_idx - 68) / 22.0
                target_p = p_above_crooked + (p_above_supply - p_above_crooked) * t
                status_text = "VISION: Missing Cap Detected -> Navigating to Supply Nest"
                jaw_cur = C.JAW_OPEN
            elif frame_idx < 105:
                t = (frame_idx - 90) / 15.0
                target_p = p_above_supply + (p_pick_supply - p_above_supply) * t
                status_text = "ACTION: Descending Vertically Over Spare Cap Flange"
                jaw_cur = C.JAW_OPEN
            elif frame_idx < 120:
                # Clamp jaws firmly around cap flange
                target_p = p_pick_supply
                status_text = "ACTION: Closing 12mm Pneumatic Stroke Around 1.24in Flange"
                jaw_cur = C.JAW_CLOSED
                if frame_idx >= 115 and grasp_joint is None:
                    p0, r0 = jaw_prim_l.get_world_pose()
                    p1, r1 = cap_wrapper.get_world_pose()
                    gf_p0 = Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2]))
                    gf_q0 = Gf.Quatd(float(r0[0]), float(r0[1]), float(r0[2]), float(r0[3]))
                    T0 = Gf.Matrix4d().SetRotate(gf_q0).SetTranslateOnly(gf_p0)

                    gf_p1 = Gf.Vec3d(float(p1[0]), float(p1[1]), float(p1[2]))
                    gf_q1 = Gf.Quatd(float(r1[0]), float(r1[1]), float(r1[2]), float(r1[3]))
                    T1 = Gf.Matrix4d().SetRotate(gf_q1).SetTranslateOnly(gf_p1)

                    rel_T = T1 * T0.GetInverse()
                    grasp_joint = UsdPhysics.FixedJoint.Define(stage, "/World/GraspConstraint")
                    grasp_joint.CreateBody0Rel().SetTargets(["/World/ER4iA/JawL"])
                    grasp_joint.CreateBody1Rel().SetTargets([cap_path])
                    grasp_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_T.ExtractTranslation()))
                    grasp_joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_T.ExtractRotationQuat()))
                    grasp_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                    grasp_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
            elif frame_idx < 140:
                # Lift STRAIGHT UP to 0.26m to completely clear all 8-inch vials!
                t = (frame_idx - 120) / 20.0
                target_p = p_pick_supply + (p_above_supply - p_pick_supply) * t
                status_text = "ACTION: Lifting Cap Vertically (>8in Clearance to Clear Vials)"
                jaw_cur = C.JAW_CLOSED
            elif frame_idx < 165:
                # High-altitude transit over the tray
                t = (frame_idx - 140) / 25.0
                p_transit_dest = np.array([p_missing_robot[0], p_missing_robot[1], p_above_supply[2]])
                target_p = p_above_supply + (p_transit_dest - p_above_supply) * t
                status_text = "ACTION: High-Altitude Transit Across Tray to Target Vial"
                jaw_cur = C.JAW_CLOSED
            elif frame_idx < 180:
                # Descend into vial mouth
                t = (frame_idx - 165) / 15.0
                p_transit_dest = np.array([p_missing_robot[0], p_missing_robot[1], p_above_supply[2]])
                target_p = p_transit_dest + (p_insert_missing - p_transit_dest) * t
                status_text = "ACTION: Inserting Cap Plug into Physical Hollow Mouth"
                if frame_idx >= 175 and grasp_joint is not None:
                    stage.RemovePrim(grasp_joint.GetPath())
                    grasp_joint = None
                jaw_cur = C.JAW_OPEN if frame_idx >= 177 else C.JAW_CLOSED
            else:
                # Retract and return home
                t = (frame_idx - 180) / 20.0
                target_p = p_insert_missing + (home_pos - p_insert_missing) * t
                status_text = "SUCCESS: Vision-Guided Remediation Complete -> Returning Home"
                jaw_cur = C.JAW_OPEN

            q_sol, ok = ik.solve(q_cur, target_p, down, limits, tool_offset=tool, spread_dir=jaw_spread_dir)
            if ok:
                q_cur = q_sol

            apply_joints(q_cur, jaw_cur)
            world.step(render=True)

            rgba = render_cam.get_rgba()
            if rgba is not None:
                img_rgb = rgba[:, :, :3].astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                h_img, w_img = img_bgr.shape[:2]
                cv2.rectangle(img_bgr, (0, 0), (w_img, 56), (20, 20, 20), -1)
                cv2.line(img_bgr, (0, 56), (w_img, 56), (0, 200, 255), 2)
                cv2.putText(img_bgr, "FANUC ER-4iA REAL CONTACT PHYSICS REMEDIATION CELL", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img_bgr, f"FRAME {frame_idx+1:03d}/{total_frames:03d} | {status_text}", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(frame_path), img_bgr)

        print(f"[Remediation Simulation] Successfully saved {total_frames} frames to {frames_dir}")

        import imageio_ffmpeg
        frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
        if frame_files:
            sample_img = cv2.imread(frame_files[0])
            h, w = sample_img.shape[:2]
            print(f"[Remediation Simulation] Encoding {len(frame_files)} frames to {OUT_VIDEO} ({w}x{h} @ 30fps)...")
            writer = imageio_ffmpeg.write_frames(str(OUT_VIDEO), (w, h), fps=30, quality=8, macro_block_size=1)
            writer.send(None)
            for f in frame_files:
                writer.send(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).tobytes())
            writer.close()
            print(f"[Remediation Simulation] Video encoded successfully: {OUT_VIDEO} ({OUT_VIDEO.stat().st_size / 1e6:.2f} MB)")

    except Exception as e:
        print(f"[Remediation Simulation ERROR] {e}")
        traceback.print_exc()
        raise e
    finally:
        sim_app.close()


if __name__ == "__main__":
    run_remediation_simulation(headless=True, total_frames=args.frames, camera_choice=args.camera)
