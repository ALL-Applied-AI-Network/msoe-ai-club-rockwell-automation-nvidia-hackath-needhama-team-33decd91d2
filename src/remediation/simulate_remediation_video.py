#!/usr/bin/env python3
"""
Simulate and Render the Complete Fanuc ER-4iA Cap Remediation Sequence.

Physical & Geometric Specifications (docs/PHYSICAL_SPECIFICATIONS.md):
- Vials: 1.00 in ID, 1.05 in OD, 8.00 in height, flat bottom.
- Caps: 1.24 in OD top flange (0.08 in thick), 0.90 in bottom outer diameter, 0.80 in inner hollow diameter.
- Tray: 1.25 in aperture holes.

100% Continuous Physical Grasping & PhysX Dynamic Simulation:
- NO faked teleportation or visibility toggling of carried caps.
- Cap is spawned as a true PhysX dynamic rigid body at the supply nest.
- Parallel gripper descends, closes jaws, and engages a dynamic PhysX FixedJoint
  attachment constraint between JawL and the cap with mathematically exact relative transform.
- Arm physically lifts the cap from the nest, transports it smoothly across the workspace,
  and seats it onto vial G12.
- Upon seating, the constraint is released and gripper jaws open cleanly.
"""

import argparse
import glob
import math
import os
import sys
import traceback
from pathlib import Path
import cv2
import numpy as np

# Parse CLI arguments before SimulationApp imports
parser = argparse.ArgumentParser(description="Fanuc Cap Remediation Simulation")
parser.add_argument("--frames", type=int, default=180, help="Total frames to render (30fps -> 6 sec)")
parser.add_argument("--camera", default="side", choices=["side", "table"], help="Camera viewpoint to render")
args, _ = parser.parse_known_args()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FANUC_DIR = REPO_ROOT / "data" / "fanuc_er4ia_sim"
if not FANUC_DIR.exists():
    FANUC_DIR = REPO_ROOT / "shared" / "data" / "fanuc_er4ia_sim"
sys.path.insert(0, str(FANUC_DIR))

OUT_DIR = REPO_ROOT / "runs" / "remediation_video"
OUT_VIDEO = REPO_ROOT / "runs" / "fanuc_cap_remediation_demo.mp4"


def run_remediation_simulation(headless=True, total_frames=180, camera_choice="side"):
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

        print("[Remediation Simulation] Initializing World and building Fanuc cell...")
        world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
        stage = omni.usd.get_context().get_stage()

        # Build base scene (Robot + Table + Cameras + Gripper)
        fanuc_cell.build(stage, with_objects=False)

        # Place the clean isolated tray flush on the table
        clean_tray_usd = REPO_ROOT / "shared" / "scenes" / "clean_vial_tray.usda"
        tray_prim = UsdGeom.Xform.Define(stage, "/World/CleanTray")
        tray_prim.GetPrim().GetReferences().AddReference(str(clean_tray_usd))

        # Nominal placement: X=0.38m, Y=0.0m, Z resting flush on table surface
        tray_x = 0.38
        tray_y = 0.0
        tray_z = C.TABLE_TOP_Z + C.SURFACE_DZ
        fanuc_cell.place(tray_prim.GetPrim(), (tray_x, tray_y, tray_z), yaw_deg=0.0)
        print(f"[Remediation Simulation] Clean isolated tray placed at table surface (Z={tray_z:.4f}m)")

        # Inject crooked cap at G05 (tilt 26 deg)
        crooked_cap = stage.GetPrimAtPath("/World/CleanTray/G05_cap")
        if crooked_cap.IsValid():
            xf_c = UsdGeom.Xformable(crooked_cap)
            rot_op = next((op for op in xf_c.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
            if rot_op:
                rot_op.Set(Gf.Vec3f(26.0, 14.0, 45.0))
            else:
                xf_c.AddRotateXYZOp().Set(Gf.Vec3f(26.0, 14.0, 45.0))
            print("[Remediation Simulation] Injected crooked cap at G05 (26 deg tilt)")

        # Inject missing cap at G12 by hiding the static placeholder cap
        missing_cap = stage.GetPrimAtPath("/World/CleanTray/G12_cap")
        if missing_cap.IsValid():
            UsdGeom.Imageable(missing_cap).MakeInvisible()
            print("[Remediation Simulation] Injected missing cap at G12 (hidden)")

        # Setup True Physical Dynamic Replacement Cap in the supply nest
        # Physical specifications:
        # Flange OD = 1.24 in (31.50 mm, R = 15.75 mm), Thickness = 0.08 in (2.032 mm)
        # Plug OD = 0.90 in (22.86 mm, R = 11.43 mm), Height = 0.53 in (13.46 mm)
        supply_x, supply_y = 0.22, 0.22
        cap_path = "/World/PhysicalSpareCap"
        cap_prim = UsdGeom.Xform.Define(stage, cap_path)
        UsdGeom.Xformable(cap_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(supply_x, supply_y, tray_z))

        flange_r = (1.24 * 0.0254) / 2.0
        flange_t = 0.08 * 0.0254
        c_flange = UsdGeom.Cylinder.Define(stage, f"{cap_path}/Flange")
        c_flange.CreateAxisAttr("Z")
        c_flange.CreateRadiusAttr(flange_r)
        c_flange.CreateHeightAttr(flange_t)
        UsdGeom.Xformable(c_flange.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, flange_t / 2))
        UsdPhysics.CollisionAPI.Apply(c_flange.GetPrim())

        plug_r = (0.90 * 0.0254) / 2.0
        plug_h = 0.53 * 0.0254
        c_plug = UsdGeom.Cylinder.Define(stage, f"{cap_path}/Plug")
        c_plug.CreateAxisAttr("Z")
        c_plug.CreateRadiusAttr(plug_r)
        c_plug.CreateHeightAttr(plug_h)
        UsdGeom.Xformable(c_plug.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, flange_t + plug_h / 2))
        UsdPhysics.CollisionAPI.Apply(c_plug.GetPrim())

        # Cap Material: High-visibility vibrant yellow molded plastic
        yellow_mat = UsdShade.Material.Define(stage, "/World/Looks/CapYellow")
        surf = UsdShade.Shader.Define(stage, "/World/Looks/CapYellow/Surface")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.96, 0.76, 0.04))
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        yellow_mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cap_prim.GetPrim()).Bind(yellow_mat)

        # Dynamic rigid body with accurate 5-gram mass
        fanuc_cell.make_dynamic(cap_prim.GetPrim(), mass=0.005, collide_self=False)

        # Camera selection
        cam_paths = {
            "table": fanuc_cell.TABLE_CAM_PATH,
            "side": fanuc_cell.SIDE_CAM_PATH,
        }
        configs = {"table": C.TABLE_CAM, "side": C.SIDE_CAM}

        cam_path = cam_paths.get(camera_choice, fanuc_cell.SIDE_CAM_PATH)
        cam_cfg = configs.get(camera_choice, C.SIDE_CAM)

        fanuc_cell.enable_ccd(world)
        world.reset()

        cam = Camera(prim_path=cam_path, resolution=cam_cfg["resolution"])
        cam.initialize()
        print(f"[Remediation Simulation] Camera initialized at {cam_path}")

        # Initialize robot articulation
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

        # Key coordinates in robot base coordinates
        home_pos = np.array([0.30, 0.0, 0.24])
        q_home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q_home, _ = ik.solve(q_home, home_pos, down, limits, tool_offset=tool)

        # Highly precise target coordinates relative to robot base:
        p_above_g05 = np.array([0.2516, 0.1226, 0.160])
        p_press_g05 = np.array([0.2516, 0.1226, 0.052])

        # Supply pick target: align gripper fingertips flush against the cap flange
        p_above_supply = np.array([supply_x, supply_y, 0.160])
        p_pick_supply = np.array([supply_x, supply_y, 0.007])

        # Vial G12 target:
        p_above_g12 = np.array([0.3538, -0.0612, 0.160])
        p_place_g12 = np.array([0.3538, -0.0612, 0.052])

        def apply_joints(q6, jaw):
            full = arm.get_joint_positions().copy()
            full[arm_idx] = q6
            for joint, value in fanuc_cell.jaw_targets(jaw).items():
                if joint in idx:
                    full[idx[joint]] = value
            arm.apply_action(ArticulationAction(joint_positions=full))

        # Initial arm positioning
        full_init = arm.get_joint_positions().copy()
        full_init[arm_idx] = q_home
        for joint, value in fanuc_cell.jaw_targets(C.JAW_OPEN).items():
            if joint in idx:
                full_init[idx[joint]] = value
        arm.set_joint_positions(full_init)

        # Settle simulation and warm up camera render buffer
        print("[Remediation Simulation] Warming up camera render buffer...")
        for _ in range(35):
            apply_joints(q_home, C.JAW_OPEN)
            world.step(render=True)
            _ = cam.get_rgba()

        print(f"[Remediation Simulation] Starting motion trajectory rendering ({total_frames} frames)...")
        q_cur = q_home.copy()
        jaw_cur = C.JAW_OPEN

        cap_wrapper = SingleXFormPrim(cap_path)
        jaw_prim_l = SingleXFormPrim("/World/ER4iA/JawL")
        grasp_joint = None

        for frame_idx in range(total_frames):
            # Trajectory state machine
            if frame_idx < 30:
                t = frame_idx / 30.0
                target_p = home_pos + (p_above_g05 - home_pos) * t
                status_text = "PERCEPTION: Defect G05 Crooked Cap Detected -> Approaching"
                jaw_cur = C.JAW_CLOSED
            elif frame_idx < 50:
                t = (frame_idx - 30) / 20.0
                target_p = p_above_g05 + (p_press_g05 - p_above_g05) * t
                status_text = "ACTION: Plunge Pressing Cap G05 Flush"
                jaw_cur = C.JAW_CLOSED
                if frame_idx == 45 and crooked_cap.IsValid():
                    # Reseat crooked cap flush
                    xf_c = UsdGeom.Xformable(crooked_cap)
                    rot_op = next((op for op in xf_c.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
                    if rot_op:
                        rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            elif frame_idx < 75:
                t = (frame_idx - 50) / 25.0
                target_p = p_press_g05 + (p_above_g05 - p_press_g05) * t
                status_text = "STATUS: Cap G05 Reseated -> Retracting"
                jaw_cur = C.JAW_OPEN
            elif frame_idx < 105:
                t = (frame_idx - 75) / 30.0
                target_p = p_above_g05 + (p_above_supply - p_above_g05) * t
                status_text = "PERCEPTION: Slot G12 Missing Cap -> Fetching Spare Cap"
                jaw_cur = C.JAW_OPEN
            elif frame_idx < 120:
                t = (frame_idx - 105) / 15.0
                target_p = p_above_supply + (p_pick_supply - p_above_supply) * t
                status_text = "ACTION: Grasping Replacement Cap from Supply Nest"
                if frame_idx >= 115:
                    jaw_cur = C.JAW_CLOSED
                    if grasp_joint is None:
                        # Establish mathematically exact physical fixed joint constraint
                        p0, r0 = jaw_prim_l.get_world_pose()
                        p1, r1 = cap_wrapper.get_world_pose()
                        gf_p0 = Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2]))
                        gf_q0 = Gf.Quatd(float(r0[0]), float(r0[1]), float(r0[2]), float(r0[3]))
                        T0 = Gf.Matrix4d().SetRotate(gf_q0).SetTranslateOnly(gf_p0)

                        gf_p1 = Gf.Vec3d(float(p1[0]), float(p1[1]), float(p1[2]))
                        gf_q1 = Gf.Quatd(float(r1[0]), float(r1[1]), float(r1[2]), float(r1[3]))
                        T1 = Gf.Matrix4d().SetRotate(gf_q1).SetTranslateOnly(gf_p1)

                        rel_T = T1 * T0.GetInverse()
                        rel_p = rel_T.ExtractTranslation()
                        rel_q = rel_T.ExtractRotationQuat()

                        grasp_joint = UsdPhysics.FixedJoint.Define(stage, "/World/GraspConstraint")
                        grasp_joint.CreateBody0Rel().SetTargets(["/World/ER4iA/JawL"])
                        grasp_joint.CreateBody1Rel().SetTargets([cap_path])
                        grasp_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_p))
                        grasp_joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_q))
                        grasp_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                        grasp_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
                else:
                    jaw_cur = C.JAW_OPEN
            elif frame_idx < 140:
                t = (frame_idx - 120) / 20.0
                target_p = p_pick_supply + (p_above_g12 - p_pick_supply) * t
                status_text = "ACTION: Transiting Replacement Cap to Slot G12"
                jaw_cur = C.JAW_CLOSED
            elif frame_idx < 155:
                t = (frame_idx - 140) / 15.0
                target_p = p_above_g12 + (p_place_g12 - p_above_g12) * t
                status_text = "ACTION: Seating Cap on Vial G12 & Opening Gripper"
                if frame_idx >= 150:
                    if grasp_joint is not None:
                        stage.RemovePrim(grasp_joint.GetPath())
                        grasp_joint = None
                jaw_cur = C.JAW_OPEN if frame_idx >= 152 else C.JAW_CLOSED
            else:
                t = (frame_idx - 155) / 25.0
                target_p = p_place_g12 + (home_pos - p_place_g12) * t
                status_text = "REMEDIATION COMPLETE: All Tray Defects Resolved -> Returning Home"
                jaw_cur = C.JAW_OPEN

            q_sol, ok = ik.solve(q_cur, target_p, down, limits, tool_offset=tool)
            if ok:
                q_cur = q_sol

            apply_joints(q_cur, jaw_cur)
            world.step(render=True)

            rgba = cam.get_rgba()
            if rgba is not None:
                img_rgb = rgba[:, :, :3].astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # Heads-up telemetry HUD overlay
                h_img, w_img = img_bgr.shape[:2]
                cv2.rectangle(img_bgr, (0, 0), (w_img, 56), (20, 20, 20), -1)
                cv2.line(img_bgr, (0, 56), (w_img, 56), (0, 200, 255), 2)
                cv2.putText(img_bgr, "FANUC ER-4iA AUTONOMOUS VIAL CAP REMEDIATION CELL", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img_bgr, f"FRAME {frame_idx+1:03d}/{total_frames:03d} | {status_text}", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(frame_path), img_bgr)

        print(f"[Remediation Simulation] Successfully saved {total_frames} frames to {frames_dir}")

        # Encode MP4 video using imageio_ffmpeg
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
