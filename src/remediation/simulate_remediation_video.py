#!/usr/bin/env python3
"""
Simulate and Render the Complete Fanuc ER-4iA Cap Remediation Sequence.

Execution Pipeline:
1. Spawns Fanuc ER-4iA cell + table + isolated clean tray fixture with defects:
     - G05: Crooked cap (tilted at 26°)
     - G12: Missing cap (uncapped vial)
     - /World/CapSupply: Spare replacement cap in supply nest
2. Fixed Assets & Realistic Sizing:
     - Uses properly scaled CAD visual matching Dataset 3 (CAP_RADIUS, CAP_HEIGHT, yellow polypropylene)
     - Clean tray material bindings mapped to realistic brushed aluminium, glass, and yellow caps
     - Carried cap dynamically attaches to gripper between jaw fingertips during transit
3. Sequence:
     - Move 1 (G05 Reseat): Approach above G05 -> descend & plunge press cap flush -> retract
     - Move 2 (G12 Backfill): Transit to Cap Supply -> grasp spare cap -> carry to G12 -> seat on vial -> return home
4. Captures 180 frames with heads-up telemetry HUD overlay and encodes to MP4 video.
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
FANUC_DIR = Path("/data/ai_club/team_14_2026/data/fanuc_er4ia_sim")
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
        from pxr import UsdGeom, Gf, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.utils.stage import add_reference_to_stage
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
        cap_usda = FANUC_DIR / "assets" / "usd_assets" / "vial_cap.usda"

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

        # Inject missing cap at G12 (hidden initially)
        missing_cap = stage.GetPrimAtPath("/World/CleanTray/G12_cap")
        if missing_cap.IsValid():
            UsdGeom.Imageable(missing_cap).MakeInvisible()
            print("[Remediation Simulation] Injected missing cap at G12")

        # Spawn accurately scaled spare replacement cap in the supply nest (X=0.22, Y=0.22)
        supply_x, supply_y = 0.22, 0.22
        supply_cap_prim = UsdGeom.Xform.Define(stage, "/World/SupplyCap")
        UsdGeom.Xformable(supply_cap_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(supply_x, supply_y, tray_z))
        vis_supply = "/World/SupplyCap/Visual"
        add_reference_to_stage(str(cap_usda), vis_supply)
        fanuc_cell.scale_cad_visual(stage, vis_supply, (C.CAP_RADIUS / 0.015875, C.CAP_RADIUS / 0.015875, C.CAP_HEIGHT / 0.01570))
        vis_xf = UsdGeom.Xformable(stage.GetPrimAtPath(vis_supply))
        tr_op = vis_xf.AddTranslateOp(opSuffix="rimRest")
        tr_op.Set(Gf.Vec3d(0, 0, C.CAP_FLANGE_THICK))
        rot_op = vis_xf.AddRotateXOp(opSuffix="domeUp")
        rot_op.Set(180.0)  # upright, flange resting on table
        existing_ops = [op for op in vis_xf.GetOrderedXformOps() if op.GetOpName() not in (tr_op.GetOpName(), rot_op.GetOpName())]
        vis_xf.SetXformOpOrder([tr_op] + existing_ops + [rot_op])
        print("[Remediation Simulation] Accurately sized spare cap spawned in supply nest")

        # Create carried cap attached to gripper (hidden until grasped)
        tool_flange_path = "/World/ER4iA/J6_link"
        carried_cap_prim = UsdGeom.Xform.Define(stage, f"{tool_flange_path}/CarriedCap")
        vis_carried = f"{tool_flange_path}/CarriedCap/Visual"
        add_reference_to_stage(str(cap_usda), vis_carried)
        fanuc_cell.scale_cad_visual(stage, vis_carried, (C.CAP_RADIUS / 0.015875, C.CAP_RADIUS / 0.015875, C.CAP_HEIGHT / 0.01570))
        c_vis_xf = UsdGeom.Xformable(stage.GetPrimAtPath(vis_carried))
        c_tr_op = c_vis_xf.AddTranslateOp(opSuffix="rimRest")
        c_tr_op.Set(Gf.Vec3d(0, 0, C.CAP_FLANGE_THICK))
        c_rot_op = c_vis_xf.AddRotateXOp(opSuffix="domeUp")
        c_rot_op.Set(180.0)
        c_existing_ops = [op for op in c_vis_xf.GetOrderedXformOps() if op.GetOpName() not in (c_tr_op.GetOpName(), c_rot_op.GetOpName())]
        c_vis_xf.SetXformOpOrder([c_tr_op] + c_existing_ops + [c_rot_op])

        # Position right between gripper fingertips (tip offset = 0.223m)
        tip_offset = fanuc_cell.tip_offset()
        UsdGeom.Xformable(carried_cap_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, tip_offset - 0.005))
        UsdGeom.Imageable(carried_cap_prim.GetPrim()).MakeInvisible()

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

        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.array(C.TOOL_AXIS) * tip_offset)
        down = [0.0, 0.0, -1.0]

        # Key coordinates in robot base coordinates
        home_pos = np.array([0.30, 0.0, 0.24])
        q_home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q_home, _ = ik.solve(q_home, home_pos, down, limits, tool_offset=tool)

        # Precise target coordinates relative to robot base:
        p_above_g05 = np.array([0.242, 0.106, 0.160])
        p_press_g05 = np.array([0.242, 0.106, 0.076])

        p_above_supply = np.array([supply_x, supply_y, 0.160])
        p_pick_supply = np.array([supply_x, supply_y, 0.015])

        p_above_g12 = np.array([0.344, -0.078, 0.160])
        p_place_g12 = np.array([0.344, -0.078, 0.076])

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
                    # Cap is grasped: hide table cap and show carried cap in gripper
                    UsdGeom.Imageable(supply_cap_prim.GetPrim()).MakeInvisible()
                    UsdGeom.Imageable(carried_cap_prim.GetPrim()).MakeVisible()
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
                    # Release onto vial
                    UsdGeom.Imageable(carried_cap_prim.GetPrim()).MakeInvisible()
                    if missing_cap.IsValid():
                        UsdGeom.Imageable(missing_cap).MakeVisible()
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
