#!/usr/bin/env python3
"""
Fanuc Robotic Tray Remediation Scene Builder (Isaac Sim).

Composes:
1. FANUC ER-4iA Robotic Arm + Parallel Gripper
2. Worktable with randomized Tray placement (simulating mechanical conveyor play / parallax)
3. 30 Vials in the Tray with injected defects (crooked cap, missing cap)
4. Replacement Cap Buffer Nest with spare yellow caps ready for pick-and-place
5. Multiple calibrated cameras (Overhead TableCam, SideCam, and WristCam)
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FANUC_SIM_DIR = REPO_ROOT / "shared" / "data" / "fanuc_er4ia_sim"
sys.path.insert(0, str(FANUC_SIM_DIR))

CSI_DIR = REPO_ROOT / "shared" / "data" / "csi_digital_twin_v1"
TRAY_STAGE_FILE = CSI_DIR / "stations" / "tray_inspection" / "tray_inspection.usdc"
VIAL_CAP_USDA = FANUC_SIM_DIR / "assets" / "usd_assets" / "vial_cap.usda"


def build_remediation_scene(
    out_usd="shared/scenes/fanuc_remediation_scene.usd",
    headless=True,
    randomize_tray=True,
    crooked_slot=5,
    missing_slot=12
):
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": headless})

    try:
        import omni.usd
        from pxr import UsdGeom, Gf, Sdf, UsdPhysics
        import cell as fanuc_cell
        import scene_config as C

        stage = omni.usd.get_context().get_stage()
        # 1. Build base FANUC cell (Robot + Table + Wrist/Table Cameras)
        fanuc_cell.build(stage, with_objects=False)

        # 2. Tray Placement with Randomized Position & Heading
        # Nominal reach center for Fanuc ER-4iA: X ~ 0.35m, Y ~ 0.0m, Z ~ table top
        base_x = 0.34
        base_y = -0.02
        base_z = 0.002  # Rest flush on tabletop

        if randomize_tray:
            # Mechanical conveyor tolerance: +/- 18mm translation, +/- 3.5 deg yaw rotation
            tray_x = base_x + random.uniform(-0.018, 0.018)
            tray_y = base_y + random.uniform(-0.018, 0.018)
            tray_yaw = random.uniform(-3.5, 3.5)
        else:
            tray_x = base_x
            tray_y = base_y
            tray_yaw = 0.0

        print(f"[Remediation Scene] Placing Tray at X={tray_x:.4f}m, Y={tray_y:.4f}m, Yaw={tray_yaw:.2f}°")

        # Reference tray inspection USD
        tray_prim_path = "/World/VialTray"
        tray_xform = UsdGeom.Xform.Define(stage, tray_prim_path)
        tray_prim = tray_xform.GetPrim()
        tray_prim.GetReferences().AddReference(str(TRAY_STAGE_FILE))

        # Apply randomized 2D transform to tray
        tray_xform.AddTranslateOp().Set(Gf.Vec3d(tray_x, tray_y, base_z))
        tray_xform.AddRotateZOp().Set(tray_yaw)

        # 3. Inject Specific Defects for the Robot to Remediate
        # (a) Crooked Cap (tilted unseated cap to press)
        crooked_prim = stage.GetPrimAtPath(f"{tray_prim_path}/G{crooked_slot:02d}_cap")
        if crooked_prim.IsValid():
            xf_c = UsdGeom.Xformable(crooked_prim)
            rot_op = next((op for op in xf_c.GetOrderedXformOps() if "rotate" in op.GetOpName()), None)
            if rot_op:
                rot_op.Set(Gf.Vec3f(26.0, 14.0, 45.0))
            print(f"[Remediation Scene] Injected crooked cap at G{crooked_slot:02d}")

        # (b) Missing Cap (uncapped vial body to replace)
        missing_prim = stage.GetPrimAtPath(f"{tray_prim_path}/G{missing_slot:02d}_cap")
        if missing_prim.IsValid():
            UsdGeom.Imageable(missing_prim).MakeInvisible()
            print(f"[Remediation Scene] Injected missing cap at G{missing_slot:02d}")

        # 4. Create Replacement Cap Buffer Nest (Spare caps for picking)
        # Positioned conveniently on the robot's left flank (X=0.20m, Y=0.20m)
        supply_prim_path = "/World/CapSupply"
        supply_xform = UsdGeom.Xform.Define(stage, supply_prim_path)
        supply_xform.AddTranslateOp().Set(Gf.Vec3d(0.22, 0.18, base_z))

        # Add 3 spare caps in the supply nest
        for i in range(3):
            cap_sub_path = f"{supply_prim_path}/SpareCap_{i+1}"
            spare_xform = UsdGeom.Xform.Define(stage, cap_sub_path)
            spare_xform.AddTranslateOp().Set(Gf.Vec3d(i * 0.035, 0.0, 0.0))
            if VIAL_CAP_USDA.exists():
                spare_xform.GetPrim().GetReferences().AddReference(str(VIAL_CAP_USDA))

        print("[Remediation Scene] Added spare replacement caps at /World/CapSupply")

        # 5. Save Composed USD Stage
        out_p = Path(out_usd).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(out_p))
        print(f"[Remediation Scene] Successfully composed and saved stage to: {out_p}")

        return {
            "stage_path": str(out_p),
            "tray_pose": {"x": tray_x, "y": tray_y, "yaw_deg": tray_yaw},
            "crooked_slot": crooked_slot,
            "missing_slot": missing_slot
        }

    finally:
        sim_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Fanuc Robotic Remediation Scene")
    parser.add_argument("--out", default="shared/scenes/fanuc_remediation_scene.usd")
    parser.add_argument("--crooked", type=int, default=5, help="Slot with crooked cap")
    parser.add_argument("--missing", type=int, default=12, help="Slot with missing cap")
    parser.add_argument("--no-random", action="store_true", help="Disable random tray placement")
    args = parser.parse_args()

    build_remediation_scene(
        out_usd=args.out,
        headless=True,
        randomize_tray=not args.no_random,
        crooked_slot=args.crooked,
        missing_slot=args.missing
    )
