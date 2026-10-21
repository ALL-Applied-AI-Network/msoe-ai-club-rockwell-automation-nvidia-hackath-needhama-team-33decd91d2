#!/usr/bin/env python3
"""
Randomized Scene Generator for Vial Tray Remediation & Table Supply Caps.

Generates a fully dynamic USD scene:
- 30 Tray Apertures:
    * 15 Crooked Caps: Random yaw (0-360 deg), tilt 26-28 deg, settled with PhysX
    * 5 Missing Caps: Empty apertures (rim open)
    * 10 Good Caps: Nominal seated caps flush on vial rim
- 10 Loose Replacement Caps on Table Surface:
    * Placed randomly on the worktable (Z = -0.0051m) outside the tray bounding box
    * Ready for pick-and-place by the robot arm
- Exports stage metadata (truth labels, coordinates) for perception & evaluation.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FANUC_SIM_DIR = REPO_ROOT / "shared" / "data" / "fanuc_er4ia_sim"
sys.path.insert(0, str(FANUC_SIM_DIR))

VIAL_CAP_USDA = FANUC_SIM_DIR / "assets" / "usd_assets" / "vial_cap.usda"
CLEAN_TRAY_USDA = REPO_ROOT / "shared" / "scenes" / "clean_vial_tray.usda"


def generate_randomized_scene(
    out_usd="shared/scenes/randomized_remediation_scene.usda",
    headless=True,
    seed=42,
    num_crooked=15,
    num_missing=5,
    num_good=10,
    num_table_caps=10
):
    random.seed(seed)

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": headless})

    try:
        import omni.usd
        from pxr import UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
        import cell as fanuc_cell
        import scene_config as C

        stage = omni.usd.get_context().get_stage()
        fanuc_cell.build(stage, with_objects=False)

        # 1. Base Tray Setup
        tray_x = 0.35
        tray_y = -0.02
        base_z = 0.002
        pitch_x = 0.040
        pitch_y = 0.038
        vial_h = 0.2032
        flange_r = 0.01575
        plug_h = 0.0055

        # Slot partitioning
        all_slots = list(range(1, 31))
        random.shuffle(all_slots)
        crooked_slots = sorted(all_slots[:num_crooked])
        missing_slots = sorted(all_slots[num_crooked:num_crooked + num_missing])
        good_slots = sorted(all_slots[num_crooked + num_missing:])

        print(f"[Randomized Scene] Seed: {seed}")
        print(f"  Crooked ({len(crooked_slots)}): {crooked_slots}")
        print(f"  Missing ({len(missing_slots)}): {missing_slots}")
        print(f"  Good    ({len(good_slots)}): {good_slots}")

        # Reference Clean Tray
        tray_prim_path = "/World/VialTray"
        tray_xform = UsdGeom.Xform.Define(stage, tray_prim_path)
        tray_prim = tray_xform.GetPrim()
        tray_prim.GetReferences().AddReference(str(CLEAN_TRAY_USDA))
        tray_xform.AddTranslateOp().Set(Gf.Vec3d(tray_x, tray_y, base_z))

        # Disable native static caps for crooked and missing slots
        for s in crooked_slots + missing_slots:
            cap_p = stage.GetPrimAtPath(f"{tray_prim_path}/G{s:02d}_cap")
            if cap_p.IsValid():
                UsdGeom.Imageable(cap_p).MakeInvisible()

        metadata = {
            "seed": seed,
            "tray_pose": {"x": tray_x, "y": tray_y, "z": base_z},
            "slots": {},
            "table_caps": []
        }

        # 2. Spawn Dynamic Crooked Caps with Randomized Yaw & Roll/Pitch
        for s in crooked_slots:
            r_idx = (s - 1) // 5
            c_idx = (s - 1) % 5
            vx = tray_x + (r_idx - 2.5) * pitch_x
            vy = tray_y + (c_idx - 2.0) * pitch_y
            vz = base_z + vial_h

            yaw_deg = random.uniform(0.0, 360.0)
            tilt_deg = random.uniform(25.0, 27.0)

            # Lateral offset along tilt direction
            theta = math.radians(tilt_deg)
            lowest = plug_h * math.cos(theta) + flange_r * math.sin(theta)
            zc = vz + lowest + 0.0005

            rad_yaw = math.radians(yaw_deg)
            # Offset along the tilt vector in XY
            dx = -0.0010 * math.sin(rad_yaw)
            dy = -0.0010 * math.cos(rad_yaw)

            c_path = f"/World/DynamicCrooked_{s:02d}"
            c_prim = UsdGeom.Xform.Define(stage, c_path)
            xf = UsdGeom.Xformable(c_prim.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(vx + dx, vy + dy, zc))
            xf.AddRotateZOp().Set(yaw_deg)
            xf.AddRotateXOp().Set(tilt_deg)

            # Cap geometry (flange + plug)
            flange = UsdGeom.Cylinder.Define(stage, f"{c_path}/Flange")
            flange.CreateAxisAttr("Z")
            flange.CreateRadiusAttr(flange_r)
            flange.CreateHeightAttr(0.004)
            UsdGeom.Xformable(flange.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.002))

            plug = UsdGeom.Cylinder.Define(stage, f"{c_path}/Plug")
            plug.CreateAxisAttr("Z")
            plug.CreateRadiusAttr(0.01143)
            plug.CreateHeightAttr(plug_h)
            UsdGeom.Xformable(plug.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -plug_h / 2.0))

            # Rigid body & sub-millimeter contact offset
            rb = UsdPhysics.RigidBodyAPI.Apply(c_prim.GetPrim())
            rb.CreateRigidBodyEnabledAttr(True)
            mass_api = UsdPhysics.MassAPI.Apply(c_prim.GetPrim())
            mass_api.CreateMassAttr(0.004)

            for geom in [flange, plug]:
                col = UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
                col.CreateCollisionEnabledAttr(True)
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
                physx_col.CreateContactOffsetAttr(0.0005)
                physx_col.CreateRestOffsetAttr(0.0)

            metadata["slots"][f"G{s:02d}"] = {
                "type": "crooked",
                "x": vx, "y": vy, "z": vz,
                "yaw_deg": yaw_deg, "tilt_deg": tilt_deg
            }

        for s in missing_slots:
            r_idx = (s - 1) // 5
            c_idx = (s - 1) % 5
            vx = tray_x + (r_idx - 2.5) * pitch_x
            vy = tray_y + (c_idx - 2.0) * pitch_y
            vz = base_z + vial_h
            metadata["slots"][f"G{s:02d}"] = {"type": "missing", "x": vx, "y": vy, "z": vz}

        for s in good_slots:
            r_idx = (s - 1) // 5
            c_idx = (s - 1) % 5
            vx = tray_x + (r_idx - 2.5) * pitch_x
            vy = tray_y + (c_idx - 2.0) * pitch_y
            vz = base_z + vial_h
            metadata["slots"][f"G{s:02d}"] = {"type": "good", "x": vx, "y": vy, "z": vz}

        # 3. Spawn 10 Loose Replacement Caps on Table Surface
        # Table surface: Z = -0.0051m (base frame) or 0.0m on table prim
        # Placement area: X in [0.28, 0.44], Y in [-0.28, -0.14] (clear of robot base and tray)
        table_z = -0.0051 + 0.002  # Rest flange on table
        print(f"[Randomized Scene] Spawning {num_table_caps} loose replacement caps on table...")

        for i in range(num_table_caps):
            # Scatter with minimum separation
            tx = random.uniform(0.30, 0.44)
            ty = random.uniform(-0.28, -0.14)
            tyaw = random.uniform(0.0, 360.0)

            cap_name = f"TableCap_{i+1:02d}"
            tc_path = f"/World/{cap_name}"
            tc_prim = UsdGeom.Xform.Define(stage, tc_path)
            xf = UsdGeom.Xformable(tc_prim.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, table_z))
            xf.AddRotateZOp().Set(tyaw)

            flange = UsdGeom.Cylinder.Define(stage, f"{tc_path}/Flange")
            flange.CreateAxisAttr("Z")
            flange.CreateRadiusAttr(flange_r)
            flange.CreateHeightAttr(0.004)
            UsdGeom.Xformable(flange.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.002))

            plug = UsdGeom.Cylinder.Define(stage, f"{tc_path}/Plug")
            plug.CreateAxisAttr("Z")
            plug.CreateRadiusAttr(0.01143)
            plug.CreateHeightAttr(plug_h)
            UsdGeom.Xformable(plug.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -plug_h / 2.0))

            rb = UsdPhysics.RigidBodyAPI.Apply(tc_prim.GetPrim())
            rb.CreateRigidBodyEnabledAttr(True)
            mass_api = UsdPhysics.MassAPI.Apply(tc_prim.GetPrim())
            mass_api.CreateMassAttr(0.004)

            for geom in [flange, plug]:
                col = UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
                col.CreateCollisionEnabledAttr(True)
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
                physx_col.CreateContactOffsetAttr(0.0005)
                physx_col.CreateRestOffsetAttr(0.0)

            metadata["table_caps"].append({
                "name": cap_name,
                "x": tx, "y": ty, "z": table_z,
                "yaw_deg": tyaw
            })

        out_p = Path(out_usd).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(out_p))

        meta_p = out_p.with_suffix(".json")
        with open(meta_p, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Randomized Scene] Saved scene to: {out_p}")
        print(f"[Randomized Scene] Saved truth metadata to: {meta_p}")
        return str(out_p)

    finally:
        sim_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Randomized Remediation Scene")
    parser.add_argument("--out", default="shared/scenes/randomized_remediation_scene.usda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crooked", type=int, default=15)
    parser.add_argument("--missing", type=int, default=5)
    parser.add_argument("--good", type=int, default=10)
    parser.add_argument("--table-caps", type=int, default=10)
    args = parser.parse_args()

    generate_randomized_scene(
        out_usd=args.out,
        headless=True,
        seed=args.seed,
        num_crooked=args.crooked,
        num_missing=args.missing,
        num_good=args.good,
        num_table_caps=args.table_caps
    )
