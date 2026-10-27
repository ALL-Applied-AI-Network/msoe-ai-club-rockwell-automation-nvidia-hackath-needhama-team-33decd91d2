#!/usr/bin/env python3
"""Render randomized YOLO samples from the qualified remediation scene.

The immutable template is ``black_silver_remediation_qualified_v3.usda``: the
FANUC cell/camera and qualified black-base, silver-top tray used by the physical
remediation setup. Each worker loads it once and varies the cap state/pose and,
optionally, small deployment-domain details.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FANUC = ROOT / "shared" / "data" / "fanuc_er4ia_sim"
sys.path[:0] = [str(ROOT), str(FANUC)]

WIDTH, HEIGHT = 1920, 1080
CLASS_NAMES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")
VIAL_RADIUS = 0.027178 / 2.0
VIAL_HEIGHT = 0.203
GOOD_CAP_LIFT = 0.0002
CROOKED_CAP_LIFT = 0.0075
TABLE_CAP_Z = 0.9089
TRAY_BASE_POSITION = np.array([0.385, 0.035, 0.8949])
CAMERA_BASE_POSITION = np.array([0.7770000100135803, -0.04259999841451645,
                                 1.7023999691009521])


def split_for(index: int, train_count: int, val_count: int) -> str:
    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def quaternion(Gf, tilt_deg: float, yaw_deg: float):
    tilt, yaw = np.radians([tilt_deg, yaw_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rotation = rz @ rx
    value = Gf.Matrix3d(*rotation.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    return Gf.Quatf(value)


def jittered_quaternion(Gf, base, rx_deg: float, ry_deg: float, rz_deg: float):
    """Compose small world-axis rotations with an authored camera quaternion."""
    q = Gf.Quatd(base)
    for axis, angle in ((Gf.Vec3d(1, 0, 0), rx_deg),
                        (Gf.Vec3d(0, 1, 0), ry_deg),
                        (Gf.Vec3d(0, 0, 1), rz_deg)):
        q = Gf.Rotation(axis, float(angle)).GetQuat() * q
    return q


def rotate_xy(point, center, angle_deg):
    angle = np.radians(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return np.asarray(center) + rotation @ (np.asarray(point) - np.asarray(center))


def camera_model_for(CameraModel, UsdGeom, camera_prim, camera_geom):
    return CameraModel({
        "resolution": [WIDTH, HEIGHT],
        "focal_length_mm": float(camera_geom.GetFocalLengthAttr().Get()),
        "horizontal_aperture_mm": float(camera_geom.GetHorizontalApertureAttr().Get()),
        "vertical_aperture_mm": float(camera_geom.GetVerticalApertureAttr().Get()),
        "camera_to_world": np.array(
            UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0.0)
        ).tolist(),
        "table_top_z": 0.8949,
    })


def set_pose(prim, Gf, position, orientation) -> None:
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, position)))
    prim.GetAttribute("xformOp:orient").Set(orientation)


def object_bounds(instance_data, path: str):
    image = np.asarray(instance_data["data"]).squeeze()
    mapping = instance_data["info"]["idToLabels"]
    ids = [int(key) for key, value in mapping.items()
           if isinstance(value, str) and (value == path or value.startswith(path + "/"))]
    yy, xx = np.where(np.isin(image, ids))
    if len(xx) < 20:
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]


def yolo_line(class_id: int, bounds) -> str:
    x0, y0, x1, y1 = bounds
    return (f"{class_id} {(x0 + x1) / 2 / WIDTH:.8f} {(y0 + y1) / 2 / HEIGHT:.8f} "
            f"{(x1 - x0) / WIDTH:.8f} {(y1 - y0) / HEIGHT:.8f}")


def missing_bounds(camera_model, rim) -> list[int]:
    angles = np.arange(96) * 2 * np.pi / 96
    points = np.asarray(rim) + np.column_stack([
        VIAL_RADIUS * np.cos(angles), VIAL_RADIUS * np.sin(angles), np.zeros(96)
    ])
    uv = np.array([camera_model.project(point) for point in points])
    x0, y0 = np.floor(uv.min(axis=0)).astype(int)
    x1, y1 = np.ceil(uv.max(axis=0)).astype(int)
    if not (0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT):
        raise RuntimeError(f"Missing-cap rim projects out of frame: {rim}")
    return [int(x0), int(y0), int(x1), int(y1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=1000)
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=27000)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--lighting-preset", choices=("balanced", "true", "physical"),
                        default="physical")
    parser.add_argument("--pink-table-paper", action="store_true",
                        help="Place a matte pink sheet beneath the loose table caps")
    parser.add_argument("--domain-randomization", action="store_true",
                        help="Randomize deployment geometry, optics, lighting, and robot pose")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    if args.train_count + args.val_count > args.scenes:
        parser.error("train-count + val-count cannot exceed scenes")

    out = args.out.resolve()
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    source_scene = ROOT / "shared" / "scenes" / "black_silver_remediation_qualified_v3.usda"
    archived_root = "/home/ad.msoe.edu/needhama/Documents/GitHub/innovation-lab-fall-2026"
    localized_scene = out / "manifests" / f"localized_cad_scene_shard_{args.shard:02d}.usda"
    localized_scene.write_text(source_scene.read_text().replace(archived_root, str(ROOT)))

    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    records = []
    try:
        import cv2
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        from backproject import CameraModel
        import scene_config as C
        from src.remediation.custom_gripper import attach_visual_pair

        omni.usd.get_context().open_stage(str(localized_scene))
        app.update()
        stage = omni.usd.get_context().get_stage()
        layer = stage.GetRootLayer()

        # Match the real remediation simulator. The scene's original fingers
        # are visual stand-ins; this mounts data/gripper.usd to both live SMC
        # jaw carriers and hides the retired finger/extender geometry.
        gripper_mount = attach_visual_pair(stage)
        print(f"[gripper] mounted {gripper_mount['asset']} at {gripper_mount['paths']}",
              flush=True)

        # The physical cell is lit by several ceiling fixtures. Their overlapping
        # soft shadows are visible around loose caps, while broad ambient light
        # keeps the upward-facing cap cavities yellow instead of nearly black.
        dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
        key = UsdLux.RectLight(stage.GetPrimAtPath("/World/KeyLight"))
        fill = UsdLux.RectLight(stage.GetPrimAtPath("/World/FillLight"))
        rim_light = UsdLux.RectLight(stage.GetPrimAtPath("/World/RimLight"))
        ceiling_4 = UsdLux.RectLight.Define(stage, "/World/CeilingLight4")
        ceiling_4.CreateNormalizeAttr(True)
        ceiling_4_xform = UsdGeom.Xformable(ceiling_4.GetPrim())
        ceiling_4_xform.AddTranslateOp()
        ceiling_4_xform.AddRotateXYZOp()

        if args.lighting_preset == "physical":
            lighting = {
                "dome": 320.0,
                "key": 125000.0, "fill": 108000.0,
                "rim": 118000.0, "ceiling_4": 102000.0,
                "rect_width": 0.22, "rect_height": 0.09,
            }
            fixtures = (
                (key, (0.03, -0.50, 2.52), (1.00, 0.965, 0.91)),
                (fill, (0.76, -0.46, 2.58), (0.93, 0.96, 1.00)),
                (rim_light, (0.02, 0.56, 2.55), (0.97, 0.985, 1.00)),
                (ceiling_4, (0.77, 0.52, 2.50), (1.00, 0.95, 0.88)),
            )
            for light, position, color in fixtures:
                light.GetWidthAttr().Set(lighting["rect_width"])
                light.GetHeightAttr().Set(lighting["rect_height"])
                light.GetColorAttr().Set(Gf.Vec3f(*color))
                prim = light.GetPrim()
                translate = prim.GetAttribute("xformOp:translate")
                rotate = prim.GetAttribute("xformOp:rotateXYZ")
                translate.Set(Gf.Vec3d(*position))
                rotate.Set(Gf.Vec3f(180.0, 0.0, 0.0))
        elif args.lighting_preset == "true":
            lighting = {"dome": 650.0, "key": 475000.0,
                        "fill": 30000.0, "rim": 40000.0, "ceiling_4": 0.0,
                        "rect_width": 0.07, "rect_height": 0.07}
        else:
            lighting = {"dome": 650.0, "key": 320000.0,
                        "fill": 30000.0, "rim": 40000.0, "ceiling_4": 0.0,
                        "rect_width": 0.18, "rect_height": 0.18}
        dome.GetIntensityAttr().Set(lighting["dome"])
        key.GetIntensityAttr().Set(lighting["key"])
        key.GetWidthAttr().Set(lighting["rect_width"])
        key.GetHeightAttr().Set(lighting["rect_height"])
        fill.GetIntensityAttr().Set(lighting["fill"])
        rim_light.GetIntensityAttr().Set(lighting["rim"])
        ceiling_4.GetIntensityAttr().Set(lighting["ceiling_4"])

        paper = None
        if args.pink_table_paper:
            paper_path = "/World/TrueLightingPinkPaper"
            paper = UsdGeom.Mesh.Define(stage, paper_path)
            paper.CreatePointsAttr([
                Gf.Vec3f(0.245, -0.285, 0.8954),
                Gf.Vec3f(0.505, -0.285, 0.8954),
                Gf.Vec3f(0.505, -0.175, 0.8954),
                Gf.Vec3f(0.245, -0.175, 0.8954),
            ])
            paper.CreateFaceVertexCountsAttr([4])
            paper.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            paper.CreateSubdivisionSchemeAttr("none")
            material = UsdShade.Material.Define(stage, paper_path + "/PinkMaterial")
            shader = UsdShade.Shader.Define(stage, paper_path + "/PinkMaterial/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.92, 0.20, 0.52)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(paper.GetPrim()).Bind(material)
        template_cap_root = "/World/RemediationCaps"
        source_cap = Sdf.Path(template_cap_root + "/G01")
        cap_root = "/World/GeneratedRemediationCaps"
        UsdGeom.Xform.Define(stage, cap_root)

        # Use fresh static prim paths for the full inventory. Reusing the two
        # template rigid-body paths can leave stale Fabric renderer instances
        # on some GPUs after their schemas are replaced.
        destinations = [cap_root + f"/G{slot:02d}" for slot in range(1, 31)]
        destinations += [cap_root + f"/Table_{index:02d}" for index in range(10)]
        for destination in destinations:
            if not Sdf.CopySpec(layer, source_cap, layer, Sdf.Path(destination)):
                raise RuntimeError(f"Could not copy CAD cap to {destination}")
        # RemediationCaps is an untyped container, so inherited Imageable
        # visibility is not reliable in Fabric. Deactivate the complete source
        # hierarchy after copying one cap; otherwise its original caps remain
        # beneath generated caps and at stale table positions.
        stage.GetPrimAtPath(template_cap_root).SetActive(False)
        app.update()

        cap_prims = {slot: stage.GetPrimAtPath(cap_root + f"/G{slot:02d}")
                     for slot in range(1, 31)}
        table_prims = [stage.GetPrimAtPath(cap_root + f"/Table_{index:02d}")
                       for index in range(10)]
        if not all(prim.IsValid() for prim in [*cap_prims.values(), *table_prims]):
            raise RuntimeError("The complete CAD cap inventory did not compose")
        cap_color = stage.GetPrimAtPath(
            "/World/RemediationLooks/Cap/Surface").GetAttribute("inputs:diffuseColor")
        cavity_shader = UsdShade.Shader(stage.GetPrimAtPath(
            "/World/RemediationLooks/CapCavity/Surface"))
        cavity_color = cavity_shader.GetInput("diffuseColor")
        cavity_emissive = cavity_shader.CreateInput(
            "emissiveColor", Sdf.ValueTypeNames.Color3f)
        # Real table-camera frames show a yellow cavity with mild edge shading,
        # not a dark interior. This tiny ambient lift is deliberately below the
        # body color so the cavity keeps shape without becoming self-lit.
        cavity_color.Set(Gf.Vec3f(0.88, 0.68, 0.025))
        cavity_emissive.Set(Gf.Vec3f(0.025, 0.019, 0.001))

        # PreviewSurface opacity rendered these CAD vials as opaque grey.
        # Rebind both wall and bottom meshes to the scene's physically based
        # OmniGlass material so they are almost transparent like the real vial.
        vial_glass = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/vial_glass"))
        if not vial_glass.GetPrim().IsValid():
            raise RuntimeError("The source scene's OmniGlass vial material is missing")
        polymer_shader = stage.GetPrimAtPath("/World/Looks/vial_glass/Shader")
        polymer_shader.GetAttribute("inputs:glass_ior").Set(1.57)
        polymer_shader.GetAttribute("inputs:frosting_roughness").Set(0.035)
        polymer_shader.GetAttribute("inputs:glass_color").Set(
            Gf.Vec3f(0.985, 0.995, 1.0)
        )
        for slot in range(1, 31):
            tube_path = f"/World/RemediationTray/G{slot:02d}_tube"
            for part in ("Wall", "Bottom"):
                part_prim = stage.GetPrimAtPath(f"{tube_path}/{part}")
                if not part_prim.IsValid():
                    raise RuntimeError(f"Missing vial geometry: {part_prim.GetPath()}")
                UsdShade.MaterialBindingAPI.Apply(part_prim).Bind(vial_glass)

        # The deployment tray top is machined aluminum. Keep the posts and
        # lower structure unchanged while rebinding its top/support surfaces.
        aluminum = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/alu_machined"))
        if not aluminum.GetPrim().IsValid():
            raise RuntimeError("The source scene's machined-aluminum material is missing")
        aluminum_shader = stage.GetPrimAtPath("/World/Looks/alu_machined/shader")
        aluminum_shader.GetAttribute("inputs:diffuse_color_constant").Set(
            Gf.Vec3f(0.84, 0.85, 0.88))
        aluminum_shader.GetAttribute("inputs:metallic_constant").Set(0.98)
        aluminum_shader.GetAttribute("inputs:reflection_roughness_constant").Set(0.14)
        aluminum_shader.GetAttribute(
            "inputs:reflection_roughness_texture_influence").Set(0.45)
        tray = stage.GetPrimAtPath("/World/RemediationTray")
        for child in tray.GetChildren():
            if child.GetName().endswith("_plate") or child.GetName() == "tray_good_support":
                UsdShade.MaterialBindingAPI.Apply(child).Bind(aluminum)
        app.update()

        # Derive slot rims from the corrected CAD vial transforms, not from a
        # hard-coded grid or the legacy clean tray.
        rims = {}
        vial_prims = {}
        vial_orient_ops = {}
        vial_parts = {}
        for slot in range(1, 31):
            tube = stage.GetPrimAtPath(f"/World/RemediationTray/G{slot:02d}_tube")
            vial_prims[slot] = tube
            if args.domain_randomization:
                vial_orient_ops[slot] = UsdGeom.Xformable(tube).AddOrientOp()
            matrix = np.array(UsdGeom.Xformable(tube).ComputeLocalToWorldTransform(0.0))
            rims[slot] = (np.array([0.0, 0.0, VIAL_HEIGHT, 1.0]) @ matrix)[:3]
            vial_parts[slot] = [tube.GetChild("Wall"), tube.GetChild("Bottom")]

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0,
                      rendering_dt=1.0 / 30.0)
        world.reset()
        arm = SingleArticulation(C.ROBOT_PRIM)
        arm.initialize()
        arm_positions = arm.get_joint_positions().copy()
        arm_names = list(arm.dof_names)
        arm_indices = [arm_names.index(f"J{joint}") for joint in range(1, 7)]
        camera = Camera("/World/TableCam", resolution=(WIDTH, HEIGHT))
        camera.initialize()
        camera_prim = stage.GetPrimAtPath("/World/TableCam")
        camera_geom = UsdGeom.Camera(camera_prim)
        camera_base_orientation = Gf.Quatd(camera_prim.GetAttribute("xformOp:orient").Get())
        camera_model = camera_model_for(CameraModel, UsdGeom, camera_prim, camera_geom)
        instance = rep.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation", init_params={"colorize": False}
        )
        render_product = camera.get_render_product_path()
        instance.attach([render_product])
        for _ in range(20):
            world.step(render=True)

        identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        indices = list(range(args.shard, args.scenes, args.num_shards))
        for sequence, index in enumerate(indices, 1):
            rng = random.Random(args.seed + index)
            randomization = {}
            if args.domain_randomization:
                # Production-batch cap tint: all caps in a frame remain the
                # same material while hue/value shift gently between frames.
                outer_rgb = [rng.uniform(0.90, 1.00), rng.uniform(0.68, 0.84),
                             rng.uniform(0.025, 0.075)]
                cavity_rgb = [outer_rgb[0] * 0.93, outer_rgb[1] * 0.90,
                              max(outer_rgb[2] * 0.82, 0.022)]
                cavity_emissive_rgb = [component * 0.027 for component in outer_rgb]
                cap_color.Set(Gf.Vec3f(*outer_rgb))
                cavity_color.Set(Gf.Vec3f(*cavity_rgb))
                cavity_emissive.Set(Gf.Vec3f(*cavity_emissive_rgb))

                if args.lighting_preset == "physical":
                    frame_lighting = {
                        "dome": rng.uniform(260.0, 390.0),
                        "key": rng.uniform(105000.0, 145000.0),
                        "fill": rng.uniform(90000.0, 126000.0),
                        "rim": rng.uniform(98000.0, 138000.0),
                        "ceiling_4": rng.uniform(84000.0, 120000.0),
                        "rect_width": rng.uniform(0.17, 0.27),
                        "rect_height": rng.uniform(0.07, 0.12),
                    }
                else:
                    frame_lighting = {
                        "dome": rng.uniform(500.0, 820.0),
                        "key": rng.uniform(390000.0, 560000.0),
                        "fill": rng.uniform(22000.0, 43000.0),
                        "rim": rng.uniform(28000.0, 56000.0),
                        "ceiling_4": 0.0,
                        "rect_width": rng.uniform(0.055, 0.105),
                        "rect_height": rng.uniform(0.055, 0.105),
                    }
                dome.GetIntensityAttr().Set(frame_lighting["dome"])
                key.GetIntensityAttr().Set(frame_lighting["key"])
                for light in (key, fill, rim_light, ceiling_4):
                    light.GetWidthAttr().Set(frame_lighting["rect_width"])
                    light.GetHeightAttr().Set(frame_lighting["rect_height"])
                fill.GetIntensityAttr().Set(frame_lighting["fill"])
                rim_light.GetIntensityAttr().Set(frame_lighting["rim"])
                ceiling_4.GetIntensityAttr().Set(frame_lighting["ceiling_4"])

                tray_position = TRAY_BASE_POSITION + np.array([
                    rng.uniform(-0.014, 0.014), rng.uniform(-0.014, 0.014), 0.0])
                tray_angle = rng.uniform(-2.5, 2.5)
                tray.GetAttribute("xformOp:translate").Set(
                    Gf.Vec3d(*tray_position.tolist()))
                tray.GetAttribute("xformOp:rotateZ").Set(float(tray_angle))

                vial_tilts = {}
                for slot, op in vial_orient_ops.items():
                    tilt = rng.uniform(0.0, 1.8)
                    yaw = rng.uniform(0.0, 360.0)
                    op.Set(quaternion(Gf, tilt, yaw))
                    vial_tilts[slot] = {"tilt_deg": tilt, "yaw_deg": yaw}

                camera_position = CAMERA_BASE_POSITION + np.array([
                    rng.uniform(-0.010, 0.010), rng.uniform(-0.010, 0.010),
                    rng.uniform(-0.008, 0.008)])
                camera_angles = [rng.uniform(-1.0, 1.0) for _ in range(3)]
                camera_prim.GetAttribute("xformOp:translate").Set(
                    Gf.Vec3d(*camera_position.tolist()))
                camera_prim.GetAttribute("xformOp:orient").Set(
                    jittered_quaternion(Gf, camera_base_orientation, *camera_angles))

                # Keep the arm in its known-safe observation posture while
                # varying its silhouette by only a few degrees.
                pose = arm_positions.copy()
                arm_delta_deg = [rng.uniform(-2.0, 2.0), rng.uniform(-1.2, 1.2),
                                 rng.uniform(-1.2, 1.2), rng.uniform(-2.0, 2.0),
                                 rng.uniform(-1.2, 1.2), rng.uniform(-2.5, 2.5)]
                pose[arm_indices] = arm_positions[arm_indices] + np.radians(arm_delta_deg)
                arm.set_joint_positions(pose)
                arm.apply_action(ArticulationAction(joint_positions=pose))

                randomization = {
                    "cap_rgb": outer_rgb, "cap_cavity_rgb": cavity_rgb,
                    "lighting": frame_lighting,
                    "tray_position_m": tray_position.tolist(), "tray_yaw_deg": tray_angle,
                    "vial_tilts": vial_tilts,
                    "camera_position_m": camera_position.tolist(),
                    "camera_jitter_xyz_deg": camera_angles,
                    "arm_joint_delta_deg": arm_delta_deg,
                }

            # Tray and vial xforms may vary per frame, so derive the true world
            # rim locations only after those changes have been authored.
            for slot, tube in vial_prims.items():
                matrix = np.array(UsdGeom.Xformable(tube).ComputeLocalToWorldTransform(0.0))
                rims[slot] = (np.array([0.0, 0.0, VIAL_HEIGHT, 1.0]) @ matrix)[:3]
            camera_model = camera_model_for(CameraModel, UsdGeom, camera_prim, camera_geom)
            slots = list(range(1, 31))
            rng.shuffle(slots)
            no_vial_count = rng.randint(3, 6)
            missing_count = rng.randint(6, 9)
            crooked_count = rng.randint(6, 10)
            no_vial = sorted(slots[:no_vial_count])
            missing_end = no_vial_count + missing_count
            crooked_end = missing_end + crooked_count
            missing = sorted(slots[no_vial_count:missing_end])
            crooked = sorted(slots[missing_end:crooked_end])
            good = sorted(slots[crooked_end:])

            for slot, parts in vial_parts.items():
                for part in parts:
                    imageable = UsdGeom.Imageable(part)
                    if slot in no_vial:
                        imageable.MakeInvisible()
                    else:
                        imageable.MakeVisible()

            for slot, prim in cap_prims.items():
                if slot in missing or slot in no_vial:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    continue
                UsdGeom.Imageable(prim).MakeVisible()
                rim = rims[slot].copy()
                if slot in crooked:
                    yaw = rng.uniform(0.0, 360.0)
                    tilt = rng.uniform(31.5, 34.0)
                    rim[0] += rng.uniform(-0.0015, 0.0015)
                    rim[1] += rng.uniform(-0.0015, 0.0015)
                    rim[2] += CROOKED_CAP_LIFT
                    orient = quaternion(Gf, tilt, yaw)
                else:
                    rim[2] += GOOD_CAP_LIFT
                    orient = identity
                set_pose(prim, Gf, rim, orient)

            # The robot consumes loose caps, so each frame may contain any
            # number from zero through ten. Every present table cap is upright
            # and occupies one non-overlapping paper location.
            table_count = rng.randint(0, 10)
            table_locations = sorted(rng.sample(range(10), table_count))
            for prim in table_prims:
                UsdGeom.Imageable(prim).MakeInvisible()
            table_truth = []
            visible_table_indices = []
            # Most randomized scenes put the paper on the requested right side;
            # a minority retain the opposite side so the class is not tied to
            # one absolute image region. Paper and caps move together.
            if args.domain_randomization:
                paper_side = "right" if rng.random() < 0.80 else "left"
                side_sign = 1.0 if paper_side == "right" else -1.0
                paper_center = np.array([
                    tray_position[0] + rng.uniform(-0.012, 0.012),
                    tray_position[1] + side_sign * rng.uniform(0.255, 0.285),
                ])
                paper_angle = tray_angle + rng.uniform(-3.0, 3.0)
            else:
                paper_side = "left"
                paper_center = np.array([0.375, -0.230])
                paper_angle = 0.0
            paper_half_size = np.array([0.130, 0.055])
            paper_corners = []
            for delta in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corner = paper_center + paper_half_size * np.asarray(delta)
                corner = rotate_xy(corner, paper_center, paper_angle)
                paper_corners.append(Gf.Vec3f(float(corner[0]), float(corner[1]), 0.8954))
            if paper is not None:
                paper.GetPointsAttr().Set(paper_corners)

            for table_prim_index, location_index in enumerate(table_locations):
                prim = table_prims[table_prim_index]
                row, column = divmod(location_index, 5)
                local_xy = paper_center + np.array([
                    (column - 2.0) * 0.045,
                    (row - 0.5) * 0.043,
                ])
                xy = rotate_xy(local_xy, paper_center, paper_angle)
                position = [float(xy[0]), float(xy[1]), TABLE_CAP_Z]
                yaw = rng.uniform(0.0, 360.0)
                set_pose(prim, Gf, position, quaternion(Gf, 0.0, yaw))
                UsdGeom.Imageable(prim).MakeVisible()
                visible_table_indices.append(table_prim_index)
                table_truth.append({"prim_index": table_prim_index,
                                    "paper_location": location_index,
                                    "position": position, "yaw_deg": yaw})
            randomization["table_paper"] = {
                "side": paper_side, "center_m": paper_center.tolist(),
                "yaw_deg": paper_angle,
            }

            for _ in range(5):
                world.step(render=True)

            # RTX and the instance annotator can occasionally be one frame
            # behind after many visibility changes. Retry the capture instead
            # of aborting an otherwise valid shard on a transient empty mask.
            visible_paths = [cap_root + f"/G{slot:02d}" for slot in [*good, *crooked]]
            for capture_attempt in range(4):
                rgba = camera.get_rgba()
                segmentation = instance.get_data()
                absent_paths = [path for path in visible_paths
                                if object_bounds(segmentation, path) is None]
                if rgba is not None and rgba.size and not absent_paths:
                    break
                for _ in range(3):
                    world.step(render=True)
            else:
                raise RuntimeError(
                    f"Incomplete camera/instance frame for sample {index}; "
                    f"absent after retries: {absent_paths}"
                )

            lines = []
            bounds_report = {}
            for slot in good:
                path = cap_root + f"/G{slot:02d}"
                bounds = object_bounds(segmentation, path)
                if bounds is None:
                    raise RuntimeError(f"Good cap {path} is not visible in sample {index}")
                lines.append(yolo_line(0, bounds)); bounds_report[f"G{slot:02d}"] = bounds
            for slot in crooked:
                path = cap_root + f"/G{slot:02d}"
                bounds = object_bounds(segmentation, path)
                if bounds is None:
                    raise RuntimeError(f"Crooked cap {path} is not visible in sample {index}")
                lines.append(yolo_line(1, bounds)); bounds_report[f"G{slot:02d}"] = bounds
            for slot in missing:
                bounds = missing_bounds(camera_model, rims[slot])
                lines.append(yolo_line(2, bounds)); bounds_report[f"G{slot:02d}"] = bounds
            occluded_table_indices = []
            for table_index in visible_table_indices:
                path = cap_root + f"/Table_{table_index:02d}"
                bounds = object_bounds(segmentation, path)
                if bounds is None:
                    # A moving arm/gripper can fully hide a loose cap. It has no
                    # visible pixels and therefore must not receive a detector
                    # target; retain it in scene truth for auditability.
                    occluded_table_indices.append(table_index)
                    continue
                lines.append(yolo_line(3, bounds)); bounds_report[f"Table_{table_index:02d}"] = bounds
            for item in table_truth:
                item["visible"] = item["prim_index"] not in occluded_table_indices

            split = split_for(index, args.train_count, args.val_count)
            stem = f"remediation_{index:06d}"
            image_path = out / "images" / split / f"{stem}.png"
            label_path = out / "labels" / split / f"{stem}.txt"
            image = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Failed to write {image_path}")
            label_path.write_text("\n".join(lines) + "\n")
            records.append({
                "index": index, "seed": args.seed + index, "split": split,
                "image": str(image_path), "label_count": len(lines),
                "slots": {"good": good, "crooked": crooked, "missing": missing,
                          "no_vial": no_vial},
                "table_caps": table_truth, "bounds_xyxy": bounds_report,
                "occluded_table_prim_indices": occluded_table_indices,
                "randomization": randomization,
            })
            if sequence == 1 or sequence % 10 == 0 or sequence == len(indices):
                print(f"[shard {args.shard}] {sequence}/{len(indices)} sample={index} split={split}", flush=True)

        instance.detach([render_product])
        manifest = {
            "schema_version": 5,
            "source_scene": str(source_scene),
            "scene_contract": "qualified v3 black-base, silver-top physical remediation setup",
            "cad_dimensions_m": {"vial_od": 0.027178, "vial_id": 0.0263652,
                                 "vial_height": VIAL_HEIGHT, "cap_flange_od": 0.03175},
            "render_geometry": {"vial_material": "/World/Looks/vial_glass",
                                "vial_material_approximation": "clear PETG/polycarbonate",
                                "vial_ior": 1.57,
                                "vial_frosting_roughness": 0.035,
                                "cap_visual": "full Dataset 1 v2 CAD cap",
                                "gripper": gripper_mount,
                                "source_cap_hierarchy_active": False,
                                "tray_top_material": "/World/Looks/alu_machined",
                                "tray_top_metallic": 0.98,
                                "tray_top_roughness": 0.14},
            "sample_distribution": {
                "no_vial_slots": [3, 6], "missing_caps": [6, 9],
                "crooked_caps": [6, 10], "good_caps": "remainder",
                "table_caps": [0, 10], "table_caps_always_upright": True,
            },
            "lighting": {"preset": args.lighting_preset,
                         "dome_intensity": lighting["dome"],
                         "key_intensity": lighting["key"],
                         "rect_size_m": [lighting["rect_width"], lighting["rect_height"]],
                         "fill_intensity": lighting["fill"],
                         "rim_intensity": lighting["rim"],
                         "ceiling_4_intensity": lighting["ceiling_4"],
                         "ceiling_fixture_count": 4 if args.lighting_preset == "physical" else 3,
                         "pink_table_paper": args.pink_table_paper,
                         "purpose": "physical multi-source contact shadows with bright cap cavities"},
            "domain_randomization": {
                "enabled": args.domain_randomization,
                "cap_rgb": {"r": [0.90, 1.00], "g": [0.68, 0.84],
                            "b": [0.025, 0.075]},
                "tray_translation_xy_m": [-0.014, 0.014],
                "tray_yaw_deg": [-2.5, 2.5],
                "vial_tilt_deg": [0.0, 1.8],
                "camera_translation_xy_m": [-0.010, 0.010],
                "camera_translation_z_m": [-0.008, 0.008],
                "camera_rotation_xyz_deg": [-1.0, 1.0],
                "table_paper_side_probability": {"right": 0.8, "left": 0.2},
                "arm_joint_delta_deg": [[-2.0, 2.0], [-1.2, 1.2], [-1.2, 1.2],
                                        [-2.0, 2.0], [-1.2, 1.2], [-2.5, 2.5]],
            },
            "shard": args.shard, "num_shards": args.num_shards,
            "scenes": args.scenes, "train_count": args.train_count,
            "val_count": args.val_count, "records": records,
        }
        (out / "manifests" / f"shard_{args.shard:02d}.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        return 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
