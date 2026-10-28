#!/usr/bin/env python3
"""Render randomized YOLO samples from the CAD remediation deployment scene.

The immutable template is ``remediation_dataset1_v2_cad.usda``: the FANUC
cell/camera from the original remediation scene plus the corrected Dataset 1
tray, CAD vials, and hollow MOCAP caps. Each worker loads it once and varies
only cap state/pose, making 1,000 full-resolution samples practical.
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
    parser.add_argument("--lighting-preset", choices=("balanced", "true"),
                        default="balanced")
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

    source_scene = ROOT / "shared" / "scenes" / "remediation_dataset1_v2_cad.usda"
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

        # The source's large studio lights nearly eliminate contact shadows.
        # "balanced" keeps the first synthetic set readable; "true" uses a
        # smaller dominant source so loose caps cast unmistakable shadows.
        dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
        key = UsdLux.RectLight(stage.GetPrimAtPath("/World/KeyLight"))
        fill = UsdLux.RectLight(stage.GetPrimAtPath("/World/FillLight"))
        rim_light = UsdLux.RectLight(stage.GetPrimAtPath("/World/RimLight"))
        if args.lighting_preset == "true":
            lighting = {"dome": 650.0, "key": 475000.0, "key_size": 0.07,
                        "fill": 30000.0, "rim": 40000.0}
        else:
            lighting = {"dome": 650.0, "key": 320000.0, "key_size": 0.18,
                        "fill": 30000.0, "rim": 40000.0}
        dome.GetIntensityAttr().Set(lighting["dome"])
        key.GetIntensityAttr().Set(lighting["key"])
        key.GetWidthAttr().Set(lighting["key_size"])
        key.GetHeightAttr().Set(lighting["key_size"])
        fill.GetIntensityAttr().Set(lighting["fill"])
        rim_light.GetIntensityAttr().Set(lighting["rim"])

        paper = None
        paper_color = None
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
            # The default matches the light physical sheet photographed under
            # TableCam-like lighting. Domain-randomized runs also sample the
            # darker physical sheet and the original saturated synthetic pink.
            paper_color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            paper_color.Set(Gf.Vec3f(0.82, 0.47, 0.56))
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
        cavity_color = stage.GetPrimAtPath(
            "/World/RemediationLooks/CapCavity/Surface").GetAttribute("inputs:diffuseColor")

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
                outer_rgb = [rng.uniform(0.78, 1.00), rng.uniform(0.58, 0.90),
                             rng.uniform(0.02, 0.16)]
                cavity_rgb = [outer_rgb[0] * 0.54, outer_rgb[1] * 0.49,
                              outer_rgb[2] * 0.28]
                cap_color.Set(Gf.Vec3f(*outer_rgb))
                cavity_color.Set(Gf.Vec3f(*cavity_rgb))
                if paper_color is not None:
                    paper_palette = (
                        (0.82, 0.47, 0.56),  # measured light physical paper
                        (0.48, 0.27, 0.35),  # measured dark physical paper
                        (0.92, 0.20, 0.52),  # retained saturated synthetic pink
                    )
                    paper_base = paper_palette[rng.randrange(len(paper_palette))]
                    paper_rgb = [min(1.0, max(0.0, value + rng.uniform(-0.045, 0.045)))
                                 for value in paper_base]
                    paper_color.Set(Gf.Vec3f(*paper_rgb))
                    randomization["paper_rgb"] = paper_rgb

                frame_lighting = {
                    "dome": rng.uniform(500.0, 820.0),
                    "key": rng.uniform(390000.0, 560000.0),
                    "key_size": rng.uniform(0.055, 0.105),
                    "fill": rng.uniform(22000.0, 43000.0),
                    "rim": rng.uniform(28000.0, 56000.0),
                }
                dome.GetIntensityAttr().Set(frame_lighting["dome"])
                key.GetIntensityAttr().Set(frame_lighting["key"])
                key.GetWidthAttr().Set(frame_lighting["key_size"])
                key.GetHeightAttr().Set(frame_lighting["key_size"])
                fill.GetIntensityAttr().Set(frame_lighting["fill"])
                rim_light.GetIntensityAttr().Set(frame_lighting["rim"])

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
                    "cap_rgb": outer_rgb, "lighting": frame_lighting,
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
            "schema_version": 4,
            "source_scene": str(source_scene),
            "scene_contract": "FANUC remediation cell + Dataset 1 v2 CAD tray/vials/caps",
            "cad_dimensions_m": {"vial_od": 0.027178, "vial_id": 0.0263652,
                                 "vial_height": VIAL_HEIGHT, "cap_flange_od": 0.03175},
            "render_geometry": {"vial_material": "/World/Looks/vial_glass",
                                "vial_material_approximation": "clear PETG/polycarbonate",
                                "vial_ior": 1.57,
                                "vial_frosting_roughness": 0.035,
                                "cap_visual": "full Dataset 1 v2 CAD cap",
                                "gripper": gripper_mount,
                                "source_cap_hierarchy_active": False,
                                "tray_top_material": "/World/Looks/alu_machined"},
            "sample_distribution": {
                "no_vial_slots": [3, 6], "missing_caps": [6, 9],
                "crooked_caps": [6, 10], "good_caps": "remainder",
                "table_caps": [0, 10], "table_caps_always_upright": True,
            },
            "lighting": {"preset": args.lighting_preset,
                         "dome_intensity": lighting["dome"],
                         "key_intensity": lighting["key"],
                         "key_size_m": [lighting["key_size"], lighting["key_size"]],
                         "fill_intensity": lighting["fill"],
                         "rim_intensity": lighting["rim"],
                         "pink_table_paper": args.pink_table_paper,
                         "purpose": "visible table-cap contact shadows"},
            "domain_randomization": {
                "enabled": args.domain_randomization,
                "cap_rgb": {"r": [0.78, 1.00], "g": [0.58, 0.90],
                            "b": [0.02, 0.16]},
                "paper_rgb_bases": [[0.82, 0.47, 0.56], [0.48, 0.27, 0.35],
                                    [0.92, 0.20, 0.52]],
                "paper_rgb_jitter": [-0.045, 0.045],
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
