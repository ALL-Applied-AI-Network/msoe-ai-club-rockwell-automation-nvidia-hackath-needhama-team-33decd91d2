"""
Universal Render & Animation Pipeline for Innovation Lab Fall 2026.
Supports single-frame renders and multi-frame animation sequences across:
  - CSI Digital Twin (FillStation FillCam, CapSealCam)
  - FANUC ER-4iA Robot Cell (side, table, wrist cameras)
  - Persistent scene builds (shared/scenes/fanuc_scene.usd)

Outputs are organized cleanly into:
  runs/<YYYY-MM-DD_HH-MM-SS>_<target>_<tag>/
    ├── frame_0001.png (or render.png)
    └── meta.json
"""

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = REPO_ROOT / "shared"
CSI_DIR = SHARED_DIR / "data" / "csi_digital_twin_v1"
FANUC_DIR = SHARED_DIR / "data" / "fanuc_er4ia_sim"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified render and multi-frame generation runner")
    parser.add_argument("--target", choices=["vial", "cap_seal", "fanuc", "build_fanuc"], default="vial",
                        help="Target scene or action")
    parser.add_argument("--state", choices=["filled", "empty", "sequence"], default="filled",
                        help="Preset state for CSI vial filling (filled=686, empty=1, sequence=start..end)")
    parser.add_argument("--camera", default=None,
                        help="Camera override (e.g., FillCam, CapSealCam, side, table, wrist)")
    parser.add_argument("--stage", default=None,
                        help="USD stage path override")
    parser.add_argument("--frame", type=int, default=None,
                        help="Single frame number to render")
    parser.add_argument("--start-frame", type=int, default=1,
                        help="Start frame for sequence rendering")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="End frame for sequence rendering")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame stride / step interval for sequence rendering")
    parser.add_argument("--subframes", type=int, default=8,
                        help="Rendering quality subframes (path-tracing / ray-tracing)")
    parser.add_argument("--tag", default=None,
                        help="Custom label for the output run folder")
    parser.add_argument("--out-dir", default=None,
                        help="Explicit output directory override (skips auto-timestamped naming)")
    parser.add_argument("--renderer", choices=["RayTracedLighting", "PathTracing"], default="RayTracedLighting",
                        help="RTX renderer engine")
    return parser.parse_args()


def get_run_dir(target_name, tag=None, out_dir=None):
    if out_dir:
        d = Path(out_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        label = f"_{tag}" if tag else ""
        d = REPO_ROOT / "runs" / f"{timestamp}_{target_name}{label}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_build_fanuc():
    target_usd = SHARED_DIR / "scenes" / "fanuc_scene.usd"
    target_usd.parent.mkdir(parents=True, exist_ok=True)
    print(f"[render.py] Building persistent FANUC scene at: {target_usd}")
    
    from isaacsim import SimulationApp
    kit = SimulationApp({"headless": True})
    
    sys.path.insert(0, str(FANUC_DIR))
    import omni.usd
    import cell
    from isaacsim.core.api import World
    
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage)
    stage.Export(str(target_usd))
    kit.close()
    print(f"[render.py] Successfully exported: {target_usd}")


def render_csi(args, run_dir):
    import cv2
    import numpy as np
    from isaacsim import SimulationApp

    # Resolve stage and camera defaults
    if args.target == "cap_seal":
        camera_name = args.camera or "CapSealCam"
        stage_file = args.stage or "stations/tray_inspection/tray_inspection.usdc"
        frames_to_render = [args.frame or 1]
    else:  # vial
        camera_name = args.camera or "FillCam"
        stage_file = args.stage or "stations/fill_station/fill_station.usda"
        if args.frame is not None:
            frames_to_render = [args.frame]
        elif args.state == "filled":
            frames_to_render = [686]
        elif args.state == "empty":
            frames_to_render = [1]
        elif args.state == "sequence":
            end_f = args.end_frame or 686
            frames_to_render = list(range(args.start_frame, end_f + 1, args.step))
        else:
            frames_to_render = [686]

    cam_json_path = CSI_DIR / "cameras" / f"{camera_name}.json"
    with open(cam_json_path) as f:
        cam_cfg = json.load(f)

    rw, rh = cam_cfg["render_px"]
    resolved_stage = CSI_DIR / stage_file if not Path(stage_file).is_absolute() else Path(stage_file)
    if not resolved_stage.exists():
        raise FileNotFoundError(f"Stage not found: {resolved_stage}")

    print(f"[render.py] Initializing SimulationApp (Renderer: {args.renderer}, {rw}x{rh})...")
    kit = SimulationApp({
        "headless": True,
        "renderer": args.renderer,
        "width": rw,
        "height": rh
    })

    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.usd, omni.timeline
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(resolved_stage)):
            raise RuntimeError(f"Could not open stage: {resolved_stage}")

        while ctx.get_stage_loading_status()[2] > 0:
            kit.update()

        stage = ctx.get_stage()
        prim_candidates = [
            cam_cfg.get("prim_in_csi_line"),
            cam_cfg.get("prim_in_station_file")
        ]
        prim_path = next((p for p in prim_candidates if p and stage.GetPrimAtPath(p).IsValid()), None)
        if not prim_path:
            raise RuntimeError(f"Camera {camera_name} not found in stage {stage_file}")

        rp = rep.create.render_product(prim_path, (rw, rh))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        rep.orchestrator.set_capture_on_play(False)
        kit.update()

        # Resync camera clipping range if needed
        cam_prim = stage.GetPrimAtPath(prim_path)
        clip = UsdGeom.Camera(cam_prim).GetClippingRangeAttr()
        near_far = clip.Get()
        stage.SetEditTarget(stage.GetSessionLayer())
        clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
        kit.update()
        clip.Set(near_far)
        kit.update()

        tl = omni.timeline.get_timeline_interface()
        time_codes_per_sec = stage.GetTimeCodesPerSecond() or 30.0

        is_multi = len(frames_to_render) > 1
        rendered_files = []

        print(f"[render.py] Rendering {len(frames_to_render)} frame(s) for camera: {camera_name}...")
        for idx, fnum in enumerate(frames_to_render):
            tl.set_current_time(fnum / time_codes_per_sec)
            kit.update()

            # Warmup / capture step
            data = None
            for _ in range(6):
                kit.update()
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
                curr = np.asarray(ann.get_data())
                if curr.ndim == 3 and curr.shape[:2] == (rh, rw) and curr.shape[2] >= 3:
                    data = curr
                    if data.max() > 0:
                        break

            if data is None or data.max() == 0:
                print(f"[render.py] Warning: Frame {fnum} captured empty/zero pixels")

            img = data[..., :3].copy()
            out_filename = f"frame_{fnum:04d}.png" if is_multi else "render.png"
            out_path = run_dir / out_filename

            # Post-process lens distortion if overscan exists
            if cam_cfg.get("overscan", 1.0) != 1.0:
                tmp = str(out_path) + ".pinhole.png"
                cv2.imwrite(tmp, img[..., ::-1])
                try:
                    subprocess.run([
                        sys.executable,
                        str(CSI_DIR / "tools" / "apply_lens.py"),
                        tmp,
                        str(cam_json_path),
                        str(out_path)
                    ], check=True)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
            else:
                cv2.imwrite(str(out_path), img[..., ::-1])

            rendered_files.append(str(out_path))
            print(f"[{idx+1}/{len(frames_to_render)}] Saved {out_path.name} (Frame {fnum})", flush=True)

        meta = {
            "target": args.target,
            "camera": camera_name,
            "stage": str(resolved_stage),
            "frames": frames_to_render,
            "subframes": args.subframes,
            "renderer": args.renderer,
            "timestamp": datetime.now().isoformat(),
            "rendered_files": rendered_files
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print("\n" + "=" * 60)
        print(f"[render.py] SUCCESS: Render complete!")
        print(f"[render.py] Target:   {args.target} ({camera_name})")
        print(f"[render.py] Frames:   {len(rendered_files)} frame(s) rendered")
        print(f"[render.py] Output:   {run_dir}")
        if len(rendered_files) == 1:
            print(f"[render.py] Image:    {rendered_files[0]}")
        print("=" * 60 + "\n", flush=True)

    finally:
        kit.close()


def render_fanuc(args, run_dir):
    import cv2
    import numpy as np
    from isaacsim import SimulationApp

    camera_name = args.camera or "side"
    num_frames = args.end_frame if args.end_frame else (1 if args.state != "sequence" else 30)

    print(f"[render.py] Initializing SimulationApp for FANUC cell...")
    kit = SimulationApp({"headless": True, "renderer": args.renderer})

    try:
        sys.path.insert(0, str(FANUC_DIR))
        import omni.usd
        import scene_config as C, cell
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera

        # If loading prebuilt stage, open it before initializing World
        cached_scene = SHARED_DIR / "scenes" / "fanuc_scene.usd"
        if cached_scene.exists() and not args.stage:
            print(f"[render.py] Loading prebuilt scene: {cached_scene}")
            omni.usd.get_context().open_stage(str(cached_scene))
            while omni.usd.get_context().get_stage_loading_status()[2] > 0:
                kit.update()
            stage = omni.usd.get_context().get_stage()
            world = World(stage_units_in_meters=1.0)
        else:
            world = World(stage_units_in_meters=1.0)
            stage = omni.usd.get_context().get_stage()
            cell.build(stage)

        _CAMS = {
            "table": (cell.TABLE_CAM_PATH, C.TABLE_CAM),
            "side": (cell.SIDE_CAM_PATH, C.SIDE_CAM),
            "wrist": (cell.WRIST_CAM_PATH, C.WRIST_CAM),
        }
        if camera_name not in _CAMS:
            raise ValueError(f"Unknown camera {camera_name}. Choose from: {list(_CAMS.keys())}")

        cam_path, cfg = _CAMS[camera_name]
        cam = Camera(prim_path=cam_path, resolution=cfg["resolution"])
        cell.enable_ccd(world)
        world.reset()
        cam.initialize()

        # Let materials settle
        for _ in range(30):
            world.step(render=True)

        is_multi = num_frames > 1
        rendered_files = []

        print(f"[render.py] Capturing {num_frames} frame(s) for FANUC camera '{camera_name}'...")
        for f in range(num_frames):
            world.step(render=True)
            rgba = cam.get_rgba()
            if rgba is None:
                raise RuntimeError("Camera returned no image data")
            rgb = rgba[:, :, :3].astype(np.uint8)
            out_name = f"frame_{f:04d}.png" if is_multi else "render.png"
            out_path = run_dir / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rendered_files.append(str(out_path))

        meta = {
            "target": "fanuc",
            "camera": camera_name,
            "frames_count": num_frames,
            "timestamp": datetime.now().isoformat(),
            "rendered_files": rendered_files
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print("\n" + "=" * 60)
        print(f"[render.py] SUCCESS: Render complete!")
        print(f"[render.py] Target:   FANUC Robot Cell ({camera_name})")
        print(f"[render.py] Frames:   {len(rendered_files)} frame(s) rendered")
        print(f"[render.py] Output:   {run_dir}")
        if len(rendered_files) == 1:
            print(f"[render.py] Image:    {rendered_files[0]}")
        print("=" * 60 + "\n", flush=True)

    finally:
        kit.close()


def main():
    args = parse_args()

    if args.target == "build_fanuc":
        run_build_fanuc()
        return

    tag = args.tag or (f"{args.target}_{args.state}" if args.target == "vial" else f"{args.target}_{args.camera or 'cam'}")
    run_dir = get_run_dir(args.target, tag=tag, out_dir=args.out_dir)
    print(f"[render.py] Output directory: {run_dir}")

    if args.target in ["vial", "cap_seal"]:
        render_csi(args, run_dir)
    elif args.target == "fanuc":
        render_fanuc(args, run_dir)


if __name__ == "__main__":
    main()

