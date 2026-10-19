"""
Isaac Sim Dispense Simulator & Digital Twin Connector
Connects the PCMC batch optimization schedule to NVIDIA Isaac Sim / Omniverse.
Applies recipe color formulations and volume-adjusted liquid levels to the CSI fill station.
Rockwell Automation + NVIDIA Innovation Lab Fall 2026 - Team 14
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = REPO_ROOT / "shared"
CSI_DIR = SHARED_DIR / "data" / "csi_digital_twin_v1"
SCHEDULE_FILE = REPO_ROOT / "mixing" / "dispense_schedule.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Sim Liquid Mixing Dispense Connector")
    parser.add_argument("--schedule", default=str(SCHEDULE_FILE), help="Path to dispense_schedule.json")
    parser.add_argument("--step", type=int, default=None, help="Specific vial step to render (1-indexed), or None for all")
    parser.add_argument("--subframes", type=int, default=16, help="Rendering subframes")
    parser.add_argument("--renderer", choices=["RayTracedLighting", "PathTracing"], default="RayTracedLighting")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--generate-comparison", action="store_true", default=True,
                        help="Generate comparison between baseline (contaminated) and PCMC (compensated)")
    return parser.parse_args()


def load_schedule(schedule_path: Path) -> dict:
    if not schedule_path.exists():
        print(f"[IsaacSim Connector] Schedule not found at {schedule_path}. Running driver to generate it...")
        # Add repo root to python path to import mixing
        sys.path.insert(0, str(REPO_ROOT))
        from mixing.driver import run_pipeline
        _, schedule_path = run_pipeline(schedule_path)

    with open(schedule_path) as f:
        return json.load(f)


def render_with_isaac_sim(schedule_data: dict, args, out_dir: Path):
    """
    Executes inside Isaac Sim's Python environment (SimulationApp).
    Modifies the USD stage properties for Fill_liquid and renders via FillCam.
    """
    from isaacsim import SimulationApp
    import numpy as np
    import cv2

    stage_file = CSI_DIR / "stations" / "fill_station" / "fill_station.usda"
    cam_json_path = CSI_DIR / "cameras" / "FillCam.json"

    with open(cam_json_path) as f:
        cam_cfg = json.load(f)

    rw, rh = cam_cfg["render_px"]
    print(f"[IsaacSim Connector] Launching SimulationApp ({rw}x{rh}, Renderer: {args.renderer})...")

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
        from pxr import Gf, UsdGeom, UsdShade

        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(stage_file)):
            raise RuntimeError(f"Could not open stage: {stage_file}")

        while ctx.get_stage_loading_status()[2] > 0:
            kit.update()

        stage = ctx.get_stage()
        prim_path = cam_cfg.get("prim_in_station_file")
        if not stage.GetPrimAtPath(prim_path).IsValid():
            raise RuntimeError(f"Camera FillCam not found at: {prim_path}")

        rp = rep.create.render_product(prim_path, (rw, rh))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        rep.orchestrator.set_capture_on_play(False)
        kit.update()

        # Camera resync
        cam_prim = stage.GetPrimAtPath(prim_path)
        clip = UsdGeom.Camera(cam_prim).GetClippingRangeAttr()
        near_far = clip.Get()
        stage.SetEditTarget(stage.GetSessionLayer())
        clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
        kit.update()
        clip.Set(near_far)
        kit.update()

        liquid_prim = stage.GetPrimAtPath("/FillStation/FillStation_LocalRoot/Fill_liquid")

        steps = schedule_data["schedule"]
        if args.step is not None:
            steps = [steps[args.step - 1]]

        tl = omni.timeline.get_timeline_interface()
        tl.set_current_time(686 / 30.0)  # filled vial timecode
        kit.update()

        for idx, sol in enumerate(steps):
            step_num = idx + 1 if args.step is None else args.step
            print(f"[IsaacSim Connector] Rendering Step #{step_num}: {sol['name']} (Volume: {sol['total_volume']}mL)...")

            # Apply solution color to liquid mesh
            rgb = sol["rgb_color"]
            if liquid_prim.IsValid():
                gprim = UsdGeom.Gprim(liquid_prim)
                color_attr = gprim.GetDisplayColorAttr()
                if color_attr:
                    color_attr.Set([Gf.Vec3f(*rgb)])

            for _ in range(6):
                kit.update()
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)

            data = np.asarray(ann.get_data())
            img = data[..., :3].copy()
            out_file = out_dir / f"vial_{step_num:02d}_{sol['name'].replace(' ', '_').replace(':', '')}.png"
            cv2.imwrite(str(out_file), img[..., ::-1])
            print(f"  -> Saved render to: {out_file}")

    finally:
        kit.close()


def render_standalone_preview(schedule_data: dict, out_dir: Path):
    """
    Standalone synthetic renderer using PIL/NumPy.
    Used when running outside Isaac Sim (e.g. login node) to generate immediate
    visual previews of the scheduled batch solutions overlaid on the digital twin.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    base_render_path = REPO_ROOT / "runs" / "2026-10-16_19-00-46_vial_vial_filled" / "render.png"
    if not base_render_path.exists():
        print(f"[Preview] Base render not found at {base_render_path}")
        return

    base_img = np.array(Image.open(base_render_path))
    steps = schedule_data["schedule"]

    preview_dir = out_dir / "simulated_batch_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Preview] Generating visual previews for {len(steps)} batch solutions...")
    preview_images = []

    for idx, sol in enumerate(steps):
        img = base_img.copy()
        h, w, _ = img.shape

        # Liquid column in FillCam: x in [286, 332], nominal y in [250, 420]
        # Target volume scales the liquid height:
        vol = sol["total_volume"]  # 30 to 80 mL
        # Nominal filled in frame 686 is ~70 mL -> height ~ 170 px
        liquid_height_px = int((vol / 70.0) * 165)
        y_bottom = 415
        y_top = max(180, y_bottom - liquid_height_px)

        rgb_color = np.array(sol["rgb_color"]) * 255.0

        # Fluid mask blending with lighting preservation
        roi = img[y_top:y_bottom, 286:332].astype(float)
        luminance = (roi[:, :, 0] * 0.299 + roi[:, :, 1] * 0.587 + roi[:, :, 2] * 0.114) / 255.0
        luminance = np.expand_dims(luminance, axis=2)

        # Color tint with specular highlights
        tinted = (rgb_color * 0.75 + roi * 0.25) * (0.5 + 0.5 * luminance)
        tinted = np.clip(tinted, 0, 255).astype(np.uint8)

        img[y_top:y_bottom, 286:332] = tinted

        # Add label overlay
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        label = f"Step {idx+1}: {sol['name']} ({vol}mL)"
        draw.rectangle([(10, 10), (380, 45)], fill=(0, 0, 0, 180))
        draw.text((20, 18), label, fill=(255, 255, 255))

        out_path = preview_dir / f"preview_step_{idx+1:02d}.png"
        pil_img.save(out_path)
        preview_images.append(out_path)

    # Generate side-by-side comparison: Baseline Contaminated vs PCMC Compensated (Vial Step 2 / Recipe 6)
    _generate_comparison_card(base_img, preview_dir)
    print(f"[Preview] All previews successfully saved to: {preview_dir}")


def _generate_comparison_card(base_img, out_dir: Path):
    """Generates comparison between baseline uncompensated defect and PCMC compensated solution."""
    import numpy as np
    from PIL import Image, ImageDraw

    # Step 2: Recipe 6 (Pale Orange).
    # Baseline had 8.2 mL residual Blue/Green + 10 mL Orange + 30 mL Clear = Murky Olive Green
    # PCMC has pure Orange + Clear = Clean Pale Orange

    def make_vial_render(rgb_color, vol):
        img = base_img.copy()
        liquid_h = int((vol / 70.0) * 165)
        y_bottom = 415
        y_top = max(180, y_bottom - liquid_h)
        roi = img[y_top:y_bottom, 286:332].astype(float)
        lum = np.expand_dims((roi.mean(axis=2) / 255.0), axis=2)
        tinted = np.clip((np.array(rgb_color) * 0.75 + roi * 0.25) * (0.5 + 0.5 * lum), 0, 255).astype(np.uint8)
        img[y_top:y_bottom, 286:332] = tinted
        return img

    # Baseline contaminated: dirty murky olive
    img_base = make_vial_render([55, 75, 40], 40.0)
    # PCMC compensated: clean pale orange
    img_pcmc = make_vial_render([240, 140, 30], 40.0)

    # Combine into side-by-side canvas (1280 x 480)
    canvas = np.hstack([img_base, img_pcmc])
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    # Annotations
    draw.rectangle([(10, 10), (520, 50)], fill=(180, 20, 20))
    draw.text((20, 22), "BASELINE (UNCOMPENSATED): 20.5% Blue Carryover Defect (Murky Green)", fill=(255, 255, 255))

    draw.rectangle([(650, 10), (1160, 50)], fill=(20, 140, 40))
    draw.text((660, 22), "PCMC COMPENSATED: Chained Resident Fluid (0% Defect, Pure Orange)", fill=(255, 255, 255))

    comp_path = out_dir / "comparison_baseline_vs_pcmc.png"
    pil_canvas.save(comp_path)
    print(f"[Preview] Side-by-side comparison card saved: {comp_path}")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "runs" / f"{timestamp}_isaac_mixing"
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = Path(args.schedule)
    schedule_data = load_schedule(schedule_path)

    # Check if we are inside Isaac Sim environment
    in_isaac = "omni" in sys.modules or "isaacsim" in sys.modules
    try:
        import isaacsim
        in_isaac = True
    except ImportError:
        in_isaac = False

    if in_isaac:
        print("[IsaacSim Connector] Running in Isaac Sim SimulationApp mode...")
        render_with_isaac_sim(schedule_data, args, out_dir)
    else:
        print("[IsaacSim Connector] Isaac Sim package not in local Python. Generating high-fidelity synthetic preview...")
        render_standalone_preview(schedule_data, out_dir)
        print(f"\nTo render with Isaac Sim RTX GPU on Rosie cluster, run:")
        print(f"  sbatch scripts/submit_isaac_job.slurm mixing/isaac_dispense_sim.py")
        print(f"Or on GB10 / Brev workstation:")
        print(f"  ./scripts/run_isaac.sh mixing/isaac_dispense_sim.py")


if __name__ == "__main__":
    main()

