#!/usr/bin/env python3
"""Encode simulation frames into either fast or realistic presentation MP4 videos.

In 'fast' mode:
  Direct continuous 60 fps encoding without compute pauses (pipelined simulation time).

In 'realistic' mode:
  Accurately injects real-world inspection dwell times into the video sequence:
  - TableCam overhead perception scan: 1.5 seconds (90 frames)
  - Wrist-camera LocateAnything-3B inference dwell: 2.4 seconds (144 frames)
  - Endpoint physical servo settling dwell: 0.5 seconds (30 frames)
  - Post-remediation verification dwell: 1.5 seconds (90 frames)

Matches the exact cadence and timing of the physical Fanuc ER-4iA workcell 1:1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def find_ffmpeg() -> str:
    # Prefer imageio-ffmpeg static binary bundled in the repo's conda env
    candidate = ROOT / ".conda_env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
    if candidate.is_file():
        return str(candidate)
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("No suitable ffmpeg binary found in environment.")


def get_inspection_pause_frames(trace_file: Path | None) -> dict[int, int]:
    """Identify frame indices where the robot holds for camera inspection / VLM inference."""
    pauses: dict[int, int] = {0: 90}  # Initial TableCam scan dwell (1.5s = 90 frames)
    if not trace_file or not trace_file.is_file():
        return pauses

    try:
        data = json.loads(trace_file.read_text())
        last_phase = None
        for idx, entry in enumerate(data):
            phase = entry.get("phase", "")
            if not phase or phase == last_phase:
                continue
            last_phase = phase

            # Wrist VLM refinement poses (await_refine_*)
            if "await_refine_spare" in phase:
                pauses[idx] = 144  # 2.4s LocateAnything inference
            elif "await_refine_missing" in phase:
                pauses[idx] = 144  # 2.4s LocateAnything inference
            elif "await_refine_crooked" in phase:
                pauses[idx] = 144  # 2.4s LocateAnything inference
            elif "gripper" in phase:
                pauses[idx] = 30   # 0.5s physical jaw motion & settling
            elif "settle" in phase:
                pauses[idx] = 90   # 1.5s post-verification dwell
    except Exception as e:
        print(f"Warning: Failed to parse trace file {trace_file}: {e}", file=sys.stderr)

    return pauses


def encode_video(
    frame_dir: Path,
    out_path: Path,
    fps: int = 60,
    crf: int = 24,
    mode: str = "fast",
    trace_file: Path | None = None,
) -> None:
    frame_files = sorted(
        [p for p in frame_dir.glob("*.png")] + [p for p in frame_dir.glob("*.jpg")]
    )
    if not frame_files:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")

    ffmpeg_bin = find_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pauses = get_inspection_pause_frames(trace_file) if mode == "realistic" else {}

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "image2pipe",
        "-vcodec", "png" if frame_files[0].suffix == ".png" else "mjpeg",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    print(f"[{mode.upper()}] Encoding {len(frame_files)} frames from {frame_dir} -> {out_path} ...")
    if mode == "realistic":
        print(f"  Injecting realistic camera & VLM inspection pauses at frames: {sorted(pauses.keys())}")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    total_written = 0

    try:
        for idx, fpath in enumerate(frame_files):
            raw = fpath.read_bytes()
            proc.stdin.write(raw)
            total_written += 1

            # Inject dwell pause frames
            if idx in pauses:
                repeat_count = pauses[idx]
                for _ in range(repeat_count):
                    proc.stdin.write(raw)
                    total_written += 1

            if (idx + 1) % 500 == 0 or idx == len(frame_files) - 1:
                print(f"  Processed {idx + 1}/{len(frame_files)} source frames (total stream: {total_written})...", flush=True)

        proc.stdin.close()
        proc.wait()
    except Exception:
        proc.kill()
        raise

    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg encoding failed (code {proc.returncode}):\n{err}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    duration_s = total_written / fps
    print(f"[{mode.upper()}] Successfully encoded: {out_path} ({size_mb:.2f} MB, {duration_s:.1f}s @ {fps} fps)\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", "-i", type=Path, required=True, help="Directory containing frames")
    p.add_argument("--out", "-o", type=Path, required=True, help="Output MP4 file path")
    p.add_argument("--fps", type=int, default=60, help="Framerate (default: 60)")
    p.add_argument("--crf", type=int, default=24, help="H.264 CRF quality (default: 24)")
    p.add_argument("--mode", choices=("fast", "realistic", "both"), default="both",
                   help="'fast' (unpaused), 'realistic' (with VLM & TableCam dwell pauses), or 'both'")
    p.add_argument("--trace", type=Path, help="Path to oracle_evaluation_trace.json for phase transitions")
    args = p.parse_args()

    if args.mode in ("fast", "both"):
        fast_out = args.out if args.mode == "fast" else args.out.with_name(args.out.stem + "_fast.mp4")
        encode_video(args.frames, fast_out, fps=args.fps, crf=args.crf, mode="fast", trace_file=args.trace)

    if args.mode in ("realistic", "both"):
        real_out = args.out if args.mode == "realistic" else args.out.with_name(args.out.stem + "_realistic.mp4")
        encode_video(args.frames, real_out, fps=args.fps, crf=args.crf, mode="realistic", trace_file=args.trace)


if __name__ == "__main__":
    main()
