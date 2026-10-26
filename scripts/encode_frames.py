#!/usr/bin/env python3
"""Encode a directory of image frames into an MP4 video using OpenCV.

Usage:
    ./scripts/encode_frames.py --frames <path_to_frames_dir> --out <output.mp4> [--fps 20]

Example:
    ./scripts/encode_frames.py --frames runs/compliant_remediation_test/action/frames --out demo.mp4
"""
import argparse
from pathlib import Path
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', '-i', type=Path, required=True, help='Path to directory containing PNG/JPG frames')
    parser.add_argument('--out', '-o', type=Path, required=True, help='Output path for the .mp4 video')
    parser.add_argument('--fps', type=float, default=20.0, help='Playback frame rate (default: 20 fps)')
    args = parser.parse_args()

    frame_dir = args.frames.resolve()
    if not frame_dir.is_dir():
        print(f"Error: '{frame_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Search for frames (png, jpg, jpeg) sorted naturally
    frame_files = sorted(
        [p for p in frame_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    )
    if not frame_files:
        print(f"Error: No image frames found in '{frame_dir}'.", file=sys.stderr)
        sys.exit(1)

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"Error: Could not read first frame '{frame_files[0]}'.", file=sys.stderr)
        sys.exit(1)

    height, width = first_frame.shape[:2]
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try mp4v or avc1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))

    if not writer.isOpened():
        # Fallback to MJPG / avi if mp4v fails
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_path = out_path.with_suffix('.avi')
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))

    print(f"Encoding {len(frame_files)} frames ({width}x{height} @ {args.fps} fps) -> {out_path} ...")
    for idx, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath))
        if img is not None:
            writer.write(img)
        if (idx + 1) % 50 == 0 or idx == len(frame_files) - 1:
            print(f"  Processed {idx + 1}/{len(frame_files)} frames...", flush=True)

    writer.release()
    size_mb = out_path.stat().st_size / (1024 * 1024)
    duration_s = len(frame_files) / args.fps
    print(f"Successfully generated: {out_path} ({size_mb:.2f} MB, {duration_s:.1f}s)")


if __name__ == '__main__':
    main()
