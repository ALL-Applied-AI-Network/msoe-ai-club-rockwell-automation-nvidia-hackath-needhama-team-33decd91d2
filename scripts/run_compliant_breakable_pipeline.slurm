#!/usr/bin/env bash
#SBATCH --job-name=compliant_remediation
#SBATCH --partition=teaching
#SBATCH --exclude=dh-node11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Physically compliant, fracture-aware digital twin remediation pipeline.
# Executes TableCam perception, YOLO defect detection, compliant zero-clipping
# pivot press simulation with breakable glass vial monitoring, and frame annotation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch $0 <scene.usda> <yolo_checkpoint.pt> <output_directory>" >&2
    exit 2
fi

scene="$1"
checkpoint="$2"
out="$3"
perception="$out/perception"
action="$out/action"

mkdir -p "$out"
rm -rf "$action/frames" "$action/yolo_frames"

echo "[pipeline] 1/4: Capturing live TableCam observation..."
./scripts/run_isaac.sh src/remediation/capture_live_tablecam.py --scene "$scene" --out "$perception"

echo "[pipeline] 2/4: Running YOLO perception and 3D localization..."
./.conda_env/bin/python src/inspection/inspect_remediation_capture.py \
    --checkpoint "$checkpoint" \
    --image "$perception/current_scene_tablecam.png" \
    --depth "$perception/current_scene_tablecam_depth_m.npy" \
    --capture-report "$perception/capture_report.json" \
    --out "$perception/current_scene_detections.json" --device 0

echo "[pipeline] 3/4: Running compliant, breakable digital twin simulation..."
./scripts/run_isaac.sh src/remediation/run_compliant_breakable_remediation.py \
    --scene "$scene" --detections "$perception/current_scene_detections.json" --out "$action"

echo "[pipeline] 4/4: Overlaying per-frame YOLO detections..."
./.conda_env/bin/python src/remediation/overlay_yolo_frames.py \
    --frames "$action/frames" --checkpoint "$checkpoint" --out "$action/yolo_frames" --device 0

FFMPEG_BIN=$(command -v ffmpeg || echo /usr/bin/ffmpeg)
if [[ -x "$FFMPEG_BIN" ]]; then
    "$FFMPEG_BIN" -y -framerate 20 -i "$action/yolo_frames/%04d.png" -c:v libx264 -pix_fmt yuv420p "$out/tablecam_yolo_compliant_remediation.mp4"
    echo "[pipeline] Video compiled successfully at $out/tablecam_yolo_compliant_remediation.mp4"
else
    echo "[pipeline] Note: ffmpeg not found on compute node; login node will compile the video."
fi

