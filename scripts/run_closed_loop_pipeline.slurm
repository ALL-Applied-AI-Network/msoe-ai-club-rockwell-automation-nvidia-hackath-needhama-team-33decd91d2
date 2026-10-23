#!/usr/bin/env bash
#SBATCH --job-name=remediation_closed_loop
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# One Slurm job, four sequential processes. Isaac Sim and Ultralytics cannot
# share one interpreter because their protobuf runtimes conflict; keeping them
# as child processes preserves a single allocated GPU job and one auditable run.
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
./scripts/run_isaac.sh src/remediation/capture_live_tablecam.py --scene "$scene" --out "$perception"
./.conda_env/bin/python src/inspection/inspect_remediation_capture.py \
    --checkpoint "$checkpoint" \
    --image "$perception/current_scene_tablecam.png" \
    --depth "$perception/current_scene_tablecam_depth_m.npy" \
    --capture-report "$perception/capture_report.json" \
    --out "$perception/current_scene_detections.json" --device 0
./scripts/run_isaac.sh src/remediation/run_best_effort_closed_loop.py \
    --scene "$scene" --detections "$perception/current_scene_detections.json" --out "$action"
./.conda_env/bin/python src/remediation/overlay_yolo_frames.py \
    --frames "$action/frames" --checkpoint "$checkpoint" --out "$action/yolo_frames" --device 0
ffmpeg -y -framerate 20 -i "$action/yolo_frames/%04d.png" -c:v libx264 -pix_fmt yuv420p "$out/tablecam_yolo_per_frame.mp4"
