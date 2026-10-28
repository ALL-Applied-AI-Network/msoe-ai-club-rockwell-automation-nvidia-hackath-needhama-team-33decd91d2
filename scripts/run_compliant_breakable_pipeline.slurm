#!/usr/bin/env bash
#SBATCH --job-name=compliant_remediation
#SBATCH --partition=teaching
#SBATCH --exclude=dh-node11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Physically compliant, fracture-aware digital twin remediation pipeline.
# Executes TableCam perception, YOLO defect detection, shared-policy collision
# simulation with clear-plastic vial evaluation, and frame annotation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

canonical_scene="shared/scenes/tablecam_right_pink_spares_qualified_v2.usda"
if [[ $# -ge 3 && "$1" == *.usd* ]]; then
    scene="$1"
    checkpoint="$2"
    out="$3"
    shift 3
elif [[ $# -ge 2 ]]; then
    scene="$canonical_scene"
    checkpoint="$1"
    out="$2"
    shift 2
else
    echo "Usage: sbatch $0 [scene.usda] <yolo_checkpoint.pt> <output_directory> [remediation options]" >&2
    echo "  With no scene argument, uses $canonical_scene" >&2
    exit 2
fi
remediation_args=("$@")

# Keep the original project-local environment as the first choice, but support
# the team HPC environment used by the current YOLO training/evaluation jobs.
if [[ -x "./.conda_env/bin/python" ]]; then
    yolo_python="./.conda_env/bin/python"
elif [[ -x "./shared/cache/locateanything_3b/venv/bin/python" ]]; then
    yolo_python="./shared/cache/locateanything_3b/venv/bin/python"
else
    echo "No YOLO Python environment found (.conda_env or shared HPC venv)" >&2
    exit 2
fi
export REMEDIATION_PYTHON="$yolo_python"
echo "[pipeline] YOLO Python: $yolo_python"

# All outputs live under shared/runs/ (bare names are placed there).
case "$out" in
    shared/runs/*) ;;
    *) out="shared/runs/${out#/}" ;;
esac
echo "[pipeline] outputs: $out"

if [[ ! -f "$scene" ]]; then
    echo "Scene does not exist: $scene" >&2
    exit 2
fi
perception="$out/perception"
action="$out/action"

mkdir -p "$out"
rm -rf "$action/frames" "$action/yolo_frames"

echo "[pipeline] 1/4: Capturing live TableCam observation..."
./scripts/run_isaac.sh src/remediation/capture_live_tablecam.py --scene "$scene" --out "$perception"

echo "[pipeline] 2/4: Running YOLO perception and 3D localization..."
"$yolo_python" src/inspection/inspect_remediation_capture.py \
    --checkpoint "$checkpoint" \
    --image "$perception/current_scene_tablecam.png" \
    --depth "$perception/current_scene_tablecam_depth_m.npy" \
    --capture-report "$perception/capture_report.json" \
    --out "$perception/current_scene_detections.json" --device 0

echo "[pipeline] 3/4: Running physics-based remediation with the selected shared crooked-cap action..."
./scripts/run_isaac.sh src/remediation/run_compliant_breakable_remediation.py \
    --scene "$scene" --detections "$perception/current_scene_detections.json" --out "$action" \
    --depth "$perception/current_scene_tablecam_depth_m.npy" --capture-report "$perception/capture_report.json" \
    --controller-policy shared --checkpoint "$checkpoint" --confidence 0.70 --inference-device 0 \
    "${remediation_args[@]}"

for required in attempt_report.json trace.json policy_decisions.json oracle_evaluation_trace.json; do
    if [[ ! -s "$action/$required" ]]; then
        echo "[pipeline] remediation did not produce required audit artifact: $action/$required" >&2
        [[ -f "$action/failure_report.json" ]] && cat "$action/failure_report.json" >&2
        exit 1
    fi
done
"$yolo_python" src/remediation/verify_policy_parity_run.py "$action"

if ! find "$action/frames" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q .; then
    echo "[pipeline] compliant simulation produced no frames; refusing to run overlay" >&2
    if [[ -f "$action/failure_report.json" ]]; then
        echo "[pipeline] planning failure report:" >&2
        cat "$action/failure_report.json" >&2
    fi
    exit 1
fi

echo "[pipeline] 4/4: Overlaying per-frame YOLO detections..."
"$yolo_python" src/remediation/overlay_yolo_frames.py \
    --frames "$action/frames" --checkpoint "$checkpoint" --out "$action/yolo_frames" --device 0

# No system ffmpeg on the cluster nodes: fall back to imageio-ffmpeg's static build (has libx264).
FFMPEG_BIN=$(command -v ffmpeg || "$yolo_python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || echo /usr/bin/ffmpeg)
if [[ -x "$FFMPEG_BIN" ]]; then
    "$FFMPEG_BIN" -y -framerate 60 -i "$action/yolo_frames/%04d.png" -c:v libx264 -pix_fmt yuv420p "$out/tablecam_yolo_compliant_remediation.mp4"
    echo "[pipeline] Video compiled successfully at $out/tablecam_yolo_compliant_remediation.mp4"
else
    echo "[pipeline] ffmpeg unavailable; encoding MP4 with the HPC Python/OpenCV fallback..."
    "$yolo_python" scripts/tools/encode_frames.py \
        --frames "$action/yolo_frames" \
        --out "$out/tablecam_yolo_compliant_remediation.mp4" \
        --fps 60
fi

