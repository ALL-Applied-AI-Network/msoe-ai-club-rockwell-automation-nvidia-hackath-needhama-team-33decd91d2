#!/usr/bin/env bash
#SBATCH --job-name=compliant_remediation
#SBATCH --partition=teaching
#SBATCH --exclude=dh-node11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Isaac/YOLO use GPU 0 while the isolated wrist LocateAnything worker uses
# GPU 1.  Request both explicitly so --wrist-refinement works without an
# sbatch command-line override and both devices are guaranteed Tesla T4s.
#SBATCH --gres=gpu:t4:2
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
remediation_args=()
perception_plane_args=()
wrist_refinement=false
skip_overlay=false
raw_args=("$@")
for ((i=0; i<${#raw_args[@]}; i++)); do
    case "${raw_args[$i]}" in
        --skip-overlay)
            skip_overlay=true
            continue
            ;;
        --wrist-refinement)
            wrist_refinement=true
            ;;
        --work-surface-z-m|--vial-rim-z-m)
            if (( i + 1 >= ${#raw_args[@]} )); then
                echo "Missing value for ${raw_args[$i]}" >&2
                exit 2
            fi
            perception_plane_args+=("${raw_args[$i]}" "${raw_args[$((i+1))]}")
            remediation_args+=("${raw_args[$i]}" "${raw_args[$((i+1))]}")
            ((i+=1))
            continue
            ;;
        --work-surface-z-m=*|--vial-rim-z-m=*)
            perception_plane_args+=("${raw_args[$i]}")
            ;;
    esac
    remediation_args+=("${raw_args[$i]}")
done

if [[ "$wrist_refinement" == true ]]; then
    visible_gpu_count=$(nvidia-smi -L | wc -l)
    if (( visible_gpu_count < 2 )); then
        echo "Wrist-VLM Isaac runs isolate Isaac and LocateAnything on two T4s." >&2
        echo "Submit with: sbatch --gres=gpu:t4:2 $0 ... --wrist-refinement" >&2
        exit 2
    fi
fi

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
rm -rf "$action/frames" "$action/side_frames" "$action/wrist_frames" "$action/yolo_frames"

echo "[pipeline] 1/4: Capturing live TableCam observation..."
./scripts/run_isaac.sh src/remediation/capture_live_tablecam.py --scene "$scene" --out "$perception"

echo "[pipeline] 2/4: Running YOLO perception and 3D localization..."
"$yolo_python" src/inspection/inspect_remediation_capture.py \
    --checkpoint "$checkpoint" \
    --image "$perception/current_scene_tablecam.png" \
    --depth "$perception/current_scene_tablecam_depth_m.npy" \
    --capture-report "$perception/capture_report.json" \
    --out "$perception/current_scene_detections.json" --device 0 \
    "${perception_plane_args[@]}"

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

if ! find "$action/side_frames" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q .; then
    echo "[pipeline] compliant simulation produced no synchronized SideCam frames" >&2
    exit 1
fi
if [[ "$wrist_refinement" == true ]] && \
   ! find "$action/wrist_frames" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q .; then
    echo "[pipeline] compliant simulation produced no synchronized WristCam frames" >&2
    exit 1
fi

# Encode the diagnostic side view before the slower per-frame YOLO overlay so
# close gripper/cap contact can be reviewed as soon as remediation completes.
FFMPEG_BIN=$(command -v ffmpeg || "$yolo_python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || echo /usr/bin/ffmpeg)
if [[ -x "$FFMPEG_BIN" ]]; then
    "$FFMPEG_BIN" -y -framerate 60 -i "$action/side_frames/%04d.png" -c:v libx264 -pix_fmt yuv420p "$out/sidecam_raw_remediation.mp4"
else
    "$yolo_python" scripts/tools/encode_frames.py \
        --frames "$action/side_frames" \
        --out "$out/sidecam_raw_remediation.mp4" \
        --fps 60
fi
echo "[pipeline] SideCam video compiled at $out/sidecam_raw_remediation.mp4"

encode_raw_view() {
    local frames="$1"
    local video="$2"
    if [[ -x "$FFMPEG_BIN" ]]; then
        "$FFMPEG_BIN" -y -framerate 60 -i "$frames/%04d.png" \
            -c:v libx264 -pix_fmt yuv420p "$video"
    else
        "$yolo_python" scripts/tools/encode_frames.py \
            --frames "$frames" --out "$video" --fps 60
    fi
}

encode_raw_view "$action/frames" "$out/tablecam_raw_remediation.mp4"
echo "[pipeline] raw TableCam video compiled at $out/tablecam_raw_remediation.mp4"
if [[ "$wrist_refinement" == true ]]; then
    encode_raw_view "$action/wrist_frames" "$out/wristcam_raw_remediation.mp4"
    echo "[pipeline] raw WristCam video compiled at $out/wristcam_raw_remediation.mp4"
fi

if [[ "$skip_overlay" == true ]]; then
    echo "[pipeline] Temporary run requested; skipping the slow per-frame YOLO overlay."
    exit 0
fi

echo "[pipeline] 4/4: Overlaying per-frame YOLO detections..."
streaming_marker="$action/yolo_frames/.streaming"
streaming_done="$action/yolo_frames/.complete"
if [[ -e "$streaming_marker" && ! -e "$streaming_done" ]]; then
    echo "[pipeline] Waiting for the concurrent YOLO overlay worker..."
    for _ in $(seq 1 900); do
        [[ -e "$streaming_done" ]] && break
        sleep 2
    done
    if [[ ! -e "$streaming_done" ]]; then
        echo "[pipeline] concurrent YOLO overlay did not finish within 30 minutes" >&2
        exit 1
    fi
fi
"$yolo_python" src/remediation/overlay_yolo_frames.py \
    --frames "$action/frames" --checkpoint "$checkpoint" --out "$action/yolo_frames" --device 0

# No system ffmpeg on the cluster nodes: fall back to imageio-ffmpeg's static build (has libx264).
if [[ -x "$FFMPEG_BIN" ]]; then
    "$FFMPEG_BIN" -y -framerate 60 -i "$action/yolo_frames/%04d.png" -c:v libx264 -pix_fmt yuv420p "$out/tablecam_annotated_remediation.mp4"
    echo "[pipeline] Video compiled successfully at $out/tablecam_annotated_remediation.mp4"
else
    echo "[pipeline] ffmpeg unavailable; encoding MP4 with the HPC Python/OpenCV fallback..."
    "$yolo_python" scripts/tools/encode_frames.py \
        --frames "$action/yolo_frames" \
        --out "$out/tablecam_annotated_remediation.mp4" \
        --fps 60
fi

