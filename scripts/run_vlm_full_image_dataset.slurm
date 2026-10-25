#!/usr/bin/env bash
#SBATCH --job-name=vlm-full-image
#SBATCH --partition=teaching
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/vlm-full-image-%j.out
#SBATCH --error=logs/vlm-full-image-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/vlm_labels

# Build into the shared repository workspace rather than installing anything
# on the compute nodes. The resulting binary is visible to both allocated nodes.
LLAMA_BINARY_PATH="${LLAMA_SERVER_BIN:-${REPO_DIR}/llama.cpp/build/bin/llama-server}"
if [ ! -x "$LLAMA_BINARY_PATH" ]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        if ! type module >/dev/null 2>&1 && [ -f /etc/profile.d/lmod.sh ]; then
            # shellcheck disable=SC1091
            source /etc/profile.d/lmod.sh
        fi
        module load cuda/12.5
    fi
    ./scripts/build_llama_cpp.sh
fi
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${REPO_DIR}/llama.cpp/build/bin/llama-server}"

if [ -n "${LLAMA_MODEL_GGUF:-}" ] || [ -n "${LLAMA_MMPROJ_GGUF:-}" ]; then
    if [ ! -f "${LLAMA_MODEL_GGUF:-}" ] || [ ! -f "${LLAMA_MMPROJ_GGUF:-}" ]; then
        echo "Both LLAMA_MODEL_GGUF and LLAMA_MMPROJ_GGUF must point to existing files." >&2
        exit 1
    fi
else
    ./scripts/download_qwen_vlm.sh
    export LLAMA_MODEL_GGUF="${REPO_DIR}/llama.cpp/models/Qwen3.8-27B-UD-IQ3_S.gguf"
    export LLAMA_MMPROJ_GGUF="${REPO_DIR}/llama.cpp/models/mmproj-F16.gguf"
fi

SERVER_COUNT="${LLAMA_SERVER_COUNT:-8}"
SERVERS_PER_NODE="${LLAMA_SERVERS_PER_NODE:-4}"
BASE_PORT="${LLAMA_BASE_PORT:-8000}"
SERVER_LOG_DIR="${REPO_DIR}/runs/vlm_labels/llama_servers_${SLURM_JOB_ID:-manual}"
mkdir -p "$SERVER_LOG_DIR"

if [ "$SERVER_COUNT" -ne 8 ] || [ "$SERVERS_PER_NODE" -ne 4 ]; then
    echo "Warning: expected 8 total servers as 4 per node; got ${SERVER_COUNT} total and ${SERVERS_PER_NODE} per node." >&2
fi

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#ALLOCATED_NODES[@]}" -lt 2 ]; then
    echo "Expected two allocated nodes; got ${#ALLOCATED_NODES[@]}." >&2
    exit 1
fi

SERVER_URLS=()
cleanup() {
    if [ -n "${SERVER_LAUNCH_PID:-}" ] && kill -0 "$SERVER_LAUNCH_PID" 2>/dev/null; then
        kill "$SERVER_LAUNCH_PID" 2>/dev/null || true
        wait "$SERVER_LAUNCH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for ((node_index = 0; node_index < 2; node_index++)); do
    node="${ALLOCATED_NODES[$node_index]}"
    for ((local_index = 0; local_index < SERVERS_PER_NODE; local_index++)); do
        port=$((BASE_PORT + node_index * SERVERS_PER_NODE + local_index))
        SERVER_URLS+=("http://${node}:${port}/v1")
    done
done

# Start four one-GPU servers on each of the two allocated nodes.
LLAMA_SERVERS_PER_NODE="$SERVERS_PER_NODE" \
    LLAMA_BASE_PORT="$BASE_PORT" \
    LLAMA_SERVER_LOG_DIR="$SERVER_LOG_DIR" \
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 \
        ./scripts/run_llama_vlm_servers_on_node.sh >"${SERVER_LOG_DIR}/server-launch.log" 2>&1 &
SERVER_LAUNCH_PID="$!"

SERVER_URLS_CSV="$(IFS=,; echo "${SERVER_URLS[*]}")"
PYTHON_BIN="${VLM_PYTHON_BIN:-python3}"
READY=0
echo "Waiting for ${SERVER_COUNT} llama-server replicas..."
for ((attempt = 1; attempt <= 360; attempt++)); do
    READY=1
    for url in "${SERVER_URLS[@]}"; do
        if ! "$PYTHON_BIN" - "$url" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen
urlopen(sys.argv[1] + "/models", timeout=3).read()
PY
        then
            READY=0
            break
        fi
    done
    if [ "$READY" -eq 1 ]; then
        break
    fi
    if ! kill -0 "$SERVER_LAUNCH_PID" 2>/dev/null; then
        echo "The llama-server launch step exited during startup. Recent logs:" >&2
        tail -n 80 "${SERVER_LOG_DIR}"/*.log >&2 || true
        exit 1
    fi
    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "Not all llama-server replicas became ready within 60 minutes." >&2
    tail -n 100 "${SERVER_LOG_DIR}"/*.log >&2 || true
    exit 1
fi

echo "All llama-server replicas are ready; launching full-image labeling."
export VLM_SERVER_URLS="$SERVER_URLS_CSV"
LABELER_ARGS=()
if [ -n "${VLM_LABELER_ARGS:-}" ]; then
    read -r -a LABELER_ARGS <<< "$VLM_LABELER_ARGS"
fi
LABELER_COMMAND=(
    ./scripts/run_vlm_full_image_labeler.sh
    --images "${VLM_IMAGE_DIR:-${REPO_DIR}/images/pruned_images}" \
    --examples-dir "${VLM_EXAMPLES_DIR:-${REPO_DIR}/images/VLM_examples}" \
    --model "${VLM_MODEL:-qwen3.8-27b-ud-iq3-s}" \
    --workers "$SERVER_COUNT" \
    --resume
)
if [ -n "${VLM_PER_IMAGE_DIR:-}" ]; then
    LABELER_COMMAND+=(--per-image-dir "$VLM_PER_IMAGE_DIR")
else
    LABELER_COMMAND+=(--out "${VLM_OUTPUT_JSON:-${REPO_DIR}/images/VLM_samples/full_image_labels.json}")
fi
if [ -n "${VLM_SKIP_LIST:-}" ]; then
    LABELER_COMMAND+=(--skip-list "$VLM_SKIP_LIST")
fi
"${LABELER_COMMAND[@]}" "${LABELER_ARGS[@]}"
