#!/usr/bin/env bash
set -euo pipefail

# Start four one-GPU llama-server replicas on the current Slurm node.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_SERVER_COUNT="${LLAMA_SERVERS_PER_NODE:-4}"
NODE_RANK="${SLURM_PROCID:-0}"
BASE_PORT="${LLAMA_BASE_PORT:-8000}"
LOG_DIR="${LLAMA_SERVER_LOG_DIR:-${REPO_ROOT}/runs/vlm_labels/llama_servers_manual}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [ "${#ALLOCATED_GPUS[@]}" -lt "$LOCAL_SERVER_COUNT" ]; then
    echo "Expected ${LOCAL_SERVER_COUNT} GPUs on ${SLURM_NODE_NAME:-this node}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi

SERVER_PIDS=()
cleanup() {
    for pid in "${SERVER_PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for ((local_index = 0; local_index < LOCAL_SERVER_COUNT; local_index++)); do
    port=$((BASE_PORT + NODE_RANK * LOCAL_SERVER_COUNT + local_index))
    gpu="${ALLOCATED_GPUS[$local_index]}"
    log_file="${LOG_DIR}/${SLURM_NODE_NAME:-node${NODE_RANK}}_server_${local_index}_gpu_${gpu}.log"
    echo "Starting llama-server on ${SLURM_NODE_NAME:-node${NODE_RANK}} GPU ${gpu}, port ${port} (log: ${log_file})"
    CUDA_VISIBLE_DEVICES="$gpu" \
        LLAMA_HOST="0.0.0.0" \
        LLAMA_PORT="$port" \
        ./scripts/run_llama_vlm_server.sh >"$log_file" 2>&1 &
    SERVER_PIDS+=("$!")
done

# Keep the srun step alive while all four server processes are serving.
while true; do
    for pid in "${SERVER_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "A llama-server exited on ${SLURM_NODE_NAME:-node${NODE_RANK}}." >&2
            exit 1
        fi
    done
    sleep 10
done
