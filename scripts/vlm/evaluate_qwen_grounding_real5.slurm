#!/usr/bin/env bash
#SBATCH --job-name=qwen-ground-real5
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:t4:4
#SBATCH --time=02:00:00
#SBATCH --output=runs/locateanything_3b/logs/%x_%j.out
#SBATCH --error=runs/locateanything_3b/logs/%x_%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR}"
RUN_ROOT="${REPO_DIR}/runs/locateanything_3b"
OUTPUT="${RUN_ROOT}/qwen3_8_27b_iq3_real5_concise"
SERVER_LOG_ROOT="${RUN_ROOT}/logs/qwen_servers_${SLURM_JOB_ID}"
LLAMA_ROOT="${LLAMA_CPP_ROOT:-${REPO_DIR}/shared/cache/llama.cpp}"
SERVER_BIN="${LLAMA_ROOT}/build/bin/llama-server"
MODEL="${LLAMA_ROOT}/models/Qwen3.8-27B-UD-IQ3_S.gguf"
MMPROJ="${LLAMA_ROOT}/models/mmproj-F16.gguf"
PYTHON_BIN="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
DATASET="${REPO_DIR}/runs/yolo26X_runs/03_mixed_replay_1280/evaluation/real_dataset"
BASE_PORT=8127

mkdir -p "${RUN_ROOT}/logs" "${SERVER_LOG_ROOT}" "${OUTPUT}"
for path in "${SERVER_BIN}" "${MODEL}" "${MMPROJ}" "${PYTHON_BIN}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required file is missing: ${path}" >&2
        exit 1
    fi
done

cleanup() {
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPUS[@]}" -lt 4 ]]; then
    echo "Expected four allocated GPUs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi

SERVER_PIDS=()
SERVER_URLS=()
for index in 0 1 2 3; do
    port=$((BASE_PORT + index))
    SERVER_URLS+=("http://127.0.0.1:${port}/v1")
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${SERVER_BIN}" \
        --model "${MODEL}" \
        --mmproj "${MMPROJ}" \
        --alias qwen3.8-27b-ud-iq3-s \
        --host 127.0.0.1 \
        --port "${port}" \
        --no-webui \
        --jinja \
        --split-mode none \
        --n-gpu-layers all \
        --ctx-size 16384 \
        --parallel 1 \
        --batch-size 512 \
        --ubatch-size 128 \
        --image-min-tokens 1024 \
        --threads-http 4 \
        --cont-batching >"${SERVER_LOG_ROOT}/server_${index}.log" 2>&1 &
    SERVER_PIDS+=("$!")
done

READY=0
for _ in $(seq 1 180); do
    READY=1
    for url in "${SERVER_URLS[@]}"; do
        if ! curl --silent --fail "${url}/models" >/dev/null 2>&1; then
            READY=0
            break
        fi
    done
    [[ "${READY}" -eq 1 ]] && break
    for pid in "${SERVER_PIDS[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "A llama-server exited during startup" >&2
            tail -n 100 "${SERVER_LOG_ROOT}"/*.log >&2 || true
            exit 1
        fi
    done
    sleep 2
done
if [[ "${READY}" -ne 1 ]]; then
    echo "llama-server did not become ready" >&2
    tail -n 100 "${SERVER_LOG_ROOT}"/*.log >&2 || true
    exit 1
fi

echo "Qwen llama-server is ready; starting real-five grounding evaluation."
SERVER_URLS_CSV="$(IFS=,; echo "${SERVER_URLS[*]}")"
srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
    "${REPO_DIR}/scripts/vlm/evaluate_qwen_grounding_real5.py" \
    --servers "${SERVER_URLS_CSV}" \
    --workers 40 \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --views raw,clahe \
    --max-tokens 2048

chmod -R g+rwX "${RUN_ROOT}"
