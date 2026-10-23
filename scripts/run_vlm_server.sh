#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-${REPO_ROOT}/.venv_vllm}"
MODEL_ID="nvidia/Qwen3.8-27B-NVFP4"

# Previous Qwen checkpoint attempt, kept for reference:
# MODEL_ID="nvidia/Qwen3.8-Flash-Next-NVFP4"
# The GB10 restarted during vLLM model initialization before the API became ready.
#
# Previous model rollback: the original Nemotron vLLM environment is untouched.
# Use its original server command with:
# VLLM_ENV_DIR="${REPO_ROOT}/.conda_vllm"
# MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"

VLLM_BIN="${VLLM_ENV_DIR}/bin/vllm"
if [ ! -x "$VLLM_BIN" ]; then
    echo "vLLM is not installed in ${VLLM_ENV_DIR}." >&2
    exit 1
fi
export PATH="${VLLM_ENV_DIR}/bin:${PATH}"

exec "$VLLM_BIN" serve "$MODEL_ID" \
    --host 127.0.0.1 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
    --kv-cache-dtype fp8 \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-8}" \
    --disable-flashinfer-autotune \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 26}'
