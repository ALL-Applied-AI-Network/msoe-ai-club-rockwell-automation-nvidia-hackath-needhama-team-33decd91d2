#!/usr/bin/env bash
set -euo pipefail

# Start one headless llama.cpp multimodal server. The Slurm launcher starts
# this once per GPU, assigning each replica a different port.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
if [ -z "$LLAMA_SERVER_BIN" ]; then
    LLAMA_SERVER_BIN="$(command -v llama-server 2>/dev/null || true)"
fi
if [ -z "$LLAMA_SERVER_BIN" ] && [ -x "${REPO_ROOT}/llama.cpp/build/bin/llama-server" ]; then
    LLAMA_SERVER_BIN="${REPO_ROOT}/llama.cpp/build/bin/llama-server"
fi
if [ -z "$LLAMA_SERVER_BIN" ]; then
    echo "llama-server was not found. Set LLAMA_SERVER_BIN to its executable." >&2
    exit 1
fi

MODEL_GGUF="${LLAMA_MODEL_GGUF:-${REPO_ROOT}/llama.cpp/models/Qwen3.8-27B-UD-IQ3_S.gguf}"
MMPROJ_GGUF="${LLAMA_MMPROJ_GGUF:-${REPO_ROOT}/llama.cpp/models/mmproj-F16.gguf}"
if [ ! -f "$MODEL_GGUF" ]; then
    echo "Model GGUF not found: $MODEL_GGUF" >&2
    echo "Run scripts/download_qwen_vlm.sh or set LLAMA_MODEL_GGUF." >&2
    exit 1
fi
if [ ! -f "$MMPROJ_GGUF" ]; then
    echo "Multimodal projector GGUF not found: $MMPROJ_GGUF" >&2
    echo "Run scripts/download_qwen_vlm.sh or set LLAMA_MMPROJ_GGUF." >&2
    exit 1
fi

PORT="${LLAMA_PORT:-8000}"
MODEL_NAME="${LLAMA_MODEL_NAME:-qwen3.8-27b-ud-iq3-s}"

exec "$LLAMA_SERVER_BIN" \
    --model "$MODEL_GGUF" \
    --mmproj "$MMPROJ_GGUF" \
    --alias "$MODEL_NAME" \
    --host "${LLAMA_HOST:-127.0.0.1}" \
    --port "$PORT" \
    --no-webui \
    --jinja \
    --split-mode none \
    --n-gpu-layers all \
    --ctx-size "${LLAMA_CTX_SIZE:-16384}" \
    --parallel 1 \
    --batch-size "${LLAMA_BATCH_SIZE:-512}" \
    --ubatch-size "${LLAMA_UBATCH_SIZE:-128}" \
    --threads-http "${LLAMA_HTTP_THREADS:-4}" \
    --cont-batching
