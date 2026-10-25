#!/usr/bin/env bash
set -euo pipefail

# Stage the exact GGUF and multimodal projector required by llama-server.
# Files are downloaded once to shared repository storage and reused by all
# eight server replicas.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${LLAMA_MODEL_DIR:-${REPO_ROOT}/llama.cpp/models}"
MODEL_FILE="${LLAMA_MODEL_FILENAME:-Qwen3.8-27B-UD-IQ3_S.gguf}"
MMPROJ_FILE="${LLAMA_MMPROJ_FILENAME:-mmproj-F16.gguf}"
MODEL_REPO="${LLAMA_HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
HF_BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/main"

mkdir -p "$MODEL_DIR"

download() {
    local url="$1"
    local destination="$2"
    if [ -s "$destination" ]; then
        echo "Using existing $(basename "$destination")"
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to download the Qwen GGUF files." >&2
        exit 1
    fi
    echo "Downloading $(basename "$destination")"
    curl --fail --location --retry 5 --retry-all-errors --continue-at - \
        --output "$destination" "${url}?download=true"
}

download "${HF_BASE_URL}/${MODEL_FILE}" "${MODEL_DIR}/${MODEL_FILE}"
download "${HF_BASE_URL}/${MMPROJ_FILE}" "${MODEL_DIR}/${MMPROJ_FILE}"

echo "Model: ${MODEL_DIR}/${MODEL_FILE}"
echo "Projector: ${MODEL_DIR}/${MMPROJ_FILE}"
