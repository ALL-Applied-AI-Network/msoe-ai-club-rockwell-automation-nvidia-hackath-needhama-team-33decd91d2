#!/usr/bin/env bash
set -euo pipefail

# Build llama.cpp's CUDA-enabled headless server once on shared repository
# storage. The VLM launcher discovers the resulting binary automatically at
# llama.cpp/build/bin/llama-server.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${LLAMA_SOURCE_DIR:-${REPO_ROOT}/llama.cpp}"
BUILD_DIR="${LLAMA_BUILD_DIR:-${SOURCE_DIR}/build}"
REPO_URL="${LLAMA_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"

if [ ! -d "${SOURCE_DIR}/.git" ]; then
    if [ -e "$SOURCE_DIR" ]; then
        echo "Source path exists but is not a git checkout: ${SOURCE_DIR}" >&2
        exit 1
    fi
    echo "Cloning llama.cpp into ${SOURCE_DIR}"
    git clone --depth 1 "$REPO_URL" "$SOURCE_DIR"
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to build llama.cpp." >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is not on PATH. Load a CUDA module before building, for example:" >&2
    echo "  module load cuda/12.5" >&2
    exit 1
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF
cmake --build "$BUILD_DIR" --target llama-server -j "${LLAMA_BUILD_JOBS:-16}"

SERVER_BIN="${BUILD_DIR}/bin/llama-server"
if [ ! -x "$SERVER_BIN" ]; then
    echo "Build completed but llama-server was not found at ${SERVER_BIN}" >&2
    exit 1
fi
echo "Built ${SERVER_BIN}"
