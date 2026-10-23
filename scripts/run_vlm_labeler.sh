#!/usr/bin/env bash
set -euo pipefail

# Run Offline In-Context VLM Auto-Labeler on Real Factory Images
# Usage:
#   ./scripts/run_vlm_labeler.sh [--max-images N]
# Requires the VLM server in another terminal:
#   ./scripts/run_vlm_server.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.conda_vllm/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Local Conda environment missing. Run the project setup that installs vLLM in .conda_vllm first." >&2
    exit 1
fi

exec "$PYTHON_BIN" "${REPO_ROOT}/src/inspection/vlm_auto_labeler.py" "$@"
