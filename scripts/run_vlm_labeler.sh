#!/usr/bin/env bash
set -euo pipefail

# Run Offline In-Context VLM Auto-Labeler on Real Factory Images
# Usage:
#   ./scripts/run_vlm_labeler.sh [--max-images N]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.venv_vlm/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "${REPO_ROOT}/src/inspection/vlm_auto_labeler.py" "$@"
