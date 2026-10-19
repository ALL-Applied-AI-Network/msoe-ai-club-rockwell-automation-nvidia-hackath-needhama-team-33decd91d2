#!/usr/bin/env bash
set -euo pipefail

# Train Full-Frame YOLOv11 Detector
# Usage:
#   ./scripts/run_train_detector.sh [--epochs N] [--batch B]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.venv_vlm/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "${REPO_ROOT}/src/inspection/train_fullframe_detector.py" "$@"
