#!/usr/bin/env bash
set -euo pipefail

# Train YOLOv11 Crop Classifier on Synthetic Digital Twin Dataset
# Usage:
#   ./scripts/run_train_classifier.sh [--epochs 25] [--batch 16]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.venv_vlm/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "${REPO_ROOT}/src/inspection/train_crop_classifier.py" "$@"
