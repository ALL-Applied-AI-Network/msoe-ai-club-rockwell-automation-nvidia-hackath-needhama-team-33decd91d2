#!/usr/bin/env bash
set -euo pipefail

# Run Tier 1 Edge Fixture Inspection (YOLOv11 + Golden Delta Ensemble)
# Usage:
#   ./scripts/run_inspection.sh [optional extra args...]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.venv_vlm/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "${REPO_ROOT}/src/inspection/aperture_inspector.py" "$@"
