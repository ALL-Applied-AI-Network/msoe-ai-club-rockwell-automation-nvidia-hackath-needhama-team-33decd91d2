#!/usr/bin/env bash
set -euo pipefail

# Generate Full-Frame YOLO Detection Dataset from Isaac Sim
# Usage:
#   ./scripts/run_generate_fullframe_data.sh [--scenes N]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec "${REPO_ROOT}/scripts/run_isaac.sh" "${REPO_ROOT}/src/data_generation/generate_fullframe_dataset.py" "$@"
