#!/usr/bin/env bash
set -euo pipefail

# Run Isaac Sim Synthetic Twin Data Generation
# Usage:
#   ./scripts/run_data_generation.sh [--scenes N]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec "${REPO_ROOT}/scripts/run_isaac.sh" "${REPO_ROOT}/src/data_generation/generate_synthetic_crops.py" "$@"
