#!/usr/bin/env bash
set -euo pipefail

# Render clean, nominal Golden Reference image through CapSealCam
# Usage:
#   ./scripts/render_golden.sh [--out /path/to/custom_golden.png]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec "${REPO_ROOT}/scripts/run_isaac.sh" "${REPO_ROOT}/src/data_generation/render_golden_reference.py" "$@"
