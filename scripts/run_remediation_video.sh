#!/usr/bin/env bash
set -euo pipefail

# Render Fanuc Robotic Remediation Video via Isaac Sim
# Usage:
#   ./scripts/run_remediation_video.sh [--frames N]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec "${REPO_ROOT}/scripts/run_isaac.sh" "${REPO_ROOT}/src/remediation/simulate_remediation_video.py" "$@"
