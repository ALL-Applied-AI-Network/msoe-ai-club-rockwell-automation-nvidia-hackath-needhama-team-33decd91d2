#!/usr/bin/env bash
set -euo pipefail

# Build Fanuc Tray Remediation Scene in Isaac Sim
# Usage:
#   ./scripts/run_build_remediation_scene.sh [--crooked N] [--missing N]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec "${REPO_ROOT}/scripts/run_isaac.sh" "${REPO_ROOT}/src/remediation/build_remediation_scene.py" "$@"
