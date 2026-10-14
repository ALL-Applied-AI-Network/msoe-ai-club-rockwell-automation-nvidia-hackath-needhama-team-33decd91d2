#!/usr/bin/env bash
set -euo pipefail

# Unified runner for Isaac Sim (Rosie HPC cluster, Brev, local workstation)
# Usage:
#   ./scripts/run_isaac.sh <script.py> [args...]
#
# Examples:
#   ./scripts/run_isaac.sh shared/data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --out runs/fill.png

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

# Candidate install paths for Isaac Sim standalone python.sh
CANDIDATE_PATHS=(
    "${ISAAC:-}"
    "${REPO_ROOT}/shared/isaacsim_6.1"
    "${HOME}/isaacsim"
    "/isaac-sim"
)

FOUND_ISAAC=""
for p in "${CANDIDATE_PATHS[@]}"; do
    if [ -n "$p" ] && [ -x "$p/python.sh" ]; then
        FOUND_ISAAC="$p"
        break
    fi
done

if [ -n "$FOUND_ISAAC" ]; then
    echo "[run_isaac] Detected Isaac Sim at: $FOUND_ISAAC"
    export ISAAC="$FOUND_ISAAC"
    exec "$FOUND_ISAAC/python.sh" "$@"
fi

echo "Error: Isaac Sim not found." >&2
echo "Please run ./setup.sh or set ISAAC=/path/to/isaacsim" >&2
exit 1
