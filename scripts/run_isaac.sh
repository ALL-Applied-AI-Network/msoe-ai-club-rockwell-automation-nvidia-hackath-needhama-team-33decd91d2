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
    # Seed the essential kernel/library paths explicitly before handing off to
    # NVIDIA's generated environment script.
    export LD_LIBRARY_PATH="$FOUND_ISAAC/kit:$FOUND_ISAAC/kit/kernel/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$FOUND_ISAAC/kit/kernel/py:$FOUND_ISAAC/python_packages:$FOUND_ISAAC/exts/isaacsim.simulation_app${PYTHONPATH:+:$PYTHONPATH}"
    # The shared bundle's Python executable currently carries a setgid bit.
    # Linux secure-execution mode then discards the library variables Isaac
    # needs. Run an ordinary-permission copy of the same 330-KiB executable,
    # while PYTHONHOME keeps its standard library in the shared bundle.
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        ISAAC_JOB_TMP="${SLURM_TMPDIR:-/tmp}"
        ISAAC_PYTHON_COPY="$ISAAC_JOB_TMP/isaac-python3-${SLURM_JOB_ID}-${SLURM_PROCID:-0}"
        install -m 0755 "$FOUND_ISAAC/kit/python/bin/python3" "$ISAAC_PYTHON_COPY"
        export PYTHONEXE="$ISAAC_PYTHON_COPY"
        export PYTHONHOME="$FOUND_ISAAC/kit/python"
        echo "[run_isaac] Using non-setgid job-local Python launcher: $ISAAC_PYTHON_COPY"
    fi
    exec "$FOUND_ISAAC/python.sh" "$@"
fi

echo "Error: Isaac Sim not found." >&2
echo "Please run ./setup.sh or set ISAAC=/path/to/isaacsim" >&2
exit 1
