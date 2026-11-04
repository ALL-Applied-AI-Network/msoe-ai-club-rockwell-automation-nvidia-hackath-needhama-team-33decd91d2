#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Universal Runner for Innovation Lab Fall 2026.
# Replaces render.sh.
#
# Automatically detects environment:
#   - On Rosie login/management node (Slurm sbatch available and no active job):
#       Submits via sbatch to a GPU compute node.
#   - On Rosie compute node, Brev, or local workstation (GPU directly available):
#       Runs directly via run_isaac.sh or bash.
#
# Usage:
#   ./run.sh [script_or_command] [args...]
#
# Examples:
#   ./run.sh src/remediation/capture_live_tablecam.py --scene <scene.usda> --out <dir>
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
    echo "Usage: ./run.sh [script_or_command] [args...]" >&2
    exit 1
fi
TARGET_SCRIPT="$1"
shift

# Handle help locally without submitting a batch job
if [ "$TARGET_SCRIPT" = "-h" ] || [ "$TARGET_SCRIPT" = "--help" ]; then
    echo "Usage: ./run.sh [script_or_command] [args...]"
    echo ""
    echo "For the main pipeline, use: sbatch scripts/run_compliant_breakable_pipeline.slurm (see README.md)"
    exit 0
fi

# Detect Slurm login/management node on Rosie:
# (sbatch command exists AND we are not already inside an active SLURM job allocation)
if command -v sbatch >/dev/null 2>&1 && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[run.sh] Rosie login/mgmt node detected. Submitting via sbatch to GPU compute node..."
    mkdir -p logs runs
    exec sbatch scripts/submit_isaac_job.slurm "$TARGET_SCRIPT" "$@"
else
    echo "[run.sh] Running directly in local/compute environment..."
    mkdir -p logs runs

    # If target is an executable shell script, execute it directly
    if [ -x "$TARGET_SCRIPT" ] && [[ "$TARGET_SCRIPT" == *.sh ]]; then
        exec "$TARGET_SCRIPT" "$@"
    # If target is a Python script, invoke via Isaac runner
    elif [[ "$TARGET_SCRIPT" == *.py ]]; then
        exec ./scripts/run_isaac.sh "$TARGET_SCRIPT" "$@"
    else
        # Fallback to direct execution
        exec "$TARGET_SCRIPT" "$@"
    fi
fi
