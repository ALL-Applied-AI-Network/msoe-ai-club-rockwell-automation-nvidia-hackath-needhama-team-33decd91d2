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
#   ./run.sh scripts/render.py --target vial --state filled
#   ./run.sh scripts/run_all.sh
#   ./run.sh scripts/run_generate_fullframe_data.sh --scenes 30
#   ./run.sh scripts/run_train_detector.sh --epochs 35
#   ./run.sh scripts/run_inspection.sh
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Default to scripts/render.py if no arguments are provided
TARGET_SCRIPT="${1:-scripts/render.py}"
shift || true

# Handle help locally without submitting a batch job
if [ "$TARGET_SCRIPT" = "-h" ] || [ "$TARGET_SCRIPT" = "--help" ]; then
    echo "Usage: ./run.sh [script_or_command] [args...]"
    echo ""
    echo "Available scripts:"
    echo "  scripts/render.py                    - Standalone / twin rendering"
    echo "  scripts/run_all.sh                   - Full pipeline (golden + data gen + train + inspect)"
    echo "  scripts/run_generate_fullframe_data.sh - Generate synthetic full-frame YOLO dataset"
    echo "  scripts/run_train_detector.sh        - Train YOLOv11 detector on GPU"
    echo "  scripts/run_inspection.sh            - Run real-time detection & slot assignment"
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
