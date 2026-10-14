#!/usr/bin/env bash
set -euo pipefail

# Universal render runner for Innovation Lab Fall 2026.
# Automatically detects environment:
#   - On Rosie login node (Slurm available): Submits via sbatch to GPU compute nodes.
#   - On Rosie compute node, GB10, or Brev (GPU directly available): Runs directly via run_isaac.sh.
#
# Usage:
#   ./render.sh [options...]
#
# Examples:
#   ./render.sh --target vial --state filled
#   ./render.sh --target vial --state sequence --start-frame 100 --end-frame 500 --step 20
#   ./render.sh --target fanuc --camera side
#   ./render.sh --target build_fanuc

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Handle help locally without submitting a batch job
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    python3 scripts/render.py --help
    exit 0
fi

# Check if we are on a Slurm head/login node (sbatch exists AND no active SLURM_JOB_ID)
if command -v sbatch >/dev/null 2>&1 && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[render.sh] Slurm cluster detected (Rosie). Submitting background batch job..."
    mkdir -p logs runs
    exec sbatch scripts/submit_isaac_job.slurm scripts/render.py "$@"
else
    # Direct execution (GB10, Brev, or inside a Slurm allocation)
    echo "[render.sh] Running directly with Isaac Sim..."
    exec ./scripts/run_isaac.sh scripts/render.py "$@"
fi
