#!/usr/bin/env bash
set -euo pipefail

# Submit the legacy Fanuc remediation video through Slurm.
# It remains a visualization only; it is not the closed-loop qualification path.
# Usage:
#   ./scripts/run_remediation_video.sh [--frames N]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec sbatch "${REPO_ROOT}/scripts/submit_isaac_job.slurm" \
    "${REPO_ROOT}/src/remediation/simulate_remediation_video.py" "$@"
