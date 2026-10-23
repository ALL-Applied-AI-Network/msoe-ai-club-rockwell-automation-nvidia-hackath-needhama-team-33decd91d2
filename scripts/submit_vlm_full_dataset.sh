#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available in this shell. Run this command from the cluster login node." >&2
    exit 127
fi

mkdir -p logs
exec sbatch scripts/run_vlm_full_dataset.slurm "$@"
