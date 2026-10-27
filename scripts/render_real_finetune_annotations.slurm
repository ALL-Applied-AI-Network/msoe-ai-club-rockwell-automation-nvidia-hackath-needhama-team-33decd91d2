#!/usr/bin/env bash
# Render provisional real-image annotations for operator inspection.
#SBATCH --job-name=render-real-labels
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/render-real-labels-%j.out
#SBATCH --error=logs/render-real-labels-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs

PKG_DIR="/tmp/render-real-labels-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'from PIL import Image' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir pillow
    export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

srun --export=ALL --ntasks=1 python3 scripts/build_real_finetune_annotations.py
