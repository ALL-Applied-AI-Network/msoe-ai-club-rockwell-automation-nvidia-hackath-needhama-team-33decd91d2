#!/usr/bin/env bash
#SBATCH --job-name=innovation_lab
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs runs

echo "=========================================================="
echo "Running on $(hostname) with GPU allocation:"
nvidia-smi
echo "=========================================================="

TARGET="${1:-scripts/render.py}"
shift || true

if [ -x "$TARGET" ] && [[ "$TARGET" == *.sh ]]; then
    echo "[Slurm] Executing shell script: $TARGET $@"
    exec "$TARGET" "$@"
elif [[ "$TARGET" == *.py ]]; then
    echo "[Slurm] Executing Python with Isaac Sim runner: $TARGET $@"
    exec ./scripts/run_isaac.sh "$TARGET" "$@"
else
    echo "[Slurm] Executing: $TARGET $@"
    exec "$TARGET" "$@"
fi
