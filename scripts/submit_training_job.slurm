#!/usr/bin/env bash
#SBATCH --job-name=train_yolo
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

# Preserve the project-local desktop environment when present; cluster jobs
# use the shared team venv that produced the current YOLO checkpoint.
if [ -x "${REPO_DIR}/.conda_env/bin/python" ]; then
    PYTHON_BIN="${REPO_DIR}/.conda_env/bin/python"
elif [ -x "${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python" ]; then
    PYTHON_BIN="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
else
    echo "Error: no YOLO Python environment found (.conda_env or shared HPC venv)" >&2
    exit 1
fi

echo "[Slurm] Training with: $PYTHON_BIN $@"
exec "$PYTHON_BIN" "$@"
