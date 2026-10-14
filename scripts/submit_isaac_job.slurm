#!/usr/bin/env bash
#SBATCH --job-name=isaac_sim
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

echo "Running on $(hostname) with GPU allocation:"
nvidia-smi

# Execute command passed to sbatch, or default to scripts/render.py
./scripts/run_isaac.sh "${@:-scripts/render.py}"
