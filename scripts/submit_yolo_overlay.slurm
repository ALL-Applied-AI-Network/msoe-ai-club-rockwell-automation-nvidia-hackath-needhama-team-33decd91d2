#!/usr/bin/env bash
#SBATCH --job-name=remediation_yolo_overlay
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec ./.conda_env/bin/python src/remediation/overlay_yolo_frames.py "$@"
