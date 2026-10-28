#!/usr/bin/env bash
#SBATCH --job-name=hybrid-blends
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=shared/runs/2026-10-27_yolo26x_vlm_deployment/blends-%j.out
#SBATCH --error=shared/runs/2026-10-27_yolo26x_vlm_deployment/blends-%j.err

set -euo pipefail
umask 002
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
srun --export=ALL --ntasks=1 shared/cache/locateanything_3b/venv/bin/python \
    scripts/training/evaluate_refinement_blends.py \
    --predictions shared/runs/2026-10-27_hybrid_vlm_yolo_synthetic_eval/predictions.json \
    --output shared/runs/2026-10-27_yolo26x_vlm_deployment/blend_sweep.json \
    --confidence 0.40
