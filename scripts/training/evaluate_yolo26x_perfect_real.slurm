#!/usr/bin/env bash
#SBATCH --job-name=eval-yolo26x-perfect
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/eval-%j.out
#SBATCH --error=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/eval-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_PYTHON="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
ROOT="${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune"
cd "${REPO_DIR}"

srun --export=ALL --ntasks=1 "${ENV_PYTHON}" \
    scripts/training/evaluate_yolo26x_perfect_real.py \
    --models \
      "baseline=${REPO_DIR}/shared/yolo26x/best.pt" \
      "corrected_box_finetune=${ROOT}/yolo26x_vlm_box_finetune_1280/weights/best.pt" \
    --synthetic-data "${ROOT}/mixed_replay_dataset/data.yaml" \
    --real-root "${ROOT}/refined_real" \
    --output "${ROOT}/evaluation" \
    --imgsz 1280 \
    --confidence 0.40 \
    --synthetic-samples 4

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
echo "[complete] ${ROOT}/evaluation/yolo26x_perfect_real_finetune_report.png"
