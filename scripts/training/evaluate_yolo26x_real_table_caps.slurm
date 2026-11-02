#!/usr/bin/env bash
#SBATCH --job-name=eval-yolo26x-table
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=shared/runs/2026-10-28_yolo26x_real_table_finetune/eval-%j.out
#SBATCH --error=shared/runs/2026-10-28_yolo26x_real_table_finetune/eval-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
ROOT="${REPO_DIR}/shared/runs/2026-10-28_yolo26x_real_table_finetune"
PRIOR="${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune"
OLD_MODEL="${PRIOR}/yolo26x_vlm_box_finetune_1280/weights/best.pt"
NEW_MODEL="${ROOT}/yolo26x_real_table_finetune_1280/weights/best.pt"
cd "${REPO_DIR}"

srun --export=ALL --ntasks=1 "${PYTHON}" \
    scripts/training/evaluate_yolo26x_perfect_real.py \
    --models \
      "corrected_box_finetune=${OLD_MODEL}" \
      "table_cap_finetune=${NEW_MODEL}" \
    --synthetic-data "${ROOT}/dataset_build/mixed_replay_dataset/data.yaml" \
    --real-root "${PRIOR}/refined_real" \
    --output "${ROOT}/evaluation" \
    --imgsz 1280 \
    --confidence 0.60 \
    --synthetic-samples 4

srun --export=ALL --ntasks=1 "${PYTHON}" \
    scripts/training/test_table_cap_samples.py \
    --input sample_tables \
    --models \
      "corrected_box_finetune=${OLD_MODEL}" \
      "table_cap_finetune=${NEW_MODEL}" \
    --output "${ROOT}/table_photo_evaluation" \
    --imgsz 1280 \
    --operational-confidence 0.60 \
    --display-confidence 0.05

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
echo "[complete] ${ROOT}/evaluation/yolo26x_perfect_real_finetune_report.png"
echo "[complete] ${ROOT}/table_photo_evaluation/table_cap_model_comparison.png"
