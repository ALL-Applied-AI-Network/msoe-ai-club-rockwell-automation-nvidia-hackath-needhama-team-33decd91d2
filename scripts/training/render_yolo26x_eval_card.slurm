#!/usr/bin/env bash
#SBATCH --job-name=yolo26x-eval-card
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=shared/runs/2026-10-28_yolo26x_real_table_finetune/eval-card-%j.out
#SBATCH --error=shared/runs/2026-10-28_yolo26x_real_table_finetune/eval-card-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
ROOT="${REPO_DIR}/shared/runs/2026-10-28_yolo26x_real_table_finetune"
OUTPUT="${ROOT}/new_photo_eval_card"
MODEL="${ROOT}/yolo26x_real_table_finetune_1280/weights/best.pt"
SYNTH="${REPO_DIR}/runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000"
REAL="${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune/refined_real"
cd "${REPO_DIR}"

srun --export=ALL --ntasks=1 "${PYTHON}" \
    scripts/training/render_yolo26x_eval_card.py \
    --model "${MODEL}" \
    --synthetic-image "${SYNTH}/images/test/remediation_000900.png" \
    --synthetic-label "${SYNTH}/labels/test/remediation_000900.txt" \
    --real-image "${REAL}/images/all/run_155442_0000.png" \
    --real-label "${REAL}/labels/all/run_155442_0000.txt" \
    --table-images sample_tables/20261028_020055.jpg sample_tables/20261028_020101.jpg \
    --output "${OUTPUT}" \
    --confidence 0.60 \
    --imgsz 1280 \
    --heldout-map 0.9724

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
echo "[complete] ${OUTPUT}/yolo26x_new_table_photo_eval_card.png"
