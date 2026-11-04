#!/usr/bin/env bash
#SBATCH --job-name=test-table-caps
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=shared/runs/2026-10-28_table_cap_paper_test/job-%j.out
#SBATCH --error=shared/runs/2026-10-28_table_cap_paper_test/job-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT="${REPO_DIR}/shared/runs/2026-10-28_table_cap_paper_test"
OUTPUT="${ROOT}/results"
PYTHON="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
mkdir -p "${ROOT}"
cd "${REPO_DIR}"

srun --export=ALL --ntasks=1 "${PYTHON}" \
    scripts/training/test_table_cap_samples.py \
    --input sample_tables \
    --models \
      "baseline=${REPO_DIR}/shared/yolo26x/best.pt" \
      "corrected_box_finetune=${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune/yolo26x_vlm_box_finetune_1280/weights/best.pt" \
    --output "${OUTPUT}" \
    --imgsz 1280 \
    --operational-confidence 0.40 \
    --display-confidence 0.05

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
echo "[complete] ${OUTPUT}/table_cap_model_comparison.png"
