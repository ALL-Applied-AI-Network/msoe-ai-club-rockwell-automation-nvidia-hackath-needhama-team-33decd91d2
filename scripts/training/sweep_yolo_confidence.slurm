#!/usr/bin/env bash
#SBATCH --job-name=sweep-yolo-conf
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=shared/runs/2026-10-28_yolo26x_real_table_finetune/sweep-%j.out
#SBATCH --error=shared/runs/2026-10-28_yolo26x_real_table_finetune/sweep-%j.err

set -euo pipefail
umask 002
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT="${REPO_DIR}/shared/runs/2026-10-28_yolo26x_real_table_finetune"
MODEL="${1:?model checkpoint required}"
OUTPUT_NAME="${2:?output name required}"
cd "${REPO_DIR}"

srun --export=ALL --ntasks=1 shared/cache/locateanything_3b/venv/bin/python \
    scripts/training/sweep_yolo_confidence.py \
    --model "${MODEL}" \
    --datasets \
      "synthetic=${ROOT}/dataset_build/mixed_replay_dataset/images/test,${ROOT}/dataset_build/mixed_replay_dataset/labels/test" \
      "prior_real=${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune/refined_real/images/all,${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune/refined_real/labels/all" \
      "table_real=${ROOT}/dataset_build/reviewed_table_caps/images/all,${ROOT}/dataset_build/reviewed_table_caps/labels/all" \
    --output "${ROOT}/${OUTPUT_NAME}" \
    --imgsz 1280

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
