#!/usr/bin/env bash
#SBATCH --job-name=build-perfect-real
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/build-%j.out
#SBATCH --error=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/build-%j.err

set -euo pipefail
umask 002
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT="${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune/refined_real"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT}"
srun --export=ALL --ntasks=1 shared/cache/locateanything_3b/venv/bin/python \
    scripts/training/build_vlm_refined_real_labels.py \
    --images runs/yolo26X_runs/03_mixed_replay_1280/evaluation/real_dataset/images/test \
    --reviewed-labels runs/yolo26X_runs/03_mixed_replay_1280/evaluation/real_dataset/labels/test \
    --vlm-predictions runs/locateanything_3b/lora_finetune/evaluation_real5/predictions_and_raw_responses.json \
    --output "${OUTPUT}"
chgrp -R ai_club "${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune"
chmod -R g+rwX "${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune"
