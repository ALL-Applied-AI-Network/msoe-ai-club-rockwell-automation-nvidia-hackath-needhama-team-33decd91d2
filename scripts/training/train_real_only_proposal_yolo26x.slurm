#!/usr/bin/env bash
#SBATCH --job-name=real-proposal-yolo
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --output=shared/runs/2026-10-29_real_only_proposal_yolo/train-v100-%j.out
#SBATCH --error=shared/runs/2026-10-29_real_only_proposal_yolo/train-v100-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${REPO_DIR}/.conda_env/bin/python"
RUN_ROOT="${REPO_DIR}/shared/runs/2026-10-29_real_only_proposal_yolo"
DATA="${RUN_ROOT}/dataset/data.yaml"
TRAINING="${RUN_ROOT}/yolo26x_one_class_1280"

cd "${REPO_DIR}"
test -x "${PYTHON_BIN}"
test -f "${DATA}"
test ! -e "${TRAINING}"

export YOLO_CONFIG_DIR="${REPO_DIR}/shared/cache/ultralytics"
export TORCH_HOME="${REPO_DIR}/shared/cache/torch"
export PYTHONUNBUFFERED=1

srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
    scripts/training/train_real_only_seed_yolo.py \
    --model yolo26x.pt \
    --data "${DATA}" \
    --output "${TRAINING}" \
    --epochs 300 \
    --batch 1 \
    --imgsz 1280 \
    --device 0 \
    --learning-rate 0.001 \
    --freeze 0 \
    --patience 0

for SPEC in "050:0.50" "015:0.15"; do
    SUFFIX="${SPEC%%:*}"
    CONFIDENCE="${SPEC##*:}"
    srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
        scripts/training/render_seed_yolo_raw_samples.py \
        --model "${TRAINING}/weights/best.pt" \
        --raw-root "${REPO_DIR}/runs/archive" \
        --output "${RUN_ROOT}/tablecam_raw_predictions_conf${SUFFIX}" \
        --per-sequence 3 \
        --confidence "${CONFIDENCE}" \
        --imgsz 1280
done

chgrp -R ai_club "${RUN_ROOT}"
chmod -R g+rwX "${RUN_ROOT}"
