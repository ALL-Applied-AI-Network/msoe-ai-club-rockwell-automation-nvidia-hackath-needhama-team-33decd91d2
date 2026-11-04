#!/usr/bin/env bash
#SBATCH --job-name=yolo26x-real-table
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=shared/runs/2026-10-28_yolo26x_real_table_finetune/train-%j.out
#SBATCH --error=shared/runs/2026-10-28_yolo26x_real_table_finetune/train-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT="${REPO_DIR}/shared/runs/2026-10-28_yolo26x_real_table_finetune"
DATA="${ROOT}/dataset_build/mixed_replay_dataset/data.yaml"
OUTPUT="${ROOT}/yolo26x_real_table_finetune_1280"
BASELINE="${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune/yolo26x_vlm_box_finetune_1280/weights/best.pt"
PYTHON="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"

cd "${REPO_DIR}"
test -f "${BASELINE}"
test -f "${DATA}"
test ! -e "${OUTPUT}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

srun --export=ALL --ntasks=1 "${PYTHON}" \
    scripts/training/finetune_yolo26x_perfect_real.py \
    --model "${BASELINE}" \
    --data "${DATA}" \
    --output "${OUTPUT}" \
    --epochs 8 \
    --batch 8 \
    --imgsz 1280 \
    --device 0,1,2,3

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
echo "[complete] ${OUTPUT}/weights/best.pt"
