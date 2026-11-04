#!/usr/bin/env bash
#SBATCH --job-name=yolo26x-perfect-real
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/train-%j.out
#SBATCH --error=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/train-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_PYTHON="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
ROOT="${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune"
DATA="${ROOT}/mixed_replay_dataset/data.yaml"
OUTPUT="${ROOT}/yolo26x_vlm_box_finetune_1280"
BASELINE="${REPO_DIR}/shared/yolo26x/best.pt"

cd "${REPO_DIR}"
test -x "${ENV_PYTHON}"
test -f "${BASELINE}"
test -f "${DATA}"
test ! -e "${OUTPUT}"

# These teaching nodes expose four PCIe T4s without a reliable direct-peer
# NCCL path.  Force same-node collectives through shared memory instead of
# letting NCCL select the peer/IB transports that can spin indefinitely.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[train] baseline=${BASELINE}"
echo "[train] data=${DATA}"

srun --export=ALL --ntasks=1 "${ENV_PYTHON}" \
    scripts/training/finetune_yolo26x_perfect_real.py \
    --model "${BASELINE}" \
    --data "${DATA}" \
    --output "${OUTPUT}" \
    --epochs 12 \
    --batch 8 \
    --imgsz 1280 \
    --device 0,1,2,3

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
echo "[complete] ${OUTPUT}/weights/best.pt"
