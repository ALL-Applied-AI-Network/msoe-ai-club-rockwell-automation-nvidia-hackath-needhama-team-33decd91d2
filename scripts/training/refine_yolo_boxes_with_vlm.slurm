#!/usr/bin/env bash
#SBATCH --job-name=vlm-box-refine
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=shared/runs/2026-10-27_vlm_box_refinement/slurm-%j.out
#SBATCH --error=shared/runs/2026-10-27_vlm_box_refinement/slurm-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT="${REPO_DIR}/shared/runs/2026-10-27_vlm_box_refinement"
PYTHON_BIN="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PKG_DIR="/tmp/vlm-box-refine-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "${PKG_DIR}"
if ! srun --export=ALL --ntasks=1 "${PYTHON_BIN}" -c 'import ultralytics' >/dev/null 2>&1; then
    srun --export=ALL --ntasks=1 "${PYTHON_BIN}" -m pip install --target "${PKG_DIR}" --no-cache-dir --no-deps \
        'ultralytics>=8.4,<9' 'ultralytics-thop>=2.0.18' 'polars' 'scipy' 'py-cpuinfo'
fi
export PYTHONPATH="${PKG_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
    scripts/training/refine_yolo_boxes_with_vlm.py \
    --model shared/yolo26x/best.pt \
    --images runs/yolo26X_runs/03_mixed_replay_1280/evaluation/real_dataset/images/test \
    --reviewed-labels runs/yolo26X_runs/03_mixed_replay_1280/evaluation/real_dataset/labels/test \
    --vlm-predictions runs/locateanything_3b/lora_finetune/evaluation_real5/predictions_and_raw_responses.json \
    --vlm-view raw \
    --output "${OUTPUT}" \
    --imgsz 1280 \
    --confidence 0.40

chgrp -R ai_club "${OUTPUT}"
chmod -R g+rwX "${OUTPUT}"
