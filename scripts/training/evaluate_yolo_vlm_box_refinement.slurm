#!/usr/bin/env bash
#SBATCH --job-name=hybrid-synth-eval
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=shared/runs/2026-10-27_hybrid_vlm_yolo_synthetic_eval/eval-%j.out
#SBATCH --error=shared/runs/2026-10-27_hybrid_vlm_yolo_synthetic_eval/eval-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT="${REPO_DIR}/shared/runs/2026-10-27_hybrid_vlm_yolo_synthetic_eval"
PYTHON_BIN="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
PKG_DIR="/tmp/hybrid-synth-eval-${SLURM_JOB_ID:-manual}/packages"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT}" "${PKG_DIR}"
if ! srun --export=ALL --ntasks=1 "${PYTHON_BIN}" -c 'import ultralytics' >/dev/null 2>&1; then
    srun --export=ALL --ntasks=1 "${PYTHON_BIN}" -m pip install --target "${PKG_DIR}" --no-cache-dir --no-deps \
        'ultralytics>=8.4,<9' 'ultralytics-thop>=2.0.18' 'polars' 'scipy' 'py-cpuinfo'
fi
export PYTHONPATH="${PKG_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
    scripts/training/evaluate_yolo_vlm_box_refinement.py \
    --model shared/yolo26x/best.pt \
    --dataset runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000 \
    --split test \
    --vlm-predictions "${OUTPUT}/predictions_and_raw_responses.json" \
    --output "${OUTPUT}" \
    --imgsz 1280 \
    --confidence 0.40

chgrp -R ai_club "${OUTPUT}"
chmod -R g+rwX "${OUTPUT}"
