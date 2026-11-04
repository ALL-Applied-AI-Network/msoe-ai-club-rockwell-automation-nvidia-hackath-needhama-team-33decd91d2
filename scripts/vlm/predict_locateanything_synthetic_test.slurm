#!/usr/bin/env bash
#SBATCH --job-name=locate-synth100
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --output=shared/runs/2026-10-27_hybrid_vlm_yolo_synthetic_eval/vlm-%j.out
#SBATCH --error=shared/runs/2026-10-27_hybrid_vlm_yolo_synthetic_eval/vlm-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT="${REPO_DIR}/shared/runs/2026-10-27_hybrid_vlm_yolo_synthetic_eval"
PYTHON_BIN="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT}"
export HF_HOME="${REPO_DIR}/shared/cache/locateanything_3b/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
    scripts/vlm/predict_locateanything_dataset.py \
    --model nvidia/LocateAnything-3B \
    --adapter-root runs/locateanything_3b/lora_finetune/full/final \
    --dataset runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000 \
    --split test \
    --output "${OUTPUT}" \
    --max-new-tokens 4096

chgrp -R ai_club "${OUTPUT}"
chmod -R g+rwX "${OUTPUT}"
