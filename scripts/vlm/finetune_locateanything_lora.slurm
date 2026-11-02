#!/usr/bin/env bash
#SBATCH --job-name=locate-lora
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=08:00:00
#SBATCH --output=runs/locateanything_3b/logs/%x_%j.out
#SBATCH --error=runs/locateanything_3b/logs/%x_%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR}"
ENV_DIR="${REPO_DIR}/shared/cache/locateanything_3b/venv"
HF_HOME_DIR="${REPO_DIR}/shared/cache/locateanything_3b/huggingface"
SYNTHETIC="${REPO_DIR}/runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000"
REAL="${REPO_DIR}/runs/yolo26X_runs/03_mixed_replay_1280/evaluation/real_dataset"
OUTPUT="${LOCATE_LORA_OUTPUT:-${REPO_DIR}/runs/locateanything_3b/lora_finetune/full}"
NPROC="${LOCATE_LORA_GPUS:-2}"

mkdir -p "${REPO_DIR}/runs/locateanything_3b/logs" "${OUTPUT}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

EXTRA_ARGS=()
if [[ "${LOCATE_LORA_MODE:-full}" == "smoke" ]]; then
    EXTRA_ARGS+=(--max-steps 4 --grad-accum 1 --num-workers 0 --log-every 1 --real-repeat 1 --missing-repeat 1 --clahe-prob 0.0)
fi
if [[ -n "${LOCATE_LORA_EXTRA_ARGS:-}" ]]; then
    read -r -a USER_EXTRA_ARGS <<< "${LOCATE_LORA_EXTRA_ARGS}"
    EXTRA_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi

echo "node=$(hostname) output=${OUTPUT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --export=ALL --ntasks=1 "${ENV_DIR}/bin/torchrun" \
    --standalone \
    --nproc_per_node="${NPROC}" \
    "${REPO_DIR}/scripts/vlm/finetune_locateanything_lora.py" \
    --model nvidia/LocateAnything-3B \
    --synthetic "${SYNTHETIC}" \
    --real "${REAL}" \
    --output "${OUTPUT}" \
    --epochs 2 \
    --grad-accum 4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --learning-rate 1e-4 \
    --projector-learning-rate 2e-5 \
    --real-repeat 20 \
    --missing-repeat 2 \
    --clahe-prob 0.5 \
    "${EXTRA_ARGS[@]}"

chmod -R g+rwX "${REPO_DIR}/runs/locateanything_3b/lora_finetune"
