#!/usr/bin/env bash
#SBATCH --job-name=locateanything-real5
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --output=runs/locateanything_3b/logs/%x_%j.out
#SBATCH --error=runs/locateanything_3b/logs/%x_%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR}"
RUN_ROOT="${REPO_DIR}/runs/locateanything_3b"
CACHE_ROOT="${REPO_DIR}/shared/cache/locateanything_3b"
ENV_DIR="${CACHE_ROOT}/venv"
HF_HOME_DIR="${CACHE_ROOT}/huggingface"
DATASET="${REPO_DIR}/runs/yolo26X_runs/03_mixed_replay_1280/evaluation/real_dataset"
OUTPUT="${LOCATE_EVAL_OUTPUT:-${RUN_ROOT}/official_bf16_real5_concise}"
ADAPTER_ROOT="${LOCATE_ADAPTER_ROOT:-}"

mkdir -p "${RUN_ROOT}/logs" "${CACHE_ROOT}" "${HF_HOME_DIR}" "${OUTPUT}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    python3 -m venv "${ENV_DIR}"
fi

if ! "${ENV_DIR}/bin/python" -c 'import torch, transformers, cv2, matplotlib' >/dev/null 2>&1; then
    "${ENV_DIR}/bin/python" -m pip install --upgrade pip
    "${ENV_DIR}/bin/python" -m pip install \
        torch torchvision --index-url https://download.pytorch.org/whl/cu128
    "${ENV_DIR}/bin/python" -m pip install \
        opencv-python-headless==4.11.0.86 \
        transformers==4.57.1 \
        numpy==1.26.4 \
        Pillow==11.1.0 \
        peft torchvision decord==0.6.0 lmdb==1.7.5 \
        accelerate huggingface-hub matplotlib
fi

export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${ENV_DIR}/bin/python" -c 'import torch, transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda}")'

ADAPTER_ARGS=()
if [[ -n "${ADAPTER_ROOT}" ]]; then
    ADAPTER_ARGS+=(--adapter-root "${ADAPTER_ROOT}")
fi

srun --export=ALL --ntasks=1 "${ENV_DIR}/bin/python" \
    "${REPO_DIR}/scripts/vlm/evaluate_locateanything_real5.py" \
    --model nvidia/LocateAnything-3B \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --views raw,clahe \
    --max-new-tokens 4096 \
    "${ADAPTER_ARGS[@]}"

chmod -R g+rwX "${RUN_ROOT}" "${CACHE_ROOT}"
