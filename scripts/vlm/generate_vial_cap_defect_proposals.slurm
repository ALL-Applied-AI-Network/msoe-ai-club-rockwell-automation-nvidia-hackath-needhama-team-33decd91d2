#!/usr/bin/env bash
#SBATCH --job-name=vial-cap-proposals
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --output=shared/runs/2026-10-28_vial_cap_weak_labeling/proposals-%j.out
#SBATCH --error=shared/runs/2026-10-28_vial_cap_weak_labeling/proposals-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_ROOT="${REPO_DIR}/shared/runs/2026-10-28_vial_cap_weak_labeling"
PYTHON_BIN="${REPO_DIR}/.conda_env/bin/python"
IMAGES="${DEFECT_IMAGES:?Set DEFECT_IMAGES to the real-image directory}"
SPLITS="${DEFECT_SPLITS:-${REPO_DIR}/config/vial_cap_reviewed_pilot_splits.json}"
OUTPUT="${DEFECT_PROPOSAL_OUTPUT:-${RUN_ROOT}/proposals}"
ADAPTER_ROOT="${LOCATE_ADAPTER_ROOT:-}"
INCLUDE_SPLITS="${DEFECT_INCLUDE_SPLITS:-}"
CONFIG="${DEFECT_CONFIG:-${REPO_DIR}/config/vial_cap_defect_trainer.yaml}"

cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}"
test -x "${PYTHON_BIN}"

"${PYTHON_BIN}" -c \
    'import PIL, torch, transformers; print(f"Pillow={PIL.__version__} torch={torch.__version__} transformers={transformers.__version__} cuda={torch.cuda.is_available()}")'

export HF_HOME="${REPO_DIR}/shared/cache/locateanything_3b/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ADAPTER_ARGS=()
if [[ -n "${ADAPTER_ROOT}" ]]; then
    ADAPTER_ARGS+=(--adapter-root "${ADAPTER_ROOT}")
fi

SPLIT_ARGS=()
if [[ -n "${INCLUDE_SPLITS}" ]]; then
    SPLIT_ARGS+=(--include-splits "${INCLUDE_SPLITS}")
fi

srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
    scripts/vlm/generate_defect_proposals.py \
    --config "${CONFIG}" \
    --images "${IMAGES}" \
    --splits "${SPLITS}" \
    --output "${OUTPUT}" \
    "${SPLIT_ARGS[@]}" \
    "${ADAPTER_ARGS[@]}"

chgrp -R ai_club "${RUN_ROOT}"
chmod -R g+rwX "${RUN_ROOT}"
