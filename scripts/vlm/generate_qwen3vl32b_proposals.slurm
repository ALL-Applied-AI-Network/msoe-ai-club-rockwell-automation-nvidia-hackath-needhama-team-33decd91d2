#!/usr/bin/env bash
#SBATCH --job-name=qwen32b-grounding
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=06:00:00
#SBATCH --output=shared/runs/2026-10-28_vial_cap_weak_labeling/qwen32b-%j.out
#SBATCH --error=shared/runs/2026-10-28_vial_cap_weak_labeling/qwen32b-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_ROOT="${REPO_DIR}/shared/runs/2026-10-28_vial_cap_weak_labeling"
PYTHON_BIN="${REPO_DIR}/shared/cache/locateanything_3b/venv/bin/python"
IMAGES="${DEFECT_IMAGES:?Set DEFECT_IMAGES to the real-image directory}"
SPLITS="${DEFECT_SPLITS:-${REPO_DIR}/config/vial_cap_reviewed_pilot_splits.json}"
OUTPUT="${DEFECT_PROPOSAL_OUTPUT:-${RUN_ROOT}/qwen32b_proposals_train_val}"
CONFIG="${DEFECT_CONFIG:-${REPO_DIR}/config/vial_cap_qwen32b_grounding_experiment.yaml}"

cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}"
test -x "${PYTHON_BIN}"

export HF_HOME="${REPO_DIR}/shared/cache/qwen3vl_32b/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

srun --export=ALL --ntasks=1 "${PYTHON_BIN}" \
    scripts/vlm/generate_qwen3vl_defect_proposals.py \
    --config "${CONFIG}" \
    --images "${IMAGES}" \
    --splits "${SPLITS}" \
    --output "${OUTPUT}" \
    --include-splits train,val

chgrp -R ai_club "${RUN_ROOT}"
chmod -R g+rwX "${RUN_ROOT}"
