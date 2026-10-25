#!/usr/bin/env bash
# Continuation job for images without a successful per-image classification.
# Timeout cases are listed separately for a later stronger-model pass.
#SBATCH --job-name=vlm-full-remain
#SBATCH --partition=teaching
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --output=logs/vlm-full-image-remaining-%j.out
#SBATCH --error=logs/vlm-full-image-remaining-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

export VLM_IMAGE_DIR="${VLM_IMAGE_DIR:-${REPO_DIR}/images/pruned_images}"
export VLM_EXAMPLES_DIR="${VLM_EXAMPLES_DIR:-${REPO_DIR}/images/VLM_examples}"
export VLM_PER_IMAGE_DIR="${VLM_PER_IMAGE_DIR:-${REPO_DIR}/images/VLM_samples}"
export VLM_SKIP_LIST="${VLM_SKIP_LIST:-${REPO_DIR}/images/VLM_samples/timeout_images.txt}"

# The dataset launcher uses --resume and writes one atomic JSON file per image.
exec bash "${REPO_DIR}/scripts/run_vlm_full_image_dataset.slurm"
