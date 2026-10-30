#!/usr/bin/env bash
#SBATCH --job-name=setup-locate-env
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup-locate-env-%j.out
#SBATCH --error=logs/setup-locate-env-%j.err

set -euo pipefail
umask 002

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${REPO_DIR}/.conda_env/bin/python"

cd "${REPO_DIR}"
mkdir -p logs
test -x "${PYTHON_BIN}"

"${PYTHON_BIN}" -m pip install \
    transformers==4.57.1 \
    accelerate \
    huggingface-hub \
    peft \
    decord==0.6.0 \
    lmdb==1.7.5 \
    Pillow==11.1.0

"${PYTHON_BIN}" -c \
    'import PIL, accelerate, decord, lmdb, peft, torch, transformers; print(f"Pillow={PIL.__version__} torch={torch.__version__} transformers={transformers.__version__} cuda={torch.cuda.is_available()}")'
