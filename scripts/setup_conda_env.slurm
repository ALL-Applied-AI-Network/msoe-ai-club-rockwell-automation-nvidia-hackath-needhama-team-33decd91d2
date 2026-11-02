#!/usr/bin/env bash
#SBATCH --job-name=setup_conda_env
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs

echo "=== Initializing .conda_env on $(hostname) ==="
rm -rf .conda_env

/usr/local/anaconda3/condabin/conda create -y -p "${REPO_DIR}/.conda_env" python=3.10

"${REPO_DIR}/.conda_env/bin/python" -m pip install --upgrade pip
"${REPO_DIR}/.conda_env/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
"${REPO_DIR}/.conda_env/bin/python" -m pip install ultralytics opencv-python-headless numpy pyyaml scipy imageio-ffmpeg

echo "=== Verifying Environment ==="
"${REPO_DIR}/.conda_env/bin/python" -c "import torch, ultralytics, cv2; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Ultralytics:', ultralytics.__version__)"

echo "=== .conda_env Setup Complete ==="

