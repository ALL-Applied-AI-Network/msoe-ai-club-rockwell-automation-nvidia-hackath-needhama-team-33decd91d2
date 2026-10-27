#!/usr/bin/env bash
# Train the largest standard YOLO26 detector on the qualified synthetic scene.
#SBATCH --job-name=y26x-01-synth
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/y26x-01-synth-%j.out
#SBATCH --error=logs/y26x-01-synth-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/yolo26X_runs

PKG_DIR="/tmp/y26x-01-${SLURM_JOB_ID:-manual}/packages"
PYTHON_BIN="/usr/local/miniforge/miniforge3/bin/python"
mkdir -p "$PKG_DIR"
if ! "$PYTHON_BIN" -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --target "$PKG_DIR" --no-cache-dir \
      --extra-index-url 'https://download.pytorch.org/whl/cu118' \
      'torch==2.5.1+cu118' 'torchvision==0.20.1+cu118' \
      'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

ROOT="runs/yolo26X_runs"
DATA="runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000/data.yaml"
ANNOTATIONS="images/Real_Finetune/five_image_annotations.json"
MODEL_OUT="$ROOT/01_synthetic_only_1280"
EVAL_OUT="$MODEL_OUT/evaluation"

test -f "$DATA"
test -f "$ANNOTATIONS"
test ! -e "$MODEL_OUT"

echo "[hardware] $(hostname) job=${SLURM_JOB_ID:-manual}"
nvidia-smi
echo '[train] YOLO26x / 1280 px / qualified synthetic data only'
srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/train_yolo26x_stage.py \
  --model yolo26x.pt \
  --data "$DATA" \
  --out "$MODEL_OUT" \
  --stage synthetic \
  --epochs 50 --batch 4 --imgsz 1280 --device 0

echo '[evaluate] fixed synthetic test and all five reviewed physical frames'
srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$DATA" \
  --annotations "$ANNOTATIONS" \
  --out "$EVAL_OUT" \
  --imgsz 1280 \
  --title 'YOLO26x 1280 — synthetic-only training' \
  --limitation 'The physical audit has five reviewed frames. This synthetic-only stage has not seen them during training.' \
  --training-state '{"stage":"synthetic_only","base":"yolo26x.pt","epochs_requested":50,"imgsz":1280,"physical_frames_in_training":0}' \
  --report-name yolo26x_synthetic_only_eval_report.png

echo '[complete] synthetic-only stage'
ls -lh "$MODEL_OUT/weights/best.pt" "$MODEL_OUT/weights/last.pt"
