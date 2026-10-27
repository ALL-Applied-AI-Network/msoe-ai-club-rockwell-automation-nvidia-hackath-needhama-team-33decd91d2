#!/usr/bin/env bash
# Adapt the synthetic-only YOLO26x checkpoint to all five reviewed real frames.
#SBATCH --job-name=y26x-02-real
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/y26x-02-real-%j.out
#SBATCH --error=logs/y26x-02-real-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/yolo26X_runs

PKG_DIR="/tmp/y26x-02-${SLURM_JOB_ID:-manual}/packages"
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
SYNTHETIC="runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000/data.yaml"
REAL="runs/remediation_scene_finetune/real_finetune_5/data.yaml"
ANNOTATIONS="images/Real_Finetune/five_image_annotations.json"
BASE="$ROOT/01_synthetic_only_1280/weights/best.pt"
MODEL_OUT="$ROOT/02_real_finetuned_1280"
EVAL_OUT="$MODEL_OUT/evaluation"

test -f "$BASE"
test -f "$SYNTHETIC"
test -f "$REAL"
test -f "$ANNOTATIONS"
test ! -e "$MODEL_OUT"

echo "[hardware] $(hostname) job=${SLURM_JOB_ID:-manual}"
nvidia-smi
echo '[train] conservative adaptation on all five reviewed physical frames'
srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/train_yolo26x_stage.py \
  --model "$BASE" \
  --data "$REAL" \
  --out "$MODEL_OUT" \
  --stage physical \
  --epochs 120 --batch 4 --imgsz 1280 --device 0

echo '[evaluate] fixed synthetic test and all five physical training frames'
srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$SYNTHETIC" \
  --annotations "$ANNOTATIONS" \
  --out "$EVAL_OUT" \
  --imgsz 1280 \
  --title 'YOLO26x 1280 — five-frame physical fine-tune' \
  --limitation 'All five physical frames were used for both fine-tuning and evaluation; physical metrics are resubstitution and do not estimate unseen-frame generalization.' \
  --training-state '{"stage":"physical_only_adaptation","base":"01_synthetic_only_1280","epochs_requested":120,"imgsz":1280,"physical_frames":5,"train_equals_physical_eval":true}' \
  --report-name yolo26x_real_finetuned_eval_report.png

echo '[complete] physical adaptation stage'
ls -lh "$MODEL_OUT/weights/best.pt" "$MODEL_OUT/weights/last.pt"
