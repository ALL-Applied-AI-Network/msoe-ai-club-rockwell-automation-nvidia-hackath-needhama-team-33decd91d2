#!/usr/bin/env bash
# Generalize the physical-adapted checkpoint with balanced synthetic/real replay.
#SBATCH --job-name=y26x-03-mixed
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/y26x-03-mixed-%j.out
#SBATCH --error=logs/y26x-03-mixed-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/yolo26X_runs

PKG_DIR="/tmp/y26x-03-${SLURM_JOB_ID:-manual}/packages"
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
SYNTHETIC_ROOT="runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000"
SYNTHETIC="$SYNTHETIC_ROOT/data.yaml"
REAL_ROOT="runs/remediation_scene_finetune/real_finetune_5"
ANNOTATIONS="images/Real_Finetune/five_image_annotations.json"
MIXED="$ROOT/datasets/balanced_mixed_replay"
BASE="$ROOT/02_real_finetuned_1280/weights/best.pt"
MODEL_OUT="$ROOT/03_mixed_replay_1280"
EVAL_OUT="$MODEL_OUT/evaluation"
INCUMBENT="runs/remediation_scene_finetune/yolo26l_real5_960_ft/weights/best.pt"
INCUMBENT_OUT="$ROOT/00_incumbent_yolo26l_1280/evaluation"

test -f "$BASE"
test -f "$SYNTHETIC"
test -f "$REAL_ROOT/data.yaml"
test -f "$ANNOTATIONS"
test -f "$INCUMBENT"
test ! -e "$MODEL_OUT"

echo "[hardware] $(hostname) job=${SLURM_JOB_ID:-manual}"
nvidia-smi
if [[ ! -f "$MIXED/data.yaml" ]]; then
  echo '[prepare] balanced replay: 800 synthetic + 800 physical entries'
  srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/build_yolo26x_mixed_replay_dataset.py \
    --synthetic "$SYNTHETIC_ROOT" \
    --real "$REAL_ROOT" \
    --out "$MIXED" \
    --real-total 800
fi

echo '[train] mixed synthetic/physical replay from physical-adapted checkpoint'
srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/train_yolo26x_stage.py \
  --model "$BASE" \
  --data "$MIXED/data.yaml" \
  --out "$MODEL_OUT" \
  --stage mixed \
  --epochs 30 --batch 4 --imgsz 1280 --device 0

echo '[evaluate] mixed-replay model on fixed synthetic test and all five physical frames'
srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$SYNTHETIC" \
  --annotations "$ANNOTATIONS" \
  --out "$EVAL_OUT" \
  --imgsz 1280 \
  --title 'YOLO26x 1280 — balanced synthetic/physical replay' \
  --limitation 'The replay set contains repeated copies of all five physical evaluation frames; physical metrics are resubstitution. The synthetic metrics use the untouched 100-image test split.' \
  --training-state '{"stage":"balanced_mixed_replay","base":"02_real_finetuned_1280","epochs_requested":30,"imgsz":1280,"synthetic_entries":800,"physical_entries":800,"physical_unique_frames":5}' \
  --report-name yolo26x_mixed_replay_eval_report.png

if [[ ! -f "$INCUMBENT_OUT/metrics.json" ]]; then
  echo '[compare] re-evaluate incumbent YOLO26l at the same 1280-pixel setting'
  srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/evaluate_partial_yolo26l_289047.py \
    --checkpoint "$INCUMBENT" \
    --synthetic-data "$SYNTHETIC" \
    --annotations "$ANNOTATIONS" \
    --out "$INCUMBENT_OUT" \
    --imgsz 1280 \
    --title 'Incumbent YOLO26l — evaluated at deployment resolution' \
    --limitation 'All five physical frames were used for both fine-tuning and evaluation; physical metrics are resubstitution and do not estimate unseen-frame generalization.' \
    --training-state '{"stage":"incumbent_comparison","trained_imgsz":960,"evaluated_imgsz":1280,"physical_frames":5}' \
    --report-name incumbent_yolo26l_1280_eval_report.png
fi

echo '[summarize] four-model comparison'
srun --export=ALL --ntasks=1 "$PYTHON_BIN" scripts/summarize_yolo26x_pipeline.py --root "$ROOT"

echo '[complete] mixed replay and comparison'
ls -lh "$MODEL_OUT/weights/best.pt" "$MODEL_OUT/weights/last.pt"
echo "$ROOT/comparison.md"
