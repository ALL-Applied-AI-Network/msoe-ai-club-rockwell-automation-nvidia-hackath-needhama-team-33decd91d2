#!/usr/bin/env bash
# Baseline -> synthetic fine-tune -> synthetic + Qwen/GPT fine-tune.
# All three evaluations use the unchanged remediation validation set.
#SBATCH --job-name=remediation-ft-eval
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/remediation-ft-eval-%j.out
#SBATCH --error=logs/remediation-ft-eval-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_finetune

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

# The teaching nodes do not provide a consistent Python ML environment.
# Keep the fallback CUDA PyTorch/Ultralytics bundle node-local and reuse it
# for every srun below.
PKG_DIR="/tmp/remediation-ft-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir 'torch' 'ultralytics<9'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
srun --export=ALL --ntasks=1 python3 -c 'import torch, ultralytics; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ultralytics", ultralytics.__version__)'

ROOT="runs/remediation_finetune"
SYNTH_DATA="$ROOT/synthetic_dataset"
REAL_DATA="$ROOT/real_dataset"
MIXED_DATA="$ROOT/mixed_dataset"
EVAL_DATA="shared/inspection_models/remediation_detection_dataset/data.yaml"

srun --export=ALL --ntasks=1 python3 scripts/datasets/prepare_remediation_finetune_data.py \
  --synthetic-source images/yolo_detection_dataset_800 \
  --base-remediation shared/inspection_models/remediation_detection_dataset \
  --synthetic-out "$SYNTH_DATA" \
  --real-images images/pruned_images \
  --vlm-dir images/VLM_samples \
  --gpt-dir images/GPT_samples \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --real-out "$REAL_DATA" \
  --mixed-out "$MIXED_DATA"

echo '[stage baseline] evaluating existing remediation checkpoint'
srun --export=ALL --ntasks=1 python3 scripts/training/evaluate_remediation_detector.py \
  --model shared/inspection_models/remediation_yolo_best.pt \
  --data "$EVAL_DATA" \
  --out "$ROOT/baseline.json"

echo '[stage synthetic] fine-tuning on translated synthetic data'
srun --export=ALL --ntasks=1 python3 scripts/training/finetune_remediation_detector.py \
  --model shared/inspection_models/remediation_yolo_best.pt \
  --data "$SYNTH_DATA/data.yaml" \
  --eval-data "$EVAL_DATA" \
  --out "$ROOT/synthetic_ft" \
  --epochs 30 --batch 8 --imgsz 640

echo '[stage mixed] fine-tuning synthetic checkpoint with accepted Qwen/GPT frames'
srun --export=ALL --ntasks=1 python3 scripts/training/finetune_remediation_detector.py \
  --model "$ROOT/synthetic_ft/weights/best.pt" \
  --data "$MIXED_DATA/data.yaml" \
  --eval-data "$EVAL_DATA" \
  --out "$ROOT/synthetic_plus_real_ft" \
  --epochs 30 --batch 8 --imgsz 640

echo '[complete] reports:'
find "$ROOT" -maxdepth 2 -name '*.json' -print | sort
