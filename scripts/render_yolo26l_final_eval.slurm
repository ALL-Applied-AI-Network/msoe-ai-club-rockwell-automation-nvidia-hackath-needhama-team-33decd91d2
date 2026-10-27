#!/usr/bin/env bash
# Regenerate the final synthetic/real evaluation and its compact-label report.
#SBATCH --job-name=render-y26l-eval
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/render-y26l-eval-%j.out
#SBATCH --error=logs/render-y26l-eval-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs

PKG_DIR="/tmp/render-y26l-eval-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

ROOT="runs/remediation_scene_finetune"
DATA="$ROOT/remediation_black_silver_physical_pink_1000/data.yaml"
MODEL="$ROOT/yolo26l_black_silver_physical_pink_960_ft/weights/best.pt"
OUT="$ROOT/eval_yolo26l_black_silver_physical_pink_960"
ANNOTATIONS="$ROOT/eval_partial_yolo26l_289047/real_annotations.json"

test -f "$MODEL"
test -f "$DATA"
test -f "$ANNOTATIONS"

srun --export=ALL --ntasks=1 python3 scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$MODEL" \
  --synthetic-data "$DATA" \
  --annotations "$ANNOTATIONS" \
  --out "$OUT" \
  --imgsz 960 \
  --title 'YOLO26l — qualified black/silver physical-lighting scene' \
  --limitation 'The four-frame physical audit contains known annotation errors and is directional, not a deployment-grade estimate.' \
  --training-state '{"completed_epochs":50,"planned_epochs":50,"base_model":"yolo26l.pt","scene":"black_silver_remediation_qualified_v3.usda"}' \
  --report-name yolo26l_black_silver_physical_eval_report.png

echo "[complete] $OUT/yolo26l_black_silver_physical_eval_report.png"
