#!/usr/bin/env bash
# Train the accuracy-first detector on the qualified black/silver remediation
# scene with physical ceiling lighting and the planned pink table-cap paper.
# Single-GPU training avoids the cluster's T4 NCCL initialization deadlock.
#SBATCH --job-name=y26l-black-silver
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/y26l-black-silver-%j.out
#SBATCH --error=logs/y26l-black-silver-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/y26l-black-silver-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

ROOT="runs/remediation_scene_finetune"
DATA="$ROOT/remediation_black_silver_physical_pink_1000"
MODEL_OUT="$ROOT/yolo26l_black_silver_physical_pink_960_ft"
EVAL_OUT="$ROOT/eval_yolo26l_black_silver_physical_pink_960"
ANNOTATIONS="$ROOT/eval_partial_yolo26l_289047/real_annotations.json"

test -f yolo26l.pt
test -f "$DATA/data.yaml"
test -f "$ANNOTATIONS"
test ! -e "$MODEL_OUT"

echo '[train] YOLO26l at 960px on 800 qualified-scene samples using one T4'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model yolo26l.pt \
  --data "$DATA/data.yaml" \
  --eval-data "$DATA/data.yaml" \
  --out "$MODEL_OUT" \
  --epochs 50 --batch 4 --imgsz 960 --device 0

echo '[evaluate] held-out corrected synthetic test plus manually labeled physical audit'
srun --export=ALL --ntasks=1 python3 scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$DATA/data.yaml" \
  --annotations "$ANNOTATIONS" \
  --out "$EVAL_OUT" \
  --imgsz 960 \
  --title 'YOLO26l — qualified black/silver physical-lighting scene' \
  --limitation 'The physical audit contains four curated frames and is directional, not a deployment-grade confidence interval.' \
  --training-state '{"completed_epochs":50,"planned_epochs":50,"base_model":"yolo26l.pt","scene":"black_silver_remediation_qualified_v3.usda"}' \
  --report-name yolo26l_black_silver_physical_eval_report.png

echo '[complete] checkpoint and evaluation report:'
ls -lh "$MODEL_OUT/weights/best.pt"
echo "$EVAL_OUT/yolo26l_black_silver_physical_eval_report.png"
