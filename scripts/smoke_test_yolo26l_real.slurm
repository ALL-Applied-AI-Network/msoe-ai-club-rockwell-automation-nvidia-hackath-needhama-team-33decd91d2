#!/usr/bin/env bash
# Run the reusable synthetic + manually labeled physical audit on a checkpoint
# snapshot without interfering with an active training job.
#SBATCH --job-name=y26l-real-smoke
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/y26l-real-smoke-%j.out
#SBATCH --error=logs/y26l-real-smoke-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs

CHECKPOINT="${1:?checkpoint path is required}"
OUT="${2:?output directory is required}"
SYNTHETIC_DATA="${3:-runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000/data.yaml}"
ANNOTATIONS="${4:-runs/remediation_scene_finetune/eval_partial_yolo26l_289047/real_annotations.json}"
EXISTING_PACKAGES="${5:-}"

test -f "$CHECKPOINT"
test -f "$SYNTHETIC_DATA"
test -f "$ANNOTATIONS"

if [[ -n "$EXISTING_PACKAGES" && -d "$EXISTING_PACKAGES" ]]; then
    export PYTHONPATH="$EXISTING_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    PKG_DIR="/tmp/y26l-real-smoke-${SLURM_JOB_ID:-manual}/packages"
    mkdir -p "$PKG_DIR"
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
    export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID:-unknown} checkpoint=$CHECKPOINT"
nvidia-smi

srun --export=ALL --ntasks=1 python3 scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$CHECKPOINT" \
  --synthetic-data "$SYNTHETIC_DATA" \
  --annotations "$ANNOTATIONS" \
  --out "$OUT" \
  --imgsz 960 \
  --title 'Interim YOLO26l physical smoke test — epoch 26 snapshot' \
  --limitation 'Interim checkpoint and four-frame physical audit; use this only as an early domain-gap signal.' \
  --training-state '{"checkpoint_epoch":26,"training_in_progress":true,"base_model":"yolo26l.pt"}' \
  --report-name interim_epoch26_real_smoke_report.png

echo "[complete] $OUT/interim_epoch26_real_smoke_report.png"
