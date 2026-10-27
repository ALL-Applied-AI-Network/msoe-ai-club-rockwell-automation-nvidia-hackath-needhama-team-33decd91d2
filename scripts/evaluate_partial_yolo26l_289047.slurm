#!/usr/bin/env bash
# Diagnostic-only evaluation of the canceled epoch-22 YOLO26l checkpoint.
# The checkpoint was trained on the now-known wrong synthetic tray scene.
#SBATCH --job-name=eval-y26l-289047
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-y26l-289047-%j.out
#SBATCH --error=logs/eval-y26l-289047-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune/eval_partial_yolo26l_289047

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/eval-y26l-289047-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUT="runs/remediation_scene_finetune/eval_partial_yolo26l_289047"
CHECKPOINT="runs/remediation_scene_finetune/yolo26l_pink_scene_960_ft_wrong_tray_cancelled_289047/weights/best.pt"
SYNTHETIC="runs/remediation_scene_finetune/remediation_cad_true_lighting_pink_1000/data.yaml"
ANNOTATIONS="$OUT/real_annotations.json"

test -f "$CHECKPOINT"
test -f "$SYNTHETIC"
test -f "$ANNOTATIONS"

srun --export=ALL --ntasks=1 python3 scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$CHECKPOINT" \
  --synthetic-data "$SYNTHETIC" \
  --annotations "$ANNOTATIONS" \
  --out "$OUT" \
  --imgsz 960 \
  --title 'Diagnostic evaluation — partial YOLO26l epoch 22 — wrong synthetic tray scene' \
  --limitation 'PARTIAL CHECKPOINT: training was canceled after epoch 22 (23/50), and the model was trained on the now-known WRONG synthetic tray scene. Results are diagnostic only.' \
  --training-state '{"last_completed_epoch":22,"completed_epochs":23,"planned_epochs":50,"slurm_job":289047,"canceled":true}' \
  --report-name partial_yolo26l_wrong_tray_eval_report.png

echo '[complete] evaluation artifacts:'
find "$OUT" -maxdepth 2 -type f -printf '%p\n' | sort
