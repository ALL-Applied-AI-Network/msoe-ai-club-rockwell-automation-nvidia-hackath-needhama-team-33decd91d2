#!/usr/bin/env bash
#SBATCH --job-name=remediation-model-comparison
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/remediation-model-comparison-%j.out
#SBATCH --error=logs/remediation-model-comparison-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs/remediation_finetune/model_comparison

PKG_DIR="/tmp/remediation-model-comparison-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! srun --export=ALL --ntasks=1 python3 -c 'import torch, ultralytics, matplotlib' >/dev/null 2>&1; then
    srun --export=ALL --ntasks=1 python3 -m pip install --target "$PKG_DIR" --no-cache-dir 'torch' 'ultralytics<9' 'matplotlib'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

FULL_SYNTH="runs/remediation_finetune/full_synthetic_test_raw/data.yaml"
if [[ -n "${COMPARE_SYNTH_DATA:-}" ]]; then
    SYNTH_DATA="$COMPARE_SYNTH_DATA"
    SYNTH_SPLIT="${COMPARE_SYNTH_SPLIT:-val}"
elif [[ -f "$FULL_SYNTH" ]] && grep -q 'test' "$FULL_SYNTH"; then
    SYNTH_DATA="$FULL_SYNTH"
    SYNTH_SPLIT="test"
else
    echo '[warning] independent full synthetic test is unavailable; using the translated held-out synthetic validation set'
    SYNTH_DATA="runs/remediation_finetune/synthetic_dataset/data.yaml"
    SYNTH_SPLIT="val"
fi

BASE="shared/inspection_models/remediation_yolo_best.pt"
ROOT="runs/remediation_finetune"
OUT="${COMPARE_OUT:-$ROOT/model_comparison}"

srun --export=ALL --ntasks=1 python3 scripts/training/evaluate_remediation_comparison.py \
  --models \
    "baseline=$BASE" \
    "synthetic=$ROOT/synthetic_ft/weights/best.pt" \
    "synthetic_plus_real=$ROOT/synthetic_plus_real_ft/weights/best.pt" \
    "table_background=$ROOT/table_background_ft/weights/best.pt" \
  --synthetic-data "$SYNTH_DATA" \
  --synthetic-split "$SYNTH_SPLIT" \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$OUT"

echo '[complete] comparison files:'
find "$OUT" -maxdepth 1 -type f -printf '%p\n' | sort
