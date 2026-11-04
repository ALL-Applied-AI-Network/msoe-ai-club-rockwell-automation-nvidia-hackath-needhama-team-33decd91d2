#!/usr/bin/env bash
#SBATCH --job-name=remediation-prediction-samples
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/remediation-prediction-samples-%j.out
#SBATCH --error=logs/remediation-prediction-samples-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

PKG_DIR="/tmp/remediation-prediction-samples-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! srun --export=ALL --ntasks=1 python3 -c 'import torch, ultralytics, PIL, yaml' >/dev/null 2>&1; then
    srun --export=ALL --ntasks=1 python3 -m pip install --target "$PKG_DIR" --no-cache-dir 'torch' 'ultralytics<9' 'pillow' 'pyyaml'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

FULL_SYNTH="runs/remediation_finetune/full_synthetic_test_raw/data.yaml"
if [[ -f "$FULL_SYNTH" ]] && grep -q 'test' "$FULL_SYNTH"; then
    SYNTH_DATA="$FULL_SYNTH"
    SYNTH_SPLIT="test"
else
    echo '[info] independent test set is not ready; rendering translated synthetic validation samples'
    SYNTH_DATA="runs/remediation_finetune/synthetic_dataset/data.yaml"
    SYNTH_SPLIT="val"
fi

ROOT="runs/remediation_finetune"
srun --export=ALL --ntasks=1 python3 scripts/training/render_remediation_prediction_samples.py \
  --models \
    "baseline=shared/inspection_models/remediation_yolo_best.pt" \
    "synthetic=$ROOT/synthetic_ft/weights/best.pt" \
    "synthetic_plus_real=$ROOT/synthetic_plus_real_ft/weights/best.pt" \
    "table_background=$ROOT/table_background_ft/weights/best.pt" \
  --synthetic-data "$SYNTH_DATA" \
  --synthetic-split "$SYNTH_SPLIT" \
  --synthetic-count 3 \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$ROOT/prediction_samples"

find "$ROOT/prediction_samples" -maxdepth 2 -type f -printf '%p\n' | sort
