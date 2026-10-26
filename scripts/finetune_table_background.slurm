#!/usr/bin/env bash
# Generate colored-paper table-cap scenes, merge them with the accepted
# synthetic+real training set, and fine-tune/evaluate the saved mixed model.
#SBATCH --job-name=table-cap-background-ft
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/table-cap-background-ft-%j.out
#SBATCH --error=logs/table-cap-background-ft-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs/remediation_finetune/checkpoints

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/table-background-ft-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir 'torch' 'ultralytics<9'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
srun --export=ALL --ntasks=1 python3 -c 'import torch, ultralytics; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ultralytics", ultralytics.__version__)'

ROOT="runs/remediation_finetune"
RAW="$ROOT/table_background_raw"
TABLE_DATA="$ROOT/table_background_mix"
EVAL_DATA="shared/inspection_models/remediation_detection_dataset/data.yaml"
BASE_DATA="$ROOT/mixed_dataset"
CHECKPOINT="$ROOT/synthetic_plus_real_ft/weights/best.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing completed mixed-stage checkpoint: $CHECKPOINT" >&2
    exit 2
fi

if [[ ! -f "$RAW/data.yaml" ]]; then
    echo '[stage table data] generating physics-qualified colored-paper scenes'
    srun --export=ALL --ntasks=1 ./scripts/run_isaac.sh \
      src/data_generation/generate_remediation_dataset.py \
      --out "$RAW" --scenes 40 --seed 6100 --crooked 0 --missing 0 \
      --table-caps 5 --fixed-inventory \
      --paper-colors '0.85,0.15,0.12;0.10,0.45,0.85;0.18,0.65,0.25;0.78,0.48,0.08'
fi

srun --export=ALL --ntasks=1 python3 scripts/merge_remediation_datasets.py \
  --base "$BASE_DATA" --extra "$RAW" --out "$TABLE_DATA"

echo '[stage table background] fine-tuning the saved mixed checkpoint'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model "$CHECKPOINT" --data "$TABLE_DATA/data.yaml" --eval-data "$EVAL_DATA" \
  --out "$ROOT/table_background_ft" --epochs 20 --batch 8 --imgsz 640

# Stable, explicitly named copies make the hand-off between stages obvious.
cp -p "$ROOT/synthetic_ft/weights/best.pt" "$ROOT/checkpoints/synthetic_ft_best.pt"
cp -p "$ROOT/synthetic_plus_real_ft/weights/best.pt" "$ROOT/checkpoints/synthetic_plus_real_ft_best.pt"
cp -p "$ROOT/table_background_ft/weights/best.pt" "$ROOT/checkpoints/table_background_ft_best.pt"
echo '[complete] checkpoints:'
ls -lh "$ROOT/checkpoints"/*.pt
