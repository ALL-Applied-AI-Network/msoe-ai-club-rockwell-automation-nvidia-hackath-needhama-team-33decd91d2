#!/usr/bin/env bash
# Build an independent held-out synthetic test set, then evaluate every
# available remediation checkpoint on exactly that test split.
#SBATCH --job-name=remediation-synthetic-test
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/remediation-synthetic-test-%j.out
#SBATCH --error=logs/remediation-synthetic-test-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs/remediation_finetune/synthetic_test_eval

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/remediation-synthetic-test-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir 'torch' 'ultralytics<9'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
srun --export=ALL --ntasks=1 python3 -c 'import torch, ultralytics; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ultralytics", ultralytics.__version__)'

ROOT="runs/remediation_finetune"
TEST_DATA="$ROOT/full_synthetic_test_raw"
EVAL_ROOT="$ROOT/synthetic_test_eval"

if [[ ! -f "$TEST_DATA/data.yaml" ]]; then
    echo '[stage test data] generating independent 50-scene test source'
    srun --export=ALL --ntasks=1 ./scripts/run_isaac.sh \
      src/data_generation/generate_remediation_dataset.py \
      --out "$TEST_DATA" --scenes 50 --seed 7200 \
      --crooked 5 --missing 4 --table-caps 5 --fixed-inventory
fi

declare -A MODELS=(
  [baseline]="shared/inspection_models/remediation_yolo_best.pt"
  [synthetic]="$ROOT/synthetic_ft/weights/best.pt"
  [synthetic_plus_real]="$ROOT/synthetic_plus_real_ft/weights/best.pt"
  [table_background]="$ROOT/table_background_ft/weights/best.pt"
)

for name in baseline synthetic synthetic_plus_real table_background; do
    model="${MODELS[$name]}"
    if [[ ! -f "$model" ]]; then
        echo "Missing checkpoint: $model" >&2
        exit 2
    fi
    echo "[stage eval] $name"
    srun --export=ALL --ntasks=1 python3 scripts/evaluate_remediation_detector.py \
      --model "$model" --data "$TEST_DATA/data.yaml" --split test \
      --out "$EVAL_ROOT/$name.json" --imgsz 640
done

echo '[complete] synthetic test reports:'
find "$EVAL_ROOT" -maxdepth 1 -name '*.json' -print | sort
