#!/usr/bin/env bash
# Generate 800 train + 100 validation + 100 held-out deployment-scene samples.
# Override resources for a pilot, e.g. sbatch --gres=gpu:1 --ntasks=1 ... OUT 4 1 4 0
#SBATCH --job-name=remediation-scene-data
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/remediation-scene-data-%j.out
#SBATCH --error=logs/remediation-scene-data-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune

OUT="${1:-runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000}"
SCENES="${2:-1000}"
SHARDS="${3:-4}"
TRAIN_COUNT="${4:-800}"
VAL_COUNT="${5:-100}"
WORKERS="${6:-$SHARDS}"
SHARD_OFFSET="${7:-0}"
LIGHTING_PRESET="${8:-physical}"
PINK_TABLE_PAPER="${9:-1}"
FINALIZE="${10:-1}"
DOMAIN_RANDOMIZATION="${11:-1}"

echo "[generation] host=$(hostname) job=${SLURM_JOB_ID:-unknown} out=$OUT scenes=$SCENES shards=$SHARDS"
nvidia-smi

export REMEDIATION_DATASET_OUT="$OUT"
export REMEDIATION_DATASET_SCENES="$SCENES"
export REMEDIATION_DATASET_SHARDS="$SHARDS"
export REMEDIATION_TRAIN_COUNT="$TRAIN_COUNT"
export REMEDIATION_VAL_COUNT="$VAL_COUNT"
export REMEDIATION_SHARD_OFFSET="$SHARD_OFFSET"
export REMEDIATION_LIGHTING_PRESET="$LIGHTING_PRESET"
export REMEDIATION_PINK_TABLE_PAPER="$PINK_TABLE_PAPER"
export REMEDIATION_DOMAIN_RANDOMIZATION="$DOMAIN_RANDOMIZATION"

srun --ntasks="$WORKERS" --cpus-per-task=6 --gpu-bind=single:1 \
  bash scripts/run_deployment_remediation_generator_worker.sh

if [[ "$FINALIZE" == "1" ]]; then
  srun --ntasks=1 --cpus-per-task=2 python3 scripts/finalize_deployment_remediation_dataset.py \
    --out "$OUT" --scenes "$SCENES" --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" --num-shards "$SHARDS"
  echo "[generation] complete: $OUT/data.yaml"
else
  echo "[generation] worker shard complete; finalization deferred"
fi
