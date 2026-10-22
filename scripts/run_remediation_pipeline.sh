#!/usr/bin/env bash
# Submit generation, then training only after the dataset qualification succeeds.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
OUT_DIR="${1:?Usage: run_remediation_pipeline.sh runs/archive/YYYY-MM-DD_task [scenes]}"
SCENES="${2:-100}"
mkdir -p logs
if [[ -e "$OUT_DIR" ]]; then
    echo "Choose a fresh output directory: $OUT_DIR" >&2
    exit 1
fi
GEN_JOB=$(sbatch --parsable scripts/submit_isaac_job.slurm src/data_generation/generate_remediation_dataset.py \
    --out "$OUT_DIR/dataset" --scenes "$SCENES")
GEN_JOB="${GEN_JOB%%;*}"
TRAIN_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$GEN_JOB" scripts/submit_training_job.slurm \
    src/inspection/train_remediation_detector.py --data "$OUT_DIR/dataset/data.yaml" --out "$OUT_DIR/training")
echo "Generation job: $GEN_JOB"
echo "Dependent training job: $TRAIN_JOB"
echo "Review qualification manifests and evaluation.json before using the checkpoint."
