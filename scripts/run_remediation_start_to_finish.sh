#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATE_TAG="$(date +%Y-%m-%d)"
TIMESTAMP="$(date +%H%M%S)"
OUT_DIR="${1:-runs/archive/${DATE_TAG}_${TIMESTAMP}_full_remediation}"
SCENES="${2:-100}"
EPOCHS="${3:-50}"

mkdir -p logs runs "$OUT_DIR"

echo "=========================================================="
echo "Submitting Canonical Full Remediation Pipeline (Runbook Spec)"
echo "Output Directory: $OUT_DIR"
echo "Dataset Scenes:   $SCENES"
echo "Training Epochs:  $EPOCHS"
echo "=========================================================="

# 1. Deterministic Scene Qualification (Guarantees 1 crooked, 1 missing, 1 table cap)
echo "-> Step 1: Submitting Deterministic Scene Qualification..."
QUAL_JOB=$(sbatch --parsable scripts/submit_isaac_job.slurm src/remediation/qualify_scene.py \
    --out "$OUT_DIR/qualification" \
    --seed 20261022 \
    --crooked 1 --missing 1 --table-caps 1 \
    --steps 360 --fixed)
QUAL_JOB="${QUAL_JOB%%;*}"
echo "   [Job ID] Scene Qualification: $QUAL_JOB"

# 2. Synthetic Dataset Generation (100 physics-settled scenes for YOLO training)
echo "-> Step 2: Submitting Synthetic Dataset Generation ($SCENES scenes)..."
GEN_JOB=$(sbatch --parsable scripts/submit_isaac_job.slurm src/data_generation/generate_remediation_dataset.py \
    --out "$OUT_DIR/yolo/dataset" --scenes "$SCENES")
GEN_JOB="${GEN_JOB%%;*}"
echo "   [Job ID] Dataset Generation: $GEN_JOB"

# 3. YOLO 4-Class Training (Depends on Step 2 completion)
echo "-> Step 3: Submitting YOLO Detector Training ($EPOCHS epochs)..."
TRAIN_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$GEN_JOB" scripts/submit_training_job.slurm \
    src/inspection/train_remediation_detector.py --data "$OUT_DIR/yolo/dataset/data.yaml" --out "$OUT_DIR/yolo/training" --epochs "$EPOCHS")
TRAIN_JOB="${TRAIN_JOB%%;*}"
echo "   [Job ID] Detector Training: $TRAIN_JOB"

# 4. Closed-Loop Remediation Pipeline (Depends on Step 1 and Step 3 completion)
echo "-> Step 4: Submitting Closed-Loop Remediation Pipeline..."
LOOP_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$QUAL_JOB:$TRAIN_JOB" scripts/run_closed_loop_pipeline.slurm \
    "$OUT_DIR/qualification/scene.usda" \
    "$OUT_DIR/yolo/training/weights/best.pt" \
    "$OUT_DIR/closed_loop")
LOOP_JOB="${LOOP_JOB%%;*}"
echo "   [Job ID] Closed-Loop Pipeline: $LOOP_JOB"

echo "=========================================================="
echo "Full Remediation Pipeline DAG scheduled in Slurm:"
echo "  Scene Qualification: $QUAL_JOB"
echo "  Dataset Generation:  $GEN_JOB"
echo "  YOLO Training:       $TRAIN_JOB (after $GEN_JOB)"
echo "  Closed-Loop Run:     $LOOP_JOB (after $QUAL_JOB and $TRAIN_JOB)"
echo ""
echo "Output Directory:      $OUT_DIR"
echo "Final Review Video:    $OUT_DIR/closed_loop/tablecam_yolo_per_frame.mp4"
echo "=========================================================="
