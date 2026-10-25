#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

RUN_DIR="runs/archive/2026-10-24_190844_full_remediation"

echo "=========================================================="
echo "Resuming Full Remediation Pipeline from Sample 88"
echo "Output Directory: $RUN_DIR"
echo "=========================================================="

# 1. Resume Dataset Generation (88 of 100 already complete)
echo "-> Resuming Dataset Generation (12 remaining scenes)..."
GEN_JOB=$(sbatch --parsable scripts/submit_isaac_job.slurm src/data_generation/generate_remediation_dataset.py \
    --out "$RUN_DIR/yolo/dataset" --scenes 100)
GEN_JOB="${GEN_JOB%%;*}"
echo "   [Job ID] Dataset Generation: $GEN_JOB"

# 2. YOLO Detector Training (Depends on dataset completion)
echo "-> Submitting YOLO Detector Training (50 epochs)..."
TRAIN_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$GEN_JOB" scripts/submit_training_job.slurm \
    src/inspection/train_remediation_detector.py --data "$RUN_DIR/yolo/dataset/data.yaml" --out "$RUN_DIR/yolo/training" --epochs 50)
TRAIN_JOB="${TRAIN_JOB%%;*}"
echo "   [Job ID] Detector Training: $TRAIN_JOB"

# 3. Closed-Loop Remediation Pipeline & Video Encoding (Depends on training completion)
echo "-> Submitting Closed-Loop Remediation Pipeline..."
LOOP_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$TRAIN_JOB" scripts/run_closed_loop_pipeline.slurm \
    "$RUN_DIR/qualification/scene.usda" \
    "$RUN_DIR/yolo/training/weights/best.pt" \
    "$RUN_DIR/closed_loop")
LOOP_JOB="${LOOP_JOB%%;*}"
echo "   [Job ID] Closed-Loop Pipeline: $LOOP_JOB"

echo "=========================================================="
echo "Resumed Pipeline DAG scheduled on Rosie Slurm:"
echo "  Step 1 (Generate remaining 12 scenes): $GEN_JOB"
echo "  Step 2 (Train 50 epochs):              $TRAIN_JOB (after $GEN_JOB)"
echo "  Step 3 (Closed Loop & Video Encode):   $LOOP_JOB (after $TRAIN_JOB)"
echo ""
echo "Final Video Target: $RUN_DIR/closed_loop/tablecam_yolo_per_frame.mp4"
echo "=========================================================="

