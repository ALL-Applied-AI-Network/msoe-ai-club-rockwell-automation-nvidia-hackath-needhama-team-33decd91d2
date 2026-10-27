#!/usr/bin/env bash
# Fine-tune the qualified-scene YOLO26l on all five reviewed real frames.
# The same five frames are evaluated afterward; those metrics are resubstitution.
#SBATCH --job-name=y26l-real5-ft
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/y26l-real5-ft-%j.out
#SBATCH --error=logs/y26l-real5-ft-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune

PKG_DIR="/tmp/y26l-real5-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

ROOT="runs/remediation_scene_finetune"
SYNTHETIC="$ROOT/remediation_black_silver_physical_pink_1000/data.yaml"
BASE_MODEL="$ROOT/yolo26l_black_silver_physical_pink_960_ft/weights/best.pt"
REAL_DATA="$ROOT/real_finetune_5/data.yaml"
ANNOTATIONS="images/Real_Finetune/five_image_annotations.json"
MODEL_OUT="$ROOT/yolo26l_real5_960_ft"
BASE_EVAL="$ROOT/eval_yolo26l_real5_baseline"
FINAL_EVAL="$ROOT/eval_yolo26l_real5_finetuned"

test -f "$BASE_MODEL"
test -f "$SYNTHETIC"
test ! -e "$MODEL_OUT"

echo "[prepare] regenerate approved labels and five-frame YOLO dataset"
srun --export=ALL --ntasks=1 python3 scripts/build_real_finetune_annotations.py
srun --export=ALL --ntasks=1 python3 scripts/build_real5_yolo_dataset.py

echo "[baseline] evaluate qualified-scene checkpoint on synthetic test and all five real frames"
srun --export=ALL --ntasks=1 python3 scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$BASE_MODEL" \
  --synthetic-data "$SYNTHETIC" \
  --annotations "$ANNOTATIONS" \
  --out "$BASE_EVAL" \
  --imgsz 960 \
  --title 'YOLO26l baseline — five reviewed physical frames' \
  --limitation 'All five real frames are reused for fine-tuning and evaluation; real metrics measure training-set fit, not generalization.' \
  --training-state '{"stage":"before_real5_finetune","base":"qualified_black_silver_synthetic"}' \
  --report-name yolo26l_real5_baseline_eval_report.png

echo "[train] conservative real-domain fine-tune on five frames"
srun --export=ALL --ntasks=1 python3 scripts/finetune_real5_detector.py \
  --model "$BASE_MODEL" \
  --data "$REAL_DATA" \
  --out "$MODEL_OUT" \
  --epochs 120 \
  --imgsz 960

echo "[evaluate] fine-tuned checkpoint on synthetic test and all five training frames"
srun --export=ALL --ntasks=1 python3 scripts/evaluate_partial_yolo26l_289047.py \
  --checkpoint "$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$SYNTHETIC" \
  --annotations "$ANNOTATIONS" \
  --out "$FINAL_EVAL" \
  --imgsz 960 \
  --title 'YOLO26l real-five fine-tune — synthetic retention and physical training fit' \
  --limitation 'All five real frames were used for both fine-tuning and evaluation; real metrics are resubstitution performance and do not estimate generalization.' \
  --training-state '{"stage":"after_real5_finetune","epochs_requested":120,"base":"qualified_black_silver_synthetic","real_frames":5,"train_equals_eval":true}' \
  --report-name yolo26l_real5_finetuned_eval_report.png

echo "[complete]"
ls -lh "$MODEL_OUT/weights/best.pt"
echo "$BASE_EVAL/yolo26l_real5_baseline_eval_report.png"
echo "$FINAL_EVAL/yolo26l_real5_finetuned_eval_report.png"
