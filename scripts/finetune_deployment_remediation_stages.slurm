#!/usr/bin/env bash
# Baseline -> corrected remediation simulation -> simulation + current real data.
# All final comparisons use the untouched 100-image CAD simulation test split.
#SBATCH --job-name=remediation-cad-ft-eval
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/remediation-cad-ft-eval-%j.out
#SBATCH --error=logs/remediation-cad-ft-eval-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/remediation-cad-ft-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics<9' 'matplotlib' 'pillow' 'pyyaml'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
srun --export=ALL --ntasks=1 python3 -c \
  'import torch, ultralytics; print("torch", torch.__version__, "gpus", torch.cuda.device_count(), "ultralytics", ultralytics.__version__)'

ROOT="runs/remediation_scene_finetune"
SIM_BALANCED="$ROOT/remediation_cad_1000"
SIM_TRUE_LIGHT="$ROOT/remediation_cad_true_lighting_pink_1000"
SIM_DATA="$ROOT/remediation_cad_dual_lighting"
REAL_DATA="runs/remediation_finetune/real_dataset"
OTHER_REAL_DATA="${OTHER_REAL_DATASET:-runs/remediation_finetune/other_arm_real_dataset}"
MIXED_DATA="$ROOT/remediation_cad_plus_real"
BASELINE="shared/inspection_models/remediation_yolo_best.pt"
MEDIUM_PRETRAINED="${MEDIUM_PRETRAINED:-yolo11m.pt}"
NANO_SIM_MODEL="$ROOT/yolo11n_cad_simulation_ft"
NANO_MIXED_MODEL="$ROOT/yolo11n_cad_simulation_plus_real_ft"
MEDIUM_SIM_MODEL="$ROOT/yolo11m_cad_simulation_ft"
MEDIUM_MIXED_MODEL="$ROOT/yolo11m_cad_simulation_plus_real_ft"
COMPARE_OUT="$ROOT/model_comparison"
SAMPLES_OUT="$ROOT/prediction_samples"

test -f "$SIM_BALANCED/data.yaml"
test -f "$SIM_TRUE_LIGHT/data.yaml"
test -f "$BASELINE"
test -f "$REAL_DATA/data.yaml"

echo '[dataset] merging 1,600 CAD simulation frames across balanced and true-lighting domains'
srun --export=ALL --ntasks=1 python3 scripts/merge_multiple_remediation_datasets.py \
  --dataset "balanced=$SIM_BALANCED" \
  --dataset "true_light_pink=$SIM_TRUE_LIGHT" \
  --out "$SIM_DATA"

echo '[dataset] merging both CAD lighting domains with available real-arm datasets'
MERGE_DATASETS=(--dataset "cad_sim=$SIM_DATA" --dataset "real_current=$REAL_DATA")
if [[ -f "$OTHER_REAL_DATA/data.yaml" ]]; then
    echo "[dataset] including additional other-arm dataset: $OTHER_REAL_DATA"
    MERGE_DATASETS+=(--dataset "real_other_arm=$OTHER_REAL_DATA")
else
    echo "[dataset] other-arm dataset is not present yet: $OTHER_REAL_DATA"
fi
srun --export=ALL --ntasks=1 python3 scripts/merge_multiple_remediation_datasets.py \
  "${MERGE_DATASETS[@]}" --out "$MIXED_DATA"

echo '[stage 1/5] nano baseline is preserved and will be evaluated unchanged'
echo '[stage 2/5] fine-tuning the nano baseline on both corrected CAD lighting domains'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model "$BASELINE" \
  --data "$SIM_DATA/data.yaml" \
  --eval-data "$SIM_DATA/data.yaml" \
  --out "$NANO_SIM_MODEL" \
  --epochs 30 --batch 32 --imgsz 640 --device 0,1,2,3

echo '[stage 3/5] fine-tuning the saved nano checkpoint on simulation + real labels'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model "$NANO_SIM_MODEL/weights/best.pt" \
  --data "$MIXED_DATA/data.yaml" \
  --eval-data "$SIM_DATA/data.yaml" \
  --out "$NANO_MIXED_MODEL" \
  --epochs 30 --batch 32 --imgsz 640 --device 0,1,2,3

echo '[stage 4/5] fine-tuning a higher-capacity YOLO11m from its pretrained checkpoint'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model "$MEDIUM_PRETRAINED" \
  --data "$SIM_DATA/data.yaml" \
  --eval-data "$SIM_DATA/data.yaml" \
  --out "$MEDIUM_SIM_MODEL" \
  --epochs 30 --batch 32 --imgsz 640 --device 0,1,2,3

echo '[stage 5/5] fine-tuning the saved medium checkpoint on simulation + real labels'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model "$MEDIUM_SIM_MODEL/weights/best.pt" \
  --data "$MIXED_DATA/data.yaml" \
  --eval-data "$SIM_DATA/data.yaml" \
  --out "$MEDIUM_MIXED_MODEL" \
  --epochs 30 --batch 32 --imgsz 640 --device 0,1,2,3

echo '[evaluation] same held-out CAD simulation test and same two hand-reviewed real frames for every checkpoint'
srun --export=ALL --ntasks=1 python3 scripts/evaluate_remediation_comparison.py \
  --models \
    "baseline=$BASELINE" \
    "yolo11n_cad_simulation=$NANO_SIM_MODEL/weights/best.pt" \
    "yolo11n_cad_plus_real=$NANO_MIXED_MODEL/weights/best.pt" \
    "yolo11m_cad_simulation=$MEDIUM_SIM_MODEL/weights/best.pt" \
    "yolo11m_cad_plus_real=$MEDIUM_MIXED_MODEL/weights/best.pt" \
  --synthetic-data "$SIM_DATA/data.yaml" \
  --synthetic-split test \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$COMPARE_OUT"

echo '[render] prediction examples on the two real images and four held-out CAD simulation images'
srun --export=ALL --ntasks=1 python3 scripts/render_remediation_prediction_samples.py \
  --models \
    "baseline=$BASELINE" \
    "yolo11n_cad_simulation=$NANO_SIM_MODEL/weights/best.pt" \
    "yolo11n_cad_plus_real=$NANO_MIXED_MODEL/weights/best.pt" \
    "yolo11m_cad_simulation=$MEDIUM_SIM_MODEL/weights/best.pt" \
    "yolo11m_cad_plus_real=$MEDIUM_MIXED_MODEL/weights/best.pt" \
  --synthetic-data "$SIM_DATA/data.yaml" \
  --synthetic-split test \
  --synthetic-count 4 \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$SAMPLES_OUT"

echo '[complete] preserved checkpoints:'
ls -lh "$BASELINE" \
  "$NANO_SIM_MODEL/weights/best.pt" "$NANO_MIXED_MODEL/weights/best.pt" \
  "$MEDIUM_SIM_MODEL/weights/best.pt" "$MEDIUM_MIXED_MODEL/weights/best.pt"
echo "[complete] comparison: $COMPARE_OUT/remediation_model_comparison.png"
echo "[complete] samples: $SAMPLES_OUT"
