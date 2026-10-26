#!/usr/bin/env bash
# Accuracy-first detector for 1080p remediation inspection frames.
# Uses 960px input so cap-state details survive preprocessing.
#SBATCH --job-name=yolo26l-pink-960
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/yolo26l-pink-960-%j.out
#SBATCH --error=logs/yolo26l-pink-960-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/yolo26l-pink-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

ROOT="runs/remediation_scene_finetune"
DATA="$ROOT/remediation_cad_true_lighting_pink_1000"
BASELINE="shared/inspection_models/remediation_yolo_best.pt"
MODEL_OUT="$ROOT/yolo26l_pink_scene_960_ft"
COMPARE_OUT="$ROOT/yolo26l_pink_scene_960_comparison"
SAMPLES_OUT="$ROOT/yolo26l_pink_scene_960_samples"

test -f "$DATA/data.yaml"
test -f "$BASELINE"

echo '[train] YOLO26l at 960px on 800 corrected deployment scenes'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model yolo26l.pt \
  --data "$DATA/data.yaml" \
  --eval-data "$DATA/data.yaml" \
  --out "$MODEL_OUT" \
  --epochs 50 --batch 4 --imgsz 960 --device 0

echo '[evaluate] both models at 960px on identical synthetic and real tests'
srun --export=ALL --ntasks=1 python3 scripts/evaluate_remediation_comparison.py \
  --models \
    "yolo11n_baseline=$BASELINE" \
    "yolo26l_pink_960=$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$DATA/data.yaml" \
  --synthetic-split test \
  --imgsz 960 \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$COMPARE_OUT"

echo '[render] prediction audit at deployment resolution'
srun --export=ALL --ntasks=1 python3 scripts/render_remediation_prediction_samples.py \
  --models \
    "yolo11n_baseline=$BASELINE" \
    "yolo26l_pink_960=$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$DATA/data.yaml" \
  --synthetic-split test \
  --synthetic-count 4 \
  --imgsz 960 \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$SAMPLES_OUT"

echo '[complete] checkpoint and reports:'
ls -lh "$MODEL_OUT/weights/best.pt"
echo "$COMPARE_OUT/remediation_model_comparison.png"
echo "$SAMPLES_OUT"
