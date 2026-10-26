#!/usr/bin/env bash
# Train the higher-capacity detector on the corrected deployment simulation.
# Incoming real data is intentionally excluded until a reviewed test split is frozen.
#SBATCH --job-name=yolo11m-pink-scene
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/yolo11m-pink-scene-%j.out
#SBATCH --error=logs/yolo11m-pink-scene-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/yolo11m-pink-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics<9' 'matplotlib' 'pillow' 'pyyaml'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

ROOT="runs/remediation_scene_finetune"
DATA="$ROOT/remediation_cad_true_lighting_pink_1000"
BASELINE="shared/inspection_models/remediation_yolo_best.pt"
MODEL_OUT="$ROOT/yolo11m_pink_scene_ft"
COMPARE_OUT="$ROOT/yolo11m_pink_scene_comparison"
SAMPLES_OUT="$ROOT/yolo11m_pink_scene_samples"

test -f "$DATA/data.yaml"
test -f "$BASELINE"

echo '[train] YOLO11m on 800 corrected pink-paper deployment scenes'
srun --export=ALL --ntasks=1 python3 scripts/finetune_remediation_detector.py \
  --model yolo11m.pt \
  --data "$DATA/data.yaml" \
  --eval-data "$DATA/data.yaml" \
  --out "$MODEL_OUT" \
  --epochs 50 --batch 32 --imgsz 640 --device 0,1,2,3

echo '[evaluate] untouched 100-image synthetic test and current hand-reviewed real tests'
srun --export=ALL --ntasks=1 python3 scripts/evaluate_remediation_comparison.py \
  --models \
    "yolo11n_baseline=$BASELINE" \
    "yolo11m_pink_scene=$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$DATA/data.yaml" \
  --synthetic-split test \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$COMPARE_OUT"

echo '[render] comparable predictions on real and held-out synthetic frames'
srun --export=ALL --ntasks=1 python3 scripts/render_remediation_prediction_samples.py \
  --models \
    "yolo11n_baseline=$BASELINE" \
    "yolo11m_pink_scene=$MODEL_OUT/weights/best.pt" \
  --synthetic-data "$DATA/data.yaml" \
  --synthetic-split test \
  --synthetic-count 4 \
  --hand-dir images/Hand_Samples \
  --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json \
  --out "$SAMPLES_OUT"

echo '[complete] checkpoint and reports:'
ls -lh "$MODEL_OUT/weights/best.pt"
echo "$COMPARE_OUT/remediation_model_comparison.png"
echo "$SAMPLES_OUT"
