#!/usr/bin/env bash
# Continue the preserved corrected-scene YOLO26l checkpoint on the broader
# deployment domain-randomized synthetic set. Evaluations keep both stages.
#SBATCH --job-name=yolo26l-domain-960
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/yolo26l-domain-960-%j.out
#SBATCH --error=logs/yolo26l-domain-960-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/remediation_scene_finetune

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
nvidia-smi

PKG_DIR="/tmp/yolo26l-domain-${SLURM_JOB_ID:-manual}/packages"
mkdir -p "$PKG_DIR"
if ! python3 -c 'import torch, ultralytics, matplotlib, PIL, yaml, cloudpickle' >/dev/null 2>&1; then
    python3 -m pip install --target "$PKG_DIR" --no-cache-dir \
      'torch' 'ultralytics>=8.4,<9' 'matplotlib' 'pillow' 'pyyaml' 'cloudpickle>=3.1.1'
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"

ROOT="runs/remediation_scene_finetune"
BASE_DATA="$ROOT/remediation_cad_true_lighting_pink_1000"
DOMAIN_DATA="$ROOT/remediation_cad_domain_randomized_1000"
BASE_MODEL="$ROOT/yolo26l_pink_scene_960_ft/weights/best.pt"
MODEL_OUT="$ROOT/yolo26l_domain_randomized_960_ft"
BASE_EVAL_OUT="$ROOT/yolo26l_domain_randomized_960_base_test"
DOMAIN_EVAL_OUT="$ROOT/yolo26l_domain_randomized_960_domain_test"
SAMPLES_OUT="$ROOT/yolo26l_domain_randomized_960_samples"
ORIGINAL_BASELINE="shared/inspection_models/remediation_yolo_best.pt"

test -f "$BASE_DATA/data.yaml"
test -f "$DOMAIN_DATA/data.yaml"
test -f "$BASE_MODEL"
test -f "$ORIGINAL_BASELINE"

echo '[train] continuing corrected-scene YOLO26l on 800 domain-randomized scenes'
srun --export=ALL --ntasks=1 python3 scripts/training/finetune_remediation_detector.py \
  --model "$BASE_MODEL" \
  --data "$DOMAIN_DATA/data.yaml" \
  --eval-data "$DOMAIN_DATA/data.yaml" \
  --out "$MODEL_OUT" \
  --epochs 30 --batch 4 --imgsz 960 --device 0

MODELS=(
  "yolo11n_original=$ORIGINAL_BASELINE"
  "yolo26l_corrected_scene=$BASE_MODEL"
  "yolo26l_domain_randomized=$MODEL_OUT/weights/best.pt"
)

echo '[evaluate] all preserved stages on the fixed corrected-scene synthetic test'
srun --export=ALL --ntasks=1 python3 scripts/training/evaluate_remediation_comparison.py \
  --models "${MODELS[@]}" \
  --synthetic-data "$BASE_DATA/data.yaml" --synthetic-split test --imgsz 960 \
  --hand-dir images/Hand_Samples --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json --out "$BASE_EVAL_OUT"

echo '[evaluate] all preserved stages on the fixed domain-randomized synthetic test'
srun --export=ALL --ntasks=1 python3 scripts/training/evaluate_remediation_comparison.py \
  --models "${MODELS[@]}" \
  --synthetic-data "$DOMAIN_DATA/data.yaml" --synthetic-split test --imgsz 960 \
  --hand-dir images/Hand_Samples --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json --out "$DOMAIN_EVAL_OUT"

echo '[render] prediction audit for the domain-randomized test and real samples'
srun --export=ALL --ntasks=1 python3 scripts/training/render_remediation_prediction_samples.py \
  --models "${MODELS[@]}" \
  --synthetic-data "$DOMAIN_DATA/data.yaml" --synthetic-split test \
  --synthetic-count 4 --imgsz 960 \
  --hand-dir images/Hand_Samples --image-dir images/pruned_images \
  --apertures src/inspection/config/fixture_g_apertures.json --out "$SAMPLES_OUT"

echo '[complete] preserved checkpoints:'
ls -lh "$BASE_MODEL" "$MODEL_OUT/weights/best.pt"
echo "$BASE_EVAL_OUT/remediation_model_comparison.png"
echo "$DOMAIN_EVAL_OUT/remediation_model_comparison.png"
echo "$SAMPLES_OUT"
