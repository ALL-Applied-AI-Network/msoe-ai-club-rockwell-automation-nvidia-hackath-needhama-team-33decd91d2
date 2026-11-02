#!/usr/bin/env bash
#SBATCH --job-name=physical_stack_smoke
#SBATCH --partition=teaching
#SBATCH --exclude=dh-node11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/physical-stack-smoke-%j.out
#SBATCH --error=logs/physical-stack-smoke-%j.err

# Non-contact deployment preflight. This loads and infers with the exact
# one-T4 YOLO + LocateAnything stack but never imports or contacts the robot.
set -euo pipefail
umask 002
cd "${SLURM_SUBMIT_DIR:-$PWD}"

python_bin="${PHYSICAL_POLICY_PYTHON:-$PWD/shared/cache/locateanything_3b/venv/bin/python}"
hf_home="${LOCATEANYTHING_HF_HOME:-$PWD/shared/cache/locateanything_3b/huggingface}"
checkpoint="${YOLO_CHECKPOINT:-$PWD/shared/yolo26x/best.pt}"
reference_image="${DEPLOYMENT_SMOKE_IMAGE:-$PWD/compliant_remediation_test/perception/current_scene_tablecam.png}"
run_name="$(date +%F)_physical_stack_smoke_${SLURM_JOB_ID}"
out="${DEPLOYMENT_SMOKE_OUT:-$PWD/shared/runs/$run_name}"

for path in "$python_bin" "$checkpoint" "$reference_image"; do
    if [[ ! -e "$path" ]]; then
        echo "Required deployment asset is missing: $path" >&2
        exit 2
    fi
done
model_cache="$hf_home/hub/models--nvidia--LocateAnything-3B"
if [[ ! -d "$model_cache" ]]; then
    echo "Shared LocateAnything cache is incomplete: $model_cache" >&2
    exit 2
fi
mkdir -p logs "$out"
export HF_HOME="$hf_home"
export HF_HUB_CACHE="$hf_home/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
if [[ "${gpu_name^^}" != *T4* ]]; then
    echo "Expected a Tesla T4 allocation; received: $gpu_name" >&2
    exit 2
fi
echo "[physical-stack-smoke] host=$(hostname) gpu=$gpu_name"
echo "[physical-stack-smoke] output=$out"
echo "[physical-stack-smoke] no robot connection will be attempted"

exec "$python_bin" scripts/tools/smoke_test_physical_deployment.py \
    --checkpoint "$checkpoint" \
    --reference-image "$reference_image" \
    --out "$out" \
    --device 0
