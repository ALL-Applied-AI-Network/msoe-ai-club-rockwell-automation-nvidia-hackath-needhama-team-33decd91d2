#!/usr/bin/env bash
#SBATCH --job-name=physical_remediation
#SBATCH --partition=teaching
#SBATCH --exclude=dh-node11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/physical-policy-%j.out
#SBATCH --error=logs/physical-policy-%j.err

# Slurm launcher for both the API-compatible mock and an approved physical-cell
# run. The policy defaults to non-contact hover mode. Real contact still needs
# explicit --mode full --allow-provisional-contact arguments from the operator.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${PHYSICAL_POLICY_PYTHON:-}" ]]; then
    policy_python="$PHYSICAL_POLICY_PYTHON"
elif [[ -x ./.conda_env/bin/python ]]; then
    policy_python=./.conda_env/bin/python
elif [[ -x ./shared/cache/locateanything_3b/venv/bin/python ]]; then
    policy_python=./shared/cache/locateanything_3b/venv/bin/python
else
    echo "Physical policy Python not found; set PHYSICAL_POLICY_PYTHON or provision the shared HPC venv" >&2
    exit 2
fi

# LocateAnything is a team deployment dependency, not a per-user desktop
# cache.  Keep the override explicit for maintenance while making the shared
# model/cache the portable default for every operator checkout.
export HF_HOME="${LOCATEANYTHING_HF_HOME:-$PWD/shared/cache/locateanything_3b/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
if [[ ! -d "${HF_HUB_CACHE}/models--nvidia--LocateAnything-3B" ]]; then
    echo "Shared LocateAnything checkpoint is incomplete: ${HF_HUB_CACHE}" >&2
    exit 2
fi

echo "[physical-policy] host: $(hostname)"
echo "[physical-policy] Python: $policy_python"
echo "[physical-policy] LocateAnything cache: $HF_HOME"
nvidia-smi --query-gpu=name --format=csv,noheader
exec "$policy_python" PhysicalAI-Hackathon-2026/run.py "$@"
