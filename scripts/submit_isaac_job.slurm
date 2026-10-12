#!/usr/bin/env bash
#SBATCH --job-name=isaac_sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

TEAM_BASE="/data/ai_club/team_14_2026"
# Configure shared team cache/temp so everyone on the team reuses downloaded layers
if [ -d "/data/ai_club" ] || [ -d "$TEAM_BASE" ]; then
    export SINGULARITY_CACHEDIR="${TEAM_BASE}/cache/singularity"
    export SINGULARITY_TMPDIR="${TEAM_BASE}/cache/tmp"
    mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
elif [ -d "/data/scratch/$USER" ]; then
    export SINGULARITY_CACHEDIR="/data/scratch/$USER/singularity_cache"
    export SINGULARITY_TMPDIR="/data/scratch/$USER/singularity_tmp"
    mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
fi

echo "Running on $(hostname) with GPU allocation:"
nvidia-smi

# Execute command passed to sbatch, or default to an example render
if [ "$#" -gt 0 ]; then
    ./scripts/run_isaac.sh "$@"
else
    echo "Running default CSI digital twin test render..."
    ./scripts/run_isaac.sh data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --frame 686 --out fill_test.png
fi
