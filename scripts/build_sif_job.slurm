#!/usr/bin/env bash
#SBATCH --job-name=build_isaac_sif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_sif-%j.out
#SBATCH --error=logs/build_sif-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs

TEAM_BASE="/data/ai_club/team_14_2026"
SHARED_CACHE="${TEAM_BASE}/cache/singularity"
SHARED_TMP="${TEAM_BASE}/cache/tmp"
mkdir -p "$SHARED_CACHE" "$SHARED_TMP"

export SINGULARITY_CACHEDIR="$SHARED_CACHE"
export SINGULARITY_TMPDIR="$SHARED_TMP"

OUTPUT_SIF="${TEAM_BASE}/isaac-sim.sif"
SOURCE_IMAGE="${ISAAC_IMAGE:-docker://nvcr.io/nvidia/isaac-sim:4.5.0}"

echo "Building SIF container on $(hostname)..."
echo "Source: $SOURCE_IMAGE"
echo "Output: $OUTPUT_SIF"
echo "Cache:  $SINGULARITY_CACHEDIR"

# Build SIF image
singularity build --force "$OUTPUT_SIF" "$SOURCE_IMAGE"

# Ensure group ownership and permissions so the whole team can execute it
echo "Setting permissions on $OUTPUT_SIF..."
chgrp ai_club "$OUTPUT_SIF" 2>/dev/null || true
chmod 775 "$OUTPUT_SIF" 2>/dev/null || true

echo "SIF build completed successfully at: $OUTPUT_SIF"

