#!/usr/bin/env bash
set -euo pipefail

# Unified runner for Isaac Sim across Brev, Local machines, and Cluster (Singularity)
# Usage:
#   ./scripts/run_isaac.sh <script.py> [args...]
#
# Examples:
#   ./scripts/run_isaac.sh shared/data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --out runs/fill.png

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

# 1. Brev / Native Host Mode: Check for standard Isaac Sim Python install paths
CANDIDATE_PATHS=(
    "${ISAAC:-}"
    "/isaac-sim"
    "${HOME}/.local/share/ov/pkg/isaac-sim-4.5.0"
    "${HOME}/.local/share/ov/pkg/isaac-sim-6.0.0"
)

FOUND_ISAAC=""
for p in "${CANDIDATE_PATHS[@]}"; do
    if [ -n "$p" ] && [ -x "$p/python.sh" ]; then
        FOUND_ISAAC="$p"
        break
    fi
done

if [ -n "$FOUND_ISAAC" ]; then
    echo "[run_isaac] Detected native Isaac Sim at: $FOUND_ISAAC"
    export ISAAC="$FOUND_ISAAC"
    exec "$FOUND_ISAAC/python.sh" "$@"
fi

# 2. Singularity / Cluster Mode
if command -v singularity >/dev/null 2>&1; then
    echo "[run_isaac] Native Isaac Sim not found. Using Singularity on cluster..."

    # Configure shared cache directory
    if [ -d "$REPO_ROOT/shared/cache" ]; then
        SHARED_CACHE="$REPO_ROOT/shared/cache/singularity"
        SHARED_TMP="$REPO_ROOT/shared/cache/tmp"
        mkdir -p "$SHARED_CACHE" "$SHARED_TMP"
        export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$SHARED_CACHE}"
        export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$SHARED_TMP}"
    elif [ -d "/data/scratch/$USER" ]; then
        export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/data/scratch/$USER/singularity_cache}"
        export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/data/scratch/$USER/singularity_tmp}"
        mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
    fi

    # Locate container image
    SHARED_SIF="$REPO_ROOT/shared/isaac-sim.sif"
    LOCAL_SIF="$REPO_ROOT/isaac-sim.sif"
    if [ -n "${ISAAC_SIF:-}" ] && [ -f "$ISAAC_SIF" ]; then
        IMAGE_TARGET="$ISAAC_SIF"
    elif [ -f "$SHARED_SIF" ]; then
        IMAGE_TARGET="$SHARED_SIF"
    elif [ -f "$LOCAL_SIF" ]; then
        IMAGE_TARGET="$LOCAL_SIF"
    else
        IMAGE_TARGET="${ISAAC_IMAGE:-docker://nvcr.io/nvidia/isaac-sim:4.5.0}"
    fi

    echo "[run_isaac] Target container: $IMAGE_TARGET"

    # Set up bind mounts (repo root and /data shared storage if present)
    BINDS="$REPO_ROOT:/workspace"
    if [ -d "/data" ]; then
        BINDS="$BINDS,/data:/data"
    fi

    exec singularity exec --nv \
        --bind "$BINDS" \
        --pwd /workspace \
        "$IMAGE_TARGET" \
        /isaac-sim/python.sh "$@"
fi

echo "Error: Neither a native Isaac Sim installation (e.g. on Brev) nor Singularity was found." >&2
echo "Please set ISAAC=/path/to/isaacsim or run on a system with Singularity/Apptainer." >&2
exit 1
