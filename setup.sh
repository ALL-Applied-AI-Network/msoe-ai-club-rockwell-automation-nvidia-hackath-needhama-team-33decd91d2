#!/usr/bin/env bash
set -euo pipefail

# Determine repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure helper scripts have execution permissions
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh scripts/*.slurm 2>/dev/null || true
fi

# Initialize and update git submodules
if [ -f ".gitmodules" ] && command -v git >/dev/null 2>&1; then
    echo "Initializing and updating git submodules..."
    git submodule update --init --recursive
fi

# Detect environment:
# 1. School Cluster (Rosie): /data/ai_club exists or /data is writable
# 2. Standalone / Brev: local ./shared directory
if [ -d "/data/ai_club" ] || [ -w "/data" 2>/dev/null ]; then
    TEAM_REMOTE_DIR="/data/ai_club/team_14_2026"
    TARGET_DATA_DIR="${TEAM_REMOTE_DIR}/data"
    SHARED_CACHE_DIR="${TEAM_REMOTE_DIR}/cache"
    IS_CLUSTER=true

    mkdir -p "$TARGET_DATA_DIR" "$SHARED_CACHE_DIR" logs runs

    if [ "$(ls -A "$TARGET_DATA_DIR" 2>/dev/null)" ]; then
        echo "Shared directory '${TARGET_DATA_DIR}' already exists and is not empty. Skipping extraction."
    else
        if [ -f "data.zip" ]; then
            echo "Extracting data.zip to ${TEAM_REMOTE_DIR}..."
            unzip -q -o data.zip -d "$TEAM_REMOTE_DIR"
            echo "Extraction complete."
        else
            echo "Error: 'data.zip' not found in $SCRIPT_DIR and '${TARGET_DATA_DIR}' is empty." >&2
            exit 1
        fi
    fi

    # Create/update ./shared symlink -> /data/ai_club/team_14_2026
    if [ -L "shared" ] || [ ! -e "shared" ]; then
        rm -f shared team_14_2026 data 2>/dev/null || true
        echo "Creating symlink shared -> ${TEAM_REMOTE_DIR}..."
        ln -s "$TEAM_REMOTE_DIR" shared
    fi

    # Set group ownership and shared permissions for team members (top-level + setgid for auto-inheritance)
    chgrp ai_club "$TEAM_REMOTE_DIR" "$TARGET_DATA_DIR" "$SHARED_CACHE_DIR" 2>/dev/null || true
    chmod g+rwxs "$TEAM_REMOTE_DIR" "$TARGET_DATA_DIR" "$SHARED_CACHE_DIR" 2>/dev/null || true
else
    # Brev / Standalone: ./shared is a local folder containing data/, cache/, etc.
    IS_CLUSTER=false
    mkdir -p shared/data shared/cache logs runs

    if [ "$(ls -A shared/data 2>/dev/null)" ]; then
        echo "Directory 'shared/data/' already exists and is not empty. Skipping extraction."
    elif [ -f "data.zip" ]; then
        echo "Extracting data.zip into shared/..."
        unzip -q -o data.zip -d shared
        echo "Extraction complete."
    else
        echo "Warning: 'data.zip' not found in $SCRIPT_DIR and 'shared/data/' is empty."
    fi
fi

# Clean up any legacy team_14_2026 or data symlinks
rm -f team_14_2026 data 2>/dev/null || true

# Bootstrap a writable cache copy; organizer data must remain immutable.
FANUC_SOURCE="${SCRIPT_DIR}/shared/data/fanuc_er4ia_sim"
FANUC_DIR="${SCRIPT_DIR}/shared/cache/fanuc_runtime"
if [ -d "$FANUC_SOURCE" ] && [ -f "$FANUC_SOURCE/bootstrap.sh" ]; then
    mkdir -p "$FANUC_DIR"
    rsync -a --chmod=u+rwX --exclude='.venv' --exclude='__pycache__' \
        --exclude='runs' --exclude='textures' "$FANUC_SOURCE/" "$FANUC_DIR/"
    echo "Preparing FANUC host tools in writable cache: $FANUC_DIR"
    (cd "$FANUC_DIR" && bash bootstrap.sh)
fi

# Ensure Isaac Sim 6.1 standalone is available
if [ -f "scripts/install_isaac_standalone.sh" ]; then
    if [ -x "shared/isaacsim_6.1/python.sh" ] || [ -x "${HOME}/isaacsim/python.sh" ]; then
        echo "Isaac Sim 6.1 is already installed."
    else
        echo "Installing Isaac Sim 6.1 standalone into ./shared/isaacsim_6.1..."
        ./scripts/install_isaac_standalone.sh
    fi
fi

# Scene construction is an explicit Slurm job, never a login-node side effect.
echo "Build/qualify remediation scenes with scripts/submit_isaac_job.slurm."

echo "Setup finished successfully. All data and assets accessible at ./shared/data/"
