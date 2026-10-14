#!/usr/bin/env bash
set -euo pipefail

# Determine repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure helper scripts have execution permissions
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh scripts/*.slurm 2>/dev/null || true
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

# Bootstrap FANUC Robot Arm Digital Twin (Twin 2)
FANUC_DIR="${SCRIPT_DIR}/shared/data/fanuc_er4ia_sim"
if [ -d "$FANUC_DIR" ] && [ -f "$FANUC_DIR/bootstrap.sh" ]; then
    # Create functional .venv using Conda with Python 3.10 if .venv is missing or python is older than 3.9
    NEED_CONDA_VENV=false
    if [ ! -x "$FANUC_DIR/.venv/bin/python" ]; then
        NEED_CONDA_VENV=true
    elif ! "$FANUC_DIR/.venv/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        NEED_CONDA_VENV=true
    fi

    if [ "$NEED_CONDA_VENV" = true ]; then
        CONDA_BIN=""
        if command -v conda >/dev/null 2>&1; then
            CONDA_BIN="$(command -v conda)"
        elif [ -x "/usr/local/anaconda3/bin/conda" ]; then
            CONDA_BIN="/usr/local/anaconda3/bin/conda"
        fi

        if [ -n "$CONDA_BIN" ]; then
            echo "Creating Python 3.10 virtual environment in $FANUC_DIR/.venv via Conda..."
            rm -rf "$FANUC_DIR/.venv"
            "$CONDA_BIN" create -y -p "$FANUC_DIR/.venv" python=3.10 pip
        fi
    fi

    echo "Bootstrapping FANUC robot arm simulation (generating textures & running unit tests)..."
    (
        cd "$FANUC_DIR"
        chmod +x bootstrap.sh *.sh 2>/dev/null || true
        ./bootstrap.sh
    )
    if [ "$IS_CLUSTER" = true ]; then
        chgrp ai_club "$FANUC_DIR" 2>/dev/null || true
        chmod g+rwxs "$FANUC_DIR" 2>/dev/null || true
    fi
    echo "FANUC robot arm simulation initialized."
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

# Ensure persistent FANUC scene USD is built in shared/scenes/
if [ ! -f "shared/scenes/fanuc_scene.usd" ]; then
    # Only build automatically if we have a direct GPU (GB10, Brev, or compute node)
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || nvidia-smi >/dev/null 2>&1; then
        echo "Building initial persistent FANUC scene at shared/scenes/fanuc_scene.usd..."
        mkdir -p shared/scenes
        ./scripts/run_isaac.sh scripts/render.py --target build_fanuc 2>/dev/null || true
    fi
fi

echo "Setup finished successfully. All data and assets accessible at ./shared/data/"
