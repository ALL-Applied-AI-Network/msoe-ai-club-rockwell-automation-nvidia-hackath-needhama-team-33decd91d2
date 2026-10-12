#!/usr/bin/env bash
set -euo pipefail

# Determine repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect environment:
# 1. School Cluster (Rosie): /data/ai_club exists or /data is writable
# 2. Standalone / Brev / Local workstation
if [ -d "/data/ai_club" ] || [ -w "/data" 2>/dev/null ]; then
    TEAM_DIR="/data/ai_club/team_14_2026"
    TARGET_DATA_DIR="${TEAM_DIR}/data"
    SHARED_CACHE_DIR="${TEAM_DIR}/cache"
    IS_CLUSTER=true
else
    TEAM_DIR="${SCRIPT_DIR}"
    TARGET_DATA_DIR="${SCRIPT_DIR}/data"
    IS_CLUSTER=false
fi

# Ensure helper scripts have execution permissions
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh scripts/*.slurm 2>/dev/null || true
fi

# 1. Setup on Rosie (Cluster)
if [ "$IS_CLUSTER" = true ]; then
    mkdir -p "$TARGET_DATA_DIR" "$SHARED_CACHE_DIR" logs runs

    if [ "$(ls -A "$TARGET_DATA_DIR" 2>/dev/null)" ]; then
        echo "Shared directory '${TARGET_DATA_DIR}' already exists and is not empty. Skipping extraction."
    else
        if [ -f "data.zip" ]; then
            echo "Extracting data.zip to ${TEAM_DIR}..."
            unzip -q -o data.zip -d "$TEAM_DIR"
            echo "Extraction complete."
        else
            echo "Error: 'data.zip' not found in $SCRIPT_DIR and '${TARGET_DATA_DIR}' is empty." >&2
            exit 1
        fi
    fi

    # Create team_14_2026 symlink in repo root -> /data/ai_club/team_14_2026
    if [ -L "team_14_2026" ] || [ ! -e "team_14_2026" ]; then
        rm -f team_14_2026
        echo "Creating symlink team_14_2026 -> ${TEAM_DIR}..."
        ln -s "$TEAM_DIR" team_14_2026
    fi

    # Ensure stale ./data symlink is removed
    rm -f data 2>/dev/null || true

    # Set group ownership and shared permissions for team members
    echo "Setting group ownership to ai_club and group permissions for ${TEAM_DIR}..."
    chgrp -R ai_club "$TEAM_DIR" 2>/dev/null || true
    chmod -R g+rwX "$TEAM_DIR" 2>/dev/null || true

# 2. Setup on Brev / Standalone
else
    mkdir -p logs runs
    if [ -d "data" ] && [ ! -L "data" ] && [ "$(ls -A data 2>/dev/null)" ]; then
        echo "Directory 'data/' already exists and is not empty. Skipping extraction."
    elif [ -f "data.zip" ]; then
        rm -f data 2>/dev/null || true
        echo "Extracting data.zip into ${SCRIPT_DIR}..."
        unzip -q -o data.zip
        echo "Extraction complete."
    else
        echo "Warning: 'data.zip' not found in $SCRIPT_DIR and 'data/' is empty."
    fi
fi

# 3. Bootstrap FANUC Robot Arm Digital Twin (Twin 2)
if [ "$IS_CLUSTER" = true ]; then
    FANUC_DIR="${TEAM_DIR}/data/fanuc_er4ia_sim"
else
    FANUC_DIR="${SCRIPT_DIR}/data/fanuc_er4ia_sim"
fi

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
        chgrp -R ai_club "$FANUC_DIR" 2>/dev/null || true
        chmod -R g+rwX "$FANUC_DIR" 2>/dev/null || true
    fi
    echo "FANUC robot arm simulation initialized."
fi

echo "Setup finished successfully."
