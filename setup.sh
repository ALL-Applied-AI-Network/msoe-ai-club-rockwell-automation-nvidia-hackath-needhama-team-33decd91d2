#!/usr/bin/env bash
set -euo pipefail

# Determine repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect environment:
# 1. School Cluster: /data/ai_club exists or /data is writable
# 2. Standalone / Brev / Local workstation
if [ -d "/data/ai_club" ] || [ -w "/data" 2>/dev/null ]; then
    TEAM_DIR="/data/ai_club/team_14_2026"
    TARGET_DATA_DIR="${TEAM_DIR}/data"
    SHARED_CACHE_DIR="${TEAM_DIR}/cache"
    IS_CLUSTER=true
else
    TARGET_DATA_DIR="${SCRIPT_DIR}/data"
    IS_CLUSTER=false
fi

# Ensure helper scripts have execution permissions
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh scripts/*.slurm 2>/dev/null || true
fi

# 1. Setup on Cluster
if [ "$IS_CLUSTER" = true ]; then
    mkdir -p "$TARGET_DATA_DIR" "$SHARED_CACHE_DIR"

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

    # Create local symlink in repository if not already pointing there
    if [ ! -e "data" ]; then
        echo "Creating symlink data -> ${TARGET_DATA_DIR}..."
        ln -s "$TARGET_DATA_DIR" data
    fi

    # Set group ownership and shared permissions for team members
    echo "Setting group ownership to ai_club and group permissions for ${TEAM_DIR}..."
    chgrp -R ai_club "$TEAM_DIR" 2>/dev/null || true
    chmod -R g+rwX "$TEAM_DIR" 2>/dev/null || true

# 2. Setup on Brev / Local Standalone
else
    if [ -d "data" ] && [ "$(ls -A data 2>/dev/null)" ]; then
        echo "Directory 'data/' already exists and is not empty. Skipping extraction."
    elif [ -f "data.zip" ]; then
        echo "Extracting data.zip into ${SCRIPT_DIR}..."
        unzip -q -o data.zip
        echo "Extraction complete."
    else
        echo "Warning: 'data.zip' not found in $SCRIPT_DIR and 'data/' is empty."
    fi
fi

echo "Setup finished successfully."
