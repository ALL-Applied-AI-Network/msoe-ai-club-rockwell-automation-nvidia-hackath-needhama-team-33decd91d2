#!/usr/bin/env bash
set -euo pipefail

# Standalone Isaac Sim 6.1.0 Installer for Rosie / Local Linux
TARGET_DIR="${1:-$REPO_ROOT/shared/isaacsim_6.1}"
DOWNLOAD_DIR="${HOME}/Downloads"
ZIP_URL="https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.1.0-linux-x86_64.zip"
ZIP_FILE="${DOWNLOAD_DIR}/isaac-sim-standalone-6.1.0-linux-x86_64.zip"

mkdir -p "$DOWNLOAD_DIR" "$TARGET_DIR"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Downloading Isaac Sim 6.1.0 standalone (~12.4 GB)..."
    wget -c "$ZIP_URL" -O "$ZIP_FILE"
else
    echo "Found existing download at $ZIP_FILE."
fi

echo "Extracting to $TARGET_DIR..."
unzip -q -o "$ZIP_FILE" -d "$TARGET_DIR"

echo "Running post-install setup..."
cd "$TARGET_DIR"
./post_install.sh

echo "=========================================================="
echo "Successfully installed Isaac Sim 6.1.0 standalone at: $TARGET_DIR"
echo "=========================================================="
