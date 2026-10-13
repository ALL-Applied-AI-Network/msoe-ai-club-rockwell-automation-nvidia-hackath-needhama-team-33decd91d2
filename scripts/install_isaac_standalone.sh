#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Standalone Isaac Sim 6.1.0 Installer for Rosie / Local Linux / GB10
TARGET_DIR="${1:-$REPO_ROOT/shared/isaacsim_6.1}"
DOWNLOAD_DIR="${HOME}/Downloads"

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ZIP_URL="https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.1.0-linux-aarch64.zip"
    ZIP_FILE="${DOWNLOAD_DIR}/isaac-sim-standalone-6.1.0-linux-aarch64.zip"
else
    ZIP_URL="https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.1.0-linux-x86_64.zip"
    ZIP_FILE="${DOWNLOAD_DIR}/isaac-sim-standalone-6.1.0-linux-x86_64.zip"
fi

mkdir -p "$DOWNLOAD_DIR" "$TARGET_DIR"

if [ -x "$TARGET_DIR/python.sh" ]; then
    echo "Isaac Sim 6.1.0 is already installed at $TARGET_DIR (python.sh exists)."
    echo "Nothing to do!"
    exit 0
fi

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
