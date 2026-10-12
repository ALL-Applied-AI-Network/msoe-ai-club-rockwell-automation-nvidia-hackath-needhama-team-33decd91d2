#!/usr/bin/env bash
set -euo pipefail

# Determine repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if data/ already exists and is non-empty
if [ -d "data" ] && [ "$(ls -A data 2>/dev/null)" ]; then
    echo "Directory 'data/' already exists and is not empty. Skipping extraction."
    exit 0
fi

# Extract data.zip if available
if [ -f "data.zip" ]; then
    echo "Extracting data.zip..."
    unzip -q -o data.zip
    echo "Extraction complete."
else
    echo "Warning: 'data.zip' not found in $SCRIPT_DIR and 'data/' directory is missing or empty."
    exit 1
fi
