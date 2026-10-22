#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# End-to-End YOLO Detection & Slot Assignment Inspection Pipeline
# 
# 1. Render clean Golden Reference frame (baseline)
# 2. Generate full-frame YOLO bounding box dataset (Isaac Sim twin)
# 3. Train YOLOv11 Object Detector on full frames
# 4. Run Real-Time Full-Frame Detection + Hungarian Slot Assignment
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:?Usage: scripts/run_all.sh runs/archive/YYYY-MM-DD_remediation [scenes]}"
SCENES="${2:-100}"
exec ./scripts/run_remediation_pipeline.sh "$OUT_DIR" "$SCENES"
