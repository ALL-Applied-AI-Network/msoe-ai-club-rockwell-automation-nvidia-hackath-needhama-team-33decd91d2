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

echo "=================================================================="
echo " [Step 1/4] Rendering Clean Golden Reference Frame..."
echo "=================================================================="
./scripts/render_golden.sh

echo ""
echo "=================================================================="
echo " [Step 2/4] Generating Synthetic YOLO Detection Dataset..."
echo "=================================================================="
./scripts/run_generate_fullframe_data.sh --scenes 30

echo ""
echo "=================================================================="
echo " [Step 3/4] Training YOLOv11 Detector on GPU..."
echo "=================================================================="
./scripts/run_train_detector.sh --epochs 35 --batch 16

echo ""
echo "=================================================================="
echo " [Step 4/4] Executing Real-Time Detection & Slot Assignment..."
echo "=================================================================="
./scripts/run_inspection.sh

echo ""
echo "=================================================================="
echo " Pipeline Complete!"
echo " Results:"
echo "   - Visualization: shared/inspection_models/test_defect_scene/annotated_detection.png"
echo "   - Decision JSON: shared/inspection_models/test_defect_scene/decision_detection.json"
echo "=================================================================="
