#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# End-to-End Two-Tier Vial Inspection Pipeline
# 
# 1. Render clean Golden Reference image
# 2. Generate synthetic defect dataset (Isaac Sim twin)
# 3. Train YOLOv11 Edge Crop Classifier
# 4. Run Real-Time Edge Aperture Inspection on a test scene
# 5. (Optional) Run Offline In-Context VLM Auto-Labeler on real factory data
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================================="
echo " [Step 1/4] Rendering Clean Golden Reference Frame..."
echo "=================================================================="
./scripts/render_golden.sh

echo ""
echo "=================================================================="
echo " [Step 2/4] Generating Procedural Synthetic Defect Dataset..."
echo "=================================================================="
./scripts/run_data_generation.sh --scenes 10

echo ""
echo "=================================================================="
echo " [Step 3/4] Training YOLOv11 Edge Classifier on GPU..."
echo "=================================================================="
./scripts/run_train_classifier.sh --epochs 25 --batch 16

# Copy best weights permanently to shared/inspection_models/
if [ -f "runs/train_yolo_classifier/vial_aperture_cls/weights/best.pt" ]; then
    cp runs/train_yolo_classifier/vial_aperture_cls/weights/best.pt shared/inspection_models/vial_aperture_cls_best.pt
fi

echo ""
echo "=================================================================="
echo " [Step 4/4] Executing Real-Time Edge Aperture Inspection..."
echo "=================================================================="
./scripts/run_inspection.sh

echo ""
echo "=================================================================="
echo " Pipeline Complete!"
echo " Results:"
echo "   - Visualization: shared/inspection_models/test_defect_scene/annotated_ensemble.png"
echo "   - Decision JSON: shared/inspection_models/test_defect_scene/decision_ensemble.json"
echo "=================================================================="
