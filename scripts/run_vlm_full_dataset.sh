#!/usr/bin/env bash
set -euo pipefail

# Label all real captures at all 30 aperture positions and keep VLM crops separate from YOLO training.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

shopt -s nullglob
images=("${REPO_ROOT}/shared/data/csi_vial_fill_dataset/cap_seal/images/"*.jpg)
IMAGE_COUNT="${#images[@]}"
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "No .jpg captures found under shared/data/csi_vial_fill_dataset/cap_seal/images." >&2
    exit 1
fi

SAMPLES_DIR="${REPO_ROOT}/shared/inspection_models/crop_dataset/VLM_samples"
if [ -d "$SAMPLES_DIR" ] && [ -n "$(find "$SAMPLES_DIR" -type f -print -quit)" ]; then
    ARCHIVE_DIR="${REPO_ROOT}/runs/vlm_labels/partial_VLM_samples_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${REPO_ROOT}/runs/vlm_labels"
    mv "$SAMPLES_DIR" "$ARCHIVE_DIR"
    echo "Archived previous VLM samples to ${ARCHIVE_DIR}"
fi

exec ./scripts/run_vlm_labeler.sh \
    --max-images "$IMAGE_COUNT" \
    --all-apertures \
    --crop-output shared/inspection_models/crop_dataset/VLM_samples \
    --out runs/vlm_labels/full_real_dataset.json \
    "$@"
