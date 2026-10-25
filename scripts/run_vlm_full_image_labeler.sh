#!/usr/bin/env bash
set -euo pipefail

# Run the one-request-per-full-image labeler against llama-server replicas.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${VLM_PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

exec "$PYTHON_BIN" -u "${REPO_ROOT}/src/inspection/vlm_full_image_labeler.py" "$@"
