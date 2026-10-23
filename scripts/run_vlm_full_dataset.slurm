#!/usr/bin/env bash
#SBATCH --job-name=vlm-label
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/vlm-%j.out
#SBATCH --error=logs/vlm-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs runs/vlm_labels

PYTHON_BIN="${REPO_DIR}/.conda_vllm/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Local vLLM environment missing: ${PYTHON_BIN}" >&2
    exit 1
fi

SERVER_LOG="${REPO_DIR}/runs/vlm_labels/vllm-server-${SLURM_JOB_ID:-manual}.log"
./scripts/run_vlm_server.sh >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for the VLM server (log: ${SERVER_LOG})"
READY=0
for ((attempt = 1; attempt <= 360; attempt++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "VLM server exited before becoming ready:" >&2
        tail -n 100 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if "$PYTHON_BIN" -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8000/v1/models", timeout=3).read()' \
        >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "VLM server did not become ready within 60 minutes. See ${SERVER_LOG}." >&2
    exit 1
fi

echo "VLM server ready; starting the full dataset labeling job."
./scripts/run_vlm_full_dataset.sh "$@"
