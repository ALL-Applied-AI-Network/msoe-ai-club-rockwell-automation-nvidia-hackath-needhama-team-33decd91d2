#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs runs/vlm_labels

PYTHON_BIN="${REPO_DIR}/.conda_vllm/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Local vLLM environment missing: ${PYTHON_BIN}" >&2
    exit 1
fi

PID_FILE="${REPO_DIR}/runs/vlm_labels/local_vlm_labeler.pid"
LOG_FILE="${REPO_DIR}/runs/vlm_labels/local_vlm_labeler_$(date +%Y%m%d_%H%M%S).log"

if [ "${1:-}" != "--worker" ]; then
    if [ -f "$PID_FILE" ]; then
        old_pid="$(cat "$PID_FILE")"
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "A local VLM job is already running (PID ${old_pid}). Log: $(ls -t runs/vlm_labels/local_vlm_labeler_*.log 2>/dev/null | head -1)" >&2
            exit 1
        fi
        rm -f "$PID_FILE"
    fi

    nohup setsid "$0" --worker "$@" </dev/null >>"$LOG_FILE" 2>&1 &
    job_pid=$!
    echo "$job_pid" > "$PID_FILE"
    echo "Started detached VLM job (PID ${job_pid})."
    echo "Log: ${LOG_FILE}"
    echo "Status: ps -p ${job_pid} -o pid,etime,cmd"
    exit 0
fi
shift

echo "$$" > "$PID_FILE"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -f "$PID_FILE" ] && [ "$(cat "$PID_FILE")" = "$$" ]; then
        rm -f "$PID_FILE"
    fi
}
trap cleanup EXIT

api_ready() {
    "$PYTHON_BIN" -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8000/v1/models", timeout=3).read()' \
        >/dev/null 2>&1
}

if api_ready; then
    echo "Using existing VLM server at http://127.0.0.1:8000/v1"
else
    echo "Starting the local VLM server."
    ./scripts/run_vlm_server.sh >>"$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    ready=0
    for ((attempt = 1; attempt <= 360; attempt++)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "VLM server exited before becoming ready."
            exit 1
        fi
        if api_ready; then
            ready=1
            break
        fi
        sleep 10
    done
    if [ "$ready" -ne 1 ]; then
        echo "VLM server did not become ready within 60 minutes."
        exit 1
    fi
fi

echo "VLM server ready; running the full dataset labeler."
./scripts/run_vlm_full_dataset.sh "$@"
