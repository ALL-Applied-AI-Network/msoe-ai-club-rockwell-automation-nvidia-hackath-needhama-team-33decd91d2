#!/usr/bin/env bash
#SBATCH --job-name=build-perfect-replay
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/replay-%j.out
#SBATCH --error=shared/runs/2026-10-27_yolo26x_perfect_real_finetune/replay-%j.err

set -euo pipefail
umask 002
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT="${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune/mixed_replay_dataset"
cd "${REPO_DIR}"
srun --export=ALL --ntasks=1 shared/cache/locateanything_3b/venv/bin/python \
    scripts/training/build_perfect_real_mixed_replay.py \
    --synthetic runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000 \
    --real shared/runs/2026-10-27_yolo26x_perfect_real_finetune/refined_real \
    --output "${OUTPUT}" \
    --real-repeats 160
chgrp -R ai_club "${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune"
chmod -R g+rwX "${REPO_DIR}/shared/runs/2026-10-27_yolo26x_perfect_real_finetune"
