#!/usr/bin/env bash
#SBATCH --job-name=build-table-replay
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=shared/runs/2026-10-28_yolo26x_real_table_finetune/build-%j.out
#SBATCH --error=shared/runs/2026-10-28_yolo26x_real_table_finetune/build-%j.err

set -euo pipefail
umask 002
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT="${REPO_DIR}/shared/runs/2026-10-28_yolo26x_real_table_finetune"
cd "${REPO_DIR}"

srun --export=ALL --ntasks=1 shared/cache/locateanything_3b/venv/bin/python \
    scripts/training/build_real_table_cap_replay.py \
    --synthetic runs/remediation_scene_finetune/remediation_black_silver_physical_pink_1000 \
    --prior-real shared/runs/2026-10-27_yolo26x_perfect_real_finetune/refined_real \
    --table-images shared/runs/2026-10-28_table_cap_paper_test/results/processed \
    --output "${ROOT}/dataset_build" \
    --prior-real-repeats 80 \
    --table-repeats 400

chgrp -R ai_club "${ROOT}"
chmod -R g+rwX "${ROOT}"
