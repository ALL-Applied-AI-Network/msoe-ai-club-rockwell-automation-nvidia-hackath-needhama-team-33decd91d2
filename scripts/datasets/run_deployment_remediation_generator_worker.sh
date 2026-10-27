#!/usr/bin/env bash
set -euo pipefail

SHARD_ID="$((${SLURM_PROCID:-0} + ${REMEDIATION_SHARD_OFFSET:-0}))"

EXTRA_ARGS=(--lighting-preset "${REMEDIATION_LIGHTING_PRESET:-balanced}")
if [[ "${REMEDIATION_PINK_TABLE_PAPER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--pink-table-paper)
fi
if [[ "${REMEDIATION_DOMAIN_RANDOMIZATION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--domain-randomization)
fi

./scripts/run_isaac.sh src/data_generation/generate_deployment_remediation_dataset.py \
  --out "$REMEDIATION_DATASET_OUT" \
  --scenes "$REMEDIATION_DATASET_SCENES" \
  --train-count "$REMEDIATION_TRAIN_COUNT" \
  --val-count "$REMEDIATION_VAL_COUNT" \
  --shard "$SHARD_ID" \
  --num-shards "$REMEDIATION_DATASET_SHARDS" \
  "${EXTRA_ARGS[@]}"

# Kit can occasionally shut down with status 0 after the Python workload has
# raised. The shard manifest is written only after every assigned image, so it
# is the authoritative completion marker.
printf -v SHARD_MANIFEST '%s/manifests/shard_%02d.json' \
  "$REMEDIATION_DATASET_OUT" "$SHARD_ID"
if [[ ! -s "$SHARD_MANIFEST" ]]; then
  echo "[generation] ERROR: missing completed shard manifest: $SHARD_MANIFEST" >&2
  exit 1
fi
