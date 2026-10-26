#!/usr/bin/env bash
set -euo pipefail

# Script to configure shared directory permissions for Team 14 (Innovation Lab)
# Usage: ./scripts/fix_shared_permissions.sh

TEAM_DIR="/data/ai_club/team_14_2026"
SHARED_SCENES="${TEAM_DIR}/scenes"
SHARED_DATA="${TEAM_DIR}/data"
SHARED_CACHE="${TEAM_DIR}/cache"
SHARED_RUNS="${TEAM_DIR}/runs"

echo "=== Configuring permissions for ${TEAM_DIR} ==="

if [ ! -d "${TEAM_DIR}" ]; then
    echo "Warning: '${TEAM_DIR}' does not exist on this machine."
    exit 0
fi

# 1. Create the collaboration directories before applying the workspace policy.
mkdir -p "${SHARED_SCENES}" "${SHARED_CACHE}" "${SHARED_RUNS}"

# 2. Make the entire team workspace owner/group-readable, writable, and
# executable recursively.  The set-group-ID bit keeps newly created entries in
# the ai_club group.  Raw data is made read-only again immediately below.
chgrp -R ai_club "${TEAM_DIR}" 2>/dev/null || true
chmod -R u+rwx,g+rwxs "${TEAM_DIR}" 2>/dev/null || true

# 3. Enforce strictly read-only raw competition data (extracted from zip).
if [ -d "${SHARED_DATA}" ]; then
    echo "Enforcing strictly Read-Only permissions on raw competition data (${SHARED_DATA})..."
    chmod -R a-w "${SHARED_DATA}" 2>/dev/null || true
    chmod -R a+rX "${SHARED_DATA}" 2>/dev/null || true
fi

echo "=== Shared directory permissions successfully configured ==="
ls -ld "${TEAM_DIR}" "${SHARED_SCENES}" "${SHARED_RUNS}" "${SHARED_DATA}"
