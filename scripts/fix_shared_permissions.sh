#!/usr/bin/env bash
set -euo pipefail

# Script to configure shared directory permissions for Team 14 (Innovation Lab)
# Usage: ./scripts/fix_shared_permissions.sh

TEAM_DIR="/data/ai_club/team_14_2026"
SHARED_SCENES="${TEAM_DIR}/scenes"
SHARED_DATA="${TEAM_DIR}/data"
SHARED_CACHE="${TEAM_DIR}/cache"

echo "=== Configuring permissions for ${TEAM_DIR} ==="

if [ ! -d "${TEAM_DIR}" ]; then
    echo "Warning: '${TEAM_DIR}' does not exist on this machine."
    exit 0
fi

# 1. Enable user + group read/write/execute on team root
chmod u+rwx,g+rwxs "${TEAM_DIR}" 2>/dev/null || true
chgrp ai_club "${TEAM_DIR}" 2>/dev/null || true

# 2. Create and configure scenes directory for team collaboration
mkdir -p "${SHARED_SCENES}" "${SHARED_CACHE}"
chgrp -R ai_club "${SHARED_SCENES}" "${SHARED_CACHE}" 2>/dev/null || true
chmod -R g+rwxs "${SHARED_SCENES}" "${SHARED_CACHE}" 2>/dev/null || true

# 3. Enforce strictly Read-Only on raw competition data (extracted from zip)
if [ -d "${SHARED_DATA}" ]; then
    echo "Enforcing strictly Read-Only permissions on raw competition data (${SHARED_DATA})..."
    chmod -R a-w "${SHARED_DATA}" 2>/dev/null || true
    chmod -R a+rX "${SHARED_DATA}" 2>/dev/null || true
fi

echo "=== Shared directory permissions successfully configured ==="
ls -ld "${TEAM_DIR}" "${SHARED_SCENES}" "${SHARED_DATA}"
