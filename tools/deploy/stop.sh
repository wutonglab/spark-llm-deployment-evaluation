#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${VARIANT_ENV:-${ENV_FILE:-$SCRIPT_DIR/.env}}"
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }
CONTAINER_NAME="${CONTAINER_NAME:-vllm}"
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "==> Stopping ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" >/dev/null
    docker rm "${CONTAINER_NAME}" >/dev/null
    echo "    Done."
else
    echo "==> ${CONTAINER_NAME} is not running"
fi
