#!/usr/bin/env bash
# Quick health probe + smoke chat completion.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${VARIANT_ENV:-${ENV_FILE:-$SCRIPT_DIR/.env}}"
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }

PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL_ID:?MODEL_ID must be set}"

echo "==> GET /v1/models"
curl -fsS "http://localhost:${PORT}/v1/models" | python3 -m json.tool

echo
echo "==> POST /v1/chat/completions (smoke)"
curl -fsS "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
  "max_tokens": 32,
  "temperature": 0
}
EOF
)" | python3 -m json.tool
