#!/usr/bin/env bash
# Generic vLLM launcher. Reads variant config from .env (or VARIANT_ENV) and
# starts a vLLM container with the chosen knobs.
#
# Usage:
#   cp variant-configs/with-mtp1.env .env
#   $EDITOR .env       # set HF_TOKEN, WORKSPACE, VLLM_PORT
#   bash launch.sh
#
# Or pick a variant inline:
#   VARIANT_ENV=variant-configs/baseline.env bash launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${VARIANT_ENV:-${ENV_FILE:-$SCRIPT_DIR/.env}}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: env file not found: $ENV_FILE" >&2
    echo "Hint: cp variant-configs/with-mtp1.env .env && \$EDITOR .env" >&2
    exit 1
fi

set -a; source "$ENV_FILE"; set +a

required=(HF_TOKEN WORKSPACE VLLM_PORT MODEL_ID VLLM_IMAGE)
for var in "${required[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is empty in $ENV_FILE" >&2
        exit 1
    fi
done

CONTAINER_NAME="${CONTAINER_NAME:-vllm}"
MODEL_DIR="${WORKSPACE}/models/$(basename "$MODEL_ID")"

# Pre-flight
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable" >&2; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not working" >&2; exit 1; }

if lsof -iTCP:"$VLLM_PORT" -sTCP:LISTEN -n -P 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: port $VLLM_PORT already in use" >&2; exit 1
fi

# Download model if absent
if [[ ! -d "$MODEL_DIR" ]] || [[ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]]; then
    echo "==> Downloading $MODEL_ID to $MODEL_DIR ..."
    mkdir -p "$MODEL_DIR"
    HF_TOKEN="$HF_TOKEN" MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" \
        python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ['MODEL_ID'],
    local_dir=os.environ['MODEL_DIR'],
    token=os.environ['HF_TOKEN'],
)
" || { echo "ERROR: model download failed" >&2; exit 1; }
fi

# Remove stale container
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# Build vLLM args
PYCMD=(
    -m vllm.entrypoints.openai.api_server
    --model /model
    --served-model-name "$MODEL_ID"
    --host 0.0.0.0 --port 8000
    --tensor-parallel-size 1
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}"
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --dtype auto
)
[[ -n "${KV_CACHE_DTYPE:-}" ]] && PYCMD+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${ATTENTION_BACKEND:-}" ]] && PYCMD+=(--attention-backend "$ATTENTION_BACKEND")
[[ "${ENABLE_PREFIX_CACHE:-false}" == "true" ]] && PYCMD+=(--enable-prefix-caching)
[[ -n "${SPEC_CONFIG:-}" ]] && PYCMD+=(--speculative-config "$SPEC_CONFIG")
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_VLLM_ARGS)
    PYCMD+=("${EXTRA_ARR[@]}")
fi

echo "==> Launching ${CONTAINER_NAME} on port ${VLLM_PORT}"
docker run -d \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia --gpus all \
    -p "${VLLM_PORT}:8000" \
    -v "${MODEL_DIR}":/model \
    --ipc=host --restart unless-stopped \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    --entrypoint python3 \
    "${VLLM_IMAGE}" \
    "${PYCMD[@]}"

# Health check
echo "==> Waiting for vLLM to become healthy (timeout 5 min)"
deadline=$(( $(date +%s) + 300 ))
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
        echo "==> Ready: http://localhost:${VLLM_PORT}/v1"
        break
    fi
    if [[ $(date +%s) -gt $deadline ]]; then
        echo "ERROR: vLLM did not become healthy" >&2
        docker logs --tail=50 "${CONTAINER_NAME}" >&2 || true
        exit 1
    fi
    sleep 5
done
