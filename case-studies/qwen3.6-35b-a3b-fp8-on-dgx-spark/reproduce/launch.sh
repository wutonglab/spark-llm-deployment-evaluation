#!/usr/bin/env bash
# Reproduce the final Qwen3.6-35B-A3B-FP8 + MTP-1 production config on DGX Spark.
#
# Prerequisites:
#   - DGX Spark (or compatible GB10 / sm_121 platform)
#   - NVIDIA Container Toolkit + Docker running
#   - HuggingFace token with access to Qwen/Qwen3.6-35B-A3B-FP8
#   - Disk free: ~ 60 GB for model + container
#
# Usage:
#   cp env-templates/with-mtp1.env .env
#   $EDITOR .env                  # set HF_TOKEN, WORKSPACE, VLLM_PORT
#   bash launch.sh
#
set -euo pipefail

# ---- Locate .env --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found." >&2
    echo "Hint: cp env-templates/with-mtp1.env .env && \$EDITOR .env" >&2
    exit 1
fi

set -a; source "$ENV_FILE"; set +a

# ---- Validate required env vars -----------------------------------------
required=(HF_TOKEN WORKSPACE VLLM_PORT MODEL_ID VLLM_IMAGE)
for var in "${required[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is empty in $ENV_FILE" >&2
        exit 1
    fi
done

CONTAINER_NAME="${CONTAINER_NAME:-qwen36-fp8}"
MODEL_DIR="${WORKSPACE}/models/$(basename "$MODEL_ID")"

# ---- Pre-flight checks --------------------------------------------------
echo "==> Pre-flight checks"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found" >&2; exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker daemon not reachable (try: sudo systemctl start docker)" >&2; exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found or not working" >&2; exit 1
fi

# Port availability
if lsof -iTCP:"$VLLM_PORT" -sTCP:LISTEN -n -P 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: port $VLLM_PORT already in use" >&2; exit 1
fi

# Disk
avail_gb=$(df -BG "$WORKSPACE" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo 0)
if [[ "${avail_gb:-0}" -lt 60 ]]; then
    echo "WARN: only ${avail_gb}GB free at $WORKSPACE (model + container needs ~60GB)" >&2
fi

# ---- Download model if absent -------------------------------------------
if [[ ! -d "$MODEL_DIR" ]] || [[ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]]; then
    echo "==> Downloading $MODEL_ID to $MODEL_DIR ..."
    mkdir -p "$MODEL_DIR"
    HF_TOKEN="$HF_TOKEN" python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=os.environ['MODEL_ID'],
    local_dir=os.environ['MODEL_DIR'],
    token=os.environ['HF_TOKEN'],
)
" || { echo "ERROR: model download failed" >&2; exit 1; }
fi

# ---- Remove stale container ---------------------------------------------
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "==> Removing stale container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# ---- Launch vLLM --------------------------------------------------------
echo "==> Launching ${CONTAINER_NAME} on port ${VLLM_PORT}"

# Build optional flags from env
SPEC_CFG="${SPEC_CONFIG:-}"
KV_DTYPE="${KV_CACHE_DTYPE:-auto}"
ATTN_BACKEND="${ATTENTION_BACKEND:-}"
EXTRA_ARGS="${EXTRA_VLLM_ARGS:-}"

# Build the python command line for vllm
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
[[ -n "$KV_DTYPE" ]] && PYCMD+=(--kv-cache-dtype "$KV_DTYPE")
[[ -n "$ATTN_BACKEND" ]] && PYCMD+=(--attention-backend "$ATTN_BACKEND")
[[ "${ENABLE_PREFIX_CACHE:-false}" == "true" ]] && PYCMD+=(--enable-prefix-caching)
[[ -n "$SPEC_CFG" ]] && PYCMD+=(--speculative-config "$SPEC_CFG")
# Append any extra args (space-separated)
if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    PYCMD+=("${EXTRA_ARR[@]}")
fi

docker run -d \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia --gpus all \
    -p "${VLLM_PORT}:8000" \
    -v "${MODEL_DIR}":/model \
    --ipc=host \
    --restart unless-stopped \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    --entrypoint python3 \
    "${VLLM_IMAGE}" \
    "${PYCMD[@]}"

# ---- Health check loop --------------------------------------------------
echo "==> Waiting for vLLM to become healthy (timeout 5 min)"
deadline=$(( $(date +%s) + 300 ))
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
        echo "==> vLLM ready at http://localhost:${VLLM_PORT}/v1"
        break
    fi
    if [[ $(date +%s) -gt $deadline ]]; then
        echo "ERROR: vLLM did not become healthy in 5 minutes" >&2
        docker logs --tail=50 "${CONTAINER_NAME}" >&2 || true
        exit 1
    fi
    sleep 5
done

echo ""
echo "Container: ${CONTAINER_NAME}"
echo "Endpoint:  http://localhost:${VLLM_PORT}/v1"
echo "Model id:  ${MODEL_ID}"
echo ""
echo "Next steps:"
echo "  - Quick test:   curl http://localhost:${VLLM_PORT}/v1/models"
echo "  - Benchmark:    cd ../../tools/benchmark && bash run_variants.sh --only with-mtp1"
echo "  - Stop:         docker stop ${CONTAINER_NAME}"
