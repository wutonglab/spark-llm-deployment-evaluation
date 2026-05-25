#!/usr/bin/env bash
# Reproduce the Qwen3.6-35B-A3B-FP8 + MTP-1 production config on NVIDIA Jetson Thor.
#
# Differences from the DGX Spark launcher (intentional):
#   1. Uses `vllm/vllm-openai:nightly-aarch64` (NGC NV image is x86_64-only).
#   2. Mounts a small entrypoint script that runs `pip install pytest` before
#      `vllm serve`, working around a `cupy.testing` import chain in the
#      nightly-aarch64 image that breaks MTP head registration without pytest.
#   3. Default `GPU_MEM_UTIL=0.75` (Spark's 0.85 OOMs at startup on Thor —
#      unified-memory `Free` after boot is ~100 GB, not 122 GB).
#   4. Default `MAX_NUM_BATCHED_TOKENS=32768` and `MAX_NUM_SEQS=64`
#      (Spark uses 16384 / 8 — measured 15-26% slower on Thor on single-stream).
#
# Prerequisites:
#   - NVIDIA Jetson Thor (sm_110) with JetPack 7 / L4T R38 or later
#   - Docker with `nvidia` runtime registered in /etc/docker/daemon.json
#   - HuggingFace token with access to Qwen/Qwen3.6-35B-A3B-FP8 (~ 35 GB on disk)
#   - Recommended: `sudo nvpmodel -m 0 && sudo jetson_clocks` for MAXN profile
#
# Usage:
#   cp env-templates/thor-tuned.env .env
#   $EDITOR .env            # set HF_TOKEN, WORKSPACE, VLLM_PORT, VLLM_API_KEY
#   bash launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found." >&2
    echo "Hint: cp env-templates/thor-tuned.env .env && \$EDITOR .env" >&2
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

required=(HF_TOKEN WORKSPACE VLLM_PORT MODEL_ID VLLM_IMAGE VLLM_API_KEY)
for var in "${required[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is empty in $ENV_FILE" >&2
        exit 1
    fi
done

CONTAINER_NAME="${CONTAINER_NAME:-qwen36-fp8}"
MODEL_DIR="${WORKSPACE}/models/$(basename "$MODEL_ID")"
ENTRY_DIR="${WORKSPACE}/vllm-entrypoints"
ENTRY_SCRIPT="${ENTRY_DIR}/prod.sh"

echo "==> Pre-flight checks"
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable" >&2; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "WARN: nvidia-smi not in PATH (Jetson sometimes hides it under /usr/bin)" >&2; }

# Check uname for aarch64
arch="$(uname -m)"
if [[ "$arch" != "aarch64" ]]; then
    echo "WARN: host arch is $arch, expected aarch64 (Jetson Thor). The nightly-aarch64 image will not run on x86_64." >&2
fi

# Port check
if command -v ss >/dev/null; then
    ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${VLLM_PORT}$" && { echo "ERROR: port $VLLM_PORT already in use" >&2; exit 1; }
fi

# Disk
mkdir -p "$WORKSPACE"
avail_gb=$(df -BG "$WORKSPACE" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo 0)
if [[ "${avail_gb:-0}" -lt 60 ]]; then
    echo "WARN: only ${avail_gb}GB free at $WORKSPACE (FP8 model ~35GB + image ~27GB ≈ 62GB)" >&2
fi

# Download model if absent
if [[ ! -d "$MODEL_DIR" ]] || [[ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]]; then
    echo "==> Downloading $MODEL_ID to $MODEL_DIR ..."
    mkdir -p "$MODEL_DIR"
    HF_TOKEN="$HF_TOKEN" MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=os.environ['MODEL_ID'],
    local_dir=os.environ['MODEL_DIR'],
    token=os.environ['HF_TOKEN'],
)
" || { echo "ERROR: model download failed" >&2; exit 1; }
fi

# Build entrypoint script
mkdir -p "$ENTRY_DIR"
SPEC_CFG="${SPEC_CONFIG:-}"
KV_DTYPE="${KV_CACHE_DTYPE:-auto}"
ENABLE_PREFIX="${ENABLE_PREFIX_CACHE:-true}"
LIMIT_MM="${LIMIT_MM_PER_PROMPT:-{\"image\":0,\"video\":0\}}"
EXTRA_ARGS="${EXTRA_VLLM_ARGS:-}"

cat > "$ENTRY_SCRIPT" <<EOF
#!/bin/bash
set -e
# Workaround for cupy.testing -> import pytest failure during MTP head registration
pip install --quiet pytest
exec vllm serve /model \\
  --served-model-name "$MODEL_ID" \\
  --host 0.0.0.0 --port 8000 \\
  --api-key "\${VLLM_API_KEY}" \\
  --tensor-parallel-size 1 \\
  --max-model-len "${MAX_MODEL_LEN:-262144}" \\
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \\
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \\
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.75}" \\
  --kv-cache-dtype "${KV_DTYPE}" \\
  $( [[ "$ENABLE_PREFIX" == "true" ]] && echo "--enable-prefix-caching" ) \\
  $( [[ -n "$SPEC_CFG" ]] && echo "--speculative-config '$SPEC_CFG'" ) \\
  --reasoning-parser qwen3 \\
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \\
  --trust-remote-code \\
  --limit-mm-per-prompt '$LIMIT_MM' \\
  --dtype auto \\
  $EXTRA_ARGS
EOF
chmod +x "$ENTRY_SCRIPT"

# Remove stale container
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "==> Removing stale container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# Drop page cache so vLLM's `Free` check passes (Jetson unified memory quirk)
echo "==> Drop page cache (sudo may prompt)"
sudo sync && sudo bash -c "echo 3 > /proc/sys/vm/drop_caches" || true

echo "==> Launching ${CONTAINER_NAME} on port ${VLLM_PORT}"
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --runtime=nvidia --gpus all \
    --ipc=host --network=host \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_API_KEY="${VLLM_API_KEY}" \
    -v "${MODEL_DIR}":/model:ro \
    -v "${ENTRY_SCRIPT}":/entry.sh:ro \
    --entrypoint bash "${VLLM_IMAGE}" /entry.sh

# Health-check loop (5 min — Thor warmup is ~3-5 min including pytest install + CUDA-graph capture)
echo "==> Waiting for vLLM to become healthy (timeout 5 min)"
deadline=$(( $(date +%s) + 300 ))
while true; do
    if curl -sf -H "Authorization: Bearer ${VLLM_API_KEY}" "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
        echo "==> vLLM ready at http://localhost:${VLLM_PORT}/v1"
        break
    fi
    if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
        echo "ERROR: container ${CONTAINER_NAME} died" >&2
        docker logs --tail=50 "${CONTAINER_NAME}" >&2 || true
        exit 1
    fi
    if [[ $(date +%s) -gt $deadline ]]; then
        echo "ERROR: vLLM did not become healthy in 5 minutes" >&2
        docker logs --tail=50 "${CONTAINER_NAME}" >&2 || true
        exit 1
    fi
    sleep 10
done

echo ""
echo "Container:   ${CONTAINER_NAME}"
echo "Endpoint:    http://localhost:${VLLM_PORT}/v1"
echo "Model id:    ${MODEL_ID}"
echo "API key:     ${VLLM_API_KEY:0:10}..."
echo ""
echo "Next steps:"
echo "  - Quick test:   curl -H 'Authorization: Bearer \${VLLM_API_KEY}' http://localhost:${VLLM_PORT}/v1/models"
echo "  - Benchmark:    cd ../../../tools/benchmark && bash run_variants.sh --only thor-tuned"
echo "  - Stop:         docker stop ${CONTAINER_NAME}"
