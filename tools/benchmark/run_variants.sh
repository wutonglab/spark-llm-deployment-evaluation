#!/usr/bin/env bash
# Sequentially launch + benchmark multiple variants.
#
# Usage:
#   bash run_variants.sh --all
#   bash run_variants.sh --only with-mtp1,nvfp4-default
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/tools/deploy"
VARIANT_DIR="$DEPLOY_DIR/variant-configs"

ALL_VARIANTS=(baseline with-kv-fp8 with-flashinfer with-mtp1 nvfp4-default with-32k-batched)

PICKED=()
if [[ "${1:-}" == "--all" ]]; then
    PICKED=("${ALL_VARIANTS[@]}")
elif [[ "${1:-}" == "--only" ]] && [[ -n "${2:-}" ]]; then
    IFS=',' read -ra PICKED <<< "$2"
else
    echo "Usage: $0 --all | --only <v1,v2,...>" >&2
    echo "Available variants: ${ALL_VARIANTS[*]}" >&2
    exit 1
fi

for v in "${PICKED[@]}"; do
    env_file="$VARIANT_DIR/${v}.env"
    if [[ ! -f "$env_file" ]]; then
        echo "SKIP: $env_file not found" >&2
        continue
    fi

    echo
    echo "============================================================"
    echo "Variant: $v"
    echo "============================================================"

    # Stop any running vLLM
    VARIANT_ENV="$env_file" bash "$DEPLOY_DIR/stop.sh" || true

    # Launch
    VARIANT_ENV="$env_file" bash "$DEPLOY_DIR/launch.sh"

    # Benchmark
    set -a; source "$env_file"; set +a
    python3 "$SCRIPT_DIR/bench.py" \
        --variant "$v" \
        --base-url "http://localhost:${VLLM_PORT}/v1" \
        --model "${MODEL_ID}" \
        --concurrency 1,4 \
        --output-lengths 512,1024 \
        --duration 60

    # Stop
    VARIANT_ENV="$env_file" bash "$DEPLOY_DIR/stop.sh"
done

echo
echo "==> All variants done. Run: python3 $SCRIPT_DIR/analyze.py --against <baseline.csv>"
