# Reproduce: Qwen3.6-35B-A3B-FP8 on NVIDIA Jetson Thor

This directory reproduces the final production configuration documented in
[`../05-decision.md`](../05-decision.md). The full benchmark sweep that
justifies the configuration is in [`../04-benchmark-results.md`](../04-benchmark-results.md).

## Quick Start

```bash
# 1. Copy the recommended env template, fill in your secrets
cp env-templates/thor-tuned.env .env
$EDITOR .env       # set HF_TOKEN, WORKSPACE, VLLM_PORT, VLLM_API_KEY

# 2. Launch (this downloads the model first if absent, ~35 GB)
bash launch.sh
```

The launcher waits up to 5 minutes for vLLM to pass `/v1/models`, then prints the
endpoint URL. First-launch warmup on Thor is typically 3-5 minutes (CUDA-graph
capture + the `pip install pytest` workaround — see below).

## Quick Test

```bash
KEY="$(grep ^VLLM_API_KEY .env | cut -d= -f2)"
curl -H "Authorization: Bearer $KEY" "http://localhost:${VLLM_PORT:-8000}/v1/models"
curl -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"hello"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' \
  "http://localhost:${VLLM_PORT:-8000}/v1/chat/completions"
```

## Variants

To run the comparison sweep:

```bash
# Single variant:
ln -sf env-templates/with-mtp1.env .env && bash launch.sh
# Then from another terminal:
cd ../../../tools/benchmark && bash run_variants.sh --only with-mtp1
```

| File | Purpose |
|---|---|
| `thor-tuned.env` ⭐ | Production config — `max-num-seqs 64`, `max-num-batched-tokens 32768`, MTP-1 |
| `with-mtp1.env` | Spark final config replicated (`max-num-seqs 8`, `max-num-batched-tokens 16384`) — 15-26% slower than `thor-tuned` on Thor |
| `baseline.env` | No MTP, reference point for MTP-1's value |
| `with-mtp2.env` | MTP-2 — measured −9% at c=4; do not use in production |
| `nvfp4-default.env` | RedHatAI NVFP4 build — measured 30× slower than `thor-tuned`; included to validate Framework 01's score |

## What's Different from the DGX Spark Launcher

This launcher diverges from [`../../qwen3.6-35b-a3b-fp8-on-dgx-spark/reproduce/launch.sh`](../../qwen3.6-35b-a3b-fp8-on-dgx-spark/reproduce/launch.sh) in **four** places:

1. **Image**: `vllm/vllm-openai:nightly-aarch64` instead of `nvcr.io/nvidia/vllm:26.03.post1-py3`. The NGC NV-distributed image is x86_64-only at the time of this writing.
2. **`pip install pytest` workaround**: the nightly-aarch64 image triggers `import pytest` via `cupy.testing._random` when the `Qwen3_5MoeMTP` architecture registers itself. `pytest` is missing from the image. The entrypoint installs it at container start (~3 s).
3. **`GPU_MEM_UTIL=0.75` default** (Spark uses 0.85): Thor's unified-memory `/proc/meminfo` `Free` is typically ~100 GB after boot (not 122 GB), and vLLM's pre-flight check OOMs at 0.85. The launcher also runs `echo 3 > /proc/sys/vm/drop_caches` (with sudo) before container start to maximize `Free`.
4. **`MAX_NUM_BATCHED_TOKENS=32768` and `MAX_NUM_SEQS=64`** defaults (Spark uses 16384 / 8): measured 15-26% faster on Thor single-stream with no concurrency regression. See [`../04-benchmark-results.md`](../04-benchmark-results.md).

## Persistence (Production)

For an always-on deployment, the `--restart unless-stopped` flag in `launch.sh` is
sufficient to survive Docker daemon restarts. To also survive Jetson reboots
(and reapply `nvpmodel -m 0 / jetson_clocks / drop_caches`), install the
`jetson-llm-prep.service` systemd unit documented in
[`../05-decision.md`](../05-decision.md).

## Stopping

```bash
docker stop "$(grep ^CONTAINER_NAME .env | cut -d= -f2)"
docker rm   "$(grep ^CONTAINER_NAME .env | cut -d= -f2)"
```
