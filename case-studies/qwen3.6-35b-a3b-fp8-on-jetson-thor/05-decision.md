# Final Decision: Qwen3.6-35B-A3B-FP8 on Jetson Thor

## The Decision Chain

| Step | Framework | Score / Choice | Why |
|---|---|---|---|
| 1. Adopt the model? | [02-new-model-selection](../../frameworks/02-new-model-selection.md) | **0.78 → adopt** | Hardware-independent. Same call as Spark; only the ecosystem dim shifts slightly because the `aarch64` vLLM image lags `x86_64`. |
| 2. FP8 or NVFP4? | [01-fp8-vs-nvfp4-decision](../../frameworks/01-fp8-vs-nvfp4-decision.md) | **~0.10 → FP8** | `sm_110` has **no native FP4 silicon at all** (hardware score = 0). Measured NVFP4 is **30× slower** than tuned FP8 — the strongest confirmation of Framework 01 in the repo so far. |
| 3. Speculative decoding? | [05-mtp-speculative-decoding](../../frameworks/05-mtp-speculative-decoding.md) | **MTP-1** | Only viable speculator (Qwen3.6 ships no EAGLE-3 draft, DFlash breaks under FP8, MTP-2 measured −9% at c=4 on Thor matching Spark direction). |
| 4. Engine tuning? | 6-variant A/B sweep | **`max-num-seqs 64` + `max-num-batched-tokens 32768`** | Spark's `16384 / 8` underperforms Thor's `32768 / 64` by 15-26% on single-stream with no concurrency regression. **This step does not transfer from the Spark case.** |

## Final Production Configuration

```bash
# 1. systemd boot prep (idempotent, runs on multi-user.target)
sudo tee /etc/systemd/system/jetson-llm-prep.service > /dev/null << EOF
[Unit]
Description=Jetson LLM serving prep (nvpmodel MAXN + drop caches)
Before=docker.service
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 0
ExecStart=/usr/bin/sync
ExecStart=/bin/sh -c "echo 3 > /proc/sys/vm/drop_caches"
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now jetson-llm-prep.service

# 2. vLLM entrypoint script (mounted into container, avoids docker -c quoting hell)
mkdir -p "${WORKSPACE}/vllm-entrypoints"
cat > "${WORKSPACE}/vllm-entrypoints/prod.sh" << 'EOF'
#!/bin/bash
set -e
# WORKAROUND: nightly-aarch64 image triggers `import pytest` via the
# cupy.testing import chain when Qwen3_5MoeMTP registers itself.
# pytest is missing from the image; pip install at startup is the simplest fix.
pip install --quiet pytest
exec vllm serve /model \
  --served-model-name qwen3.6-a3b-fp8 \
  --host 0.0.0.0 --port 8000 \
  --api-key "${VLLM_API_KEY:?VLLM_API_KEY must be set}" \
  --max-model-len 262144 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.75 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":4,"video":1}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --dtype auto
EOF
chmod +x "${WORKSPACE}/vllm-entrypoints/prod.sh"

# 3. Launch container (restart=unless-stopped so docker daemon restart re-spawns it)
docker run -d --name qwen36-fp8 \
  --restart unless-stopped \
  --runtime=nvidia --gpus all \
  --ipc=host --network=host \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  -v "${WORKSPACE}/models/Qwen3.6-35B-A3B-FP8":/model:ro \
  -v "${WORKSPACE}/vllm-entrypoints/prod.sh":/entry.sh:ro \
  --entrypoint bash vllm/vllm-openai:nightly-aarch64 /entry.sh
```

Reproducible launcher in [`reproduce/launch.sh`](reproduce/launch.sh) wraps the steps above with env validation and a health-check loop.

## Measured Outcome

| Metric | Final config (`thor-tuned`) | Best Spark-portable alternative (`with-mtp1`) | Naive NVFP4 |
|---|---:|---:|---:|
| Single-request tok/s (1024-out) | **66.7** | 56.8 | 2.2 |
| c=4 aggregate tok/s (1024-out) | **152** | 138 | n/a |
| c=8 aggregate tok/s (512-out) | **234** | n/a | n/a |
| c=16 aggregate tok/s (512-out) | **303** | n/a | n/a |
| Needle-in-haystack 150K input | **5/5 recall, 195 s prefill** | n/a | n/a |

## Production Notes — Two Pitfalls Worth Calling Out

### Pitfall 1 · `gpu-memory-utilization=0.85` (Spark default) OOMs on Thor at startup

Thor's `/proc/meminfo` `Free` field is **122 GB total but typically ~100 GB free** after boot (desktop + buff/cache eat the rest). vLLM's pre-flight memory check compares against this `Free` value — `0.85 * 122 = 104 GB > 100 GB free → ValueError`.

**Fix**: `--gpu-memory-utilization 0.75`. KV-cache capacity is still **3.78 M tokens** — plenty for 14× concurrency at 256K context, or 91× at 32K. If you want a higher utilization, run `echo 3 > /proc/sys/vm/drop_caches` first; that's what the `jetson-llm-prep.service` unit above does on boot.

### Pitfall 2 · `nightly-aarch64` image needs `pip install pytest` for MTP

When vLLM tries to register the `Qwen3_5MoeMTP` architecture (the MTP head class), an upstream `cupy.testing._random` module triggers `import pytest`. `pytest` isn't in the `nightly-aarch64` image. Workaround: add `pip install --quiet pytest` to the container entrypoint, as shown above and in `reproduce/launch.sh`. Cost is ~3 seconds per container start. An upstream fix would be: lazy-import `pytest` in `cupy.testing`, or remove the `cupy.testing` import path from the MTP head loader.

We did **not** open a vLLM upstream issue yet — this case-study PR is the documentation of the workaround. If anyone wants to file it, please link this case in the issue body.

## Validity of the Recommendation

This recommendation holds for:

- **NVIDIA Jetson Thor** Developer Kit class (`sm_110`, 128 GB LPDDR5X, ~273 GB/s)
- **vLLM `nightly-aarch64`** at or near `0.21.1rc1.dev262+g33d7cbe02`
- **`Qwen/Qwen3.6-35B-A3B-FP8`** checkpoint with its `mtp.safetensors` head
- General LLM API service workload (single-stream + c≤16 concurrency mix, mixed-length outputs)

Re-score when any of the following change:

- vLLM major version (CUDA-graph capture sizes, MoE backend selection, MTP head registration may all shift)
- NGC NV-distributed `aarch64` vLLM image becomes available (would likely close the 27% memory-bandwidth efficiency gap)
- Hardware swap to a different Blackwell variant (RTX PRO 6000 / B200 / Spark) — almost certainly flips Framework 01 to favor NVFP4
- Workload shift to exclusive long-context (>32K) — re-score Framework 01 dim 5; may also want to drop MTP speculative if KV traffic dominates
- Multimodal becomes dominant — Framework 02 capability dimension changes; current scoring is text-first

## What Would Change If…

| If | Likely new decision |
|---|---|
| You're on DGX Spark (`sm_121`) | Spark case study config: `max-num-batched-tokens 16384` (lower), MTP-1 still applies |
| You're on RTX PRO 6000 Blackwell | NVFP4 + MTP-1 (community-confirmed ~104 tok/s single-request on similar models) |
| You're on B200 / H200 | NVFP4 + MTP-1 (if NVFP4 path available) or FP8 + MTP-1 (always works) |
| Your workload is heavy multimodal | Re-score Framework 02 capability dim; may pick a vision-specialized model |
| You need open weights (Apache-2.0) | Re-run Framework 02 — Qwen community license fails this criterion |

## How Future Cases Can Improve This One

| Open question | What evidence resolves it |
|---|---|
| NGC NV-distributed `aarch64` vLLM image (when published) — does it close the 27% efficiency gap on Thor? | Re-run all variants on the new image; publish as a follow-up case |
| Long-context decode throughput steady-state — is tok/s at 100K prefix equal to tok/s at 1K? | Measure decode-only (excluding prefill) across `--max-model-len` settings |
| Multimodal decode throughput — do vision-token contexts slow text decode? | Measure tok/s with N images in context, varying N from 0 to 4 |
| MTP-1 lift non-uniformity on Thor (+32% at 256-out, +1% at 1024-out) — is it stable across prompt populations? | Replicate with a wider prompt diversity benchmark |

Contribute follow-ups via [`case-studies/_template.md`](../_template.md) and the PR template.
