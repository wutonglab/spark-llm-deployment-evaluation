# Final Decision: Qwen3.6-35B-A3B on DGX Spark

## The Decision Chain

| Step | Framework | Score / Choice | Why |
|---|---|---|---|
| 1. Adopt the model? | [02-new-model-selection](../../frameworks/02-new-model-selection.md) | **0.78 → adopt** | Full-stack ecosystem maturity, drop-in switching cost, modest but real capability lift over predecessor |
| 2. FP8 or NVFP4? | [01-fp8-vs-nvfp4-decision](../../frameworks/01-fp8-vs-nvfp4-decision.md) | **0.21 → FP8** | sm_121 lacks tcgen05; 273 GB/s bandwidth is below the NVFP4 sweet spot; mainline software stack doesn't run the fast path |
| 3. Speculative decoding? | [05-mtp-speculative-decoding](../../frameworks/05-mtp-speculative-decoding.md) | **MTP-1** | Only viable option: EAGLE-3 has no draft, DFlash breaks under FP8, MTP-2 unstable at concurrency, MTP-3 hits a known bug |
| 4. Engine tuning? | A/B sweep | **kv-cache fp8 + default attention + prefix-caching enabled** | flashinfer measured no gain; larger batched-token budget doesn't help most workloads |

## Final Production Configuration

```bash
docker run -d --name qwen36-fp8 \
 --runtime=nvidia --gpus all \
 -p 8000:8000 \
 -v ${WORKSPACE}/models/Qwen3.6-35B-A3B-FP8:/model \
 --ipc=host --restart unless-stopped \
 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
 --entrypoint python3 \
 nvcr.io/nvidia/vllm:26.03.post1-py3 \
 -m vllm.entrypoints.openai.api_server \
 --model /model \
 --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
 --host 0.0.0.0 --port 8000 \
 --tensor-parallel-size 1 \
 --max-model-len 262144 \
 --max-num-batched-tokens 16384 \
 --gpu-memory-utilization 0.85 \
 --kv-cache-dtype fp8 \
 --reasoning-parser qwen3 \
 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
 --enable-prefix-caching \
 --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
 --dtype auto
```

Reproduce with: see [`reproduce/`](reproduce/) directory.

## Measured Outcome

| Metric | Final config | Best alternative tried | Naive NVFP4 |
|---|---:|---:|---:|
| Single-request tok/s (1024 output) | **~61** | ~53 (baseline FP8 no MTP) | ~12 |
| c=4 aggregate tok/s (1024 output) | **~234** | ~190 (baseline) | ~93 |
| 60-token-output step latency (cache-hit) | **~1.08 s** | ~1.25 s | not measured |

## Validity of the Recommendation

This recommendation holds for:

- DGX Spark (GB10 / sm_121) hardware
- vLLM 0.17.x via `nvcr.io/nvidia/vllm:26.03.post1-py3`
- `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint
- General LLM API service workload (single-request and low-to-medium concurrency mix)

Re-score the frameworks when any of the following change:
- vLLM major version (likely shifts Framework 01 dim 3 software-stack score, possibly enables EAGLE-3 for this model)
- New community / vendor patch image with stable cadence for sm_121 NVFP4 (shifts Framework 01 dims 3 + op cost)
- New Qwen3 minor version (re-run Framework 02 from scratch)
- Hardware swap (e.g. moving to RTX PRO 6000 Blackwell or B200 — almost certainly flips Framework 01 to favor NVFP4)
- Workload shift (e.g. mostly long-context > 32K, mostly tool-calling agents) — may shift Framework 01 dim 5

## What Would Change If…

| If | The new decision likely becomes |
|---|---|
| You're on B200 instead of DGX Spark | Switch to NVFP4 + MTP-1 (or EAGLE-3 if a draft becomes available) |
| You're on RTX PRO 6000 Blackwell | Switch to NVFP4 + MTP-1 (community-confirmed ~104 tok/s single-request) |
| You're on Jetson Thor (sm_110) | Stay on FP8 (no native FP4 silicon at all), MTP-1 still applies |
| Your workload is exclusively long-context (>32K) | Stay on FP8; consider disabling speculative decoding entirely since KV traffic dominates |
| Your workload is heavy multimodal | Run Framework 02 from scratch — capability dimension changes; current scoring is text-only |
| You require Apache-2.0 / fully open weights for compliance | Re-run Framework 02; likely a different model entirely (the Qwen community license isn't Apache) |

## How Future Cases Can Improve This One

| Open question | What evidence would resolve it |
|---|---|
| NVFP4 with community patch image (`avarok/dgx-vllm-nvfp4-kernel:v22`) — does Marlin path reach the projected ~50 tok/s on this exact model? | Run the variant; publish a case study under [`case-studies/`](../) |
| DFlash with BF16 base — would the 35B BF16 footprint (~70 GB) actually fit at c=4 serving? | Memory-utilization measurement at c=4 with BF16 |
| MTP acceptance rate on real chat / tool-calling traffic | Capture acceptance metrics from a production deployment over a week |

Contribute via [`case-studies/_template.md`](../_template.md) and the PR template.
