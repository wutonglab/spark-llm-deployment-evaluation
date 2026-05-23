# Case Study: Qwen3.6-35B-A3B-FP8 on DGX Spark (GB10)

> **Question**: on a DGX Spark (NVIDIA GB10 / sm_121), should we adopt Qwen3.6-35B-A3B as an LLM inference service, and if so — FP8 or NVFP4? With which speculative decoding method? Which vLLM tuning?

> **Answer**: adopt the model (Layer 0 score **0.78**), keep **FP8** (Layer 1 NVFP4 score only **0.21**), enable **MTP-1** speculative decoding, deploy with `kv-cache-dtype fp8` + default attention backend.
>
> Measured single-request: **~61 tok/s** with the recommended config vs ~12 tok/s for naive NVFP4. Recommended config wins under every measured workload.

## Hardware Profile (sanitized)

| Attribute | Value |
|---|---|
| Platform | NVIDIA DGX Spark (GB10) |
| Compute architecture | sm_121 Blackwell (consumer / workstation tier) |
| Unified memory | 128 GB LPDDR5X |
| Memory bandwidth | ~273 GB/s |
| Container runtime | Docker + NVIDIA Container Toolkit |
| Inference framework | vLLM (NVIDIA NGC distribution) |

## Framework Scores Applied

| Layer | Framework | Score | Decision |
|---|---|---|---|
| 0 — Model selection | [`02-new-model-selection`](../../frameworks/02-new-model-selection.md) | **0.78** | ✅ adopt the model |
| 1 — Quantization | [`01-fp8-vs-nvfp4-decision`](../../frameworks/01-fp8-vs-nvfp4-decision.md) | **0.21** | ❌ keep FP8; do not switch to NVFP4 |
| 2 — Speculative decoding | [`05-mtp-speculative-decoding`](../../frameworks/05-mtp-speculative-decoding.md) | n/a (decision table) | ✅ enable MTP-1; reject MTP-2/3, EAGLE-3, DFlash |
| 3 — Engine tuning (A/B) | empirical | see [`04-benchmark-results.md`](04-benchmark-results.md) | ✅ `kv-cache-dtype=fp8` + default attention + prefix-caching off (MTP coexists with cache reuse) |

Each score is justified in detail in the per-layer files (`01-` through `05-`).

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
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --dtype auto
```

The full reproducible launcher lives in [`reproduce/`](reproduce/).

## Headline Measurements

| Variant | Single-request 1024-tok output (tok/s) | c=4 aggregate output 1024 (tok/s) |
|---|---:|---:|
| `baseline` (FP8, no MTP, no kv-fp8) | ~53 | ~190 |
| `with-kv-fp8` | ~52 | ~164 |
| `with-flashinfer` | ~52 | ~166 |
| **`with-mtp1`** ⭐ | **~61** | **~234** |
| `nvfp4-default` (vendor NVFP4 image) | ~12 | ~55 |

Detailed tables (including TTFT, ITL, multiple output lengths, multiple concurrency levels) are in [`04-benchmark-results.md`](04-benchmark-results.md). Raw CSV in [`data/baseline-results.csv`](data/baseline-results.csv).

## What This Case Validates About the Frameworks

- **Framework 01** predicted "stay on FP8 with score 0.21" — the NVFP4 measurement at 12 tok/s confirms the prediction is correct in the strong sense (not just slightly worse, but **5× worse** under the default software stack).
- **Framework 05** predicted MTP-1 as the only viable speculative method for this (model, quant, hardware) tuple — measurement confirmed +18% single-request, +23% at c=4 long output.
- **Framework 02** scored Qwen3.6-35B-A3B at 0.78 — the model has been in production stable use since adoption.

## What This Case Does NOT Validate (open questions)

- **NVFP4 with the Avarok community patch image**: the patched path is *expected* to reach ~50 tok/s based on third-party reports, but we have not run it in this case study (operational cost adjustment in Framework 01 dim 5 = −0.07 was based on that absence). Community reproductions welcome — see [`case-studies/_template.md`](../_template.md).
- **DFlash with BF16 base**: would require ~70 GB memory which approaches Spark's unified memory budget for 4-concurrency serving; we didn't pursue.
- **Other speculative methods landing in vLLM > 0.17**: re-validate when major framework versions ship.

## How to Reproduce This Case

Two paths:

1. **Manual**: clone the repo, copy [`reproduce/env-templates/with-mtp1.env`](reproduce/env-templates/with-mtp1.env) to `.env`, run [`reproduce/launch.sh`](reproduce/launch.sh), then [`../../tools/benchmark/run_variants.sh`](../../tools/benchmark/run_variants.sh).
2. **Agent-driven**: run [`../../agent/evaluate.py`](../../agent/evaluate.py) with the model id `Qwen/Qwen3.6-35B-A3B-FP8` and target hardware "DGX Spark / GB10". The agent walks all 4 layers and produces an evaluation report comparable to this case.

## File Map

| File | Purpose |
|---|---|
| [`00-summary.md`](00-summary.md) | This file |
| [`01-applying-framework-02.md`](01-applying-framework-02.md) | Layer 0 model-selection score 0.78 with citations |
| [`02-applying-framework-01.md`](02-applying-framework-01.md) | Layer 1 quantization score 0.21 — why NOT NVFP4 |
| [`03-applying-framework-05.md`](03-applying-framework-05.md) | Layer 2 speculative-decoding choice (MTP-1) |
| [`04-benchmark-results.md`](04-benchmark-results.md) | Layer 3 A/B tuning + full measurement tables |
| [`05-decision.md`](05-decision.md) | Consolidated final decision + rationale chain |
| [`data/baseline-results.csv`](data/baseline-results.csv) | Raw measurements |
| [`reproduce/`](reproduce/) | Reproducible launcher + env templates |
