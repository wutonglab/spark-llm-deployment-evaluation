# Case Study: `Qwen3.6-35B-A3B-FP8` on NVIDIA Jetson Thor (sm_110)

> **Question**: on a Jetson Thor 128 GB unified-memory edge platform (`sm_110`, no native FP4 silicon), should we adopt `Qwen3.6-35B-A3B` for a general-purpose LLM API service, and if so — which quantization, speculative method, and tuning?
>
> **Answer (one-line)**: **FP8 + MTP-1 + KV-cache FP8 + prefix caching + `max-num-seqs 64` + `max-num-batched-tokens 32768`**, served via vLLM `nightly-aarch64`. Measured **67 tok/s single-stream / 303 tok/s @ c=16** — that is **30× faster than the NVFP4 build** of the same model on the same hardware, and validates the porting-guide's prediction that Thor must avoid NVFP4.

## Why this case matters

The existing repository covers DGX Spark (`sm_121`) in detail. Jetson Thor is a **different Blackwell variant** (`sm_110`) with the same headline numbers (128 GB LPDDR5X, ~273 GB/s) but **no native FP4 tensor cores**. The porting guide ([`src/hardware-portability/porting-to-other-gpus.md`](../../src/hardware-portability/porting-to-other-gpus.md)) predicted "FP8 only on Thor" but no case study had validated it. This is that case.

## Hardware Profile

| Attribute | Value |
|---|---|
| Platform | NVIDIA Jetson Thor (128 GB Developer Kit class) |
| Compute architecture | **sm_110** (Blackwell-edge, no native FP4) |
| Memory | 128 GB LPDDR5X (unified, GPU + CPU share) |
| Memory bandwidth | ~273 GB/s |
| OS / driver | JetPack 7 / L4T R38.4.0, kernel 6.8.12-tegra, CUDA 13.0, Driver 580.00 |
| Power profile | nvpmodel MAXN (`-m 0`) + jetson_clocks |
| Container runtime | Docker 29.1.3 with `nvidia` runtime registered in `/etc/docker/daemon.json` |
| Inference framework | vLLM `0.21.1rc1.dev262+g33d7cbe02` via `vllm/vllm-openai:nightly-aarch64` (~27 GB image) |
| Tensor parallelism | 1 (single SoC) |

> The vLLM **`nightly-aarch64`** image is the production-relevant choice here — the NGC `nvcr.io/nvidia/vllm:26.03.post1-py3` image used by the DGX Spark case study is x86_64-only at the time of this writing.

## Framework Scores Applied

| Layer | Framework | Score | Decision |
|---|---|---|---|
| 0 — Model selection | [`02-new-model-selection`](../../frameworks/02-new-model-selection.md) | **0.78 / 1.00** (carry-over from Spark case; capability + ecosystem dims are hardware-independent) | adopt |
| 1 — Quantization | [`01-fp8-vs-nvfp4-decision`](../../frameworks/01-fp8-vs-nvfp4-decision.md) | **~0.10 / 1.00** (worse than Spark's 0.21 — hardware-support dim is **0** on `sm_110`) | **FP8** |
| 2 — Speculative decoding | [`05-mtp-speculative-decoding`](../../frameworks/05-mtp-speculative-decoding.md) | n/a (decision table) | **MTP-1** (MTP-2 measured regression at concurrency) |
| 3 — Engine tuning (A/B) | empirical 6-variant sweep | see [`04-benchmark-results.md`](04-benchmark-results.md) | **`max-num-seqs 64` + `max-num-batched-tokens 32768`** (Spark's `16384` is **not** optimal on Thor) |

Detailed per-layer reasoning lives in `01-…md` through `04-…md`.

## Final Production Configuration

```bash
docker run -d --name qwen36-fp8 \
  --runtime=nvidia --gpus all \
  --ipc=host --network=host \
  --restart unless-stopped \
  -e HF_HUB_OFFLINE=1 \
  -v "${WORKSPACE}/models/Qwen3.6-35B-A3B-FP8":/model:ro \
  -v "${WORKSPACE}/entry.sh":/entry.sh:ro \
  --entrypoint bash vllm/vllm-openai:nightly-aarch64 /entry.sh
```

…where `entry.sh` calls `vllm serve` with:

```
--max-model-len 262144
--max-num-batched-tokens 32768
--max-num-seqs 64
--gpu-memory-utilization 0.75
--kv-cache-dtype fp8
--enable-prefix-caching
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
--reasoning-parser qwen3
--enable-auto-tool-choice --tool-call-parser qwen3_xml
--trust-remote-code
--limit-mm-per-prompt '{"image":4,"video":1}'
```

A reproducible launcher with the `pytest` workaround (see [05-decision](05-decision.md)) lives in [`reproduce/`](reproduce/).

## Headline Measurements

Single-stream output token throughput (3-run median, decode-bound regime):

| Variant | 256-out | 512-out | 1024-out | 2048-out |
|---|---:|---:|---:|---:|
| `nvfp4-default` (RedHatAI NVFP4 build, `--enforce-eager`) | — | — | **2.2** | — |
| `baseline` (FP8 + KV-FP8, no MTP, `max-num-seqs 8 / batched 16384`) | 38.5 | 50.0 | 56.2 | 56.3 |
| `with-mtp1` (Spark final config replicated on Thor) | 50.7 | 52.3 | 56.8 | 66.7 |
| **`thor-tuned`** (= `with-mtp1` + `max-num-seqs 64 / batched 32768`) ⭐ | **54.1** | **65.9** | **66.7** | **66.8** |

Concurrent aggregate throughput (3-run median, `thor-tuned`):

| Output length | c=4 | c=8 | c=16 |
|---:|---:|---:|---:|
| 1024 | 152 | — | — |
| 512 | — | 234 | 303 |

Detailed tables in [`04-benchmark-results.md`](04-benchmark-results.md). Raw CSV in [`data/baseline-results.csv`](data/baseline-results.csv).

## What This Case Validates / Challenges About the Frameworks

- **Framework 01 confirmed in the strongest possible sense**: the **0.10 prediction** for "Jetson Thor + any model" maps to a **30× measured slowdown** if you naively pick NVFP4. This is the largest framework-validating signal in the repo so far.
- **Framework 05 partially modified**: MTP-1 helps on Thor as Framework 05 predicts, **but** the lift distribution is non-uniform (+32% on short outputs, +1% on medium 512-out, +18% on long 2048-out) — different from the Spark case's uniform ~+18%. MTP-2 is **worse than MTP-1 at c=4** on Thor (-9%) — same direction as Spark, larger magnitude.
- **Engine-tuning dimension is *not* Spark-portable**: Spark's best config uses `max-num-batched-tokens=16384`, `max-num-seqs=8` (implicit). On Thor, **`32768 / 64` lifts single-stream throughput by 15-26%** with no concurrency regression. This is the one Spark assumption that does **not** carry over.

## Open Questions

1. **NGC NVIDIA-distributed vLLM** (`nvcr.io/nvidia/vllm:*-py3`) on `aarch64` — does it exist for `sm_110`? If yes, does it close the ~28% efficiency gap vs Thor's 91 tok/s memory-bandwidth ceiling? (Our measured 67 tok/s = 73% efficiency.)
2. **Long-context decode steady-state** beyond the prefill measurements in this case (we measured needle-in-haystack retrieval at 150 K input but did **not** sweep decode throughput at long context). Are tokens/s the same at 100 K prefix as at 1 K? Anecdotally yes, but unmeasured.
3. **Multimodal (image + video) decode throughput** — we validated correctness (5/5 recognition tasks pass) but did not benchmark tokens/s with image tokens in the context. Vision tokens are pre-computed by the vision encoder; should not change decode throughput, but unverified.

## How to Reproduce This Case

1. **Manual**: clone the repo, `cp reproduce/env-templates/thor-tuned.env reproduce/.env`, edit, `bash reproduce/launch.sh`, then `bash tools/benchmark/run_variants.sh --only thor-tuned` (note: this case's `launch.sh` differs from the Spark version — it installs `pytest` inside the container as a workaround for a `cupy.testing` import chain triggered by MTP head registration in the `nightly-aarch64` image).
2. **Agent-driven**: pending — the agent currently emits Spark-style `--max-num-batched-tokens 16384`; if you re-run with `--target-hardware "Jetson Thor / sm_110 / 128GB / 273 GB/s LPDDR5X"`, the model-selection and quantization scores should match this case, but engine-tuning will need manual override to `32768 / 64`.

## File Map

| File | Required? | Purpose |
|---|---|---|
| `00-summary.md` | Required | This file |
| `01-applying-framework-02.md` | Required | Layer 0 (model selection) — short, carries over from Spark |
| `02-applying-framework-01.md` | Required | Layer 1 (FP8 vs NVFP4) re-scored for `sm_110` |
| `03-applying-framework-05.md` | Required | Layer 2 (MTP-1 over MTP-2 / no-spec) |
| `04-benchmark-results.md` | Required | Full A/B sweep + multimodal + needle |
| `05-decision.md` | Required | Decision chain + the `pytest` workaround you must know about |
| `data/baseline-results.csv` | Required | Raw measurements |
| `reproduce/launch.sh` | Required | Thor-adapted launcher (handles `pytest` install) |
| `reproduce/env-templates/*.env` | Required | One env per variant |
| `reproduce/README.md` | Required | Reproduce instructions |
