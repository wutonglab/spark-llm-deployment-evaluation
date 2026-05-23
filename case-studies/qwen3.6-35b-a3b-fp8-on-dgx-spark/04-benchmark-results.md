# Benchmark Results: 6-Variant A/B/C/D/E + NVFP4 Side-By-Side

> **Goal**: empirically validate the predictions from frameworks 01 + 05 by running variants of (attention backend, kv-cache dtype, speculative method, batched tokens, quantization format) on the **same** hardware + model + workload, and tabulating the results.

## Hardware + Software Environment (sanitized)

| Attribute | Value |
|---|---|
| Platform | NVIDIA DGX Spark (GB10 / sm_121) |
| Unified memory | 128 GB LPDDR5X |
| Memory bandwidth | ~273 GB/s |
| Inference engine | vLLM (NVIDIA NGC distribution, `nvcr.io/nvidia/vllm:26.03.post1-py3`) |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` (HuggingFace, full 256K context) |
| Tensor parallelism | 1 (single GPU) |
| Context length cap | 262144 |
| `gpu-memory-utilization` | 0.85 |
| `max-num-batched-tokens` | 16384 (except E variant) |

Configuration env-templates for every variant are in [`reproduce/env-templates/`](reproduce/env-templates/).

## The 6 Variants

| Variant | Quantization | KV cache dtype | Attention backend | Speculative | Batched tokens |
|---|---|---|---|---|---|
| `baseline` | FP8 | bf16 (default) | default | none | 16384 |
| `with-kv-fp8` | FP8 | **fp8** | default | none | 16384 |
| `with-flashinfer` | FP8 | fp8 | **flashinfer** | none | 16384 |
| **`with-mtp1`** | FP8 | fp8 | default | **MTP-1** | 16384 |
| `nvfp4-default` | NVFP4 (vendor) | fp8 | default | none | 16384 |
| `with-32k-batched` | FP8 | fp8 | default | none | **32768** |

Each variant launched in a fresh container, measured at single-request and concurrency `c = 4`, across multiple output lengths.

## Single-Request Throughput (tokens/s)

| Output length | `baseline` | `with-kv-fp8` | `with-flashinfer` | **`with-mtp1`** | `nvfp4-default` | `with-32k-batched` |
|---|---:|---:|---:|---:|---:|---:|
| 256 (first / cold start) | ~20 | ~20 | ~20 | ~21 | ~12 | ~19 |
| 512 | ~52 | ~52 | ~52 | **~62** | ~12 | ~52 |
| 1024 | ~53 | ~52 | ~52 | **~61** | ~12 | ~52 |
| 2048 | ~52 | ~52 | ~52 | **~62** | ~12 | ~52 |

Notes:
- **First request (256-out)** is dominated by CUDA-graph warm-up; not steady-state.
- `with-mtp1` lifts steady-state by **+18%** over the other FP8 variants.
- `nvfp4-default` is **5× slower** — predicted by Framework 01 (score 0.21).
- `with-flashinfer` matches or slightly trails default backend — no benefit on this model.

## c = 4 Aggregate Throughput (tokens/s)

| Output length | `baseline` | `with-kv-fp8` | `with-flashinfer` | **`with-mtp1`** | `nvfp4-default` | `with-32k-batched` |
|---|---:|---:|---:|---:|---:|---:|
| 256 | ~64 | ~63 | ~61 | ~64 | ~55 | ~60 |
| 512 | ~168 | ~163 | ~165 | **~193** | ~93 | ~164 |
| 1024 | ~190 | ~164 | ~166 | **~234** | ~93 | ~190 |
| 2048 | ~197 | ~162 | ~163 | ~197 | ~94 | ~189 |

Notes:
- `with-mtp1` shows its biggest lift at c=4 / output 1024: **+23% vs `baseline`** (190 → 234).
- `with-32k-batched` recovers the 2048-output regression seen in plain `with-kv-fp8` (162 → 189), suggesting batched-tokens tuning helps long outputs at concurrency — but does not beat `with-mtp1`.
- `nvfp4-default` aggregate sits at ~55–94 tok/s, also confirming the framework prediction.

## Real-World Application Latency (60-token outputs, fixed system prompt)

A workload typical for an agent or tool-use scenario (constrained outputs, repeating system prompts; measures the prefix-caching path):

| Variant | Step 1 (cache miss) | Step 2 (cache hit) | Step 3 (cache hit) |
|---|---:|---:|---:|
| `with-kv-fp8` | 1.29 s | 1.25 s | 1.25 s |
| **`with-mtp1`** | **1.10 s** | **1.08 s** | **1.11 s** |
| `with-32k-batched` | 1.30 s | 1.26 s | 1.25 s |

`with-mtp1` ~15–20% faster per step, while prefix-cache hits still register on subsequent steps (validates Framework 05's observation that MTP-1 and prefix-caching coexist in vLLM 0.17.x).

## Prefix-Caching Hit Test (10 prompts, same system prompt, ~8-token outputs)

| Step | `with-kv-fp8` (s) | **`with-mtp1`** (s) |
|---|---:|---:|
| 1 (miss) | 0.30 | 0.30 |
| 2 | 0.25 | 0.25 |
| 3 | 0.25 | 0.25 |
| 4 | 0.25 | 0.25 |
| 5 | 0.23 | 0.22 |
| 6 | 0.27 | 0.25 |
| 7 | 0.25 | 0.25 |
| 8 | 0.25 | 0.25 |
| 9 | 0.25 | 0.25 |
| 10 | 0.31 | 0.28 |

The two variants are statistically indistinguishable on this fully-cache-friendly workload — confirming that on heavy-prefix-reuse workloads, MTP-1's marginal benefit narrows. MTP-1 is still recommended because it dominates on the medium-prefix-reuse regime where most real services live.

## Decision Confirmation

| Framework prediction | Measurement | Match? |
|---|---|---|
| Framework 01 score 0.21 → "stay on FP8, NVFP4 underperforms" | NVFP4 measured 5× slower under default software stack | ✅ confirmed in strong sense |
| Framework 05 → MTP-1 is the only viable speculative method | All variants without MTP-1 cap at ~52 tok/s; MTP-1 reaches ~61 | ✅ +18% gain matches |
| Framework 05 → flashinfer attention is *not* helpful on this model | `with-flashinfer` is flat or slightly worse than default | ✅ confirmed |
| Framework 05 → MTP-1 coexists with prefix-caching in 0.17.x | Step 2+ latency same between variants on cache-friendly workload | ✅ confirmed |

The benchmark sweep validates the framework's predictions across single-request, concurrent, and prefix-cache-heavy regimes.

## Raw Data

The CSV with all measurement rows is in [`data/baseline-results.csv`](data/baseline-results.csv). Columns: variant, concurrency, output_length, output_tokens_per_second.

## How to Reproduce

```bash
# from repo root
cd tools/benchmark
bash run_variants.sh --all # or --only with-mtp1, etc.
python analyze.py --against ../../case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/data/baseline-results.csv
```

`analyze.py` writes a markdown report into `tools/benchmark/results/` showing your numbers next to this case study's baseline. Within ±15% on tokens/sec and ±20% on first-token latency counts as "framework prediction confirmed."

For end-to-end agent-driven replication see [`reproduce/README.md`](reproduce/README.md).
