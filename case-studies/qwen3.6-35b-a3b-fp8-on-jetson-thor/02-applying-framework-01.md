# Layer 1 · Applying Framework 01 (FP8 vs NVFP4) on Jetson Thor

This is the **most important re-scoring** in the case. The Spark case scored 0.21 on Framework 01 ("stay on FP8, but it's close"). Thor scores **~0.10** — lower than Spark on *two* dimensions, with the **hardware-support dimension dropping to 0**.

## Score: **~0.10 / 1.00 → strongly FP8**

| Dimension | Weight | Spark (`sm_121`) | **Thor (`sm_110`)** | Why Thor differs |
|---|---:|---:|---:|---|
| Hardware support (native FP4 silicon) | 0.30 | 0.15 (`sm_121` has limited FP4 paths but no `tcgen05`) | **0.00** | `sm_110` has **no native FP4 tensor cores at all**. Any NVFP4 GEMM goes through emulation or scale-and-dequant paths in the kernel, fully eating the 4× bandwidth advantage. |
| Memory bandwidth (favors FP4 if hardware exists) | 0.25 | 0.55 (273 GB/s, modest but real) | 0.55 (same ~273 GB/s) | Bandwidth helps FP4 *if* there's hardware to consume it. Without it, no benefit. |
| Software stack maturity (NVFP4 kernels in shipping releases) | 0.20 | 0.20 (improving) | **0.10** | `aarch64` lags `x86_64` by 1-2 weeks on most vLLM nightly fixes; NGC NV-distributed image not available for `aarch64` at the time of writing. |
| Model architecture (MoE active-params → quantization sensitivity) | 0.15 | 0.30 | 0.30 (same model) | A3B model means 3 GB active footprint at FP8, ~1.5 GB at FP4. Either fits Thor's KV-cache budget easily. |
| Workload pattern (mix of single-stream vs concurrent vs long-context) | 0.10 | 0.20 | 0.20 (same) | General-purpose API service expects all three. |
| **Weighted total** | | **~0.21** | **~0.10** | **strongly FP8** |

## Measurement that confirms the score

We **did** run NVFP4 on Thor (the `nvfp4-default` variant in [`04-benchmark-results.md`](04-benchmark-results.md)) before falling back to FP8 — for the explicit purpose of validating Framework 01 with hard numbers, not assuming.

| Variant | Same model, same hardware | Single-stream 1024-out tok/s |
|---|---|---:|
| RedHatAI NVFP4 build with `--enforce-eager` | yes | **2.2** |
| Qwen3.6-35B-A3B-FP8 + MTP-1, Thor-tuned | yes | **66.7** |

**Ratio: 30.3× slowdown for NVFP4.** This is the largest framework-validating signal in the repository. The Spark case showed 5× (61 vs 12 tok/s); Thor shows 30× because `sm_110`'s hardware-support score is **0 vs Spark's 0.15**.

The 2.2 tok/s NVFP4 number on Thor was measured **before** we knew about the engine-tuning optimizations (`--enforce-eager` was on for safety on first launch). A non-eager NVFP4 run would likely land somewhere in the 5-12 tok/s range based on the Spark comparison — but **still ~5× slower than the FP8 path**, so the qualitative conclusion (avoid NVFP4 on Thor) is robust.

## Channels Consulted (per Framework 03)

1. **NVIDIA documentation**: confirmed `sm_110` is a separate Blackwell-class variant from `sm_121` ([Jetson Thor product brief](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) and CUDA 13 release notes covering arch capabilities).
2. **vLLM source**: `vllm/model_executor/layers/quantization/nvfp4.py` and `__init__.py` — at startup vLLM logs `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM` on `sm_110`, which is a CUTLASS fallback path (not the native tcgen05 path that `sm_121` and `sm_100`/`sm_103` can use). This kernel exists but is bandwidth-limited because the multiplication unit isn't FP4-native.
3. **Repo's own porting guide**: [`src/hardware-portability/porting-to-other-gpus.md`](../../src/hardware-portability/porting-to-other-gpus.md) Jetson Thor section predicted hardware score `0` and "FP8 (or BF16) only" — this case is the empirical confirmation.

## Implication for the porting guide

The porting-guide section for Jetson Thor was written as a hypothesis. After this case, it can be cited as **empirically confirmed at the 30× level** — meaning anyone running NVFP4-targeted models on Thor should expect double-digit slowdowns relative to the FP8 path of the same model. A patch updating the porting guide is included in this PR.
