# Future Case Studies — Wanted

This directory is a placeholder for case studies we'd like to see contributed. Pick one, run it, and PR using [`../_template.md`](../_template.md).

## High Priority

### Different hardware tier, same model

- **Qwen3.6-35B-A3B-FP8 on RTX PRO 6000 Blackwell (sm_120)** — confirm Framework 01's prediction that the score flips and NVFP4 wins
- **Qwen3.6-35B-A3B-FP8 on B200 (sm_100a)** — datacenter Blackwell; NVFP4 should dominate
- **Qwen3.6-35B-A3B-FP8 on Jetson Thor (sm_110)** — different aarch64 / Blackwell-edge profile

### Different model family, same hardware (DGX Spark)

- **Llama-3.3 70B Dense on DGX Spark** — Dense vs MoE comparison; check whether NVFP4 score changes due to compute/memory ratio
- **DeepSeek-V3 on DGX Spark** — MoE with different activation profile
- **GLM-4.5 on DGX Spark** — Chinese-optimized model with different ecosystem maturity

### Different workload regime

- **Long-context-only workload** (mostly > 32K) — see whether the engine-tuning conclusion changes (e.g. flashinfer might pay off here)
- **Tool-calling agent at scale** (c ≥ 16) — high-concurrency regime
- **Multimodal (Qwen3.6-VL)** — adds Framework 02 multimodal dim

### Different quantization community work

- **NVFP4 via the community patch image** (`avarok/dgx-vllm-nvfp4-kernel`) on DGX Spark — confirm the projected ~50 tok/s with Marlin path
- **AWQ-INT4 cross-comparison** on the same model + hardware

## Lower Priority

- Closed-API model integration patterns (the frameworks need a separate evaluation path; Framework 02 has notes but no case yet)
- Edge / small-model regimes (< 14B Dense)

## How to Adopt

If you want to take one of these on:

1. Open an Issue with the `case-study-submission` label saying which case you're running
2. Run it
3. Submit a PR per [`../README.md`](../README.md)

If multiple people offer the same case, the first PR to merge wins; subsequent submissions become "alternative measurements" appended as data points to the original case's `data/` directory.
