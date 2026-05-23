# Glossary

| Term | Definition |
|---|---|
| **A3B** | "Activated 3 Billion" — refers to the active parameter count per token in a Mixture-of-Experts model. Qwen3.6-35B-A3B has 35B total parameters but only ~3B active per forward pass. |
| **Acceptance rate** | In speculative decoding, the fraction of speculated tokens that the main model verifies as correct. Higher acceptance → larger speedup. |
| **Block-scaled quantization** | A class of low-precision formats (NVFP4, MXFP4, MXFP8) where small blocks of weights share a scaling factor, preserving precision better than naive uniform quantization. |
| **BF16** | Brain Float 16 — 16-bit floating-point format used as the "high-precision" baseline for LLM inference. 1 sign + 8 exponent + 7 mantissa bits. |
| **Blackwell** | NVIDIA GPU architecture generation following Hopper. Includes datacenter parts (B100/B200, sm_100a/sm_103a) and consumer/workstation parts (RTX 5090, RTX PRO 6000 Blackwell, GB10/Spark — sm_120/sm_121). |
| **CUDA graph** | A pre-recorded sequence of CUDA operations replayed as a single submission. Reduces kernel-launch overhead during steady-state inference. |
| **CUTLASS** | NVIDIA's open-source CUDA template library for high-performance linear algebra. Often where new quantization formats land first. |
| **Dequant** | The act of converting low-precision weights back to a higher-precision format (e.g. NVFP4 → BF16) on-the-fly during matmul. Adds overhead if not done in a fused kernel. |
| **DFlash** | A block-diffusion speculative-decoding method published by z-lab. Drafts multiple future tokens in parallel rather than autoregressively. |
| **EAGLE / EAGLE-3** | A speculative-decoding method that trains a lightweight draft head reading the target model's internal hidden states. EAGLE-3 is the current production-standard variant. |
| **FP8** | 8-bit floating-point format. Two common encodings: E4M3 (4-bit exp, 3-bit mantissa) and E5M2. Standard on Hopper/Blackwell tensor cores. |
| **flashinfer** | Inference attention backend library, alternative to default Flash-Attention. Sometimes faster on some model architectures, slower on others. |
| **GB10** | NVIDIA's "Grace Blackwell 10" superchip powering DGX Spark. SM architecture: `sm_121`. Workstation-tier — has `mma.kind::mxf4` but lacks `tcgen05` instructions. |
| **HBM** | High Bandwidth Memory. Stacked DRAM used on datacenter GPUs (HBM2e, HBM3, HBM3e). Provides ≥ 1 TB/s bandwidth — far higher than consumer GDDR. |
| **KV cache** | The cached attention keys and values produced during prefill and reused during decode. Sized as `2 × layers × heads × head_dim × seq_len × batch_size`. Often dominates memory for long-context serving. |
| **`kv-cache-dtype`** | vLLM flag controlling KV cache storage precision (`auto`/`bf16` vs `fp8`). FP8 halves footprint with negligible accuracy impact in most cases. |
| **LMArena** | Crowdsourced human-judged LLM leaderboard with blind elo ratings. Often closer to user-perceived quality than benchmark numbers. |
| **LiveBench** | Monthly-refresh contamination-resistant benchmark for general LLM capability. |
| **Linear attention / Mamba** | Sub-quadratic alternatives to standard attention; use `conv1d`-style kernels that are compute-bound rather than memory-bound. Qwen3.6's "Gated DeltaNet" layers are of this family. |
| **LPDDR5X** | Low-power DDR5 memory used in mobile + edge platforms (including DGX Spark unified memory). Lower bandwidth than HBM (~273 GB/s on Spark vs ~5 TB/s on B200). |
| **Marlin** | A CUTLASS-based kernel family for quantized GEMM. The Marlin NVFP4 backend can dequant→BF16 on sm_121 where the native FP4 path is unavailable. |
| **MMLU / MMLU-Pro** | Multi-task language understanding benchmarks. MMLU-Pro is the harder, contamination-resistant variant. |
| **MoE (Mixture of Experts)** | An architecture where each token is routed to a subset of the model's "experts" rather than passing through all parameters. Trades parameter count (large) for per-token compute (small). |
| **MTP (Multi-Token Prediction)** | Speculative-decoding method where the model itself has an extra prediction head that speculates the next-next token. Used by DeepSeek-V3 and Qwen3.6. The head ships in the model checkpoint. |
| **`num_speculative_tokens`** | vLLM speculative-config parameter. Setting to 1 (MTP-1) is the most stable; higher values increase risk of instability or known bugs. |
| **NVFP4** | NVIDIA-native 4-bit floating-point format with block-scaled quantization. Designed for Blackwell tensor cores. Often confused with general FP4 — NVFP4 is the specific variant NVIDIA backs. |
| **Prefix caching** | vLLM feature that caches the KV state of repeated input prefixes (e.g. shared system prompts) across requests. Major throughput win for agent / multi-turn workloads. |
| **Prefill vs Decode** | Two phases of LLM inference: prefill processes the input tokens (parallel, compute-heavy); decode generates output tokens one at a time (memory-bandwidth-bound). |
| **PTX** | NVIDIA's intermediate assembly language. Lower-level than CUDA C++; sometimes the only way to access specific tensor-core instructions. |
| **`sm_121`** | The compute capability identifier for DGX Spark (GB10). Workstation-tier Blackwell with some datacenter features (e.g. `mma.kind::mxf4`) but lacking others (`tcgen05`, 2-SM MMA). |
| **Speculative decoding** | Family of techniques that generate multiple candidate tokens via a fast "draft" pass and verify them with the main model in parallel. Trades a small amount of GPU work for latency. |
| **Speculator** | A draft model packaged for use with a specific inference engine. RedHatAI publishes speculators for several Qwen and Llama variants. |
| **`tcgen05`** | Tensor-core gen 5 instructions on datacenter Blackwell. Used by the fastest NVFP4 paths. Not available on `sm_120`/`sm_121`. |
| **TTFT / ITL** | Time-To-First-Token and Inter-Token-Latency. Two key serving metrics: TTFT for interactivity, ITL for sustained generation rate. |
| **vLLM** | Open-source LLM serving framework. Provides OpenAI-compatible HTTP API + many production features (continuous batching, prefix caching, speculative decoding, KV-cache quantization). |
