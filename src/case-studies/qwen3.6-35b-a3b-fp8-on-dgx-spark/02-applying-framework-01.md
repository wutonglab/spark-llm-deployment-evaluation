# Applying Framework 01 (FP8 vs NVFP4) to Qwen3.6-35B-A3B on DGX Spark

> **Goal**: decide whether to deploy Qwen3.6-35B-A3B in FP8 or NVFP4 quantization on DGX Spark (GB10 / sm_121).
>
> **Framework**: [`01-fp8-vs-nvfp4-decision.md`](../../frameworks/01-fp8-vs-nvfp4-decision.md). Channel evidence per [`03-inference-research-channels.md`](../../frameworks/03-inference-research-channels.md).

## Score Summary

| Dimension | Score | Max |
|---|---:|---:|
| 1. Hardware capability (sm_121) | **0.15** | 0.30 |
| 2. Memory bandwidth (273 GB/s LPDDR5X) | **0.05** | 0.30 |
| 3. Software stack maturity | **0.00** | 0.20 |
| 4. Model architecture (MoE + linear-attention hybrid) | **0.03** | 0.10 |
| 5. Workload pattern (general LLM API service) | **0.05** | 0.10 |
| Operational cost adjustment | **−0.07** | (subtractive) |
| **Total** | **0.21** | 1.00 |

**Threshold**: < 0.30 → **stay on FP8**. ❌ NVFP4

This score is **far from the 0.30 minimum threshold** — not a close call. Subsequent measurements (12 tok/s NVFP4 vs 61 tok/s FP8+MTP-1) confirm the framework's verdict is correct in the strong sense.

---

## Dimension-by-Dimension

### Dim 1: Hardware Capability (0.15 / 0.30)

GB10 is sm_121 — a consumer/workstation-tier Blackwell. NVFP4 hardware is **partially** present but key instructions are missing.

| Hardware feature | sm_121 (GB10) | sm_100a (B200) |
|---|---|---|
| `mma.kind::mxf4` block-scaled MMA | ✅ present | ✅ present |
| `tcgen05` (tensor-core gen-5) | ❌ missing | ✅ present |
| 2-SM cooperative MMA | ❌ missing | ✅ present |
| Shared memory per SM | ~101 KB | larger |

**Score**: 0.15 — mid of the sm_120/121 range (0.10–0.20). Partial hardware support means software needs to work around the missing instructions, and even when patched the tile-size space is constrained.

**Channel citations** (per Framework 03):
- *GitHub issue tracker*: CUTLASS issue #2800 documents the `BlockScaledMmaOp.admissible_archs` exclusion of sm_121
- *Vendor forum*: DGX Spark sub-forum posts comparing sm_121 vs sm_100a for FP4 workloads
- *Community kernel research*: independent CUTLASS DSL patches demonstrating 356 TFLOPS NVFP4 is achievable on sm_121, but only with custom builds

### Dim 2: Memory Bandwidth (0.05 / 0.30)

DGX Spark uses LPDDR5X unified memory at ~273 GB/s. This is **5×–7× lower** than datacenter Blackwell (B200 HBM3e ~5 TB/s) and ~5× below workstation Blackwell (RTX PRO 6000 Blackwell GDDR7 ~1.4 TB/s).

| Architecture | Bandwidth | Score band |
|---|---|---|
| ≥ 1.5 TB/s (HBM3e datacenter, GDDR7 workstation) | best | 0.30 |
| 500–1500 GB/s | medium | 0.15–0.25 |
| **< 500 GB/s (DGX Spark 273 GB/s)** | **worst** | **0.05–0.10** |

**Score**: 0.05 — bottom of the < 500 GB/s band. NVFP4 weights are half the size of FP8, but dequant overhead on sm_121 eats most of the bandwidth savings.

**Channel citations**:
- DGX Spark hardware spec sheet (vendor publication)
- Cross-comparison blog posts measuring inference bandwidth utilization on this class of hardware

### Dim 3: Software Stack Maturity (0.00 / 0.20)

Five sub-checks; each ±0.05, net floor at 0.

| Check | + or − | Reasoning |
|---|---|---|
| CUTLASS default `admissible_archs` includes sm_121 | **−0.05** | Excluded per CUTLASS #2800 |
| vLLM mainline image runs NVFP4 fast path on sm_121 | **−0.05** | Default falls back to dequant→BF16 path |
| Upstream PR landed | **−0.05** | Patches exist in community forks only, not yet upstream-merged for sm_121 |
| Patch image updated in last 30 days | **−0.05** | Best-known community image (`avarok/dgx-vllm-nvfp4-kernel`) hadn't been updated in months at measurement time |
| ≥ 3 independent production-stable reports | **−0.05** | Community reports are positive but small-N |

Raw subtotal: **−0.25**. Floored to **0.00**.

**Channel citations**:
- *GitHub*: CUTLASS issue #2800, vLLM PRs #37700 / #37725 in progress
- *Docker Hub*: `avarok/dgx-vllm-nvfp4-kernel` last-updated timestamp + pull counts
- *Independent blogs*: third-party measurements indicate functionality but warn about lag

### Dim 4: Model Architecture (0.03 / 0.10)

Qwen3.6-35B-A3B is a **hybrid MoE + linear-attention** model:
- 35B total parameters, ~3B activated per token (A3B)
- ~70% of layers are linear-attention (Mamba-style conv1d kernels)
- ~30% are standard attention
- MTP head adds ~850 MB of BF16-kept weights (cannot quantize)

**Why this hurts NVFP4 gain**:
- Conv1d on the Mamba layers is **compute-bound**, not memory-bound. Halving weight precision doesn't accelerate the kernel.
- The MTP head, `lm_head`, and norms remain BF16. They're a single-digit % of the model but eat away at the "weight read savings" story.
- MoE expert weights are quantizable, but only the active experts are read per token — the savings are amortized over fewer reads than for a Dense model.

**Score**: 0.03 — in the "MoE small activation + heavy Mamba" range.

**Channel citations**:
- Qwen3.6 architecture documentation (vendor)
- Independent analyses of MoE+Mamba inference cost breakdowns

### Dim 5: Workload Pattern (0.05 / 0.10)

The deployment target is a general LLM API service: a mix of:
- Low-concurrency interactive use (chat-like)
- Periodic batch jobs at c = 4–8
- Occasional long-context calls (8K–32K)

This is in the **medium concurrency** band: 0.06 with a small downward adjustment for the prefix-cache opportunities (system prompts repeat across sessions). **Score: 0.05**.

### Operational Cost: −0.07

Going to NVFP4 on sm_121 today requires the community patch container (e.g. `avarok/dgx-vllm-nvfp4-kernel`). The image:
- Is built on a vLLM revision **older than the current mainline** (lacks recent features like newer tool-call parsers)
- Hasn't been updated in months at measurement time
- Requires custom flags / env vars that the user must maintain

Per the framework's operational-cost guide: "Custom docker image / rebuilt CUTLASS required → −0.07". Applied.

---

## Cross-Verification

Per Framework 03's "≥ 2 sources for any +N% / fully supported claim" rule:

| Claim | Source A | Source B |
|---|---|---|
| sm_121 lacks `tcgen05` | CUTLASS issue #2800 | Vendor forum discussion of sm_121 architecture |
| Mainline vLLM falls back to BF16 on sm_121 NVFP4 | vLLM mainline issue tracker | Independent measurement showing dequant overhead |
| Community patch image works but is stale | Docker Hub metadata | Direct repo inspection |
| Hybrid architecture limits quantization gain | Vendor architecture description | Third-party kernel profiling write-up |

---

## Outcome

Total **0.21 < 0.30** → **stay on FP8**.

The decision is robust: even if every dim were nudged by one band (e.g. bandwidth to 0.10, architecture to 0.05, op cost to −0.03), the total would still sit ~0.30, well below the "1-day side-by-side test" band.

Subsequent measurements (see [`04-benchmark-results.md`](04-benchmark-results.md)) confirm:
- `nvfp4-default` variant: ~12 tok/s single-request
- `with-mtp1` (FP8 + MTP-1): ~61 tok/s single-request

The framework's prediction is corroborated by a **5× measured throughput gap** in the predicted direction.

---

## What Would Change This Decision

| Trigger | Likely new score impact |
|---|---|
| CUTLASS merges sm_121 default support | Dim 3 +0.10 |
| Mainline vLLM nvcr image bundles the sm_121 NVFP4 path | Dim 3 +0.05; op cost +0.07 |
| New DGX Spark refresh with higher-bandwidth memory | Dim 2 + (e.g. +0.15) |
| New community-stable patch image with monthly cadence | Op cost +0.04 |

We recommend re-running this framework quarterly until upstream sm_121 NVFP4 support stabilizes.
