# Framework 01: FP8 → NVFP4 Quantization Decision

> **Purpose**: scoring rubric for "should we switch this model from FP8 to NVFP4?" — the most common quantization-format question on Blackwell-class hardware.

## Rule

Score the candidate model+hardware combination along the 5 dimensions below (minus operational cost). Threshold:

- `< 0.30`: stay on FP8
- `0.30 – 0.70`: 1-day side-by-side test required before deciding
- `≥ 0.70`: switch to NVFP4

## Why

NVFP4 is not "free 2x speedup from halving weight bits." Across measured workloads:

- On consumer-class Blackwell (sm_120 / sm_121, e.g. DGX Spark GB10): NVFP4 was **~5× slower than FP8** under the default software stack — measured `~12 tok/s` vs `~61 tok/s`.
- On datacenter-class Blackwell (sm_100a, e.g. B200): NVFP4 is the genuine "1 PFLOP" path.
- On workstation Blackwell (sm_120, e.g. RTX PRO 6000 Blackwell): community-reported `~104 tok/s` single-request with NVFP4 + speculative — clearly beats FP8.

**Same Blackwell generation, different SM, 5–10× performance gap.** The framework forces you to score each axis explicitly.

## How to Apply

Pair this with [`03-inference-research-channels.md`](../frameworks/03-inference-research-channels.md). Walk the channels, score the 5 dims, compare to threshold. See worked example: Qwen3.6-35B-A3B on DGX Spark (total score 0.21 → stay on FP8; validated by case study).

---

## The 5 Dimensions (weights sum to 1.0)

### Dimension 1: Hardware Capability (weight 0.30)

| SM architecture | Representative GPU | NVFP4 fast path | Score |
|---|---|---|---|
| **sm_100a / sm_103a** | H100/H200/B200 datacenter Blackwell | ✅ Full (tcgen05 + 2-SM MMA + large SMEM) | 0.30 |
| **sm_120 / sm_121** | RTX 5090 / RTX PRO 6000 Blackwell / GB10 (DGX Spark) | ⚠️ Partial (`mma.kind::mxf4` available but tcgen05 missing; SMEM only ~101 KB) | 0.10 – 0.20 |
| **sm_89 / sm_90** | Ada / Hopper non-datacenter | ❌ No native FP4 tensor core | 0 |
| **sm_80 and below** | Ampere etc. | ❌ | 0 |

### Dimension 2: Memory Bandwidth (weight 0.30)

| Bandwidth | NVFP4 vs FP8 expected gain | Score |
|---|---|---|
| **≥ 1.5 TB/s** (B200 HBM3e ~5 TB/s, RTX PRO 6000 Blackwell GDDR7 ~1.4 TB/s) | 30 – 50% (bandwidth-bound; halving weights matters) | 0.30 |
| **500 – 1500 GB/s** (H100/A100-class HBM) | 15 – 30% | 0.15 – 0.25 |
| **< 500 GB/s** (DGX Spark ~273 GB/s LPDDR5X, low-bandwidth consumer GPUs) | ≤ 15%, often eaten by dequant overhead | 0.05 – 0.10 |

### Dimension 3: Software Stack Maturity (weight 0.20)

Apply ±0.05 per checked item; floor at 0:

| Check | + | − |
|---|---|---|
| CUTLASS default `admissible_archs` includes target SM | +0.05 | −0.05 |
| vLLM / TRT-LLM / SGLang official image runs NVFP4 fast path by default | +0.05 | −0.05 |
| Upstream PR landed (not a community fork) | +0.05 | −0.05 |
| Image / PR updated in last 30 days | +0.05 | −0.05 |
| ≥ 3 independent community reports of production stability | +0.05 | −0.05 |

**As of late 2025 / early 2026 snapshot**: sm_120/121 default `BlockScaledMmaOp.admissible_archs` in CUTLASS 4.4 [excludes sm_121](https://github.com/NVIDIA/cutlass/issues/2800); the mainline vLLM container falls back to dequant→BF16 path; the only path to fast-path NVFP4 on sm_121 today is community patch images (e.g. Avarok's `dgx-vllm-nvfp4-kernel`) that haven't been updated in months. Net software-stack score on Spark today: **≤ 0.05**.

### Dimension 4: Model Architecture Fit (weight 0.10)

| Trait | NVFP4 benefit | Score |
|---|---|---|
| Dense ≥ 30B | High (weight read dominates) | 0.10 |
| MoE with ≥ 10B activated | Medium-high | 0.08 |
| MoE with 3–10B activated (e.g. Qwen3.6-35B-A3B) | Medium | 0.05 |
| Mamba / linear-attention occupies > 50% of layers | Low (conv1d is compute-bound; quantization barely helps) | 0.02 |
| Significant BF16-kept components (MTP head / `lm_head` / norms) > 5% of weights | Discount | −0.02 |
| < 10B model (usually compute-bound) | Weak | 0.02 |

### Dimension 5: Workload Pattern (weight 0.10)

| Pattern | NVFP4 vs FP8 | Score |
|---|---|---|
| **Low-concurrency single-request long-output** (chat / agent / long-doc generation) | Best | 0.10 |
| Medium concurrency (c = 2–8) | Medium | 0.06 |
| **High-concurrency batch** (c ≥ 16) | Small (compute-bound dominates) | 0.03 |
| **High prefix-cache hit rate** (fixed system prompt, multi-turn) | Small (prefill already amortized) | 0.03 |
| **Very long context** (> 32K) | Discounted (KV-cache traffic dominates; weights matter less) | 0.04 |
| **Accuracy-sensitive** (math / code / legal / medical) | High risk; require independent eval | −0.05 |

### Operational Cost (−0.10 to 0)

| Situation | Adjustment |
|---|---|
| Production-grade official image, drop-in | 0 |
| Minor patch needed (env-var toggle / single-file fix) | −0.03 |
| Custom docker image / rebuilt CUTLASS required | −0.07 |
| Monkey-patching upstream + maintaining a fork | −0.10 |

---

## Decision Thresholds

```
total = hardware + bandwidth + software + architecture + workload − op_cost
```

| Total | Decision |
|---|---|
| ≥ 0.70 | **Switch to NVFP4** (high ROI) |
| 0.50 – 0.70 | **Prefer NVFP4** with 1–2 day production accuracy validation |
| 0.30 – 0.50 | **1-day side-by-side test** required (FP8 vs NVFP4 head-to-head) |
| 0.15 – 0.30 | **Stay on FP8** (NVFP4 gain doesn't offset risk) |
| < 0.15 | **Definitively stay on FP8 / BF16** (NVFP4 is meaningless here) |

---

## Operational Checklist (walk in order; any ❌ kills the candidate)

1. **Hardware**: target SM = sm_100a / sm_103a?
 - Yes → likely worth it; continue
 - sm_120 / sm_121 → software stack is the deciding factor; continue
 - sm_89 or older → ❌ stay on FP8

2. **Bandwidth**: ≥ 1 TB/s?
 - Yes → continue
 - < 500 GB/s (e.g. DGX Spark) → ⚠️ likely not worth it

3. **Software**: chosen inference framework has an official production image on target SM that runs the NVFP4 fast path?
 - Yes → OK
 - Only patch forks available → assess stability carefully
 - Neither → ❌ stay on FP8

4. **Model**: Dense ≥ 30B with Mamba/linear-attention < 30%?
 - Yes → strong gain
 - MoE small-activation + heavy Mamba → weak; proceed with caution
 - < 10B → ❌ usually not worth it

5. **Workload**: low-concurrency single-request dominant?
 - Yes → maximal gain
 - High concurrency / high prefix-cache hit → small gain

6. **Accuracy**: business can tolerate 4-bit quantization error?
 - Yes + independent eval done → proceed
 - No / no eval → ❌ stay on FP8 + run an eval first

---

## Worked Example: Qwen3.6-35B-A3B on DGX Spark (GB10 / sm_121)

| Dimension | Score | Reasoning |
|---|---|---|
| 1. Hardware (GB10 sm_121) | **0.15 / 0.30** | `mma.kind::mxf4` present but tcgen05 / 2-SM MMA missing; SMEM only 101 KB |
| 2. Bandwidth (~273 GB/s LPDDR5X) | **0.05 / 0.30** | Far below datacenter threshold |
| 3. Software (mainline vLLM nvcr image + CUTLASS sm_121 exclusion) | **0.00 / 0.20** | −0.05 × 5 negative factors, floored at 0 |
| 4. Architecture (35B-A3B MoE + ~70% linear-attention) | **0.03 / 0.10** | High Mamba share dampens NVFP4 benefit |
| 5. Workload (general LLM API service) | **0.05 / 0.10** | Mixed concurrency profile |
| Op cost (community patch image, dated, missing recent features) | **−0.07** | Custom docker + ecosystem lag |
| **Total** | **0.21** | **< 0.30 → stay on FP8** ✅ |

**Empirical validation**: full A/B/C/D/E variant study under [`../case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/`](../case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/) measured single-request `with-mtp1` (FP8 + MTP-1) at **~61 tok/s** vs `nvfp4-default` at **~12 tok/s** under identical software. The framework's "stay on FP8" prediction matched.

---

## Hypothetical Examples (framework predictions)

| Scenario | Estimated total | Decision |
|---|---|---|
| B200 datacenter + Llama-3 70B Dense + chat | ~0.85 | ✅ strong push for NVFP4 |
| RTX PRO 6000 Blackwell + Qwen3.6-27B Dense + agent | ~0.65 | ✅ switch (community measurements consistent) |
| DGX Spark + a Coder-class MoE 80B FP8 model | ~0.40 | ⚠️ side-by-side test |
| **DGX Spark + Qwen3.6-35B-A3B MoE+Mamba (this case)** | **0.21** | ❌ stay on FP8 |
| A100 + Llama-3 7B Dense | ~0.15 | ❌ stay on FP8 (compute-bound + no FP4 silicon) |
| Jetson Thor sm_110 + any model | ~0.10 | ❌ stay on FP8 (hardware + bandwidth + software all weak) |

---

## Framework Boundaries

- **Snapshot in time**: software-stack scores reflect the NVFP4 ecosystem maturity in late 2025 / early 2026. sm_120/121 stack progresses fast — re-score quarterly.
- **Out of scope: INT4 (AWQ / GPTQ)**. NVFP4 is NVIDIA-native 4-bit; AWQ INT4 is broader ecosystem but typically lower accuracy. If your shortlist includes AWQ, add a comparison dimension.
- **Accuracy is empirical**: 4-bit error varies by workload. The accuracy axis in dim 5 **does not substitute for an actual eval**; always validate with at least an MMLU/MMLU-Pro shift check before production.
- **Speculative decoding is a separate decision**: see [`05-mtp-speculative-decoding.md`](05-mtp-speculative-decoding.md). Quantization and spec-decoding choices interact (some draft methods incompatible with some quant formats), but you score them in sequence, not jointly.

---

## Related Frameworks

- [`03-inference-research-channels.md`](03-inference-research-channels.md) — channel list for gathering inputs to this framework
- [`05-mtp-speculative-decoding.md`](05-mtp-speculative-decoding.md) — separate decision layer for speculative decoding
- [`02-new-model-selection.md`](02-new-model-selection.md) — upstream model-selection decision
- [`how-to-use-these-frameworks.md`](how-to-use-these-frameworks.md) — overall calling order
