# Framework 02: New Model Selection Decision Framework

> **Purpose**: scoring rubric for "should we adopt this new LLM in production?" — the most upstream decision before any quantization or deployment work.

## Rule

Evaluate the candidate LLM along the 6 dimensions below. Total score determines the recommendation:

- `< 0.20`: drop
- `0.20 – 0.35`: wait for next version
- `0.35 – 0.55`: shadow A/B for 1 week, then decide
- `0.55 – 0.75`: prefer-adopt with 1-week production canary
- `≥ 0.75`: introduce directly

## Why

Picking the wrong model wastes every downstream optimization. Vendor launch numbers diverge from production reality (e.g. great MMLU but weak SWE-bench; strong Chinese leaderboards but poor product traction; advertised "tool calling" but unstable structured output). A **uniform rubric** prevents being swayed by recency bias.

## How to Apply

Pair this with [`04-new-model-evaluation-channels.md`](04-new-model-evaluation-channels.md):
1. Use the channel list to gather evidence.
2. Score the 6 dimensions below with citations to which channels supplied each piece of evidence.
3. Compare against the threshold table.

See the worked example at the bottom — Qwen3.6-35B-A3B replacing Qwen3.5-35B-A3B (score 0.78 → introduce; subsequently validated by the case study under [`case-studies/`](../case-studies/)).

---

## The 6 Dimensions (weights sum to 1.0)

### Dimension 1: General + Domain Capability (weight 0.30)

| Sub-item | Scoring basis | Max |
|---|---|---|
| **LMArena / LiveBench ranking** | Elo delta vs current production model | 0.10 |
| **HuggingFace Open LLM Leaderboard** | MMLU-Pro / IFEval / BBH / MATH / GPQA / MuSR composite | 0.05 |
| **Domain-specific benchmark** | Whatever your business cares about (SWE-bench / MATH / RULER / τ-bench / SuperCLUE / MMMU) | 0.10 |
| **Relative improvement vs current model** | < +5% → 0; +5–15% → 0.03; > +15% → 0.05 | 0.05 |

**Scoring guide**:
- Strict lead on both general and domain: 0.25–0.30
- Partial lead (domain strong, general flat): 0.15–0.25
- Mixed (general strong but domain weak): 0.05–0.15
- Regression: 0

### Dimension 2: Business Fit (weight 0.20)

**Self-testing required**; public benchmarks alone are insufficient.

| Sub-item | Scoring basis | Max |
|---|---|---|
| **Business-prompt A/B blind test** | ≥ 50 real production prompts; record win-rate | 0.10 |
| **Output format stability** | 100 repeats of structured JSON / tool-call task; error rate | 0.05 |
| **Edge-case improvement** | Your 10 hardest cases — does the new model help? | 0.05 |

**Scoring guide**:
- Win-rate ≥ 60% + format stability ≥ 95% → 0.20
- Win-rate 50–60% → 0.10–0.15
- Win-rate < 50% → 0 (not worth replacing)

### Dimension 3: License + Compliance (weight 0.15)

| License | Score |
|---|---|
| Apache-2.0 / MIT / BSD | 0.15 |
| Custom community license (Llama / Qwen / Mistral style) allowing commercial use with DAU/MAU thresholds | 0.10 |
| Research-only / non-commercial | 0 |
| Training data contains GPL / Copyleft risk | -0.05 |

**Additional checks**:
- Attribution requirements / derivative work allowed?
- Training-data provenance traceable?
- Region compliance (EU / China / US export control)?

### Dimension 4: Inference Ecosystem Maturity (weight 0.15)

| Sub-item | Max |
|---|---|
| **Official support in vLLM / TRT-LLM / SGLang** (not pending PR) | 0.05 |
| **Quantized variants available** (vendor or community FP8 / AWQ / GPTQ / NVFP4) | 0.03 |
| **Tool-calling parser implemented** (only score if you use tools) | 0.02 |
| **Speculative-decoding draft available** (MTP / EAGLE-3 / DFlash) | 0.02 |
| **HF Transformers direct support** (no `trust_remote_code` or special fork) | 0.03 |

**Scoring guide**:
- Full-stack mature → 0.15
- Mainstream framework + quant OK, missing speculative/tool → 0.08–0.12
- Only Transformers, vLLM PR pending → 0.03–0.05
- Requires self-maintained fork → 0

### Dimension 5: Switching Cost (weight 0.10)

| Sub-item | Max |
|---|---|
| **Hardware compatible** (no need to swap GPU) | 0.03 |
| **Inference framework needs no upgrade** (current version supports it) | 0.02 |
| **Business API contract unchanged** (OpenAI-compatible) | 0.02 |
| **Existing prompts don't need rewrites** (same model family / similar style) | 0.03 |

**Scoring guide**:
- Drop-in replacement → 0.10
- Minor tweaks (upgrade framework or tune prompts) → 0.05–0.08
- Major rework (new GPU / rewrite all prompts) → 0

### Dimension 6: Sustainability (weight 0.10)

| Sub-item | Max |
|---|---|
| **Author track record** (established lab: Qwen / DeepSeek / Meta / Mistral / etc.) | 0.05 |
| **HF repo activity** (≥ 1 commit/month, issue response < 1 week) | 0.02 |
| **Roadmap clarity** (next-version plans public) | 0.02 |
| **Active community ecosystem** (≥ 10K downloads/month + multiple forks) | 0.01 |

**Scoring guide**:
- Top-tier lab + high activity → 0.10
- Top-tier lab + low activity → 0.06
- Second-tier / academic + high activity → 0.05
- Solo project / stalled → 0–0.02

---

## Decision Thresholds

```
total = capability + business_fit + license + ecosystem + switching_cost + sustainability
```

| Total | Decision |
|---|---|
| ≥ 0.75 | **Introduce directly** (strong ROI; safe to bulk-replace) |
| 0.55–0.75 | **Prefer-adopt** with 1-week production canary |
| 0.35–0.55 | **Run a 1-week shadow A/B** alongside current model |
| 0.20–0.35 | **Wait for next version** (current capability insufficient but trajectory positive) |
| < 0.20 | **Drop** (insufficient / license-blocked / ecosystem-broken) |

---

## Operational Checklist (6 quick gates)

Walk these gates in order; any ❌ kills the candidate without needing the full score:

1. **General capability**: LMArena / LiveBench / Open LLM ranking vs current?
   - Clear lead → continue
   - Flat or slight drop → ⚠️ only worth replacing if domain is strong
   - Wholesale regression → ❌ drop

2. **Domain match**: business-critical benchmark (SWE-bench / MATH / Chinese / agent) clear lead?
   - Yes → continue
   - No → ⚠️ need other dimensions to compensate

3. **Self-test**: 50+ business prompts A/B blind win-rate ≥ 50%?
   - Yes → continue
   - < 50% → ❌ drop (public benchmarks lie about your business)

4. **License**: commercial-compliant + training-data traceable?
   - Yes → continue
   - No → ❌ drop (unless your business permits non-commercial)

5. **Ecosystem**: current vLLM / TRT-LLM version supports + quantized variant available?
   - Yes → continue
   - No → ⚠️ wait 1–2 months for ecosystem maturity

6. **Switching cost**: manageable (no hardware swap, no full prompt rewrite)?
   - Yes → ready to score
   - No → ⚠️ evaluate whether upside outweighs cost

---

## Worked Example: Qwen3.6-35B-A3B Replacing Qwen3.5-35B-A3B (mid-2026)

| Dimension | Score | Reasoning |
|---|---|---|
| 1. Capability | **0.20/0.30** | LiveBench moved up one tier; Chinese/agent flat; MMLU-Pro flat |
| 2. Business fit | **0.15/0.20** | Self-test win-rate ~55% on general LLM API workload; output format more stable |
| 3. License | **0.10/0.15** | Qwen community license — commercial allowed, DAU threshold applies |
| 4. Ecosystem | **0.15/0.15** | vLLM 0.17 full support + FP8/NVFP4/AWQ three quant variants + MTP draft built in |
| 5. Switching cost | **0.10/0.10** | Same vLLM, same hardware, same OpenAI API surface, no prompt rewrites |
| 6. Sustainability | **0.08/0.10** | Top-tier lab + monthly commit cadence; roadmap public |
| **Total** | **0.78** | **≥ 0.75 → introduce directly** ✅ |

→ Adopted. Subsequent quantization/deployment work is documented in [`case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/`](../case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/).

---

## Hypothetical Examples (framework predictions)

| Candidate | Estimated total | Decision |
|---|---|---|
| Llama-3.3 70B replacing Qwen in Chinese-heavy workload | ~0.40 | ⚠️ wait — Chinese capability gap |
| DeepSeek-V3 replacing GPT-4o (API business) | ~0.60 | ✅ prefer-adopt — cost ↓, capability comparable |
| An academic fine-tune replacing Qwen | ~0.25 | ❌ drop — sustainability + ecosystem weak |
| GLM-4.5-9B replacing Qwen3-8B | ~0.50 | ⚠️ 1-week shadow — Chinese strong but ecosystem weak |
| Mistral Small 3 replacing Qwen3-32B | ~0.45 | ⚠️ shadow — English strong, Chinese weak |

---

## Boundary Conditions

- **Closed-API model evaluation**: re-weight dim 4 (ecosystem) and dim 5 (switching cost) — closed APIs have no quant/speculative, but integration is trivial.
- **Vertical domain models** (medical / legal / finance): dim 1 should use domain-specific benchmarks; general leaderboards have low signal.
- **Multimodal models**: add multimodal benchmarks to dim 1 (MMMU / MathVista / etc.).
- **Edge / small-model scenarios** (< 10B for embedded): dim 5 (switching cost) weight should rise.
- **Recency window**: < 1 month after release, data is incomplete and scores skew low; wait another month and re-evaluate.

---

## Layered Decision Stack

| Layer | Decision | Framework | Channel list |
|---|---|---|---|
| **0** | Which model | **This document** | [`04-new-model-evaluation-channels.md`](04-new-model-evaluation-channels.md) |
| **1** | Which quantization | [`01-fp8-vs-nvfp4-decision.md`](01-fp8-vs-nvfp4-decision.md) | [`03-inference-research-channels.md`](03-inference-research-channels.md) |
| **2** | Which speculative decoding | implicit in layer 1 | Same as layer 1 |
| **3** | Inference-engine tuning | A/B in case study | Same as layer 1 |

**Calling order**: layer 0 first → then 1 → 2 → 3. Never skip or reorder.

---

## Related Frameworks

- [`04-new-model-evaluation-channels.md`](04-new-model-evaluation-channels.md) — the channel list that supplies inputs to this framework
- [`01-fp8-vs-nvfp4-decision.md`](01-fp8-vs-nvfp4-decision.md) — the next decision layer (quantization)
- [`how-to-use-these-frameworks.md`](how-to-use-these-frameworks.md) — overall calling order
