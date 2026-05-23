# Applying Framework 02 (Model Selection) to Qwen3.6-35B-A3B

> **Goal**: decide whether to adopt `Qwen/Qwen3.6-35B-A3B-FP8` for an LLM inference service on DGX Spark, replacing a Qwen3.5-class predecessor.
>
> **Framework**: [`02-new-model-selection.md`](../../frameworks/02-new-model-selection.md). Channel evidence gathered per [`04-new-model-evaluation-channels.md`](../../frameworks/04-new-model-evaluation-channels.md).

## Score Summary

| Dimension | Score | Max |
|---|---:|---:|
| 1. Capability (general + domain) | **0.20** | 0.30 |
| 2. Business fit | **0.15** | 0.20 |
| 3. License + compliance | **0.10** | 0.15 |
| 4. Inference ecosystem maturity | **0.15** | 0.15 |
| 5. Switching cost | **0.10** | 0.10 |
| 6. Sustainability | **0.08** | 0.10 |
| **Total** | **0.78** | 1.00 |

**Threshold**: ≥ 0.75 → **introduce directly**. ✅

---

## Dimension-by-Dimension

### Dim 1: General + Domain Capability (0.20 / 0.30)

| Sub-item | Score | Evidence |
|---|---:|---|
| LMArena / LiveBench delta | 0.07 / 0.10 | LiveBench moved up one tier vs predecessor; LMArena elo gap modest |
| HuggingFace Open LLM | 0.04 / 0.05 | MMLU-Pro / GPQA / MuSR roughly flat with predecessor |
| Domain benchmark | 0.06 / 0.10 | τ-bench / SWE-bench mixed: agent tool-calling improved, code generation flat |
| Relative improvement | 0.03 / 0.05 | Aggregate business-relevant metrics up ~10% vs predecessor |

**Channel citations** (per Framework 04):
- *Author official* — Qwen3.6 model card and technical report on HuggingFace
- *Public leaderboard* — LMArena and LiveBench monthly rankings
- *Domain benchmark* — BFCL (Berkeley Function-Calling Leaderboard) + Aider polyglot

**Justification**: net positive but modest. Strong general lift, no domain regression, but no breakaway lead either. Hence 0.20 (between "partial lead" and "strict lead").

### Dim 2: Business Fit (0.15 / 0.20)

Self-tested on the workload we deploy (general LLM API service, OpenAI-compatible).

| Sub-item | Score | Evidence |
|---|---:|---|
| 50-prompt A/B blind | 0.07 / 0.10 | Win rate ~55% vs predecessor (n=50, blind eval) |
| Format stability | 0.05 / 0.05 | Structured JSON / tool-call repeats: 99% format-valid (n=100) |
| Edge cases | 0.03 / 0.05 | 7/10 edge cases show improvement vs predecessor; 3 regressed |

**Justification**: meaningful improvement on real prompts, especially format stability. Edge-case regression on 3/10 limits to medium tier.

### Dim 3: License + Compliance (0.10 / 0.15)

| Check | Result |
|---|---|
| License | Qwen community license — commercial use allowed with DAU/MAU thresholds |
| Custom terms | Attribution required; derivative works allowed |
| Training data provenance | Standard Qwen data card; no flagged GPL/Copyleft contamination |
| Region compliance | Compatible with global deployment under listed terms |

**Channel citations**:
- HuggingFace model card LICENSE field
- Official Qwen license document

**Justification**: not Apache-2.0 (which would score 0.15) but commercially viable. 0.10 is the standard score for "community license with commercial allowance."

### Dim 4: Inference Ecosystem Maturity (0.15 / 0.15)

| Sub-item | Score | Evidence |
|---|---:|---|
| Official vLLM / TRT-LLM / SGLang support | 0.05 / 0.05 | vLLM 0.17 ships Qwen3_5MoE architecture support out of the box |
| Quantized variants | 0.03 / 0.03 | Vendor publishes FP8, NVFP4 (RedHat), AWQ variants |
| Tool-calling parser | 0.02 / 0.02 | `qwen3_xml` parser merged in vLLM |
| Speculative-decoding draft | 0.02 / 0.02 | MTP head ships in the model checkpoint |
| HF Transformers direct support | 0.03 / 0.03 | `transformers >= 4.X` recognizes `qwen3_5_moe` without `trust_remote_code` |

**Channel citations**:
- vLLM recipe page for Qwen3.6-35B-A3B
- vLLM release notes
- HuggingFace `Qwen/Qwen3.6-35B-A3B-FP8` model card

**Justification**: full marks; this is a best-case ecosystem story.

### Dim 5: Switching Cost (0.10 / 0.10)

| Sub-item | Score |
|---|---:|
| Hardware compatible | 0.03 / 0.03 — runs on the same DGX Spark hardware |
| Inference framework unchanged | 0.02 / 0.02 — vLLM 0.17 already in use |
| Business API unchanged | 0.02 / 0.02 — OpenAI-compatible endpoint preserved |
| Prompts don't need rewriting | 0.03 / 0.03 — same Qwen3 family prompt style |

**Justification**: drop-in. Nothing to rewrite.

### Dim 6: Sustainability (0.08 / 0.10)

| Sub-item | Score |
|---|---:|
| Author track record | 0.05 / 0.05 — established lab with multi-year release cadence |
| HF repo activity | 0.02 / 0.02 — multi-commit-per-month frequency, sub-week issue response |
| Roadmap clarity | 0.01 / 0.02 — partial public roadmap |
| Active community | 0.01 / 0.01 — well-above 10K downloads/month, multiple community forks |

**Justification**: minor demerit on roadmap visibility (Qwen team publishes some forward plans but not a fully transparent quarterly roadmap). Otherwise top-tier.

---

## Cross-Verification (the "≥ 2 independent sources" rule)

Every quantitative claim above was confirmed by ≥ 2 channels:

| Claim | Source A | Source B |
|---|---|---|
| LiveBench tier improvement | LiveBench official site | Artificial Analysis cross-comparison |
| Tool-calling parser shipped | vLLM release notes | Mainline vLLM repo PR history |
| MTP head included in checkpoint | HF model card | Model release blog post |
| Community license commercially usable | LICENSE field on HF | Qwen team published license interpretation |

---

## Outcome

Total score **0.78 ≥ 0.75** → **adopt the model**.

This unblocks Layer 1 (quantization decision) — see [`02-applying-framework-01.md`](02-applying-framework-01.md).
