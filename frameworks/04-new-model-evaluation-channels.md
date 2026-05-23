# Framework 04: New Model Evaluation Channel Baseline

> **Purpose**: when evaluating whether to **adopt a new LLM** (the upstream-most decision, before any quantization / deployment work), this is the minimum set of channels you must consult before scoring it with framework 02.

## Rule

For any model-selection decision, **check at least one data point from each of the 7 channel categories below**, then feed the data into [`02-new-model-selection.md`](02-new-model-selection.md) for scoring.

## Why

Vendor marketing diverges from real-world capability in predictable ways:
- "+5 MMLU points" while completely failing SWE-bench
- "1M context" that degrades to noise past 32K
- Strong Chinese benchmark numbers that don't translate to product traction

A single source — especially the vendor's own report — is never enough.

## How to Apply

1. State the business question: agent / coding / proofreading / long-document / chat / etc., plus the current production baseline model.
2. **In parallel**, hit each of the 7 categories. Capture ≥ 1 data point per category.
3. Cross-validate: if vendor self-reported numbers differ from third-party leaderboards by > 5 points, distrust the vendor.
4. Self-test on your own prompts — no external data substitutes for this.
5. Pass collected data into [`02-new-model-selection.md`](02-new-model-selection.md) for the 6-dim score.

---

## The 7 Channel Categories

### 1. Author's Official Resources

| Source | What to read |
|---|---|
| **HuggingFace model card** | Architecture, parameters, training data description, license, recommended usage, self-reported benchmarks |
| **Official GitHub repo** | Technical report, reference implementation, known limitations |
| **arXiv preprint** | Training method, alignment technique (RLHF / DPO / RLAIF), ablations |
| **Official blog** | Release motivation, comparisons vs predecessors/competitors, demos |
| **Key namespaces** | `Qwen/`, `deepseek-ai/`, `meta-llama/`, `mistralai/`, `google/`, `microsoft/`, `THUDM/`, `01-ai/`, `internlm/` (closed-API providers: official docs sites) |
| **Official playground** | Try it before committing — chat interfaces or HF Spaces demos |

### 2. Public General-Purpose Benchmark Leaderboards

| Leaderboard | URL / Name | Value |
|---|---|---|
| **LMArena (Chatbot Arena)** | `lmarena.ai` | Blind human-judged elo, closest to perceived UX |
| **LiveBench** | `livebench.ai` | Monthly refresh, contamination-resistant |
| **HuggingFace Open LLM Leaderboard** | `huggingface.co/spaces/open-llm-leaderboard/` | MMLU-Pro / IFEval / BBH / MATH / GPQA / MuSR composite |
| **MMLU / MMLU-Pro** | Knowledge breadth | Baseline composite |
| **GPQA-Diamond** | Graduate-level reasoning | High-difficulty ceiling |
| **MATH / AIME** | Math depth | Mathematical reasoning |
| **HumanEval / MBPP** | Code (basic) | Simple code generation |
| **C-Eval / CMMLU / SuperCLUE** | Chinese composites | Required for Chinese-language scenarios |
| **Critical note** | Vendor self-reports **must** be cross-validated against third-party leaderboards; 5+ point gaps are common |

### 3. Domain-Specific Benchmarks (pick by your business)

| Domain | Benchmark | When you must check |
|---|---|---|
| **Code engineering** | SWE-bench / SWE-bench Verified / BigCodeBench / LiveCodeBench / Aider Polyglot | Any code / agent deployment |
| **Hard math** | MATH-500 / AIME 2024 / FrontierMath | Math/science scenarios |
| **Long context** | RULER / Needle-in-a-haystack / LongBench-v2 / ∞Bench | Any business with > 32K context |
| **Agent / Tool use** | τ-bench / WebArena / SWE-Agent / BFCL (Berkeley Function-Calling) | Multi-step / pipeline agents |
| **Multimodal** | MMMU / MathVista / ChartQA / DocVQA / Video-MME | Vision-inclusive business |
| **Chinese verticals** | AlignBench / CMB / FinEval | Chinese vertical industries |
| **Reasoning / thinking** | GPQA-Diamond / Humanity's Last Exam / MuSR | Reasoning-heavy business |
| **Instruction following** | IFEval / Arena-Hard | Structured output / agents |

### 4. Independent Third-Party Evaluations

| Source | Value |
|---|---|
| **Artificial Analysis** (`artificialanalysis.ai`) | Price / speed / intelligence three-axis, best cross-model coverage |
| **Vellum AI Leaderboard** | Task-utility cross-evaluation |
| **Aider Polyglot Benchmark** (`aider.chat/docs/leaderboards/`) | Multi-language coding tasks — authoritative for coding models |
| **Epoch AI** | Capability-growth trends + compute cost estimates |
| **Apollo Research / safety eval groups** | Safety / alignment independent evaluation |
| **OpenCompass** | Structured cross-task reports |

### 5. Community Real-World Feedback (most important)

Look at what real users say in the **first 6 months** after release:

| Source | What to look for |
|---|---|
| **r/LocalLLaMA** (Reddit) | First-week discussion threads (search model name + month) |
| **Hacker News** | High-quality analysis + production case mentions in comments |
| **Twitter / X researcher accounts** | Independent researchers' running threads |
| **Region-specific tech outlets** | Localized analysis; for Chinese, check WeChat tech outlets + Zhihu specialized columns |
| **LessWrong / EleutherAI Discord** | Western research-community discussion |
| **YouTube / Bilibili reviews** | Hands-on test videos often surface flaws marketing omits |
| **Search pattern** | `<model_name> production`, `<model_name> real-world`, `<model_name> issues` |

### 6. License + Commercial Terms

| Check | What |
|---|---|
| **HuggingFace model card LICENSE field** | Apache 2.0 / MIT / custom / research-only |
| **Custom license fine print** (e.g. Llama, Qwen, Mistral custom) | DAU/MAU thresholds, derivative work allowed?, commercial allowed?, attribution requirements |
| **Vendor commercial-use page** | Llama Community License, equivalents |
| **Training data provenance** | GPL / Copyleft / CC-BY-SA contamination, disclosure obligations |
| **Search pattern** | `<model_name> license commercial`, `<model_name> commercial use` |

### 7. Business PoC Self-Test (you must run it)

Not strictly a "channel," but a non-negotiable floor — no external data substitutes for testing on your prompts:

| Required test | Details |
|---|---|
| **Business-prompt subset** (≥ 50) | Your real production prompts, not benchmark sets |
| **A/B blind comparison vs current production model** | Evaluators don't know which is new |
| **Edge case set** (your 10 hardest cases) | Does the new model help where you hurt most? |
| **Format compliance / tool-calling stability** | 100 repeats — measure structural error rate |
| **Latency + throughput** | Same hardware vs current model |
| **Multi-language capability** | If non-English business, test the languages you ship in |

---

## Failure Modes (cost of skipping)

| Skipped channel | Wrong outcome you'd reach |
|---|---|
| Official technical report | Missing the architecture details (MoE / Mamba hybrid / etc.) that determine downstream quantization/deployment work |
| LMArena / LiveBench | Believing self-reported MMLU but user-perceived quality is far lower |
| Domain-specific benchmarks | Strong-general ≠ strong-your-business: e.g. great general but weak SWE-bench |
| Artificial Analysis | Picking the most expensive model when a cheaper one matches your latency/intelligence target |
| Reddit / regional outlets | Missing post-release defects like repetition loops or context-degradation past N tokens |
| License | Deploying then discovering it's not commercial-use compliant |
| Self-test | Top-of-leaderboard model performing poorly on your actual workload |

---

## Boundary Conditions

- **Time decay**: snapshot of late-2025 / 2026 channel ecosystem. Leaderboards and outlets shift.
- **Closed-API models** (OpenAI / Anthropic / Google): channel 1 (official) and channel 7 (self-test) carry more weight; channels 5/6 still matter (community feedback, ToS).
- **Open-source models**: channels 5 (community) and 6 (license) carry more weight.
- **Regulated verticals** (medical / legal / finance): add domain-specific benchmarks + compliance review.
- **Multimodal**: add the corresponding multimodal benchmark suite.

---

## Relationship to Other Layers

| Layer | Framework (the "what") | Channel list (the "where to look") |
|---|---|---|
| 0 — Model selection | [`02-new-model-selection.md`](02-new-model-selection.md) | **This document** |
| 1 — Quantization | [`01-fp8-vs-nvfp4-decision.md`](01-fp8-vs-nvfp4-decision.md) | [`03-inference-research-channels.md`](03-inference-research-channels.md) |
| 2 — Speculative decoding | implicit in layer 1 framework | Same as layer 1 |
| 3 — Inference-engine tuning | A/B test in case study | Same as layer 1 |

**Calling order**: layer 0 first (this document → framework 02). Only after a model is selected do you run layers 1-3.

---

## Related Frameworks

- [`02-new-model-selection.md`](02-new-model-selection.md) — the 6-dim scoring framework that consumes this channel list
- [`03-inference-research-channels.md`](03-inference-research-channels.md) — downstream channel list for inference decisions
- [`how-to-use-these-frameworks.md`](how-to-use-these-frameworks.md) — overall calling order
