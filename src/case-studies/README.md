# Case Studies

Worked examples that apply this repository's [frameworks](../frameworks/) to specific (model, hardware) combinations. Each case study should be reproducible from the published config.

## Case Studies in This Repo

| Case | Hardware | Model | Quant chosen | Speculative | Outcome |
|---|---|---|---|---|---|
| [qwen3.6-35b-a3b-fp8-on-dgx-spark](qwen3.6-35b-a3b-fp8-on-dgx-spark/) | DGX Spark (GB10 / sm_121) | Qwen3.6-35B-A3B-FP8 | FP8 | MTP-1 | ~61 tok/s single-request |

## Contribute a New Case Study

We welcome new cases — especially ones where the frameworks fail to predict the correct decision (we'll revise the frameworks).

### What Makes a Good Case Study

1. **A specific (model, hardware) combination** clearly stated up front
2. **Sanitized**: no proprietary or NDA-bound information; no internal hostnames / IPs / business names; no unreleased product details
3. **Framework scores applied** with citation per dimension (≥ 1 channel source per claim, ≥ 2 for "+N% gain" claims)
4. **Measurement data** (CSV preferred) with at least single-request and one concurrency level
5. **Reproducible config**: env templates + a launcher script so others can validate

### Submission Steps

1. **Fork the repo**
2. **Copy [`_template.md`](_template.md) to `case-studies/<your-model-id>-on-<your-hardware-id>/00-summary.md`** and rename `<your-model-id>` / `<your-hardware-id>` consistently across the directory
3. **Walk the frameworks** for your case, producing the equivalent of `01-…` through `05-…` (skip layers that aren't relevant — e.g. if you're not making a quantization decision, omit Framework 01)
4. **Run measurements**, save raw data to `data/`
5. **Write reproducible env templates + launcher** under `reproduce/`
6. **Run the sanitization check locally**:
 ```bash
 bash scripts/check-no-internal.sh
 ```
 It must report zero hits.
7. **Open a PR** with the `case-study-submission` label

We'll review for:
- Framework consistency (do scores match the frameworks' rubrics?)
- Sanitization (CI also enforces this)
- Reproducibility (do the env templates make sense for someone else to run?)

### When Frameworks Disagree With Reality

If your measurements show that the framework prediction was wrong — that's the **most valuable** kind of case study. Please open a Discussion under `framework-improvements` with:

- Which framework + which dimension's score you believe is mis-weighted
- Evidence (your measurements + ≥ 2 independent sources)
- Proposed change

We'll revise the framework and update affected case studies.

## Future Cases Wanted

Especially welcome:

- Llama / DeepSeek / Mistral / GLM family on DGX Spark
- Qwen3.6 on different Blackwell tier (B200, RTX PRO 6000 Blackwell)
- Smaller models (< 14B Dense) on DGX Spark — likely a different regime
- Multi-modal models
- Long-context-heavy workloads (mostly > 32K)
- Tool-calling / agent benchmarks at scale

See [`_future-cases/`](_future-cases/) for tracked candidates.
