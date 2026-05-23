# Case Study Template

> Copy this file to `case-studies/<your-model-id>-on-<your-hardware-id>/00-summary.md` and rename throughout the directory. Replace every `<...>` placeholder with your actual values.

# Case Study: `<model-id>` on `<hardware>`

> **Question**: on `<hardware-description>`, should we adopt `<model-id>` for `<workload-description>`, and if so — which quantization, speculative method, and tuning?
>
> **Answer (one-line)**: ✏️ summarize your final decision here.

## Hardware Profile (sanitized — no hostnames, IPs, business names)

| Attribute | Value |
|---|---|
| Platform | <e.g. NVIDIA DGX Spark, RTX PRO 6000 Blackwell, etc.> |
| Compute architecture | <e.g. sm_121, sm_100a> |
| Memory | <amount + type> |
| Memory bandwidth | <GB/s> |
| Container runtime | <Docker version, etc.> |
| Inference framework | <vLLM / TRT-LLM / SGLang + version> |

## Framework Scores Applied

| Layer | Framework | Score | Decision |
|---|---|---|---|
| 0 — Model selection | [`02-new-model-selection`](../../frameworks/02-new-model-selection.md) | ✏️ X.XX / 1.00 | ✏️ adopt / wait / drop |
| 1 — Quantization | [`01-fp8-vs-nvfp4-decision`](../../frameworks/01-fp8-vs-nvfp4-decision.md) | ✏️ X.XX / 1.00 | ✏️ FP8 / NVFP4 / other |
| 2 — Speculative decoding | [`05-mtp-speculative-decoding`](../../frameworks/05-mtp-speculative-decoding.md) | n/a (decision table) | ✏️ method + n |
| 3 — Engine tuning (A/B) | empirical | see `04-…md` | ✏️ summary |

Detailed score breakdowns are in the per-layer files (`01-…` to `05-…`).

## Final Production Configuration

✏️ Replace this block with the docker/compose/launch command your case ends up using. Make sure every variable is documented in `reproduce/env-templates/`.

```bash
docker run ... <your final config> ...
```

Reproducible scripts live in [`reproduce/`](reproduce/).

## Headline Measurements

| Variant | Single-request output tok/s (✏️ at output length) | Concurrent aggregate tok/s (✏️ at concurrency, output length) |
|---|---:|---:|
| `baseline` | ✏️ | ✏️ |
| `<your-chosen-variant>` ⭐ | ✏️ | ✏️ |
| `<comparison variant>` | ✏️ | ✏️ |

Detailed tables in [`04-benchmark-results.md`](04-benchmark-results.md). Raw CSV in [`data/baseline-results.csv`](data/baseline-results.csv).

## What This Case Validates / Challenges About the Frameworks

✏️ Did the frameworks predict the right answer? Did any dimension surprise you?
- Framework 01: ✏️ matched / mis-predicted because …
- Framework 02: ✏️
- Framework 05: ✏️

If a framework's prediction was wrong, open a Discussion under `framework-improvements`.

## Open Questions

✏️ Anything you didn't have time / hardware to measure that follow-up cases could cover.

## How to Reproduce This Case

Two paths:

1. **Manual**: clone the repo, copy `reproduce/env-templates/<chosen-variant>.env` to `.env`, run `reproduce/launch.sh`, then `tools/benchmark/run_variants.sh`.
2. **Agent-driven**: run `agent/evaluate.py` with the model id and target hardware string.

## File Map (must include all)

| File | Required? | Purpose |
|---|---|---|
| `00-summary.md` | Required | This file |
| `01-applying-framework-02.md` | Required if you scored model selection | Layer 0 score with citations |
| `02-applying-framework-01.md` | Required if you made a quantization decision | Layer 1 score with citations |
| `03-applying-framework-05.md` | Required if you made a speculative decision | Layer 2 with citations |
| `04-benchmark-results.md` | Required if measurements were run | Full data tables + analysis |
| `05-decision.md` | Required | Decision chain summary |
| `data/baseline-results.csv` | Required if `04-…` exists | Raw measurements |
| `reproduce/launch.sh` | Required | Reproducible launcher |
| `reproduce/env-templates/*.env` | Required | One env file per variant tested |
| `reproduce/README.md` | Required | How to run the launcher |

## Sanitization Checklist (pre-PR)

- [ ] No hostnames or internal aliases (e.g. `<your-host>`, `${SPARK_HOST}`)
- [ ] No IPs (private, public, or RFC1918 ranges hardcoded)
- [ ] No customer business names / unreleased product names
- [ ] No proprietary benchmark numbers from employer
- [ ] No HuggingFace tokens / SSH keys / credentials in any file
- [ ] All hardcoded `/home/<user>/` paths replaced with `${WORKSPACE}` or `~/...`
- [ ] `bash scripts/check-no-internal.sh` reports 0 hits
