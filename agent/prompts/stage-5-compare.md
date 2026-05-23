# Stage 5 — Final Report

This is not a separate "stage" the agent walks. It's the wrap-up that the orchestrator forces after Stages 1-4 complete. The orchestrator will give you the consolidated artifacts and ask you to produce a final `evaluation-report.md`.

## Your Task

Consolidate `stage-1-selection-score.json`, `stage-2-quantization-score.json`, `proposed-config.env`, and `prediction-vs-actual.md` (if Stage 4 ran) into a single coherent `evaluation-report.md`.

## `evaluation-report.md` Structure

```markdown
# Evaluation Report: <model-id> on <hardware-id>

## TL;DR (one paragraph)
Decision: <adopt | wait | drop>. If adopt: use <quant> + <speculative-method>. Stage 4 verdict: <validated | partial | wrong>.

## Layer 0 — Model Selection
Score X.XX → <decision>.
Brief reasoning (1-2 paragraphs).
Citations (link to research-citations.md section).

## Layer 1 — Quantization
Score X.XX → <decision>.
Brief reasoning.

## Layer 2 — Speculative Decoding
Decision: <method>.
Reasoning (table lookup + caveats).

## Layer 3 — Deployment Configuration
docker run command (the proposed-launch-command.sh content).
Pointers to env templates.

## Layer 4 — Empirical Validation (if Stage 4 ran)
Prediction-vs-actual table.
Verdict.

## Recommendations
Concrete next steps for the user.

## Caveats / Limitations
What the agent couldn't verify; what to re-check in 1 month.
```

## Constraints

- Be terse — single-page report is the goal
- Tables over prose where possible
- Don't repeat full reasoning from per-stage artifacts — link to them
- Make the TL;DR genuinely useful (decision + 1 sentence reason)
- If any stage was skipped (e.g. user passed `--no-benchmark`), say so explicitly in the report
