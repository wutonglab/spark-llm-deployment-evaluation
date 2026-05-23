# Stage 1 — Apply Framework 02 (Model Selection)

## Inputs

- `model`: {model}
- `target_hardware`: {target_hardware}
- `business_scenario`: {business_scenario}

## Your Task

1. **Load the framework** via `read_file("frameworks/02-new-model-selection.md")` and `read_file("frameworks/04-new-model-evaluation-channels.md")`. Read them carefully — they define the rubric.
2. **Gather evidence** by walking the 7 channel categories in framework 04. For each category that applies to this model, use `http_get` to fetch at least one source URL. Examples:
   - HuggingFace model card: `https://huggingface.co/{model}`
   - LMArena: `https://lmarena.ai`
   - Open LLM Leaderboard: `https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard`
   - Vendor official blog or arXiv preprint if linked from the model card
   - License text on the model card
3. **Score the 6 dimensions** of framework 02 with explicit reasoning. Each dimension's score must be backed by ≥ 1 channel citation. For any "+N%" or "fully supported" claim, cite ≥ 2 sources.
4. **Decide** per the threshold table:
   - `≥ 0.75` → adopt directly, proceed to Stage 2
   - `0.55-0.75` → prefer-adopt with canary, proceed to Stage 2
   - `0.35-0.55` → run shadow A/B; for this evaluation, still proceed to Stage 2 but flag the recommendation
   - `0.20-0.35` → wait for next version; **stop here** with a "wait" report
   - `< 0.20` → drop; **stop here** with a "drop" report

## Deliverables

Write these artifacts at the end of this stage:

### `stage-1-selection-score.json`

```json
{
  "stage": 1,
  "framework": "02-new-model-selection",
  "scores": {
    "capability": <float>,
    "business_fit": <float>,
    "license": <float>,
    "ecosystem": <float>,
    "switching_cost": <float>,
    "sustainability": <float>
  },
  "total": <float>,
  "decision": "adopt | prefer-adopt | shadow-ab | wait | drop",
  "next_stage": <bool>
}
```

### Append to `research-citations.md`

```
## Stage 1 — Model Selection (Framework 04 channels)

### 1. Author's official resources
- <URL>: <what you learned>

### 2. Public benchmarks
- <URL>: <numbers cited>

... (for each of the 7 categories you used)
```

### Append to `evaluation-report.md`

A "## Stage 1: Model Selection" section with:
- Score breakdown table (per dimension, with reasoning)
- Citations inline (or referenced by section)
- Decision + next-stage flag

## Constraints

- Do not call `run_shell` to "ssh and check things" — at this stage you're doing remote research, not local execution.
- If a channel category doesn't produce useful results in 2 attempts, note that and move on (don't burn the entire turn budget on one).
- Keep individual `http_get` calls bounded (timeout=30 default is fine).
- After scoring, write the artifacts BEFORE returning final text. The system will check the artifacts exist.
