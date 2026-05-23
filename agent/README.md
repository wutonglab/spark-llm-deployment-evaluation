# Agent: `evaluate.py`

End-to-end LLM agent that walks the 4 decision layers ([`../frameworks/`](../frameworks/)) for any (model, hardware) combination and produces an evaluation report.

## Quick Start

```bash
# from repo root
pip install -r agent/requirements.txt
cp .env.example .env && $EDITOR .env   # set OPENAI_API_KEY or ANTHROPIC_API_KEY

python agent/evaluate.py \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --target-hardware "DGX Spark / GB10 / 128GB / 273 GB/s LPDDR5X / sm_121" \
  --business-scenario "general LLM API service" \
  --provider openai
```

Output lands in `evaluation-runs/<timestamp>/`.

## CLI Reference

| Flag | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | HuggingFace repo id (e.g. `Qwen/Qwen3.6-35B-A3B-FP8`) |
| `--target-hardware` | Yes | — | Sanitized hardware description string |
| `--business-scenario` | Yes | — | High-level workload description |
| `--provider` | Yes | — | `openai` or `anthropic` |
| `--llm-model` | No | provider default | Override the LLM the agent itself uses (e.g. `gpt-4o`, `claude-3-5-sonnet-latest`) |
| `--no-benchmark` | No | benchmark enabled | Skip Stage 4 (no live measurement) |
| `--target-host` | No | local | If set, run benchmark over SSH on a remote host |
| `--dry-run` | No | false | Print planned actions without executing |
| `--max-turns-per-stage` | No | 20 | Bound the tool-use loop |

## Stages

1. **Stage 1 — Apply Framework 02 (Model Selection)**
   The agent loads [`02-new-model-selection.md`](../frameworks/02-new-model-selection.md) + [`04-new-model-evaluation-channels.md`](../frameworks/04-new-model-evaluation-channels.md) as context, then researches the model across the 7 channels and produces a 6-dimension score. If total ≥ 0.35 it proceeds to Stage 2; otherwise it stops with a "drop" recommendation.

2. **Stage 2 — Apply Framework 01 (Quantization) + Framework 05 (Speculative)**
   Loads [`01-fp8-vs-nvfp4-decision.md`](../frameworks/01-fp8-vs-nvfp4-decision.md), [`03-inference-research-channels.md`](../frameworks/03-inference-research-channels.md), and [`05-mtp-speculative-decoding.md`](../frameworks/05-mtp-speculative-decoding.md). Scores quantization (5 dims) + picks speculative method from the decision table.

3. **Stage 3 — Generate Deployment Configuration**
   Based on Stages 1+2 choices, the agent emits a parameterized `.env` config compatible with [`../tools/deploy/launch.sh`](../tools/deploy/launch.sh).

4. **Stage 4 — Benchmark Validation** *(default on; `--no-benchmark` to skip)*
   Runs [`../tools/benchmark/bench.py`](../tools/benchmark/bench.py) over a small sweep (typically: `single-request × {512, 1024}` and `c=4 × {1024}`). Compares measured numbers against Stage 2's predictions; reports "prediction validated" or "framework needs revision in dim X."

## Output Artifacts (in `evaluation-runs/<ts>/`)

| File | Content |
|---|---|
| `evaluation-report.md` | The headline narrative — what the agent decided + why, citations included |
| `framework-scores.json` | All dimension scores in machine-readable form |
| `research-citations.md` | URLs visited / fetched, organized by channel category |
| `benchmark-results.csv` | Measurements (if Stage 4 ran) |
| `prediction-vs-actual.md` | Side-by-side of predicted vs measured (if Stage 4 ran) |
| `transcript.jsonl` | Full tool-use audit log |
| `env.json` | Captured environment (driver, CUDA, vLLM versions) |

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | All stages succeeded; prediction within tolerance |
| 1 | Generic error (see stderr) |
| 2 | Stage 1 decided "drop" — no Stages 2-4 ran |
| 3 | Stage 4 measured vs predicted disagreement > tolerance (recommend re-scoring the framework) |
| 4 | Tool-use loop exhausted `--max-turns-per-stage` |

## Provider Notes

- **OpenAI**: uses chat completions with tool/function calling. Default model `gpt-4o`.
- **Anthropic**: uses messages API with tool_use blocks. Default model `claude-3-5-sonnet-latest`.

Both providers share the same `ToolRegistry` + `Orchestrator` — only the message-shape adapter differs.

## Sandboxing

- Tool writes are restricted to `evaluation-runs/<ts>/`.
- SSH (`paramiko`) is optional; only loaded if `--target-host` is set.
- Shell commands have a 600s default timeout.
- No tokens / credentials are logged to `transcript.jsonl` (the framework loader strips `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. from any tool output before recording).

## Limitations

- Channel research depends on the LLM's web access. If the agent's LLM is air-gapped, Stages 1-2 fall back to whatever is in the agent's training data (less authoritative).
- Stage 4 benchmark needs a real GPU. On `--no-benchmark`, the agent reports predictions only.
- The tool-use loop is intentionally simple (no LangChain) — debug by reading `transcript.jsonl`.

## Extending

To add a new provider (e.g. Gemini):

1. Add a subclass to `core/llm_client.py` implementing `.chat(messages, tools)` returning a normalized response.
2. Register it in the `--provider` argparse choices in `evaluate.py`.

To add a new tool:

1. Add the function to `core/tools.py` + its JSON schema in `schemas/tool_schemas.json`.
2. Update the relevant stage prompt to mention the new tool.
