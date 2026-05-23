# How to Use These Frameworks Together

This repository's 5 numbered frameworks combine into a single evaluation flow. This document shows the calling order, the inputs each consumes, and how the [`agent/`](../agent/) automates the walk.

## The Layered Decision Stack

| Layer | Decision | Framework | Channel List (inputs) |
|---|---|---|---|
| **0** | Which model to adopt | [`02-new-model-selection.md`](02-new-model-selection.md) | [`04-new-model-evaluation-channels.md`](04-new-model-evaluation-channels.md) |
| **1** | Which quantization (FP8 / NVFP4 / AWQ / GPTQ) | [`01-fp8-vs-nvfp4-decision.md`](01-fp8-vs-nvfp4-decision.md) | [`03-inference-research-channels.md`](03-inference-research-channels.md) |
| **2** | Which speculative decoding | [`05-mtp-speculative-decoding.md`](05-mtp-speculative-decoding.md) | Same as layer 1 |
| **3** | vLLM tuning (kv-cache dtype / attention backend / batched tokens / etc.) | A/B in case studies | Same as layer 1 |

Walk them strictly in order. Earlier layers' decisions are inputs to later layers'.

## Manual Walk (4 stages)

### Stage A — Pick the model (Layer 0)

1. Gather evidence using the 7 categories in framework 04.
2. Score 6 dimensions per framework 02.
3. Compare to the threshold table.

**Outputs**: a yes/no decision; if yes, the model id and the variant (quant tag if multiple offered by author).

### Stage B — Pick the quantization (Layer 1)

1. Gather evidence using the 7 categories in framework 03 — specifically: hardware capability (SM arch), bandwidth, software-stack maturity for that SM, model architecture, workload pattern.
2. Score 5 dimensions per framework 01.
3. Compare to the threshold table.

**Outputs**: which quantization format to deploy (FP8 / NVFP4 / etc.).

### Stage C — Pick the speculative method (Layer 2)

1. Read framework 05's decision table for your (model architecture, quantization) tuple.
2. Cross-check the speculator registry on HuggingFace and the inference-framework's open issues for any newly-discovered blockers.

**Outputs**: speculative method + `num_speculative_tokens` setting (or "disabled").

### Stage D — Apply A/B variants to fine-tune deployment (Layer 3)

The case studies enumerate variants typical of this layer (kv-cache dtype, attention backend, `max-num-batched-tokens`, `gpu-memory-utilization`, prefix-caching). Pick the variant that the prior case study found optimal for your (model, quant, spec) tuple. If you're in unexplored territory, run the [`tools/benchmark/run_variants.sh`](../tools/benchmark/run_variants.sh) sweep yourself.

**Outputs**: a final `.env` file ready to feed into [`tools/deploy/launch.sh`](../tools/deploy/launch.sh).

## Agent-Driven Walk

The [`agent/evaluate.py`](../agent/evaluate.py) script automates A→D end-to-end:

```bash
python agent/evaluate.py \
 --model <hf-repo-id> \
 --target-hardware "DGX Spark / GB10 / 128 GB / 273 GB/s LPDDR5X / sm_121" \
 --business-scenario "your scenario description" \
 --provider openai
```

The agent loads each framework's markdown as in-context guidance, performs channel research via web search + HTTP fetches, scores each framework, and (optionally, default on) runs the benchmark sweep to confirm predictions.

See [`agent/README.md`](../agent/README.md) for the full stage breakdown and output artifacts.

## When to Re-Run the Frameworks

Because the inference stack moves fast, scores need refreshing on a cadence. Re-run:

- **Layer 0 (model selection)**: when a candidate's vendor ships a new minor version (e.g. `Qwen3.6-X` → `Qwen3.7-X`); or quarterly to scan for new entrants
- **Layer 1 (quantization)**: when the inference framework releases a major version (likely changes the software-stack dim); or when a new community patch image lands (re-score op cost)
- **Layer 2 (speculative)**: when a new draft model lands for your target model in the speculator registry; or when a known bug (e.g. `vllm-project/vllm#40756`) is fixed
- **Layer 3 (tuning)**: when any of the above changes — earlier layers' choices invalidate prior A/B numbers

## Common Misuses

- **Skipping layer 0** ("our team already picked the model"): you lose the opportunity to fix selection mistakes before they cascade
- **Running layers in reverse**: optimizing the inference engine for a wrong-fit model
- **Treating a worked example as a recipe**: copy our [`case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/`](../case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/) config without scoring whether your model + hardware shares the same scores
- **Ignoring `accuracy` even when it's a negative-weight item**: 4-bit quantization shifts model behavior; always include a quality eval, not just throughput numbers
- **One source per dimension**: every score must be defended with ≥ 1 citation; ≥ 2 independent citations when the claim is "+N% gain" or "fully supported"

## Quick Decision Reference (for the impatient)

If you have *only* 30 seconds:

| You have | Likely outcome |
|---|---|
| Datacenter Blackwell (B200) + Dense ≥ 30B + chat workload | Adopt model if framework 02 ≥ 0.55; switch to NVFP4 if framework 01 ≥ 0.7; enable MTP if a draft is published |
| Workstation Blackwell (RTX PRO 6000 Blackwell) + same as above | Same as above; NVFP4 score will be slightly lower due to smaller SMEM |
| **DGX Spark (GB10) + Qwen3.6-35B-A3B + general workload** | **Adopt model (Qwen3.6 scores ~0.78); stay on FP8 (NVFP4 scores ~0.21); enable MTP-1 (no other method viable)** |
| DGX Spark + small model < 14B Dense | Score every layer carefully — small-model compute-bound regime may make all our advice misleading |

Whenever in doubt, run the full agent flow.
