# Stage 2 — Apply Framework 01 (Quantization) + Framework 05 (Speculative Decoding)

## Inputs

- `model`: {model} (Stage 1 confirmed this is worth deploying)
- `target_hardware`: {target_hardware}

## Your Task

1. **Load** `frameworks/01-fp8-vs-nvfp4-decision.md`, `frameworks/03-inference-research-channels.md`, and `frameworks/05-mtp-speculative-decoding.md` via `read_file`.
2. **Identify the target SM architecture** from the hardware description. Use `check_gpu` only if running locally and the description is ambiguous.
3. **Score the 5 dimensions of Framework 01** (FP8 vs NVFP4):
   - Hardware capability (SM architecture)
   - Memory bandwidth
   - Software stack maturity (research vLLM / CUTLASS / community images for the target SM)
   - Model architecture (Dense / MoE / Mamba ratio)
   - Workload pattern (from the business scenario)
   - Operational cost adjustment
4. **Decide quantization** per the threshold table.
5. **Pick speculative decoding method** using Framework 05's decision table — based on the model and the chosen quantization.

## Research Strategy

For dimension 3 (software stack maturity), specifically check:

- vLLM recipes page for this model: `https://recipes.vllm.ai/<Org>/<Model>` (e.g. `https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B`)
- vLLM mainline docs for speculative decoding feature flags
- CUTLASS GitHub issues mentioning the target SM (search `https://github.com/NVIDIA/cutlass/issues?q=sm_<SM>`)
- Speculator registry: `https://github.com/vllm-project/speculators` (does a draft model exist?)
- Community patch images on Docker Hub (search `https://hub.docker.com/`)

For dimension 5 (workload), use the business scenario the user provided.

## Deliverables

### `stage-2-quantization-score.json`

```json
{
  "stage": 2,
  "framework": "01-fp8-vs-nvfp4-decision",
  "scores": {
    "hardware": <float>,
    "bandwidth": <float>,
    "software": <float>,
    "architecture": <float>,
    "workload": <float>,
    "op_cost": <float, negative>
  },
  "total": <float>,
  "decision": "fp8 | nvfp4 | needs-side-by-side | bf16",
  "speculative_method": "mtp-1 | mtp-2 | mtp-3 | eagle-3 | dflash | none",
  "speculative_rationale": "<short>"
}
```

### Append to `evaluation-report.md`

A "## Stage 2: Quantization + Speculative" section with:
- Quantization score breakdown
- Speculative method decision matrix walk-through
- Citations

### Append to `research-citations.md`

```
## Stage 2 — Quantization / Speculative (Framework 03 channels)

### Hardware capability
- <URL>: <evidence about target SM>

### Software stack
- <URL>: <evidence about vLLM / CUTLASS support>

### Speculative draft availability
- <URL>: <speculator registry / EAGLE / DFlash availability>
```

## Constraints

- Do not assume facts about the target SM — verify by querying CUTLASS issues or vendor forum
- If software-stack evidence conflicts, prefer the more recent source
- The speculative decision must explicitly cite Framework 05's decision table or a published reason for deviating
