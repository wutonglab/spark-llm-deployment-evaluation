# Methodology

The 4 decision frameworks (plus speculative-decoding notes) form a layered stack you walk from the most upstream decision (which model) to the most downstream (vLLM tuning knobs).

| Layer | Framework | Channel list (inputs) |
|---|---|---|
| 0 — Which model to adopt | [02 — New model selection](../frameworks/02-new-model-selection.md) | [04 — New model evaluation channels](../frameworks/04-new-model-evaluation-channels.md) |
| 1 — Quantization | [01 — FP8 vs NVFP4 decision](../frameworks/01-fp8-vs-nvfp4-decision.md) | [03 — Inference research channels](../frameworks/03-inference-research-channels.md) |
| 2 — Speculative decoding | [05 — MTP speculative decoding](../frameworks/05-mtp-speculative-decoding.md) | Same as layer 1 |
| 3 — Engine tuning | A/B in case studies | Same as layer 1 |

Walk them in order. Each layer consumes the previous layer's decision.

For the full calling pattern and integration with the agent, see [How to use these frameworks](../frameworks/how-to-use-these-frameworks.md).
