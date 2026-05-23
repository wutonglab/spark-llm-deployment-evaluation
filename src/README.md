# spark-llm-deployment-evaluation

A reusable **methodology + agent toolkit** for evaluating LLM deployment decisions on NVIDIA DGX Spark (GB10 / sm_121) and similar Blackwell-class hardware. Turns *"which model / quantization / speculative decoding should I pick?"* from gut feel into a scoreable, citable, verifiable decision flow.

> **What this is**: a methodology repository.
> **What this is not**: a cookbook for reproducing one specific deployment.

## What's inside

- **5 decision frameworks** — score your (model, hardware, workload) along weighted dimensions, compare against threshold tables.
- **Multi-provider evaluation agent** (OpenAI + Anthropic) — give it a model + hardware + business scenario; it walks all 4 layers end-to-end. (Code lives at [`agent/` on GitHub](https://github.com/wutonglab/spark-llm-deployment-evaluation/tree/main/agent).)
- **First worked case study** — Qwen3.6-35B-A3B-FP8 on DGX Spark, with a complete A/B/C/D/E variant sweep and prediction-vs-measured validation.
- **Hardware portability guide** — re-score the methodology for B200, RTX PRO 6000 Blackwell, H100/H200, Jetson Thor.

## The layered decision stack

Walk these layers in order. Each consumes the previous layer's decision.

| Layer | Decision | Framework | Channel list (inputs) |
|---|---|---|---|
| **0** | Which model to adopt | [02 — New model selection](frameworks/02-new-model-selection.md) | [04 — New model evaluation channels](frameworks/04-new-model-evaluation-channels.md) |
| **1** | Which quantization (FP8 / NVFP4 / AWQ) | [01 — FP8 vs NVFP4 decision](frameworks/01-fp8-vs-nvfp4-decision.md) | [03 — Inference research channels](frameworks/03-inference-research-channels.md) |
| **2** | Which speculative decoding | [05 — MTP speculative decoding](frameworks/05-mtp-speculative-decoding.md) | Same as layer 1 |
| **3** | vLLM engine tuning | A/B in [case studies](case-studies/README.md) | Same as layer 1 |

For the full calling pattern, see [How to use these frameworks](frameworks/how-to-use-these-frameworks.md).

## Quick answer: Qwen3.6-35B-A3B on DGX Spark

| Layer | Score / Decision |
|---|---|
| Model selection (framework 02) | 0.78 → adopt |
| Quantization (framework 01) | 0.21 → stay on FP8 |
| Speculative (framework 05) | MTP-1 |
| Engine tuning | kv-cache fp8 + default attention + prefix caching |

Measured outcome: ~61 tok/s single-request, ~234 tok/s at concurrency=4 (1024-token output). NVFP4 in the same software stack: ~12 tok/s single-request.

Full reasoning: [Qwen3.6 case study](case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/00-summary.md).

## Not on DGX Spark?

The frameworks are hardware-aware but not Spark-locked. Re-score the relevant dimensions for your GPU using [Porting to other GPUs](hardware-portability/porting-to-other-gpus.md). Notes for RTX PRO 6000 Blackwell, B200, H100/H200 included.

## Sanitization policy

This repository contains **no proprietary, internal, or NDA-bound information**. Hostnames, internal IPs, customer business names, and unreleased benchmarks are blocked at CI. If you find a leak, see SECURITY.md on the [GitHub repo](https://github.com/wutonglab/spark-llm-deployment-evaluation).

## License

[Apache-2.0](https://github.com/wutonglab/spark-llm-deployment-evaluation/blob/main/LICENSE).
