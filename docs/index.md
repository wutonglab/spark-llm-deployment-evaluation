# spark-llm-deployment-evaluation

A reusable **methodology + agent toolkit** for evaluating LLM deployment decisions on NVIDIA DGX Spark (GB10 / sm_121) and similar Blackwell-class hardware. Turns "which model / quantization / speculative decoding should I pick?" from gut feel into a **scoreable, citable, verifiable** decision flow.

## What's inside

- **5 decision frameworks** — score your (model, hardware, workload) along weighted dimensions, compare against threshold tables. [→ Methodology](methodology/index.md)
- **Multi-provider evaluation agent** — give it a model + hardware + business scenario; it walks all 4 layers end-to-end. OpenAI + Anthropic supported.
- **First worked case study** — Qwen3.6-35B-A3B-FP8 on DGX Spark, with a complete A/B/C/D/E variant sweep and prediction-vs-measured validation. [→ Case Studies](case-studies/index.md)
- **Hardware portability guide** — re-score the methodology for B200, RTX PRO 6000 Blackwell, H100/H200, Jetson Thor. [→ Porting guide](hardware-portability/porting-to-other-gpus.md)

## The layered decision stack

Walk these layers in order. Each consumes the previous layer's decision.

| Layer | Decision | Framework | Channel list (inputs) |
|---|---|---|---|
| **0** | Which model to adopt | [02 — New model selection](frameworks/02-new-model-selection.md) | [04 — New model evaluation channels](frameworks/04-new-model-evaluation-channels.md) |
| **1** | Which quantization (FP8 / NVFP4 / AWQ) | [01 — FP8 vs NVFP4 decision](frameworks/01-fp8-vs-nvfp4-decision.md) | [03 — Inference research channels](frameworks/03-inference-research-channels.md) |
| **2** | Which speculative decoding | [05 — MTP speculative decoding](frameworks/05-mtp-speculative-decoding.md) | Same as layer 1 |
| **3** | vLLM engine tuning | A/B in [case studies](case-studies/index.md) | Same as layer 1 |

For the full calling pattern, see [How to use these frameworks](frameworks/how-to-use-these-frameworks.md).

## Quick answer: Qwen3.6-35B-A3B on DGX Spark

If you came here for the specific Qwen3.6 + Spark answer:

| Layer | Score / Decision |
|---|---|
| Framework 02 (model selection) | **0.78** → adopt |
| Framework 01 (quantization) | **0.21** → **stay on FP8** (do NOT switch to NVFP4) |
| Framework 05 (speculative) | **MTP-1** (the only viable option on this stack) |
| Measured outcome | ~61 tok/s single-request with the recommended config vs ~12 tok/s for naive NVFP4 |

→ Full worked example: [Qwen3.6-35B-A3B-FP8 on DGX Spark](case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/00-summary.md)

## Run the agent

Evaluate any (model, hardware) combination you care about:

```bash
git clone https://github.com/wutonglab/spark-llm-deployment-evaluation.git
cd spark-llm-deployment-evaluation
pip install -r agent/requirements.txt
cp .env.example .env  # set OPENAI_API_KEY or ANTHROPIC_API_KEY

python agent/evaluate.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --target-hardware "DGX Spark / GB10 / sm_121" \
  --business-scenario "code review agent" \
  --provider openai
```

The agent walks 4 stages — model selection, quantization decision, deployment config, optional benchmark validation — and writes an evaluation report to `evaluation-runs/<timestamp>/evaluation-report.md`.

## What this repo is and is NOT

**Is**:

- A *methodology repository*: 4 decision frameworks + a worked case + tooling that automates the walk
- *Hardware-aware*: every framework scores the GPU's compute capability, memory bandwidth, and software-stack maturity explicitly
- *Citation-driven*: every score dimension cites concrete channel evidence (HuggingFace, vendor forums, GitHub issues, independent benchmarks)

**Is NOT**:

- A cookbook for reproducing one specific deployment
- A claim that one configuration is universally optimal
- A substitute for running benchmarks on your own hardware

## Contribute

The most valuable contribution is a **new case study** showing a (model, hardware) combination we haven't tested — especially when the framework prediction turns out to be **wrong** (we'll revise the framework).

- [Case-study template](https://github.com/wutonglab/spark-llm-deployment-evaluation/blob/main/case-studies/_template.md)
- [Open an issue](https://github.com/wutonglab/spark-llm-deployment-evaluation/issues/new/choose)
- [Start a discussion](https://github.com/wutonglab/spark-llm-deployment-evaluation/discussions)

## License

[Apache-2.0](https://github.com/wutonglab/spark-llm-deployment-evaluation/blob/main/LICENSE). If you build on the frameworks or agent, citation is appreciated — see [`CITATION.cff`](https://github.com/wutonglab/spark-llm-deployment-evaluation/blob/main/CITATION.cff).
