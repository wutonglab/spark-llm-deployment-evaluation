# spark-llm-deployment-evaluation

A reusable **methodology + agent toolkit** for evaluating LLM deployment decisions on NVIDIA DGX Spark (GB10 / sm_121) and similar Blackwell-class hardware. Turns "which model / quantization / speculative decoding should I pick?" from gut feel into a scoreable, citable, verifiable decision flow.

> ⚠️ **What this is**: a methodology repository.
> ❌ **What this is not**: a cookbook for reproducing one specific deployment.

## TL;DR

- **4 decision frameworks** (model selection · quantization · speculative decoding · research channels) — see [`frameworks/`](frameworks/)
- **1 multi-provider LLM agent** that evaluates *your* model on *your* hardware end-to-end — see [`agent/`](agent/)
- **Growing case-studies library**, starting with Qwen3.6-35B-A3B-FP8 on DGX Spark — see [`case-studies/`](case-studies/)

## Quick Start: Evaluate a Model

```bash
git clone https://github.com/wutonglab/spark-llm-deployment-evaluation.git
cd spark-llm-deployment-evaluation
pip install -r agent/requirements.txt
cp .env.example .env && $EDITOR .env       # set OPENAI_API_KEY etc.

python agent/evaluate.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --target-hardware "DGX Spark / GB10 / 128GB / 273 GB/s LPDDR5X / sm_121" \
  --business-scenario "code review agent with tool calling" \
  --provider openai
```

The agent walks 4 stages:

1. **Model selection** — applies framework 02, scores 6 dimensions
2. **Quantization + speculative decoding** — applies framework 01 + 05
3. **Deployment config** — emits parameterized `.env` + `docker compose` call
4. **Benchmark validation** *(optional, on by default)* — runs `tools/benchmark/bench.py` and compares predicted vs measured

Output lands in `evaluation-runs/<timestamp>/evaluation-report.md`.

## Frameworks (the methodology)

| # | Framework | What it scores |
|---|---|---|
| 01 | [FP8 vs NVFP4 decision](frameworks/01-fp8-vs-nvfp4-decision.md) | 5 dims: hardware support · bandwidth · software stack maturity · model architecture · workload pattern |
| 02 | [New model selection](frameworks/02-new-model-selection.md) | 6 dims: capability · business fit · license · ecosystem · switching cost · sustainability |
| 03 | [Inference research channels](frameworks/03-inference-research-channels.md) | 7 channel categories you *must* check for any quantization/speculative decision |
| 04 | [New model evaluation channels](frameworks/04-new-model-evaluation-channels.md) | 7 channel categories you *must* check for any model selection decision |
| 05 | [MTP speculative decoding notes](frameworks/05-mtp-speculative-decoding.md) | When MTP-1 beats EAGLE / DFlash / no-spec on Spark-class hardware |
| — | [How to use these frameworks together](frameworks/how-to-use-these-frameworks.md) | Calling order + integration with the agent |

## Worked Example: Qwen3.6-35B-A3B-FP8 on DGX Spark

Full case study under [`case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/`](case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/):

- Framework 02 score = **0.78** → "introduce this model"
- Framework 01 score = **0.21** → "keep FP8, do NOT switch to NVFP4"
- Framework 05 → MTP-1 (FP8-compatible, ~90% acceptance)
- Benchmark across 6 variants confirms the prediction
- **Final config**: FP8 + MTP-1 — measured ~61 tok/s single-request vs ~12 tok/s for NVFP4 in the same software stack

## Not on DGX Spark?

The frameworks are hardware-aware but not Spark-locked. Re-score the relevant dimensions for your GPU using [`docs/hardware-portability/porting-to-other-gpus.md`](docs/hardware-portability/porting-to-other-gpus.md). Notes for RTX PRO 6000 Blackwell, B200, and H100/H200 included.

## Contribute a Case Study

The most useful contribution is a new case showing a different model + hardware combination — especially one where the frameworks **fail** to predict the right answer (we'll update the frameworks).

See [`case-studies/_template.md`](case-studies/_template.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Repository Map

```
frameworks/      decision frameworks (the methodology)
case-studies/    worked examples (validation that frameworks work)
agent/           multi-provider evaluation agent (OpenAI + Anthropic)
tools/           reusable deploy + benchmark tooling for any vLLM model
docs/            MkDocs site source (rendered to GitHub Pages)
scripts/        repo maintenance (sanitization lint etc.)
tests/          pytest suite
```

## Sanitization Policy

This repository **contains no proprietary, internal, or NDA-bound information**. Hostnames, internal IPs, customer business names, and unreleased benchmarks are blocked at CI via [`scripts/check-no-internal.sh`](scripts/check-no-internal.sh). See [`SECURITY.md`](SECURITY.md) if you find a leak.

## License

[Apache-2.0](LICENSE). If you build on the frameworks or agent, citation is appreciated — see [`CITATION.cff`](CITATION.cff).
