# spark-llm-deployment-evaluation

A reusable **methodology + agent toolkit** for evaluating LLM deployment decisions on NVIDIA DGX Spark (GB10 / sm_121) and similar Blackwell-class hardware.

> **What this is**: a methodology repository.
> **What this is not**: a cookbook for reproducing one specific deployment.

## Quick Navigation

- **[Methodology](methodology/)** — the 4 decision frameworks
- **[Case Studies](case-studies/)** — worked examples that validate the frameworks
- **[Hardware Portability](hardware-portability/porting-to-other-gpus.md)** — use the methodology on GPUs other than DGX Spark
- **[Glossary](glossary.md)** — MTP, NVFP4, MoE, A3B, sm_121, etc.

## Run the Agent

```bash
git clone https://github.com/wutonglab/spark-llm-deployment-evaluation.git
cd spark-llm-deployment-evaluation
pip install -r agent/requirements.txt
cp .env.example .env && $EDITOR .env

python agent/evaluate.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --target-hardware "DGX Spark / GB10 / sm_121" \
  --business-scenario "code review agent" \
  --provider openai
```

Agent output lands in `evaluation-runs/<timestamp>/evaluation-report.md`.

## Contribute

The most valuable contribution is a new case study showing a (model, hardware) combination we haven't tested — especially when the framework prediction turns out to be **wrong**. See [`CONTRIBUTING.md`](https://github.com/wutonglab/spark-llm-deployment-evaluation/blob/main/CONTRIBUTING.md).
