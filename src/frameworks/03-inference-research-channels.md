# Framework 03: Inference Research Channel Baseline

> **Purpose**: when evaluating an LLM **inference / quantization / speculative-decoding / hardware-adaptation** decision, this is the minimum set of channels you must consult before forming a recommendation.

## Rule

For any inference-stack decision (which quantization, which speculative method, which kernel backend, which container image), **check at least one data point from each of the 7 channel categories below before drawing a conclusion**. Skipping a category is the most common cause of wrong conclusions.

## Why

The most consequential facts about modern LLM inference are usually **not in the inference framework's own documentation**. Examples this repository's case studies surfaced:

- A specific consumer-grade Blackwell SM is missing a tensor-core instruction available on its datacenter cousin — only mentioned in a kernel library issue tracker.
- A community-patched container image with 10K+ pulls that fixes a quantization regression — only discoverable via Docker Hub + GPU vendor forum.
- A speculative-decoding draft model for one model family exists but not for another — only listed in a separate speculator registry.
- A correctness bug in a speculative method at high concurrency — only in a GitHub Issue thread.

A reader who only checks the inference framework's own README would have missed all four and recommended the wrong stack.

## How to Apply

1. State your question: hardware X + model Y + objective (throughput / latency / memory / accuracy).
2. **In parallel**, hit each of the 7 channel categories. Capture ≥ 1 data point per category.
3. Cross-validate: any "+N% speedup" / "fully supported" claim must be confirmed by ≥ 2 independent sources.
4. Cite source URLs in your decision write-up.
5. Do not finalize a recommendation until all 7 categories produced data (or you explicitly justify why a category was non-applicable).

---

## The 7 Channel Categories

### 1. HuggingFace Model Cards

| What to look for | Where |
|---|---|
| Official launch parameters | `huggingface.co/<official-namespace>/<model-name>` |
| Quantized variants (FP8/NVFP4/AWQ/GPTQ) | Same org, different tag |
| Speculative-decoding draft weights (MTP / EAGLE / DFlash) | `<community-org>/<model>-speculator.eagle3`, `<lab>/<model>-DFlash` |
| Community fixed forks | Namespaces like `RedHatAI/`, `unsloth/`, `QuantTrio/` often patch mainline bugs |
| Search pattern | `huggingface.co/search?q=<model_name>+FP8` (also `+NVFP4`, `+speculator`, `+eagle3`) |

### 2. Inference-Framework Official Recipes + Docs

| URL pattern | Use |
|---|---|
| `https://recipes.vllm.ai/<Org>/<Model>` | Per-model launch recipe |
| `https://docs.vllm.ai/projects/recipes/en/latest/<Org>/<Series>.html` | Series-level usage |
| `https://docs.vllm.ai/en/latest/features/speculative_decoding/<method>/` | Per-method feature pages |
| `https://docs.vllm.ai/projects/speculators/en/latest/` | Speculator standard library |
| `https://docs.vllm.ai/projects/ascend/<...>` | Hardware backend docs |
| Equivalents for SGLang, TensorRT-LLM, MLC-LLM | Each maintains its own recipe collection |

### 3. GPU Vendor Developer Forums

| Sub-forum / search | Posts to find |
|---|---|
| GPU-specific sub-forums (e.g. DGX-Spark / B200 / Hopper) | "X has landed" deployment threads, "Marlin Fix" patch threads, "CUTLASS Kernel Optimization" measured-perf threads, "broken" community complaints + vendor responses |
| CUDA Programming sub-forum | Low-level kernel / PTX instruction limitations |
| Inference-framework sub-forums | Cross-comparisons with internal constraints |
| Search pattern | `site:forums.developer.nvidia.com "<arch>" "<feature>"` (e.g. `"sm_121" "NVFP4"`) |

### 4. Key GitHub Repos + Issues

| Repo | What to look for |
|---|---|
| `vllm-project/vllm` | Issues / PRs / discussions — most production bugs land here |
| `vllm-project/speculators` | Authoritative list of which models have which speculator drafts |
| `NVIDIA/cutlass` | Architecture admissibility issues, PTX compatibility, BSD-licensed code you can patch |
| `NVIDIA/TensorRT-LLM` | Issues like FP4 kernel SMEM overflow on specific archs |
| Community deployment-recipe repos | Multi-Spark / multi-Blackwell docker compositions and their issue trackers |
| Community patch forks | Search GitHub for `dgx-vllm`, `<arch>-vllm-docker`, `<model>-speedhack` patterns |
| Upstream model repos | `<OrgName>/<Model>` GitHub repos host the reference implementation + release notes |
| Search pattern | GitHub global search for `"<arch>" "<feature>"` or `is:issue is:open <model_name>` |

### 5. Independent Blogs / Benchmarks

| Source pattern | Value |
|---|---|
| Independent reviewers' benchmark blogs | Rigorous measurements + bottleneck analysis |
| Patch-author personal blogs | First-hand explanation of why a fix was needed |
| Cross-method comparison newsletters | Speculative decoding / quantization side-by-side |
| Deployment-troubleshooting blogs | Error-symptom → root-cause maps |
| Recipe-walkthrough blogs | Pragmatic end-to-end deployment narratives |
| Cloud-provider engineering blogs | Large-cluster numbers and failure modes |
| Consumer-GPU benchmark blogs | Edge-case behavior on non-datacenter hardware |
| Concurrency-focused benchmark blogs | Multi-stream scheduling effects |
| Search pattern | WebSearch `"<model>" "<hardware>" benchmark` or `"<quant format>" "<SM_arch>"` |

### 6. Docker Hub / NGC / Container Registries

| Source | What to check |
|---|---|
| Community patch images on Docker Hub | Pull counts, last-updated date |
| Official vendor images (e.g. `nvcr.io/nvidia/vllm:*`) | Pinned version digests |
| Third-party optimized images | `<user>/<framework>-<gpu>` patterns |
| Must-check metrics | Pulls (adoption), last update (maintenance), size (what's bundled) |

### 7. Academic Papers (arXiv)

| Topic | Keywords |
|---|---|
| Speculative decoding | `"speculative decoding"` + method name (EAGLE-3, MTP, DFlash, Medusa, P-EAGLE) |
| Quantization | `"FP8 quantization"`, `"NVFP4"`, `"MXFP4"`, `"block-scale"` + LLM |
| Training/distillation for inference | `"SpecForge"`, `"TensorRT-Model-Optimizer"`, `"Marlin"` |
| Usage | Validate whether community / vendor "theoretical upper bound" claims are actually achievable |

---

## Cost of Skipping Categories (failure modes)

| Skipped category | Wrong conclusion you'd reach |
|---|---|
| HuggingFace community quant namespaces | "There's no EAGLE-3 draft for model X" — when one exists under a community org |
| GPU vendor forum | Miss the silicon-level gap (e.g. missing tensor-core instruction) and blame software |
| GitHub issues | Push a buggy speculative method to production (e.g. crash at high concurrency) |
| Docker Hub | Miss a community fix image with proven adoption signals |
| Independent blogs | Believe vendor marketing numbers that don't generalize across the same architecture family |
| arXiv | Lose the theoretical-acceptance-rate intuition for picking N in num_speculative_tokens |

---

## Boundary Conditions

- **Time decay**: this list reflects channel landscape at the time of writing. Sites move; if a URL 404s, re-search by site name + keyword before declaring the channel dead.
- **Coverage gaps**: cross-lingual / regional channels (Chinese-language tech blogs, JP/KR forums) supplement when relevant.
- **Private / proprietary channels**: customer networks, NDA channels, internal vendor docs are **explicitly out of scope** for this repository.

---

## Related Frameworks

- [`01-fp8-vs-nvfp4-decision.md`](01-fp8-vs-nvfp4-decision.md) — uses this channel list as inputs for the 5-dim quantization score
- [`05-mtp-speculative-decoding.md`](05-mtp-speculative-decoding.md) — uses this channel list to confirm draft-model availability + known bugs
- [`how-to-use-these-frameworks.md`](how-to-use-these-frameworks.md) — when this channel list is consulted in the overall flow
