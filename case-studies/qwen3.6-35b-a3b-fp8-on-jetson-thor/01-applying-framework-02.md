# Layer 0 · Applying Framework 02 (Model Selection) on Jetson Thor

Framework 02 scores model **adoption** along 6 dimensions that are mostly **hardware-independent** (capability, business fit, license, ecosystem, switching cost, sustainability). The only dimension that *can* shift between Spark and Thor is **ecosystem maturity for the target hardware** (does the model "land" in the local stack?). Everything else is identical to the Spark case.

## Score: **0.78 / 1.00 → adopt**

This is the **same score** as the DGX Spark case study ([`../qwen3.6-35b-a3b-fp8-on-dgx-spark/01-applying-framework-02.md`](../qwen3.6-35b-a3b-fp8-on-dgx-spark/01-applying-framework-02.md)) with one note on the ecosystem dimension.

| Dimension | Weight | Score | Why (Thor-specific notes only) |
|---|---:|---:|---|
| Capability lift over predecessor | 0.20 | 0.75 | Hardware-independent. Carries over from Spark case. |
| Business fit | 0.15 | 0.80 | Hardware-independent. |
| License | 0.10 | 0.70 | Qwen community license — hardware-independent. |
| Ecosystem maturity | 0.20 | **0.75** ↘ | **Slightly lower than Spark (0.85).** The `vllm/vllm-openai:nightly-aarch64` image exists and works, but the **NGC NVIDIA-distributed** image (`nvcr.io/nvidia/vllm:26.03.post1-py3`) used in the Spark case is **x86_64-only**. Patch availability for `aarch64` Blackwell is downstream of the upstream community cadence; expect ~1-2 week lag on critical fixes. Mitigated by: vLLM `nightly-aarch64` ships actively, and one user-space workaround (`pip install pytest` inside container — see [`05-decision.md`](05-decision.md)) was needed for MTP head registration. |
| Switching cost from predecessor | 0.15 | 0.85 | Hardware-independent. |
| Sustainability (continued upstream activity) | 0.20 | 0.80 | Hardware-independent. |
| **Weighted total** | | **0.78** | adopt |

## Channels Consulted (per Framework 04)

Same set as the Spark case — we did not re-derive model-selection from scratch:

1. **Author resources**: HuggingFace `Qwen/Qwen3.6-35B-A3B-FP8` model card + `RedHatAI/Qwen3.6-35B-A3B-NVFP4` derivative card.
2. **Ecosystem signal**: `vllm-project/vllm` issue tracker filtered to `Qwen3` label; checked for `aarch64` / Jetson / sm_110 issues. No critical blockers; one MTP-related transformer dependency triggers `cupy.testing` import which expects `pytest` (workaround documented in `05-decision.md`).
3. **Benchmark community**: validated headline numbers against [DGX Spark case study](../qwen3.6-35b-a3b-fp8-on-dgx-spark/) — same model, same generation kernel family.

## Why we didn't re-score from scratch

Framework 02 explicitly says (per [`how-to-use-these-frameworks.md`](../../frameworks/how-to-use-these-frameworks.md)) that **model-selection is hardware-agnostic except for the ecosystem-maturity dimension**. If the model passes 02 on Spark and the only delta is `aarch64` image availability (which is fine), the decision is *adopt* without further work.

This is the **right answer** even though the *quantization* decision (FP8 vs NVFP4) will be very different on Thor — that's Framework 01's job, not 02's. Keeping these layers separated is the whole point of the methodology.
