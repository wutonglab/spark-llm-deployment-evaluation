# Framework 05: Speculative Decoding for Qwen3.6-class MoE on Blackwell

> **Purpose**: scoring guide and known-good defaults for picking a speculative-decoding method (MTP / EAGLE-3 / DFlash / none) on Qwen3.6-style MoE+linear-attention models, targeting Blackwell-class GPUs.

## Rule

For Qwen3.6-35B-A3B (and similar MoE+Mamba hybrids) under vLLM 0.17.x on Blackwell consumer / workstation hardware: **pick MTP-1**. Avoid EAGLE-3, DFlash on FP8, and MTP-`n ≥ 2`.

## Why

Each non-MTP-1 candidate has a concrete blocker on this model class today:

- **EAGLE / EAGLE-3**: the speculator registry only ships draft models for Qwen3 *dense* (8B / 14B / 32B). No EAGLE-3 draft exists for Qwen3.6-35B-A3B (MoE + Gated DeltaNet hybrid). Training your own is hard because the target architecture is unusual.
- **DFlash**: a Qwen3.6-35B-A3B DFlash drafter exists (community), but it has been observed to **break under FP8 quantization** — only the first few speculated tokens are accepted before fallback, eliminating the gain. DFlash needs BF16 (doubling memory) to work properly, which is impractical on memory-constrained Spark-class hardware.
- **MTP-2**: at high concurrency (c = 8), independent measurements report ~17/32 requests failing due to acceptance-rate estimation errors.
- **MTP-3**: triggers a [vLLM bug](https://github.com/vllm-project/vllm/issues/40756) — long sequences (cumulative > 26K tokens, output > 1200 tokens) hit `illegal memory access` and crash the engine.
- **MTP-1**: the Qwen3.6 model ships with the MTP head as part of the checkpoint. Acceptance rate ~90% in measured workloads. Measured speedup on consumer Blackwell + FP8: **~18% single-request, +15–43% at c = 4 long output**, zero observed failures.

## How to Apply

1. **First**, run framework [`01-fp8-vs-nvfp4-decision.md`](01-fp8-vs-nvfp4-decision.md) to fix the quantization format. Speculative decoding choice depends on it.
2. **Then** apply this framework based on the chosen format + model architecture.

### Decision Table (target: Blackwell-class GPU, vLLM 0.17.x)

| Model / Quant combination | Recommended speculative method |
|---|---|
| Qwen3.6-35B-A3B + FP8 | **MTP-1** (this document) |
| Qwen3 *dense* 8B/14B/32B + FP8 | EAGLE-3 (RedHatAI speculator registry) |
| Qwen3.6-35B-A3B + BF16 | DFlash *might* work (validate acceptance length) |
| Any model + closed-source quant without MTP head | Disable speculative decoding |
| Production must use prefix-caching heavily | Test MTP-1 first; if cache invalidation costs dominate, disable spec |

### vLLM flag

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Do **not** set `num_speculative_tokens >= 2` for this model on this stack.

## Compatibility Notes

- **MTP-1 and prefix-caching**: previously believed to be mutually exclusive in vLLM. In vLLM 0.17.x both can be enabled — KV reuse still works on top of speculative decoding. Verify on your version.
- **`kv-cache-dtype fp8`**: independent of speculative decoding; safe to combine with MTP-1.
- **`flashinfer` attention backend**: measured *worse* than the default backend on this model class at multiple concurrencies. Stick with the default.

## Known Failure Modes

| Symptom | Likely cause |
|---|---|
| Acceptance rate drops to ~50% on random-prompt traffic | Expected — MTP-1 sees ~90% on representative workloads, ~70-80% on adversarial random prompts |
| Engine crash on a long output | If using MTP-`n ≥ 2`, downgrade to MTP-1 |
| First request after container restart slow (~20 tok/s) | CUDA-graph warmup; subsequent requests recover to steady-state ~60 tok/s |
| DFlash accept count never exceeds 4–6 | DFlash + FP8 incompatibility; switch to MTP-1 or change quant to BF16 |

## Cost of Skipping

- Disabling speculative decoding on Qwen3.6-35B-A3B FP8 on a memory-bandwidth-limited GPU **drops single-request throughput by ~18%** at the same quality.
- Choosing MTP-3 instead of MTP-1 risks engine crash on long sequences — a production-affecting bug.
- Choosing EAGLE-3 without checking the speculator registry leads to "no compatible draft found" error at launch.

---

## Boundary Conditions

- **Time decay**: this recommendation reflects vLLM 0.17.x + Qwen3.6 model snapshot. When either updates significantly, re-validate (the upstream speculative-decoding landscape moves fast).
- **Different model family**: the rule is Qwen3.6-A3B-specific. For Qwen3 dense, EAGLE-3 likely wins. For Llama-class, check whether a draft model has been trained.
- **vLLM version**: prior versions may have stricter MTP-vs-prefix-cache exclusion. If on < 0.17, test compatibility before enabling both.

---

## Related Frameworks

- [`01-fp8-vs-nvfp4-decision.md`](01-fp8-vs-nvfp4-decision.md) — pick the quantization first; speculative method depends on it
- [`03-inference-research-channels.md`](03-inference-research-channels.md) — check the speculator registry and vLLM issues before picking a method
- [`how-to-use-these-frameworks.md`](how-to-use-these-frameworks.md) — calling order
