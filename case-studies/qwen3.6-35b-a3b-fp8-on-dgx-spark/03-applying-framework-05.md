# Applying Framework 05 (Speculative Decoding) to Qwen3.6-35B-A3B + FP8 on DGX Spark

> **Goal**: pick the speculative-decoding method to enable for this (model, quant, hardware) tuple.
>
> **Framework**: [`05-mtp-speculative-decoding.md`](../../frameworks/05-mtp-speculative-decoding.md). Channel evidence per [`03-inference-research-channels.md`](../../frameworks/03-inference-research-channels.md).

## Decision

**Enable MTP-1**.

```yaml
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Walking the Decision Table

Framework 05 provides a direct decision table for (model architecture, quantization) tuples. For our case:

| Tuple | Qwen3.6-35B-A3B + FP8 |
|---|---|
| Recommended | **MTP-1** |
| Rejected | EAGLE / EAGLE-3 (no draft), DFlash (FP8-incompatible), MTP-2 / MTP-3 (instability / bugs) |

## Why Not Each Alternative

### EAGLE-3

**Verdict**: ❌ no compatible draft model exists.

The [`speculators`](https://github.com/vllm-project/speculators) registry maintained by the vLLM ecosystem ships EAGLE-3 drafts for the Qwen3 *dense* variants (8B / 14B / 32B). It does **not** ship one for Qwen3.6-35B-A3B (MoE + Gated DeltaNet hybrid). Training a custom EAGLE-3 draft requires deep knowledge of the target model's hidden-state structure — not practical for this case.

**Channel citation**: speculators registry repository listing.

### DFlash

**Verdict**: ❌ incompatible with FP8.

A community DFlash drafter exists for Qwen3.6-35B-A3B. However, independent measurements report that under FP8 base quantization, **only the first 4 of 15 speculated tokens are accepted** before falling back to standard decoding — eliminating the throughput gain.

DFlash works correctly with BF16 base, but that doubles model memory footprint (35 GB FP8 → 70 GB BF16), which would push the deployment uncomfortably close to the 128 GB unified-memory limit when serving multiple concurrent requests.

**Channel citations**:
- DFlash drafter HF model card (capabilities + caveats)
- Independent benchmark blog reporting the FP8 fallback behavior
- Cross-check: arXiv DFlash paper noting block-diffusion's sensitivity to base-model precision

### MTP-2

**Verdict**: ❌ instability at concurrency.

When the speculative tree depth is 2, the second MTP head's input is the *predicted* (not verified) first token. Acceptance-rate estimation degrades with depth, and at concurrency 8 independent measurements report **17 / 32 requests failing** due to acceptance-rate estimation errors.

**Channel citation**: independent benchmark blog (Spark / sm_121 measurements thread).

### MTP-3

**Verdict**: ❌ triggers known engine bug.

[vLLM issue #40756](https://github.com/vllm-project/vllm/issues/40756) documents that MTP-3 (specifically `num_speculative_tokens >= 3`) on long sequences (cumulative > 26K tokens, output > 1200 tokens) hits `illegal memory access` and crashes the engine process. Reproduced by multiple users; no fix available at the time of this case study.

**Channel citation**: the GitHub issue itself (well-documented with traces).

### No Speculative Decoding (baseline)

**Verdict**: leaves ~18% single-request throughput on the table for free.

Disabling speculative decoding entirely is safe and stable, but on a memory-bandwidth-limited GPU like DGX Spark, the cost is measurable. MTP-1 has zero observed failure modes at this stage, so disabling it is overly conservative.

## MTP-1 Caveats

- **Acceptance rate is workload-dependent**: ~90% on representative production prompts (sequential, structured); ~70–80% on adversarial random-token prompts.
- **First request after container restart**: slower than steady-state (~20 tok/s for CUDA-graph warmup); subsequent requests recover to ~61 tok/s single-request.
- **vLLM 0.17.x specific**: prior versions enforce mutual exclusion with prefix-caching. On 0.17.x, both coexist (KV reuse still works on top of speculative decoding) — verify on your version.

## Compatibility With Other Tuning

| Other knob | Compatible with MTP-1? | Notes |
|---|---|---|
| `--kv-cache-dtype fp8` | ✅ yes | Independent of spec decoding |
| `--enable-prefix-caching` | ✅ yes (vLLM ≥ 0.17.x) | Earlier docs said no; verify in your version |
| `--attention-backend flashinfer` | ⚠️ technically yes | But measurements show flashinfer is *worse* than default on this model; do not use |
| `--max-num-batched-tokens 16384` | ✅ yes | Default-ish value, works fine |
| `--reasoning-parser qwen3` | ✅ yes | Required for thinking output parsing |

## Outcome

**MTP-1 enabled in the final config.** All other speculative methods rejected with explicit channel-cited rationale.

This unblocks Layer 3 (engine tuning), where we A/B the remaining knobs (kv-cache dtype, attention backend, batched-token sizing). See [`04-benchmark-results.md`](04-benchmark-results.md).
