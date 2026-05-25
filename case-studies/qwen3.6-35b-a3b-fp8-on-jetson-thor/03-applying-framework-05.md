# Layer 2 · Applying Framework 05 (Speculative Decoding) on Jetson Thor

Framework 05 is a **decision table**, not a continuous score. The table predicts:

- **MTP-1**: viable on any hardware that runs the FP8 base, if the model ships an MTP head. Recommended default.
- **MTP-2**: unstable at concurrency on Spark.
- **EAGLE-3**: requires a draft checkpoint; Qwen3.6 doesn't ship one.
- **DFlash**: breaks under FP8.

## Decision: MTP-1

The `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint includes a separate `mtp.safetensors` (~ 850 MB) MTP head. vLLM loads it with:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Why we **did** measure MTP-2 anyway

The Spark case warned MTP-2 is unstable at concurrency. We retested on Thor because **`sm_110` is a different ISA** and the kernel paths may differ — Framework 05's conclusions are explicitly hardware-aware.

| Variant | 256-out | 512-out | 1024-out | 2048-out | c=4 / 1024 aggregate |
|---|---:|---:|---:|---:|---:|
| `with-mtp1` | 50.7 | 52.3 | 56.8 | 66.7 | 138.4 |
| `with-mtp2` | 50.9 | 57.8 | 57.5 | 58.6 | **126.2** (↓ 9% vs MTP-1) |

**Direction matches the Spark case** (MTP-2 hurts at concurrency), **magnitude is similar** (-9% on Thor vs the Spark case study's "unstable, do not use"). MTP-1 wins on the aggregate metric we care about.

We did **not** test MTP-3 — Framework 05 marks MTP-3 as hitting a known bug and the Spark case confirmed it; nothing about Thor's `sm_110` ISA would obviously change that. If a future case study disagrees, please contribute.

## Channels Consulted (per Framework 03)

1. **Speculator catalog**: [`vllm-project/speculators`](https://github.com/vllm-project/speculators) — confirms MTP-1 is the only viable speculator for Qwen3.6-A3B at the time of this writing (no EAGLE-3 draft published).
2. **vLLM tracking issues**: searched `MTP-2 concurrent` and `MTP unstable` — multiple reports across Hopper, Spark, and now Thor of degradation at c≥4.
3. **In-repo signal**: the [Spark case study `03-applying-framework-05.md`](../qwen3.6-35b-a3b-fp8-on-dgx-spark/03-applying-framework-05.md) provided the prior; this case provides cross-hardware confirmation.

## Subtle Thor-specific finding (worth flagging for Framework 05 maintainers)

MTP-1's **lift distribution** is non-uniform on Thor:

| Output length | `baseline` (no MTP) | `with-mtp1` | MTP-1 lift |
|---:|---:|---:|---:|
| 256 | 38.5 | 50.7 | **+32%** |
| 512 | 50.0 | 52.3 | +5% |
| 1024 | 56.2 | 56.8 | +1% |
| 2048 | 56.3 | 66.7 | **+18%** |

vs the Spark case showing a roughly uniform +18% across all output lengths.

This is the kind of cross-hardware signal Framework 05 was designed to surface. We have **not** opened a `framework-improvements` Discussion yet — first, we want to verify the pattern is stable across (a) different prompt populations and (b) the post-engine-tuning configuration. The bench script under `tools/benchmark/` should reproduce these numbers if pointed at this case's `data/` directory.

The conclusion does **not** change: **MTP-1 is still the right call on Thor**. The lift just isn't where Spark's lift is.
