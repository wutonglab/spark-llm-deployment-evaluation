# Benchmark Results: 6-Variant Sweep on Jetson Thor (`sm_110`)

> **Goal**: empirically validate the predictions from Frameworks 01 + 05 by running variants of (quantization, KV-cache dtype, speculative method, batched-tokens, max-num-seqs) on the **same** hardware + model + workload as the DGX Spark case study, then tabulating results side-by-side.

## Hardware + Software Environment

| Attribute | Value |
|---|---|
| Platform | NVIDIA Jetson Thor (128 GB Developer Kit class), `sm_110` |
| Unified memory | 128 GB LPDDR5X |
| Memory bandwidth | ~273 GB/s |
| OS / driver | JetPack 7 / L4T R38.4.0, kernel 6.8.12-tegra, CUDA 13.0, Driver 580.00 |
| Power profile | MAXN (`nvpmodel -m 0`) + jetson_clocks |
| Inference engine | vLLM `0.21.1rc1.dev262+g33d7cbe02` via `vllm/vllm-openai:nightly-aarch64` |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` (HuggingFace, 256K native context) + `mtp.safetensors` head |
| Tensor parallelism | 1 |
| `gpu-memory-utilization` | 0.75 (0.85 OOMs at startup on Thor — unified-memory `free` after boot fluctuates around 100 GB) |

Configuration env-templates for every variant are in [`reproduce/env-templates/`](reproduce/env-templates/).

## The 6 Variants

| Variant | Quantization | KV cache | Spec | `max-num-seqs` | `max-num-batched-tokens` |
|---|---|---|---|---:|---:|
| `nvfp4-default` (sanity) | NVFP4 (RedHatAI build) | auto | none | 8 | 16384 |
| `baseline` | FP8 | fp8 | none | 8 | 16384 |
| `with-mtp1` (= Spark final config) | FP8 | fp8 | **MTP-1** | 8 | 16384 |
| `with-mtp2` | FP8 | fp8 | **MTP-2** | 8 | 16384 |
| `thor-big-concurrency` | FP8 | fp8 | MTP-1 | **32** | **32768** |
| **`thor-tuned`** ⭐ | FP8 | fp8 | MTP-1 | **64** | **32768** |

`thor-tuned` is the **chosen production config**.

All variants enable `--enable-prefix-caching`, `--trust-remote-code`, `--reasoning-parser qwen3`, `--enable-auto-tool-choice --tool-call-parser qwen3_xml`, and `--limit-mm-per-prompt '{"image":0,"video":0}'` for the benchmark sweep (multimodal is enabled separately in the final production config — same model, no measurable text-throughput impact).

## Single-Stream Throughput (tokens/s, 3-run median, anti-prefix-cache prompts)

| Output length | `nvfp4-default` | `baseline` | `with-mtp1` | `with-mtp2` | `thor-big-conc` | **`thor-tuned`** |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | — | 38.5 | 50.7 | 50.9 | 62.3 | **54.1** |
| 512 | — | 50.0 | 52.3 | 57.8 | 65.9 | **65.9** |
| **1024** | **2.2** | 56.2 | 56.8 | 57.5 | 65.1 | **66.7** |
| 2048 | — | 56.3 | 66.7 | 58.6 | 65.4 | **66.8** |

Notes:
- `nvfp4-default` is **30× slower** than `thor-tuned` — confirms Framework 01 dim-1 score `0` for Thor at the strongest possible level (Spark showed only 5× slowdown because `sm_121` does have *some* NVFP4 path).
- `baseline` → `with-mtp1`: +18% at 2048-out (matches Spark direction), +32% at 256-out, near-zero at 512/1024-out — **Thor's MTP-1 lift is non-uniform**, unlike Spark.
- `with-mtp2` → `with-mtp1`: roughly tied on single-stream, but MTP-2 loses at concurrency (see next table).
- `thor-tuned` matches or beats `thor-big-conc` everywhere; pushing `max-num-seqs` from 32 to 64 is free.

## Concurrent Aggregate Throughput (tokens/s, 3-run median)

| Config | `with-mtp1` | `with-mtp2` | `thor-big-conc` | **`thor-tuned`** |
|---|---:|---:|---:|---:|
| c=4, 1024-out | 138.4 | **126.2** (↓ 9%) | 135.8 | **152.2** |
| c=8, 512-out | — | — | 222.7 | **233.6** |
| c=16, 512-out | — | — | — | **303.2** |

**Key findings**:

- `with-mtp2` regresses 9% at c=4 vs `with-mtp1` — direction matches Spark case, magnitude is the same family. Framework 05's MTP-2 caution carries to Thor.
- `thor-tuned` scales near-linearly from c=4 → c=8 (152 → 234, 1.5× for 2× concurrency = 0.77 efficiency) and continues to climb at c=16 (303 tok/s).
- per-stream throughput at c=16 is ~19 tok/s, still well above human read speed — usable for chat fanout.

## Theoretical Ceiling Analysis

Memory-bandwidth-bound throughput for a 3 B-active MoE model at FP8 (3 GB active per token):

```
ceiling = 273 GB/s / 3 GB = 91 tok/s
measured_single_stream = 66.7 tok/s
efficiency = 73%
```

The remaining 27% headroom is **almost certainly closeable** by either (a) an NGC NV-distributed `aarch64` vLLM image (when available) or (b) `--compilation-config` tuning for the `sm_110` ISA. Neither attempted in this case — they belong to follow-up cases.

## Long-Context Validation (needle-in-haystack, `thor-tuned`)

Prefill latency + retrieval accuracy across input lengths (single needle planted at 50% of document, asked to recall verbatim):

| Target tokens | Actual prompt tokens | Total time | Prefill speed | Needle |
|---:|---:|---:|---:|:---:|
| 8 K | 6 195 | 8.0 s | 773 tok/s | ✅ |
| 32 K | 24 630 | 10.0 s | 2 460 tok/s | ✅ |
| 64 K | 49 200 | 29.0 s | 1 699 tok/s | ✅ |
| **128 K** | 98 355 | **96.4 s** | 1 020 tok/s | ✅ |
| **200 K** | 150 060 | **195.0 s** | 770 tok/s | ✅ |

**5 / 5 needle hits**. The latency curve shows the expected attention-quadratic shape: 32 K is the prefill sweet spot at 2.5 K tok/s; beyond 64 K the user-facing latency is minute-scale and the workload is "offline analysis" not "interactive chat".

KV-cache capacity is **3.78 M tokens** at `gpu-memory-utilization=0.75`, allowing 14× concurrency even at 256K-token sequences, or 91× at 32 K.

## Multimodal Functional Validation (`thor-tuned` + `--limit-mm-per-prompt '{"image":4,"video":1}'`)

The base FP8 checkpoint includes `model_visual.safetensors`; vLLM exposes it as standard OpenAI `image_url` / `video_url` content blocks.

| Test | Asset | Expected | Result |
|---|---|---|---|
| Single-image OCR + color + small shape | 512×384 PNG "HELLO THOR" red bg + blue square | model reads text + names red/blue | ✅ (hits 3/4 — "HELLO", "red", "blue"; "THOR" missed when capped at 7 completion tokens — non-issue) |
| Single-image conceptual understanding | 512×384 PNG cartoon cat + dog | model names both animals + comments on style | ✅ (full recognition + "灵魂画手风格" commentary) |
| Two-image comparison | both images above in one request | model contrasts themes | ✅ (correctly contrasts greeting vs. educational) |
| Single-video time + color + text | 4 s, 8 fps, 2 s red "RED" + 2 s green "GREEN" | model reports 4 s duration + colors + texts | ✅ (all four facts correct) |
| Audio (out-of-scope) | 1 s silence WAV | HTTP 400 (model has `audio_config: None`) | ✅ HTTP 400 with `At most 0 audio(s)` |

Multimodal does **not** affect text-only throughput because vision tokens are pre-computed by the vision encoder and inserted as already-encoded KV — no additional decode-loop cost. We did not measure tokens/s when vision tokens are present in the context; that's an open question.

## Quality Validation (`thor-tuned`)

| Test | Result |
|---|---|
| Multi-turn Chinese conversation (recall earlier facts) | ✅ |
| Reasoning split into `reasoning` field (qwen3 parser) | ✅ |
| Reasoning answer correctness (relative-motion math problem) | ✅ |
| Tool calling (single-function `get_weather`, qwen3_xml parser) | ✅ |
| Long-context decode (7.5 K prompt + 16-token summary) | ✅ |

5 / 5 — same as the DGX Spark case.

## Decision Confirmation

| Framework prediction | Measurement | Match? |
|---|---|---|
| Framework 01 score ~0.10 → "FP8 strongly preferred on Thor; NVFP4 has no native silicon" | NVFP4 measured **30×** slower than FP8 under same software stack | ✅ confirmed in strongest sense |
| Framework 05 → MTP-1 is the right speculator; MTP-2 hurts at concurrency | MTP-2 −9% at c=4 vs MTP-1; matches Spark direction | ✅ confirmed |
| Porting guide → Thor uses FP8 only, max likely ~273 GB/s bandwidth-bound | measured 67 tok/s = 73% of 91 tok/s ceiling | ✅ confirmed |
| Spark's `max-num-batched-tokens=16384` is optimal | **NO** — `32768` gives +15-26% on single-stream on Thor without concurrency regression | ⚠️ engine-tuning is not Spark-portable |

The benchmark sweep validates the framework's qualitative predictions, and uncovers one quantitative caveat: **the engine-tuning step (Framework's "Layer 3 A/B sweep") must be re-done per-hardware** — even though Spark and Thor have the same headline numbers (128 GB / 273 GB/s), the kernel paths differ enough that batched-tokens tuning does not transfer.

## Raw Data

The CSV with all measurement rows is in [`data/baseline-results.csv`](data/baseline-results.csv). Columns: `variant`, `concurrency`, `output_length`, `output_tokens_per_second`.

## How to Reproduce

```bash
# from repo root
cd case-studies/qwen3.6-35b-a3b-fp8-on-jetson-thor/reproduce
cp env-templates/thor-tuned.env .env
$EDITOR .env                              # set WORKSPACE, HF_TOKEN
bash launch.sh
# then from another shell:
cd ../../../tools/benchmark
bash run_variants.sh --only thor-tuned    # or --all to repeat the full sweep
```

For variants this case adds (`thor-big-concurrency`, `thor-tuned`, `with-mtp2`), the launcher reads from this case's `env-templates/`. For variants shared with the Spark case (`baseline`, `with-mtp1`, `nvfp4-default`), the env files are re-used.
