# Porting the Methodology to Other GPUs

The frameworks in this repo are **hardware-aware but not Spark-locked**. To apply them to a different GPU, re-score the hardware-sensitive dimensions and re-run the agent.

## Steps

### 1. Capture Your Hardware Profile

Note (in sanitized form):

- Vendor + product class (e.g. "NVIDIA RTX PRO 6000 Blackwell", "NVIDIA B200")
- SM architecture (`sm_120`, `sm_100a`, etc.) — use `nvidia-smi --query-gpu=compute_cap --format=csv` if unsure
- Memory size + bandwidth (e.g. "80 GB HBM3e ~3.35 TB/s")
- Form factor / cooling (some GPUs sustained-clock lower than nameplate)

### 2. Re-score Framework 01 (FP8 vs NVFP4)

The two dimensions that change most across GPUs:

- **Hardware capability** (dim 1): is your SM `sm_100a/sm_103a` (datacenter Blackwell with `tcgen05`)? Or `sm_120/sm_121` (consumer/workstation, missing `tcgen05`)? Score per the rubric.
- **Memory bandwidth** (dim 2): use your GPU's actual measured bandwidth, not the nameplate.

The other dimensions (software, model architecture, workload) are usually unchanged if you're keeping the same model + the same workload.

### 3. Re-score Framework 02 (Model Selection)

Most dimensions don't change with hardware swap, but **dim 5 (switching cost)** may shift if the new hardware requires a different inference framework / container / driver.

### 4. Re-check Framework 05 (Speculative Decoding)

The decision table in framework 05 is partly architecture-aware. For example, on a GPU with full datacenter-Blackwell support, EAGLE-3 may unlock options that were blocked on Spark.

### 5. Run the Agent

```bash
python agent/evaluate.py \
  --model <your-model-hf-id> \
  --target-hardware "<your hardware string>" \
  --business-scenario "<your scenario>" \
  --provider openai
```

The agent walks all four layers and produces a fresh evaluation report.

## Specific Notes for Common Hardware

### NVIDIA B200 (sm_100a, datacenter Blackwell)

- Framework 01 hardware score: **0.30** (full FP4 silicon stack)
- Framework 01 bandwidth score: **0.30** (HBM3e ~5 TB/s)
- Software stack: mainline vLLM officially supports NVFP4 on sm_100a; no community patches needed → high score
- Most likely outcome: NVFP4 wins by a large margin over FP8

### RTX PRO 6000 Blackwell (sm_120, workstation)

- Framework 01 hardware score: **0.15–0.20** (similar gaps to sm_121, slightly larger SMEM)
- Framework 01 bandwidth score: **~0.25** (GDDR7 ~1.4 TB/s)
- Software stack: NVFP4 fast path lands earlier here than on sm_121
- Most likely outcome: NVFP4 + MTP-1 wins (community reports ~104 tok/s single-request on Qwen3.6-27B-class models)

### NVIDIA Jetson Thor (sm_110)

- Framework 01 hardware score: **0** (no native FP4 tensor cores — empirically confirmed)
- Framework 01 bandwidth score: ~0.55 (Developer Kit class: 128 GB LPDDR5X, ~273 GB/s)
- **Confirmed outcome** (case study: [`Qwen3.6-35B-A3B-FP8 on Jetson Thor`](https://github.com/wutonglab/spark-llm-deployment-evaluation/blob/main/case-studies/qwen3.6-35b-a3b-fp8-on-jetson-thor/)): FP8 + MTP-1 reaches ~67 tok/s single-stream and ~303 tok/s aggregate at c=16, while the NVFP4 build of the same model on the same hardware ran at 2.2 tok/s — a **30× measured slowdown**, the strongest framework-validating signal in the repo so far.
- Notes specific to Thor (not portable from Spark): use `vllm/vllm-openai:nightly-aarch64` (NGC NV image is x86_64-only); `--gpu-memory-utilization 0.85` OOMs (unified-memory `Free` after boot is ~100 GB, use 0.75 + `drop_caches`); `--max-num-batched-tokens=32768` and `--max-num-seqs=64` outperform Spark's `16384 / 8` by 15-26% on single-stream; one user-space `pip install pytest` workaround inside the container is required for MTP head registration.

### NVIDIA H100 / H200 (sm_90, Hopper)

- Framework 01 hardware score: **0** (no NVFP4)
- Framework 01 bandwidth: ~0.20 (HBM2e/HBM3)
- Most likely outcome: FP8 with MTP-1 (if the model ships the head); NVFP4 not applicable

## Reporting Back

If you run the methodology on hardware not covered above — please contribute the result as a new case study! See [`case-studies/_template.md`](https://github.com/wutonglab/spark-llm-deployment-evaluation/blob/main/case-studies/_template.md).

If the framework's prediction was wrong, open a Discussion under `framework-improvements`. Mismatches are how the rubrics improve.
