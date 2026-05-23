# Reproducing This Case Study

## What You Need

| Item | Detail |
|---|---|
| Hardware | NVIDIA DGX Spark or compatible GB10 / sm_121 platform |
| OS + Driver | Linux with NVIDIA driver ≥ 580 (or whatever ships with your DGX Spark) |
| Docker | with NVIDIA Container Toolkit |
| Disk | ≥ 60 GB free at `$WORKSPACE` |
| Network | HuggingFace access (model download ~35 GB) |
| HF token | with access to `Qwen/Qwen3.6-35B-A3B-FP8` |

## Step 1 — Pick an env-template and copy it to `.env`

```bash
cd reproduce/

# For the recommended production config:
cp env-templates/with-mtp1.env .env

# For the baseline (no MTP):
# cp env-templates/baseline.env .env

# For the NVFP4 "naive" comparison:
# cp env-templates/nvfp4-default.env .env

$EDITOR .env   # set HF_TOKEN; adjust WORKSPACE / VLLM_PORT if needed
```

## Step 2 — Launch

```bash
bash launch.sh
```

The script will:
1. Validate prerequisites (docker, nvidia-smi, port, disk)
2. Download the model if `$WORKSPACE/models/...` is missing (~35 GB)
3. Remove any stale container with the same name
4. Start a fresh vLLM container with the variant's flags
5. Wait up to 5 minutes for `/v1/models` to return 200

On success you'll see:
```
Endpoint:  http://localhost:8000/v1
Model id:  Qwen/Qwen3.6-35B-A3B-FP8
```

## Step 3 — Smoke test

```bash
curl http://localhost:${VLLM_PORT}/v1/models
curl http://localhost:${VLLM_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3.6-35B-A3B-FP8",
      "messages": [{"role": "user", "content": "say hi"}],
      "max_tokens": 32
    }'
```

## Step 4 — Run the benchmark sweep

```bash
cd ../../tools/benchmark
bash run_variants.sh --only with-mtp1
# or all variants: bash run_variants.sh --all
python analyze.py --against ../../case-studies/qwen3.6-35b-a3b-fp8-on-dgx-spark/data/baseline-results.csv
```

`analyze.py` writes a markdown report into `tools/benchmark/results/` showing your numbers next to this case study's published baseline. A reproduction is considered successful when:

- Single-request tok/s within **±15%** of baseline
- c=4 aggregate tok/s within **±15%** of baseline
- First-request latency within **±20%** (CUDA-graph warmup varies)

## Switching Variants

Stop the current container, switch the `.env`, re-launch:

```bash
docker stop qwen36-fp8     # or qwen36-nvfp4 etc.
cp env-templates/<other-variant>.env .env
bash launch.sh
```

## Troubleshooting

| Symptom | Likely cause + fix |
|---|---|
| `port already in use` | Pick a different `VLLM_PORT` in `.env`, or stop the conflicting service |
| `model download failed` | Bad `HF_TOKEN` or no access to gated repo; verify via `huggingface-cli whoami` |
| Container exits within seconds | Check `docker logs qwen36-fp8`. Most common: `--quantization` mismatch when switching FP8 ↔ NVFP4 (clear `WORKSPACE/models/<old>` and let it re-download) |
| First request is slow (~20 tok/s) | Expected for CUDA-graph warmup. Second request should hit steady-state (~61 tok/s for `with-mtp1`). |
| `with-mtp1` measured throughput closer to `baseline` than expected | Check `vllm` version (`docker exec qwen36-fp8 python3 -c 'import vllm; print(vllm.__version__)'`) — 0.17.x is the validated version |
| All variants give very different numbers from baseline | Your hardware spec is different (different GPU, different memory bandwidth, different driver) — your reproduction *is* still valid for **your** hardware; consider submitting it as a new case study |

## Agent-Driven Reproduction

Instead of running the scripts manually, the [`agent/`](../../../agent/) script automates the entire flow:

```bash
cd ../../../   # repo root
python agent/evaluate.py \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --target-hardware "DGX Spark / GB10 / 128GB / 273 GB/s LPDDR5X / sm_121" \
  --business-scenario "general LLM API service" \
  --provider openai
```

The agent walks all 4 framework layers (selection / quantization / speculative / tuning), produces an evaluation report comparable to this case study, and exits 0 only if measured throughput matches the predictions within tolerance.
