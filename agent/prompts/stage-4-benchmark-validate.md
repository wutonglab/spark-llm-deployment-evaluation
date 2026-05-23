# Stage 4 — Benchmark Validation

## Inputs

- Stage 3 produced: `proposed-config.env` + `proposed-launch-command.sh`
- Target host: `{target_host}` (may be `local`)
- User-specified `--no-benchmark`: `{no_benchmark}`

## Your Task

Only run this stage if `no_benchmark` is false. Otherwise, write a short note in `evaluation-report.md` saying "benchmark stage skipped" and exit.

If running:

1. **Pre-flight**:
   - Use `check_gpu` to confirm the target hardware matches what was assumed in Stages 1-2
   - Use `check_docker` to confirm Docker is reachable
2. **Launch** by running `bash proposed-launch-command.sh` (via `run_shell`). Wait up to 5 minutes for `/v1/models` to be healthy.
3. **Benchmark sweep** — minimal:
   - Single-request × output `{512, 1024}` tokens
   - c=4 aggregate × output `1024` tokens
   - Use `python tools/benchmark/bench.py --variant <stage-2-chosen> --concurrency 1,4 --output-lengths 512,1024 --duration 60`
4. **Compare** the measured numbers against Stage 2's predicted ranges:
   - Single-request 1024 prediction vs measured
   - c=4 1024 prediction vs measured
   - Tolerance: ±15% on throughput, ±20% on TTFT
5. **Write the comparison** as `prediction-vs-actual.md`.

## Deliverables

### `benchmark-results.csv`

The bench.py output (variant, concurrency, output_length, output_tokens_per_second, …).

### `prediction-vs-actual.md`

```markdown
# Stage 4: Prediction vs Actual

| Metric | Stage 2 Predicted | Measured | Within Tolerance? |
|---|---:|---:|:---:|
| Single-request 1024 tok/s | <pred> | <actual> | ✅ / ⚠️ / ❌ |
| c=4 agg 1024 tok/s | <pred> | <actual> | ✅ / ⚠️ / ❌ |

## Verdict
- ✅ "framework prediction validated" if all within tolerance
- ⚠️ "partially validated; investigate dimension X" with explanation
- ❌ "framework prediction wrong; recommend re-scoring dim X" with proposal
```

### Append to `evaluation-report.md`

A "## Stage 4: Benchmark Validation" section with the comparison table and verdict.

## Constraints

- Do not run benchmark longer than 5 minutes per variant (use `--duration 60`)
- If launch fails (container exits, `/v1/models` doesn't respond), capture `docker logs` last 50 lines into `launch-failure.log` and write a "❌ launch failed" verdict
- Do not modify the launch config to "fix" a failure — that would invalidate the validation. Instead, report the failure as evidence Framework 01 / 05 might need revision
