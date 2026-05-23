# Stage 3 — Generate Deployment Configuration

## Inputs

- Stage 1 chose: adopt model `{model}`
- Stage 2 chose: quantization `{quant}`, speculative method `{spec}`
- Hardware: `{target_hardware}`

## Your Task

Generate a parameterized `.env` and a launch command that an operator can use to bring up vLLM with the chosen configuration.

1. **Load** `tools/deploy/variant-configs/` directory listing via `read_file` for any existing variant template that matches your chosen (quant, spec) tuple. If a match exists (e.g. `with-mtp1.env`), use it as a base.
2. **Customize** with:
   - `MODEL_ID` = the model id from Stage 1 (using the quant suffix if it's a quantized variant, e.g. `Qwen/Qwen3.6-35B-A3B-FP8`)
   - `VLLM_IMAGE` = `nvcr.io/nvidia/vllm:26.03.post1-py3` (or a more recent pinned digest)
   - Variant knobs per Stage 2 decision
3. **Write** the artifact as `proposed-config.env`.
4. **Also write** the equivalent `docker run` command as `proposed-launch-command.sh`.

## Deliverables

### `proposed-config.env`

A complete `.env` file with placeholders only for the secrets the operator must fill in (HF_TOKEN, WORKSPACE).

### `proposed-launch-command.sh`

A single executable shell file that:
- Sources `proposed-config.env`
- Runs `docker run -d ...` with all the right flags assembled from the env

### Append to `evaluation-report.md`

A "## Stage 3: Deployment Configuration" section with:
- The chosen variant name
- A copy of the `docker run` command
- The reason this configuration was chosen (linking back to Stage 2's reasoning)

## Constraints

- **No defaults for HF_TOKEN, WORKSPACE, or VLLM_PORT** — these must be placeholders the operator fills in
- **No internal hostnames or IPs** in any generated artifact
- All variant knobs must trace back to either Stage 2's decisions or to documented defaults
