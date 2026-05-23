# Deployment Prerequisites

Before running `launch.sh`, confirm the following on the target host.

## 1. Hardware Visibility

```bash
nvidia-smi
```

Expected:
- The target GPU appears in the output
- Driver version is reasonable (recent enough to support your vLLM image's CUDA runtime)
- No error messages

## 2. Container Runtime

```bash
docker version
docker info | grep -i nvidia
```

Expected:
- `docker version` succeeds and shows a recent Docker (≥ 24)
- `docker info` shows `nvidia` as a registered runtime (proves NVIDIA Container Toolkit is installed)

If NVIDIA runtime is missing, install [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your distro.

## 3. Disk Space

```bash
df -h $WORKSPACE
```

Need ≥ 60 GB free under `$WORKSPACE` for the model + container layers. For 35B-class models in FP8, the model itself is ~35 GB; the vLLM container image adds ~25 GB.

## 4. Network — HuggingFace Access

```bash
curl -fsS https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B-FP8 -H "Authorization: Bearer $HF_TOKEN" | head -3
```

Expected: JSON response, not 401 / 403 / 404. If the model is gated, accept the license on HuggingFace UI first.

## 5. Port Availability

```bash
lsof -iTCP:$VLLM_PORT -sTCP:LISTEN -n -P
```

Expected: empty output (the port is free).

## 6. Python (for model download + benchmarks)

```bash
python3 -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

If `huggingface_hub` isn't installed:

```bash
pip install --user huggingface_hub
```

## 7. Optional: Sanity Pull of the Image

```bash
docker pull "${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.03.post1-py3}"
```

Caches the image before launch (~25 GB download). Saves you the launch-time wait.

## After All Checks Pass

```bash
cp variant-configs/with-mtp1.env .env
$EDITOR .env          # fill in HF_TOKEN, WORKSPACE, VLLM_PORT
bash launch.sh
bash healthcheck.sh   # smoke test
```
