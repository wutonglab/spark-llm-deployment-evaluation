#!/usr/bin/env python3
"""Generic vLLM benchmark client.

Measures single-request and concurrent throughput against an OpenAI-compatible
endpoint. Writes results as CSV; tools/benchmark/analyze.py turns CSVs into
markdown reports.

Usage:
    python bench.py --variant with-mtp1 --concurrency 1,4 --output-lengths 512,1024 --duration 60
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import httpx
except ImportError:
    sys.exit("Install httpx: pip install httpx")


@dataclass
class Result:
    variant: str
    concurrency: int
    output_length: int
    output_tokens_per_second: float
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    first_token_seconds: float | None
    n_requests: int


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    r = await client.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    data = r.json()
    usage = data.get("usage", {})
    return {
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def run_concurrency(
    base_url: str,
    model: str,
    concurrency: int,
    output_length: int,
    duration: float,
    prompts: list[str],
) -> dict:
    """Issue requests at the given concurrency for `duration` seconds. Returns aggregates."""
    async with httpx.AsyncClient() as client:
        # Warm up
        await one_request(client, base_url, model, prompts[0], max_tokens=32)

        results: list[dict] = []
        t_start = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)

        async def worker(prompt: str):
            async with sem:
                try:
                    r = await one_request(client, base_url, model, prompt, output_length)
                    results.append(r)
                except Exception as e:
                    results.append({"error": str(e)})

        tasks: list = []
        idx = 0
        while time.perf_counter() - t_start < duration:
            tasks.append(asyncio.create_task(worker(prompts[idx % len(prompts)])))
            idx += 1
            await asyncio.sleep(0.01)  # avoid runaway task creation
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - t_start
        ok = [r for r in results if "error" not in r]
        if not ok:
            return {"error": "all requests failed", "n": len(results)}
        total_out = sum(r["completion_tokens"] for r in ok)
        return {
            "elapsed": elapsed,
            "n_ok": len(ok),
            "n_failed": len(results) - len(ok),
            "total_completion_tokens": total_out,
            "output_tokens_per_second": total_out / elapsed,
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in ok) / len(ok),
        }


def load_workload(workload_path: Path, n: int = 50) -> list[str]:
    """Load prompts from JSONL ({\"text\": ...}). If file missing, fall back to a small built-in set."""
    if workload_path.is_file():
        prompts = []
        with workload_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prompts.append(json.loads(line)["text"])
                except Exception:
                    continue
        if prompts:
            return prompts[:n] if len(prompts) > n else prompts

    # Fallback
    base = [
        "Write a one-paragraph summary of the principle behind speculative decoding.",
        "Explain in plain English what FP8 quantization does to a transformer's weights.",
        "List five differences between dense and mixture-of-experts language models.",
        "Describe what a CUDA kernel is, in two sentences.",
        "Compose a four-line poem about memory bandwidth.",
    ]
    rng = random.Random(0)
    return [rng.choice(base) for _ in range(n)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM benchmark client")
    p.add_argument("--variant", required=True, help="Variant tag (used as a label in results)")
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--model", default=os.environ.get("MODEL_ID"), required=False)
    p.add_argument(
        "--concurrency",
        default="1,4",
        help="Comma-separated list of concurrency levels",
    )
    p.add_argument(
        "--output-lengths",
        default="512,1024",
        help="Comma-separated list of max_tokens values",
    )
    p.add_argument("--duration", type=float, default=60.0, help="Seconds per (concurrency, output-length) cell")
    p.add_argument("--workload", default=str(Path(__file__).parent / "workloads" / "short-prompt.jsonl"))
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path; defaults to results/<variant>-<ts>.csv",
    )
    return p.parse_args()


async def amain() -> int:
    args = parse_args()

    if not args.model:
        sys.exit("ERROR: --model not set (or MODEL_ID env not in .env)")

    # Strip /v1 if user pasted with the suffix
    base_url = args.base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    concurrency_list = [int(c) for c in args.concurrency.split(",")]
    output_lens = [int(o) for o in args.output_lengths.split(",")]
    prompts = load_workload(Path(args.workload))

    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / "results" / f"{args.variant}-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []
    print(f"==> Benchmarking variant={args.variant} against {base_url}")
    print(f"    Model: {args.model}")
    print(f"    Workload: {args.workload} ({len(prompts)} prompts)")

    for c in concurrency_list:
        for out_len in output_lens:
            print(f"    Running concurrency={c}, output={out_len} for {args.duration}s ...")
            r = await run_concurrency(base_url, args.model, c, out_len, args.duration, prompts)
            if "error" in r:
                print(f"      FAIL: {r['error']}")
                continue
            res = Result(
                variant=args.variant,
                concurrency=c,
                output_length=out_len,
                output_tokens_per_second=r["output_tokens_per_second"],
                elapsed_seconds=r["elapsed"],
                prompt_tokens=int(r["mean_prompt_tokens"]),
                completion_tokens=r["total_completion_tokens"] // r["n_ok"] if r["n_ok"] else 0,
                first_token_seconds=None,
                n_requests=r["n_ok"],
            )
            results.append(res)
            print(
                f"      tok/s: {r['output_tokens_per_second']:.1f} "
                f"({r['n_ok']} OK / {r['n_failed']} failed in {r['elapsed']:.1f}s)"
            )

    # Write CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"==> Results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
