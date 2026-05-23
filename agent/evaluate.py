#!/usr/bin/env python3
"""LLM Deployment Evaluation Agent — entry point.

Walks the 4 decision layers (model selection / quantization / speculative /
engine tuning) for a given (model, hardware, business scenario) and produces
an evaluation report.

Usage:
    python agent/evaluate.py \\
      --model Qwen/Qwen3.6-35B-A3B-FP8 \\
      --target-hardware "DGX Spark / GB10 / 128GB / 273 GB/s LPDDR5X / sm_121" \\
      --business-scenario "general LLM API service" \\
      --provider openai

See agent/README.md for full CLI reference.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "agent"))

from core.llm_client import make_client  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.tools import ToolRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM Deployment Evaluation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HuggingFace model id (e.g. Qwen/Qwen3.6-35B-A3B-FP8)")
    p.add_argument(
        "--target-hardware",
        required=True,
        help='Sanitized hardware description (e.g. "DGX Spark / GB10 / 128GB / sm_121")',
    )
    p.add_argument(
        "--business-scenario",
        required=True,
        help='High-level workload description (e.g. "general LLM API service")',
    )
    p.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    p.add_argument("--llm-model", default=None, help="Override the LLM the agent uses")
    p.add_argument("--no-benchmark", action="store_true", help="Skip Stage 4 (no live measurement)")
    p.add_argument("--target-host", default=None, help="Remote host for benchmark (SSH)")
    p.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    p.add_argument("--max-turns-per-stage", type=int, default=20)
    p.add_argument("--run-dir-base", default="evaluation-runs", help="Base dir for run artifacts")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Prepare run dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = REPO_ROOT / args.run_dir_base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Run directory: {run_dir}")

    # Build LLM client
    try:
        llm = make_client(args.provider, args.llm_model)
    except Exception as e:
        print(f"ERROR: failed to initialize LLM client: {e}", file=sys.stderr)
        return 1

    # Build tools
    tools = ToolRegistry(run_dir=run_dir, target_host=args.target_host, dry_run=args.dry_run)

    # Inputs bundle
    inputs = {
        "model": args.model,
        "target_hardware": args.target_hardware,
        "business_scenario": args.business_scenario,
        "provider": args.provider,
        "benchmark": not args.no_benchmark,
        "target_host": args.target_host,
    }

    orch = Orchestrator(
        llm=llm,
        tools=tools,
        run_dir=run_dir,
        inputs=inputs,
        max_turns_per_stage=args.max_turns_per_stage,
    )

    print(f"==> Provider: {args.provider} (model: {llm.model})")
    print(f"==> Stages: 4 ({'Stage 4 SKIPPED' if args.no_benchmark else 'all enabled'})")
    print(f"==> Max turns per stage: {args.max_turns_per_stage}")
    print()

    try:
        rc = orch.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted; partial artifacts preserved in run dir.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print()
    print(f"==> Exit code: {rc}")
    print(f"==> Report: {run_dir}/evaluation-report.md")
    return rc


if __name__ == "__main__":
    sys.exit(main())
