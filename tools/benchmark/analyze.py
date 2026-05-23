#!/usr/bin/env python3
"""Aggregate bench.py CSVs into a markdown report and (optionally) diff against a baseline.

Usage:
    python analyze.py                            # all results/*.csv → results/report.md
    python analyze.py --against ../../case-studies/qwen3.6.../data/baseline-results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def key(r: dict) -> tuple:
    return (r["variant"], int(r["concurrency"]), int(r["output_length"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    p.add_argument("--against", default=None, help="Baseline CSV to compare against (optional)")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Tolerance for prediction-validated check (default ±15%)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output markdown path (default: <results-dir>/report.md)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"results dir not found: {results_dir}")

    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"no CSVs found under {results_dir}")

    # Latest-wins per (variant, concurrency, output_length)
    measured: dict[tuple, dict] = {}
    for cp in csvs:
        for r in load_csv(cp):
            measured[key(r)] = r

    baseline: dict[tuple, dict] = {}
    if args.against:
        bp = Path(args.against)
        if not bp.is_file():
            sys.exit(f"baseline not found: {bp}")
        for r in load_csv(bp):
            baseline[key(r)] = r

    # Group for output
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in measured.values():
        by_variant[r["variant"]].append(r)

    out_path = Path(args.out) if args.out else (results_dir / "report.md")
    lines: list[str] = []
    lines.append("# Benchmark Results\n")
    lines.append("## Measured\n")
    for variant in sorted(by_variant):
        lines.append(f"### Variant: `{variant}`\n")
        lines.append("| Concurrency | Output length | tok/s | Requests |")
        lines.append("|---:|---:|---:|---:|")
        for r in sorted(by_variant[variant], key=lambda x: (int(x["concurrency"]), int(x["output_length"]))):
            lines.append(
                f"| {r['concurrency']} | {r['output_length']} | "
                f"{float(r['output_tokens_per_second']):.1f} | "
                f"{r['n_requests']} |"
            )
        lines.append("")

    pass_all = True
    if baseline:
        lines.append("## Comparison vs Baseline\n")
        lines.append("| Variant | Concurrency | Output | Baseline tok/s | Measured tok/s | Delta % | Verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|:---:|")
        for k, b in sorted(baseline.items()):
            m = measured.get(k)
            if not m:
                lines.append(
                    f"| {b['variant']} | {b['concurrency']} | {b['output_length']} | "
                    f"{float(b['output_tokens_per_second']):.1f} | — | — | ⚠️ not measured |"
                )
                continue
            base_v = float(b["output_tokens_per_second"])
            meas_v = float(m["output_tokens_per_second"])
            delta = (meas_v - base_v) / base_v if base_v else 0
            if abs(delta) <= args.tolerance:
                verdict = "✅"
            elif delta < 0:
                verdict = "❌"
                pass_all = False
            else:
                verdict = "✅ (better)"
            lines.append(
                f"| {b['variant']} | {b['concurrency']} | {b['output_length']} | "
                f"{base_v:.1f} | {meas_v:.1f} | {delta * 100:+.1f}% | {verdict} |"
            )
        lines.append("")
        lines.append(f"**Tolerance**: ±{args.tolerance * 100:.0f}%")
        lines.append("")
        lines.append(
            "**Overall verdict**: "
            + ("✅ all within tolerance" if pass_all else "❌ at least one variant outside tolerance")
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    sys.exit(main())
