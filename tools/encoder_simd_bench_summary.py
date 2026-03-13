#!/usr/bin/env python3
"""Summarize encoder SIMD microbenchmark criterion outputs.

Reads criterion estimates under a root directory and emits CSV/markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt_ns(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.1f} ns"


def collect_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for est in sorted(root.glob("encoder_*/**/new/estimates.json")):
        rel = est.relative_to(root)
        # e.g. encoder_xyb/srgb_u8_to_xyb/1024x768/new/estimates.json
        parts = rel.parts
        if len(parts) < 4:
            continue
        suite = parts[0]
        bench = "/".join(parts[1:-2])
        data = json.loads(est.read_text())
        mean_ns = float(data["mean"]["point_estimate"])
        low_ns = float(data["mean"]["confidence_interval"]["lower_bound"])
        high_ns = float(data["mean"]["confidence_interval"]["upper_bound"])
        rows.append(
            {
                "suite": suite,
                "benchmark": bench,
                "mean_ns": f"{mean_ns:.3f}",
                "mean_human": fmt_ns(mean_ns),
                "ci_low_ns": f"{low_ns:.3f}",
                "ci_high_ns": f"{high_ns:.3f}",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["suite", "benchmark", "mean_ns", "mean_human", "ci_low_ns", "ci_high_ns"],
        )
        w.writeheader()
        w.writerows(rows)


def write_md(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Encoder SIMD microbenchmark summary",
        "",
        "| Suite | Benchmark | Mean | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        ci = f"{fmt_ns(float(r['ci_low_ns']))} .. {fmt_ns(float(r['ci_high_ns']))}"
        lines.append(f"| {r['suite']} | {r['benchmark']} | {r['mean_human']} | {ci} |")
    lines.append("")
    out.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion-root", default="target/criterion")
    ap.add_argument("--out-csv", default="encoder_simd_micro.csv")
    ap.add_argument("--out-md", default="encoder_simd_micro.md")
    args = ap.parse_args()

    rows = collect_rows(Path(args.criterion_root))
    if not rows:
        raise SystemExit("no encoder_* criterion results found")
    write_csv(rows, Path(args.out_csv))
    write_md(rows, Path(args.out_md))
    print(f"wrote {len(rows)} rows -> {args.out_csv}, {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
