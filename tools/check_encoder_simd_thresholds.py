#!/usr/bin/env python3
"""Informational SIMD benchmark threshold checker.

Reads encoder SIMD CSV summary and reports whether configured speedup targets are met.
Always exits 0 unless CSV is unreadable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_mean(rows: list[dict[str, str]], suite: str, bench_prefix: str) -> float | None:
    for r in rows:
        if r.get("suite") == suite and r.get("benchmark", "").startswith(bench_prefix):
            return float(r["mean_ns"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="encoder_simd_micro.csv")
    ap.add_argument("--xyb-target", type=float, default=1.5)
    ap.add_argument("--transform-target", type=float, default=1.8)
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))

    scalar = find_mean(rows, "encoder_xyb", "srgb_u8_to_xyb_scalar")
    assisted = find_mean(rows, "encoder_xyb", "srgb_u8_to_xyb_assisted")

    print("SIMD threshold check (informational)")
    if scalar and assisted:
        speedup = scalar / assisted if assisted > 0 else 0.0
        status = "PASS" if speedup >= args.xyb_target else "NOT MET"
        print(
            f"- XYB speedup: {speedup:.2f}x (target {args.xyb_target:.2f}x) -> {status}"
        )
    else:
        print("- XYB speedup: insufficient benchmark rows")

    dct_scalar = find_mean(rows, "encoder_forward_dct_reference", "dct2d_32_scalar")
    dct_simd = find_mean(rows, "encoder_forward_dct_reference", "dct2d_32_simd")
    if dct_scalar and dct_simd:
        speedup = dct_scalar / dct_simd if dct_simd > 0 else 0.0
        status = "PASS" if speedup >= args.transform_target else "NOT MET"
        print(
            f"- Transform speedup (DCT32): {speedup:.2f}x (target {args.transform_target:.2f}x) -> {status}"
        )
    else:
        print(
            f"- Transform speedup target {args.transform_target:.2f}x: pending benchmark rows (informational)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
