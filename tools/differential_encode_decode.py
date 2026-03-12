#!/usr/bin/env python3
"""Basic differential harness: jxl-rs encode -> djxl decode smoke test.

Usage:
  python3 tools/differential_encode_decode.py image1.ppm image2.ppm
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="Input RGB images (PPM/PNG)")
    ap.add_argument("--jxle", default="./target/release/jxle")
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--effort", type=int, default=7)
    ap.add_argument("--progressive", action="store_true")
    args = ap.parse_args()

    ok = 0
    fail = 0

    for img in args.images:
        in_path = Path(img)
        with tempfile.TemporaryDirectory() as td:
            jxl = Path(td) / (in_path.stem + ".jxl")
            png = Path(td) / (in_path.stem + ".png")

            enc_cmd = [
                args.jxle,
                str(in_path),
                "-o",
                str(jxl),
                "-d",
                str(args.distance),
                "--effort",
                str(args.effort),
            ]
            if args.progressive:
                enc_cmd.append("--progressive")

            enc = run(enc_cmd)
            if enc.returncode != 0:
                print(f"FAIL encode {in_path}: {enc.stderr.strip()}")
                fail += 1
                continue

            dec = run(["djxl", str(jxl), str(png)])
            if dec.returncode != 0:
                print(f"FAIL decode {in_path}: {dec.stderr.strip()}")
                fail += 1
                continue

            print(f"OK {in_path} -> {jxl.stat().st_size} bytes")
            ok += 1

    print(f"summary: ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
