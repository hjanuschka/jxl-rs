#!/usr/bin/env python3
"""Compute simple rate/distortion comparison for jxl-rs vs libjxl.

For each input image:
- encode with jxl-rs (jxle)
- encode with libjxl (cjxl)
- decode both with djxl
- compute RGB PSNR vs source

Usage:
  python3 tools/rate_distortion_compare.py image1.ppm image2.ppm
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
import numpy as np


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="Input RGB images")
    ap.add_argument("--jxle", default="./target/release/jxle")
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--effort", type=int, default=7)
    args = ap.parse_args()

    print("image,jxlrs_bytes,libjxl_bytes,size_pct,jxlrs_psnr,libjxl_psnr,psnr_gap")

    for img in args.images:
        in_path = Path(img)
        src = np.array(Image.open(in_path).convert("RGB"), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as td:
            jxlrs_jxl = Path(td) / "ours.jxl"
            lib_jxl = Path(td) / "lib.jxl"
            jxlrs_png = Path(td) / "ours.png"
            lib_png = Path(td) / "lib.png"

            enc_ours = run(
                [
                    args.jxle,
                    str(in_path),
                    "-o",
                    str(jxlrs_jxl),
                    "-d",
                    str(args.distance),
                    "--effort",
                    str(args.effort),
                ]
            )
            if enc_ours.returncode != 0:
                print(f"{in_path.name},ERR,ERR,ERR,ERR,ERR,ERR")
                continue

            enc_lib = run(
                [
                    "cjxl",
                    str(in_path),
                    str(lib_jxl),
                    "-d",
                    str(args.distance),
                    "-e",
                    "3",
                ]
            )
            if enc_lib.returncode != 0:
                print(f"{in_path.name},ERR,ERR,ERR,ERR,ERR,ERR")
                continue

            if run(["djxl", str(jxlrs_jxl), str(jxlrs_png)]).returncode != 0:
                print(f"{in_path.name},ERR,ERR,ERR,ERR,ERR,ERR")
                continue
            if run(["djxl", str(lib_jxl), str(lib_png)]).returncode != 0:
                print(f"{in_path.name},ERR,ERR,ERR,ERR,ERR,ERR")
                continue

            ours = np.array(Image.open(jxlrs_png).convert("RGB"), dtype=np.uint8)
            lib = np.array(Image.open(lib_png).convert("RGB"), dtype=np.uint8)

            ours_b = jxlrs_jxl.stat().st_size
            lib_b = lib_jxl.stat().st_size
            size_pct = (ours_b / lib_b - 1.0) * 100.0
            ours_psnr = psnr(src, ours)
            lib_psnr = psnr(src, lib)
            gap = ours_psnr - lib_psnr

            print(
                f"{in_path.name},{ours_b},{lib_b},{size_pct:+.1f},{ours_psnr:.2f},{lib_psnr:.2f},{gap:+.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
