#!/usr/bin/env python3
"""Compare jxle codestream outputs for scalar vs auto SIMD modes on a small corpus."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def generate_ppm(path: Path, kind: str, w: int = 64, h: int = 48) -> None:
    data = bytearray()
    for y in range(h):
        for x in range(w):
            if kind == "grad":
                r = (x * 255) // max(1, w - 1)
                g = (y * 255) // max(1, h - 1)
                b = ((x + y) * 255) // max(1, w + h - 2)
            elif kind == "checker":
                c = 255 if ((x // 8 + y // 8) % 2) == 0 else 0
                r = g = b = c
            elif kind == "flat":
                if x < w // 2:
                    r, g, b = 0, 180, 255
                else:
                    r, g, b = 255, 255, 255
            else:
                r = (x * 17 + y * 31) % 256
                g = (x * 57 + y * 13) % 256
                b = (x * 91 + y * 47) % 256
            data.extend((r, g, b))
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jxle", default="target/release/jxle")
    ap.add_argument("--distance", default="1.0")
    ap.add_argument("--effort", default="7")
    args = ap.parse_args()

    jxle = Path(args.jxle)
    if not jxle.exists():
        raise SystemExit(f"jxle not found: {jxle}")

    corpus = ["grad", "checker", "flat", "noise"]

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for name in corpus:
            ppm = tdp / f"{name}.ppm"
            generate_ppm(ppm, name)

            scalar_jxl = tdp / f"{name}_scalar.jxl"
            auto_jxl = tdp / f"{name}_auto.jxl"

            env_scalar = dict(os.environ)
            env_scalar["JXL_ENC_SIMD"] = "scalar"
            run(
                [
                    str(jxle),
                    str(ppm),
                    "-o",
                    str(scalar_jxl),
                    "-d",
                    str(args.distance),
                    "--effort",
                    str(args.effort),
                ],
                env=env_scalar,
            )

            env_auto = dict(os.environ)
            env_auto["JXL_ENC_SIMD"] = "auto"
            run(
                [
                    str(jxle),
                    str(ppm),
                    "-o",
                    str(auto_jxl),
                    "-d",
                    str(args.distance),
                    "--effort",
                    str(args.effort),
                ],
                env=env_auto,
            )

            a = scalar_jxl.read_bytes()
            b = auto_jxl.read_bytes()
            if a != b:
                raise SystemExit(f"codestream mismatch for corpus item: {name}")

    print("scalar vs auto codestream corpus: identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
