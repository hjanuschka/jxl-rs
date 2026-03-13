#!/usr/bin/env python3
"""Corpus-level RD regression check for scalar vs SIMD-assisted encoder mode.

Generates a tiny deterministic RGB corpus, encodes each image twice with jxle:
- JXL_ENC_SIMD=scalar
- JXL_ENC_SIMD=assisted

Decodes with jxl_cli, computes RGB PSNR against source, and checks that assisted mode
does not regress quality/size beyond thresholds.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
from pathlib import Path


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env)


def write_ppm(path: Path, w: int, h: int, rgb: bytes) -> None:
    path.write_bytes(f"P6\n{w} {h}\n255\n".encode("ascii") + rgb)


def read_ppm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        raise ValueError(f"not a binary PPM: {path}")

    i = 2

    def skip_ws_and_comments(pos: int) -> int:
        while pos < len(data):
            b = data[pos]
            if b == 35:  # '#'
                while pos < len(data) and data[pos] not in (10, 13):
                    pos += 1
            elif b in b" \t\r\n":
                pos += 1
            else:
                break
        return pos

    def read_token(pos: int) -> tuple[bytes, int]:
        pos = skip_ws_and_comments(pos)
        start = pos
        while pos < len(data) and data[pos] not in b" \t\r\n":
            pos += 1
        return data[start:pos], pos

    tok_w, i = read_token(i)
    tok_h, i = read_token(i)
    tok_m, i = read_token(i)
    w = int(tok_w)
    h = int(tok_h)
    maxv = int(tok_m)
    if maxv != 255:
        raise ValueError(f"unsupported maxval {maxv} in {path}")
    i = skip_ws_and_comments(i)
    need = w * h * 3
    rgb = data[i : i + need]
    if len(rgb) != need:
        raise ValueError(f"short pixel payload in {path}")
    return w, h, rgb


def psnr_rgb(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        raise ValueError("mismatched rgb byte lengths")
    n = len(a)
    if n == 0:
        return 99.0
    sse = 0
    for x, y in zip(a, b):
        d = x - y
        sse += d * d
    mse = sse / n
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def make_corpus(tmp: Path) -> list[Path]:
    out: list[Path] = []
    w, h = 96, 72

    # gradient
    g = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            g[i] = (x * 255) // (w - 1)
            g[i + 1] = (y * 255) // (h - 1)
            g[i + 2] = ((x + y) * 255) // (w + h - 2)
    p = tmp / "rd_grad.ppm"
    write_ppm(p, w, h, bytes(g))
    out.append(p)

    # checker
    c = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            v = 255 if ((x // 8 + y // 8) % 2) == 0 else 0
            c[i] = v
            c[i + 1] = v
            c[i + 2] = v
    p = tmp / "rd_checker.ppm"
    write_ppm(p, w, h, bytes(c))
    out.append(p)

    # flat/logo-like
    f = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            if x < w // 2:
                rgb = (0, 180, 255)
            else:
                rgb = (255, 255, 255)
            if h // 4 <= y < 3 * h // 4 and w // 4 <= x < 3 * w // 4:
                rgb = (255, 90, 0)
            f[i], f[i + 1], f[i + 2] = rgb
    p = tmp / "rd_flat.ppm"
    write_ppm(p, w, h, bytes(f))
    out.append(p)

    # deterministic pseudo-noise
    n = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            n[i] = (x * 17 + y * 31) & 255
            n[i + 1] = (x * 57 + y * 13) & 255
            n[i + 2] = (x * 91 + y * 47) & 255
    p = tmp / "rd_noise.ppm"
    write_ppm(p, w, h, bytes(n))
    out.append(p)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jxle", default="target/release/jxle")
    ap.add_argument("--jxl-decoder", default="target/release/jxl_cli")
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--effort", type=int, default=7)
    ap.add_argument("--max-size-regress-pct", type=float, default=2.0)
    ap.add_argument("--max-psnr-drop-db", type=float, default=0.05)
    args = ap.parse_args()

    jxle = Path(args.jxle)
    dec = Path(args.jxl_decoder)
    if not jxle.exists():
        raise SystemExit(f"missing jxle binary: {jxle}")
    if not dec.exists():
        raise SystemExit(f"missing jxl decoder binary: {dec}")

    failed = False
    print(
        "image,scalar_bytes,assisted_bytes,size_regress_pct,scalar_psnr,assisted_psnr,psnr_drop_db"
    )

    with tempfile.TemporaryDirectory(prefix="simd-rd-") as td:
        tmp = Path(td)
        corpus = make_corpus(tmp)

        for img in corpus:
            src_w, src_h, src_rgb = read_ppm(img)
            scalar_jxl = tmp / f"{img.stem}.scalar.jxl"
            assisted_jxl = tmp / f"{img.stem}.assisted.jxl"
            scalar_ppm = tmp / f"{img.stem}.scalar.ppm"
            assisted_ppm = tmp / f"{img.stem}.assisted.ppm"

            env_scalar = dict(os.environ)
            env_scalar["JXL_ENC_SIMD"] = "scalar"
            env_assisted = dict(os.environ)
            env_assisted["JXL_ENC_SIMD"] = "assisted"

            cmd = [
                str(jxle),
                str(img),
                "-o",
                str(scalar_jxl),
                "-d",
                str(args.distance),
                "--effort",
                str(args.effort),
            ]
            if run(cmd, env=env_scalar).returncode != 0:
                print(f"{img.name},ERR,ERR,ERR,ERR,ERR,ERR")
                failed = True
                continue

            cmd[3] = str(assisted_jxl)
            if run(cmd, env=env_assisted).returncode != 0:
                print(f"{img.name},ERR,ERR,ERR,ERR,ERR,ERR")
                failed = True
                continue

            if run([str(dec), str(scalar_jxl), str(scalar_ppm)]).returncode != 0:
                print(f"{img.name},ERR,ERR,ERR,ERR,ERR,ERR")
                failed = True
                continue
            if run([str(dec), str(assisted_jxl), str(assisted_ppm)]).returncode != 0:
                print(f"{img.name},ERR,ERR,ERR,ERR,ERR,ERR")
                failed = True
                continue

            w1, h1, rgb_scalar = read_ppm(scalar_ppm)
            w2, h2, rgb_assisted = read_ppm(assisted_ppm)
            if (w1, h1) != (src_w, src_h) or (w2, h2) != (src_w, src_h):
                print(f"{img.name},ERR,ERR,ERR,ERR,ERR,ERR")
                failed = True
                continue

            scalar_bytes = scalar_jxl.stat().st_size
            assisted_bytes = assisted_jxl.stat().st_size
            size_regress_pct = (
                (assisted_bytes / scalar_bytes - 1.0) * 100.0 if scalar_bytes else 0.0
            )

            scalar_psnr = psnr_rgb(src_rgb, rgb_scalar)
            assisted_psnr = psnr_rgb(src_rgb, rgb_assisted)
            psnr_drop = scalar_psnr - assisted_psnr

            print(
                f"{img.name},{scalar_bytes},{assisted_bytes},{size_regress_pct:+.3f},{scalar_psnr:.3f},{assisted_psnr:.3f},{psnr_drop:+.4f}"
            )

            if size_regress_pct > args.max_size_regress_pct or psnr_drop > args.max_psnr_drop_db:
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
