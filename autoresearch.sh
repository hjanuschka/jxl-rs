#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Fast precheck
python3 -m py_compile tools/rate_distortion_compare.py >/dev/null

JXLE="${JXLE:-$REPO_DIR/target/release/jxle}"
JXL_DECODER="${JXL_DECODER:-$REPO_DIR/target/release/jxl_cli}"

# Rebuild every run so source edits are reflected in measurements.
cargo build -q --release -p jxl_cli --bin jxle --bin jxl_cli

python3 - << 'PY'
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

repo = Path('.').resolve()
imgdir = Path('/home/chrome/my-host/static-files/jxl-encode/images')

# Small but representative sampler-like RGB subset
corpus = [
    imgdir / 'kodim01_source.jpg',
    imgdir / 'kodim08_source.jpg',
    imgdir / 'kodim13_source.jpg',
    imgdir / 'zoltan_source.jpg',
    imgdir / 'Webkit-logo-P3_source.jpg',
]

jxle = Path(os.environ.get('JXLE', str(repo / 'target/release/jxle')))


def run(cmd, env=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def read_ppm_rgb(path: Path):
    data = path.read_bytes()
    if not data.startswith(b'P6'):
        raise RuntimeError(f'Not PPM P6: {path}')
    i = 2

    def skip(i):
        while i < len(data):
            b = data[i]
            if b == 35:  # '#'
                while i < len(data) and data[i] not in (10, 13):
                    i += 1
            elif b in b' \t\r\n':
                i += 1
            else:
                break
        return i

    def tok(i):
        i = skip(i)
        j = i
        while j < len(data) and data[j] not in b' \t\r\n':
            j += 1
        return data[i:j], j

    w, i = tok(i)
    h, i = tok(i)
    m, i = tok(i)
    w = int(w)
    h = int(h)
    if int(m) != 255:
        raise RuntimeError('maxval must be 255')
    i = skip(i)
    n = w * h * 3
    rgb = data[i:i+n]
    if len(rgb) != n:
        raise RuntimeError('short ppm payload')
    return w, h, rgb


def psnr(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        raise RuntimeError('size mismatch for psnr')
    sse = 0
    for x, y in zip(a, b):
        d = x - y
        sse += d * d
    mse = sse / len(a)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)

size_pcts = []
psnr_gaps = []
penalties = []
encode_total = 0.0

from PIL import Image

with tempfile.TemporaryDirectory(prefix='autoresearch-parity-') as td:
    td = Path(td)
    for src in corpus:
        src_ppm = td / f'{src.stem}.src.ppm'
        src_img = Image.open(src).convert('RGB')
        sw, sh = src_img.size
        src_rgb = src_img.tobytes()
        src_img.save(src_ppm)

        ours = td / f'{src.stem}.ours.jxl'
        lib = td / f'{src.stem}.lib.jxl'
        ours_ppm = td / f'{src.stem}.ours.ppm'
        lib_ppm = td / f'{src.stem}.lib.ppm'

        t0 = time.perf_counter()
        p = run([str(jxle), str(src_ppm), '-o', str(ours), '-d', '1.0', '--effort', '7', '--bare'])
        encode_total += time.perf_counter() - t0
        if p.returncode != 0:
            print('METRIC parity_score=999999')
            raise SystemExit(0)

        p = run(['cjxl', str(src), str(lib), '-d', '1.0', '-e', '3', '--lossless_jpeg=0', '--quiet'])
        if p.returncode != 0:
            print('METRIC parity_score=999999')
            raise SystemExit(0)

        if run(['djxl', str(ours), str(ours_ppm)]).returncode != 0:
            print('METRIC parity_score=999999')
            raise SystemExit(0)
        if run(['djxl', str(lib), str(lib_ppm)]).returncode != 0:
            print('METRIC parity_score=999999')
            raise SystemExit(0)

        ow, oh, o_rgb = read_ppm_rgb(ours_ppm)
        lw, lh, l_rgb = read_ppm_rgb(lib_ppm)
        if (ow, oh) != (sw, sh) or (lw, lh) != (sw, sh):
            print('METRIC parity_score=999999')
            raise SystemExit(0)

        ours_b = ours.stat().st_size
        lib_b = lib.stat().st_size
        size_pct = (ours_b / lib_b - 1.0) * 100.0

        ours_psnr = psnr(src_rgb, o_rgb)
        lib_psnr = psnr(src_rgb, l_rgb)
        psnr_gap = ours_psnr - lib_psnr

        penalty = max(0.0, size_pct) + 20.0 * max(0.0, -psnr_gap)

        size_pcts.append(size_pct)
        psnr_gaps.append(psnr_gap)
        penalties.append(penalty)

mean_size = sum(size_pcts) / len(size_pcts)
mean_gap = sum(psnr_gaps) / len(psnr_gaps)
max_size = max(size_pcts)
min_gap = min(psnr_gaps)
parity_score = sum(penalties) / len(penalties)

print(f'METRIC parity_score={parity_score:.6f}')
print(f'METRIC mean_size_pct={mean_size:.6f}')
print(f'METRIC mean_psnr_gap_db={mean_gap:.6f}')
print(f'METRIC max_size_pct={max_size:.6f}')
print(f'METRIC min_psnr_gap_db={min_gap:.6f}')
print(f'METRIC total_encode_s={encode_total:.6f}')
PY
