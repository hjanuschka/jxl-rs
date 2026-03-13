# Autoresearch: encoder parity with libjxl (size + PSNR)

## Objective
Reduce jxl-rs encoder parity gaps against libjxl on sampler-style RGB images.
The target is to close both:
- size gap (bytes)
- quality gap (PSNR)
at fixed settings (distance=1.0, jxl-rs effort=7, libjxl effort=3).

We are explicitly allowed to inspect libjxl C++ reference code under `/tmp/libjxl`
(read-only) to copy algorithmic behavior and constants where parity requires it.

## Metrics
- **Primary**: `parity_score` (unitless, lower is better)
  - Defined as corpus mean of: `max(0, size_pct) + 20 * max(0, -psnr_gap_db)`
  - where `size_pct = (ours/lib - 1) * 100`, `psnr_gap_db = ours_psnr - lib_psnr`.
  - This rewards equal-or-better size and equal-or-better PSNR vs libjxl.
- **Secondary**:
  - `mean_size_pct`
  - `mean_psnr_gap_db`
  - `max_size_pct`
  - `min_psnr_gap_db`
  - `total_encode_s` (jxl-rs wall-clock across corpus)

## How to Run
`./autoresearch.sh`

The script prints:
- `METRIC parity_score=<number>`
- `METRIC mean_size_pct=<number>`
- `METRIC mean_psnr_gap_db=<number>`
- `METRIC max_size_pct=<number>`
- `METRIC min_psnr_gap_db=<number>`
- `METRIC total_encode_s=<number>`

## Files in Scope
- `jxl/src/encode/vardct.rs` - VarDCT rate/distortion decisions, AQ, transforms, tokenization.
- `jxl/src/encode/xyb.rs` - XYB conversion behavior affecting RD.
- `jxl/src/encode/options.rs` - effort/distance mapping and gating.
- `jxl/src/encode/entropy/*` - entropy coding behavior affecting size.
- `jxl_transforms/src/dct8.rs` - forward transform behavior and parity-sensitive math.
- `jxl/benches/encoder_simd_micro.rs` - measurement support.
- `tools/*parity*`, `tools/*rd*` - parity instrumentation scripts.
- `/tmp/libjxl/lib/jxl/*.cc`, `*.h` (read-only reference) - libjxl behavior source for parity matching.

## Off Limits
- Decoder bitstream semantics changes unrelated to encoder parity.
- New external dependencies.
- Non-Rust encoder path / FFI encoder implementation.
- Sampler HTML changes unless needed to reflect measured outputs.

## Constraints
- Output must decode in both `djxl` and `jxl-rs`.
- Keep encoder path pure Rust.
- Maintain determinism guardrails (scalar canonical output unless explicitly gated).
- Keep changes measurable on the defined corpus.
- `/tmp/libjxl` is reference-only: inspect freely, but do not modify.

## What's Been Tried
- SIMD foundation landed (dispatch, helpers, tests, fuzz smoke, CI jobs).
- Transform SIMD kernels added for 8x8/16x16/16x8/8x16/32x32 with equivalence tests.
- AQ-related SIMD-assisted loops added (mask mapping/downsample and related helpers).
- CI checks now include scalar-vs-auto codestream equality and scalar-vs-assisted RD regression smoke.
- Current remaining macro gap is parity vs libjxl in size and PSNR at fixed settings, not just microkernel speed.
