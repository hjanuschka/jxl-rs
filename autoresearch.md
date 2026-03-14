# Autoresearch: encoder PSNR parity with libjxl (all images >= 0 dB gap)

## Objective
Achieve PSNR >= libjxl on ALL images in the corpus.
Size is secondary -- we accept larger files if needed to reach PSNR parity.
Target: parity_score = 0 (worst PSNR gap across all images >= 0 dB).

Settings: distance=1.0, jxl-rs effort=7, libjxl effort=3.

We are explicitly allowed to inspect libjxl C++ reference code under `/tmp/libjxl`
(read-only) to copy algorithmic behavior and constants where parity requires it.

## Metrics
- **Primary**: `parity_score` (dB, lower is better, target = 0)
  - Defined as: `max(0, -min_psnr_gap_db)` across all corpus images
  - where `psnr_gap_db = our_psnr - libjxl_psnr` (positive = we're better)
  - A score of 0 means ALL images have PSNR >= libjxl
- **Secondary**:
  - `mean_size_pct` (average file size vs libjxl, negative = smaller)
  - `mean_psnr_gap_db` (average PSNR difference)
  - `max_size_pct` (worst file size vs libjxl)
  - `min_psnr_gap_db` (worst PSNR gap -- this IS the primary metric negated)
  - `num_psnr_negative` (count of images with PSNR < libjxl)
  - `total_encode_s` (jxl-rs wall-clock across corpus)

## How to Run
`./autoresearch.sh`

## Corpus
Full sampler: all `*_source.jpg` files in the images directory (19 images).
Includes Kodak set, dice, zoltan, Webkit-logo-P3.

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
