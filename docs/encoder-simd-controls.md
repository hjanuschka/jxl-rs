# Encoder SIMD user controls

This document describes current runtime controls for encoder SIMD experiments.

## Environment variables

- `JXL_ENC_SIMD=auto`
  - Default behavior.
  - Current encode path keeps scalar as canonical output path unless an explicit assisted experiment mode is selected.

- `JXL_ENC_SIMD=scalar`
  - Forces scalar reference mode.
  - Used by benchmarks/CI to pin deterministic reference behavior.

- `JXL_ENC_SIMD=assisted`
  - Enables SIMD-assisted XYB preprocessing path where available.
  - Intended for measurement and parity experiments.

## Benchmark commands

- Encoder SIMD microbench (default behavior):
  - `cargo bench -p jxl --bench encoder_simd_micro --features encoder`

- Scalar-pinned benchmark run:
  - `JXL_ENC_SIMD=scalar cargo bench -p jxl --bench encoder_simd_micro --features encoder`

- XYB-focused run:
  - `cargo bench -p jxl --bench encoder_simd_micro --features encoder -- encoder_xyb`

## CI helpers

- `tools/compare_encoder_scalar_auto_codestreams.py`
  - Verifies codestream equality for scalar vs auto on generated corpus.

- `tools/check_encoder_simd_thresholds.py`
  - Reports benchmark target status as informational output.
