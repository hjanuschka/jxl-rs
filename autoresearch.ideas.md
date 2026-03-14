# Autoresearch Ideas (deferred)

## Current Status: parity_score=1.958 (94 experiments)

## Root Cause Analysis (UPDATED)

The -0.31 dB PSNR gap on kodim08 (-0.44 dB on B channel, -0.26 dB R, -0.07 dB G) comes
from the **global_scale / dequantization step mismatch**:

- Our encoder uses gs=5111 (from AQ map), libjxl uses gs=8813 (from 0.79/d)
- At gs=5111, rq=6: each quantized unit dequantizes to 2.14*dw (coarse reconstruction)
- At gs=8813, rq=6: each quantized unit dequantizes to 1.24*dw (fine reconstruction)
- Our FINER quantization step (0.468) produces MORE nonzero coefficients
- But each nonzero coefficient OVERSHOOTS the true value by more
- libjxl's COARSER quantization (0.807) produces fewer nonzeros, each closer to truth

Attempts to change gs failed because rq snaps to integers, causing discontinuities.
The AQ map's gs=5111 was chosen to give the right per-block variation range.

## Disproved Hypotheses (do NOT retry)
- FP accumulation order (gaborish, DCT): identical output (quant absorbs ULPs)
- Fast cbrt: identical output (LUT dominates)
- CfL towards_zero shrinkage: 2.6 is near-optimal
- CfL dist_mul: 1e-3 is better than 1e-6 (overcorrects)
- AdjustQuantBias-aware quantization: no coefficients in bias-sensitive range at gs=5111
- Position-dependent dead zones: no effect at our fine quantization
- extra_dc_precision=0: catastrophic -1.87 dB (DC precision is critical for us)
- extra_dc_precision=2: +0.70 dB PSNR but one image exceeds libjxl size
- Decode-based selection: works but budget too coarse (5% no change, 10% overshoots)

## Remaining Paths (All High Effort)

### 1. Dual-gs Candidate System
Encode candidates at BOTH gs=5111 and gs=8813, each with appropriate rq values.
Use decode-based PSNR to pick the best per-image. High implementation complexity
(need to restructure candidate loop to vary gs).

### 2. LZ77 for Modular Streams
libjxl uses LZ77 for DC data encoding. Could reduce DC stream overhead enough
to offset the PSNR gap by allowing extra_dc_precision=2.

### 3. ANS RebalanceHistogram
Implement libjxl's proper histogram grid fitting (RebalanceHistogram).
Our current entropy-optimal normalization doesn't match libjxl's grid structure.

## Per-Image Breakdown
- kodim01: 0 pen (-2.82%, +0.02 dB)
- kodim08: 6.20 pen (-2.71%, -0.31 dB) -- B channel: -0.44 dB!
- kodim13: 3.59 pen (-3.54%, -0.18 dB)
- zoltan: 0 pen (-1.04%, +0.36 dB)
- Webkit: 0 pen (-39.8%, +1.21 dB)

## Key Infrastructure Added
- `decode_codestream_to_f32_rgb()` - decode bare codestream in release builds
- `compute_psnr_from_decoded()` - compute PSNR from decoded f32 data
- Both available in the encoder for future RD-based selection

## Tried and Failed (94 experiments, do NOT retry)
- All quant_ac multipliers (0.82-1.40)
- Dead zones (0.505, 0.52, 0.58, position-dependent quadrant)
- All softer_rq variations (remove, rq-2, quality budget, mixed maps)
- All ep variations (2, competitive, per-candidate, content-adaptive)
- All clustering variations (more counts, lower threshold)
- Fast cbrt, gaborish FP order, bias-aware quantization
- CfL tuning (towards_zero 0.5-3.5, dist_mul 1e-3 to 1e-6)
- extra_dc_precision (0, 1, 2, candidate-based)
- Global scale changes (q=0.45, 0.55, 0.79 for gs; all cause rq discontinuities)
- Merge multiplier tuning (2.0-3.5 range)
- Decode-based selection (5%, 7%, 10% budgets)
- Force DCT8-only (merges are critical, save ~25%)
