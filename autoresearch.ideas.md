# Autoresearch Ideas (deferred)

## Current status
- Best parity_score: **1.726066** (714 experiments)
- Best configuration (stable, heavily tuned):
  - `TOWARDS_ZERO = 2.965` (CfL shrink)
  - Merge multipliers: rect=4.0, 16x16=5.5, 32x32=7.0
  - `b_dm_multiplier = 0.93`
  - `epf_iters = 2` (confirmed optimal, better than libjxl's 1)
  - `extra_dc_precision = 1`

## Remaining penalty
- kodim08: ~5.75 pen (-0.29 dB, B:-0.43 R:-0.24 G:-0.04)
- kodim13: ~3.25 pen (-0.16 dB, B:-0.24 R:-0.07 G:+0.14)

## Exhaustively confirmed (DO NOT RETRY)
### Quantization
- Dead zones (0.505, 0.535, libjxl 0.56/0.62 quadrant): always worse
- AdjustQuantBias-aware rounding (0.465/1.429 boundaries): more nonzeros, worse parity
- quant_ac multipliers (1.25-1.35): no effect (rq integers unchanged)
- Finer/coarser uniform rq candidates: finer too large, coarser too lossy
- Finer adaptive map (+1 rq per block): always loses size selection
- Y-roundtrip quantization: regresses (CfL factors fitted to original Y)

### CfL / color
- CfL shrink: 2.965 optimal (tested 2.2-3.2 at 0.001 granularity)
- Separate B vs X CfL shrink: always worse than unified
- CfL dist_mul: no gain at 8e-4 or 1.2e-3
- CfL B base-correlation: 0.98 and 1.02 both strongly regressive
- b_dm_multiplier: 0.93 optimal (0.80-0.97 tested)
- x_dm_multiplier: any deviation from 0.8 worsens parity

### Entropy
- HybridUint configs beyond kFast: no improvement
- ANS shift candidates [0,3,6,9,12]: no gain over [0,6,12]
- Dual-gs candidate system: no effect, only slower

### Loop filter
- EPF: iters=2 optimal (1 and 3 both worse)
- EPF sharpness map: has NO effect on djxl output (possible encoding bug)
- EPF sigma custom: crashed (needs F16 writer)

### Merge / transform
- Merge multipliers saturated at rect=4.0, 16=5.5, 32=7.0
- extra_dc_precision=2: always size overshoot
- Decode-based candidate selection: over-picks large candidates

## Root cause
The chroma PSNR gap is structural: our gs=5111 (from 0.39/d AQ) vs libjxl's
gs=8813 (from 0.79/d) produces different quantization steps. Our finer steps
create more nonzeros (smaller files) but each coefficient overshoots more
(worse PSNR on R/B channels).

## Only remaining paths (HIGH effort)
1. **Fix EPF sharpness encoding** -- it has zero effect currently, suggesting a
   bug. If fixed, adaptive sharpness could help chroma PSNR on photo blocks.
2. **LZ77 for modular streams** -- could save 1-3% bytes, making ep=2 viable.
3. **Dual-pass encoding with Y-roundtrip + CfL refit** -- quantize Y, compute
   CfL from dequantized Y, then quantize X/B. Requires refitting CfL maps
   per-candidate (chicken-and-egg: CfL depends on Y quant, Y quant depends on
   candidate). High implementation complexity.
4. **Custom dequant weights** -- signal slightly different weight tables to
   decoder. Requires F16 writing support for frame header fields.
