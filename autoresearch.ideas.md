# Autoresearch Ideas (deferred)

## Current Status: parity_score=2.46 (commit 354169f)

## Per-Image Breakdown (ep=1, gs=5111)
- Webkit: 0 penalty (SOLVED via extra_dc_precision=1)
- kodim08: 6.73 pen (-0.32 dB PSNR, +0.4% size) -- MAIN TARGET
- kodim13: 3.60 pen (-0.18 dB PSNR, -0.7% size)
- kodim01: 1.35 pen (+1.3% size, near-zero PSNR gap)
- zoltan: 0.64 pen (+0.6% size)

## Root Cause of Remaining Gap
- gs=5111 (from 0.39/d) gives coarser AC quantization than libjxl's gs=8813 (from 0.79/d)
- Our AC step is ~2x coarser: rq=5 * 5111/65536 = 0.39 vs libjxl rq=6 * 8813/65536 = 0.81
- But gs=8813 makes files 28% larger (pure size competition can't handle it)
- Need RD-aware competition or better AC entropy coding to close the gap

## Promising Ideas (untried)
- **RD-aware candidate selection**: use size + lambda*distortion instead of pure size.
  Could let us use gs=8813 candidates that are larger but much better PSNR.
  Lambda ~20 matches the parity_score formula weighting.
- **Custom block context map**: libjxl computes per-image block context maps that
  improve AC entropy coding efficiency. Currently we use all_default=true.
- **CfL HF optimization**: optimize chroma-from-luma factors for HF AC coefficients
  per block to reduce chroma residual entropy.
- **Per-channel ANS for HF metadata**: 4 channels with different distributions,
  splitting could improve entropy coding.
- **Smarter AC coefficient ordering**: try natural order vs current zigzag for
  better zero-run encoding.

## Tried and Failed (do NOT retry)
- quant_ac multiplier changes (1.15, 1.30): no effect (only affects adaptive candidate)
- x_qm_scale 3->2: regression, photo PSNR worsened
- gs=8813 + ep=0: massive regression (Webkit -4.64 dB)
- gs=8813 + ep=1: regression (Webkit +9.6% size)
- Hybrid gs (5111 flat, 8813 photo): 28% larger files
- 0.79/d for AQ quant_ac: regression
- EPF iters 2->1: slight regression
- Competitive ep=0/1 per candidate: ep=0 always wins pure-size
- Higher rq candidates: never win size competition
- Dead-zone removal: no effect
- Disabling gaborish: massive regression
- extra_dc_precision=3: too much size overhead
- Removing softer_rq: massive regression
- Competitive HybridUint configs: (4,2,0) already optimal
- libjxl polynomial sRGB EOTF: no difference for 8-bit
