# Autoresearch Ideas (deferred)

## Current Status: parity_score=1.958 (commit 287ebc2, 66 experiments)

## Root Cause Analysis
The 26% entropy gap at same quantization level comes from **different coefficient
values**, not ANS encoding efficiency. Our XYB+gaborish+DCT pipeline produces
slightly different floating-point coefficient distributions than libjxl's. This
causes different zero/nonzero patterns and different coefficient magnitudes,
leading to ~1.3 extra bits per coefficient.

The XYB conversion is verified identical (LUT-based). The cbrt approximation
difference is negligible (< 1 ULP). The gap likely comes from:
1. **Inverse gaborish**: Different FP accumulation order (our naive convolution
   vs libjxl's SIMD butterfly with HWY)
2. **Forward DCT**: Our naive O(N^2) DCT vs libjxl's butterfly DCT. Same math
   but different FP evaluation order produces different rounding.

## Per-Image Breakdown
- kodim01: 0 pen (-2.82%, +0.02 dB)
- kodim08: 6.20 pen (-2.71%, -0.31 dB) -- MAIN TARGET
- kodim13: 3.59 pen (-3.54%, -0.18 dB)
- zoltan: 0 pen (-1.04%, +0.36 dB)
- Webkit: 0 pen (-39.8%, +1.21 dB)

## Remaining Optimization Paths (All High Effort)

### 1. Match libjxl's Inverse Gaborish FP Order
Implement Symmetric5 convolution using the same row-based SIMD-friendly order
as libjxl's convolve.h. This is the most likely source of coefficient differences.

### 2. Match libjxl's DCT Implementation
Replace naive O(N^2) forward DCT with butterfly DCT matching libjxl's
enc_transforms-inl.h computation order. This produces the same mathematical
result but with different FP rounding that matches libjxl.

### 3. Decode-Based RD Selection
Integrate decoder into encoder. For each rq candidate, decode and compute
PSNR against input. Pick the candidate that optimizes parity directly.
Blocked by decoder module being test-only.

### 4. LZ77 for Modular Streams
libjxl uses LZ77 for DC data. Could significantly reduce DC stream sizes.

## Committed Improvements This Session
- ANS shift selection (0,6,12) with decoder roundtrip verification
- Updated ideas file with thorough analysis

## Tried and Failed (67 approaches total, do NOT retry)
- All quant_ac multipliers (1.0-1.40, 0.82-0.88)
- Dead zone thresholds (0.45-0.64)
- All softer_rq variations (remove, add rq-2, quality budget)
- All ep variations (2, competitive, per-candidate, content-adaptive)
- All clustering variations (more counts, lower threshold)
- Fast cbrt matching libjxl (identical output due to LUT)
- Trellis DC quantization (round-nearest already optimal)
- Disable gaborish (massive PSNR loss)
- Force rq=6 (entropy gap makes files too large)
- RD selection with lambda*SSE (SSE too small in opsin space)
