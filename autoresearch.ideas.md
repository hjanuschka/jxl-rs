# Autoresearch Ideas (deferred)

## Current Status: parity_score=1.958 (commit 5b1b5f8, 59 experiments)

## Per-Image Breakdown (ep=1, gs=8813, softer_rq=5 wins)
- kodim01: 0 pen (-2.82% size, +0.02 dB PSNR)
- kodim08: 6.20 pen (-2.71% size, -0.31 dB PSNR) -- MAIN TARGET
- kodim13: 3.59 pen (-3.54% size, -0.18 dB PSNR)
- zoltan: 0 pen (-1.04% size, +0.36 dB PSNR)
- Webkit: 0 pen (-39.8% size, +1.21 dB PSNR)

## Root Cause
- 26% entropy coding overhead vs libjxl at same quantization level
- Forces softer_rq=5 (coarser) to compensate -- libjxl uses rq=6 (finer)
- rq=5 vs rq=6 size difference is ~26% (exactly the entropy gap!)
- PSNR gap is entirely from rq=5 zeroing coefficients that rq=6 preserves

## Promising Untried Approaches

### 1. Trellis DC Quantization (Medium Effort)
For each DC value, try floor and ceil, pick the one that minimizes modular
prediction residual. Could save 1-3% on DC stream by reducing entropy.
Need to pre-compute left/top DC neighbor predictions during quantization.

### 2. Decode-Based RD Selection (High Effort)
Integrate decoder into encoder to compute actual PSNR for each candidate.
Pick candidate with best parity (size + quality). Currently blocked by
decoder module being test-only. Need to extract a minimal decode path
or make the test decode function available in production builds.

### 3. Better ANS Distribution Normalization (Medium Effort)
libjxl's RebalanceHistogram does proper grid-aware normalization with
entropy-optimal greedy adjustment on allowed_counts grid. Our implementation
has the shift selection framework (committed) but the rebalancing is
approximate. Could save 1-2% on ANS tables.

### 4. LZ77 for Modular Streams (High Effort)
libjxl uses LZ77 for DC/lossless data. Could significantly reduce DC
and HF metadata stream sizes. Complex implementation.

## ANS Shift Optimization (COMMITTED, minimal impact)
Shift selection (0,6,12) with decoder roundtrip is committed. Saves
~0.04% on size but doesn't change parity since we already beat libjxl
on size everywhere.

## Tried and Failed (do NOT retry)
- quant_ac multiplier 1.0/1.15/1.22/1.25/1.30/1.35/1.40
- Dead zone thresholds (0.45-0.64) -- 0.55 too aggressive, 0.45 breaks rounding
- Bias-aware AC quantization
- CfL dist_mul 1e-9, CfL dm_multiplier, CfL TOWARDS_ZERO 1.0/4.0
- Remove softer_rq (rq=6 only): +4.86% size, parity 4.37
- Add extra_softer_rq=base-2: rq=4 wins everywhere, -1.97 dB PSNR
- HybridUint (4,1,2) or competitive config selection
- AQ with 0.79/d, gs from 0.6725/d, gs override to 8813
- Disable entropy merge (2x faster but 0.01 worse parity)
- x_qm_scale 3->2
- All 1..32 cluster counts (same as 14 selected)
- 48/64 clusters (limited by number of contexts)
- extra_dc_precision=2 (zoltan +0.43% size, net worse)
- Competitive ep=1/2 (ep=1 always wins pure-size selection)
- Content-adaptive ep (is_flat_graphic heuristic, zoltan too big at ep=2)
- Per-candidate ep (softer=ep2, others=ep1): mixed ep hurts, parity 2.97
- ANS shift 12 vs 13 (identical output)
- RD selection with lambda*SSE (SSE too small in opsin space)
- Lower greedy clustering threshold (24 vs 48, no effect)
- More HF metadata predictor candidates (no effect)
- Disable gaborish: -4.56 dB PSNR, terrible
- 2% quality budget selection: rq=5 vs rq=6 gap is 26%, way over budget
