# Autoresearch Ideas (deferred)

## Current Status: parity_score=1.96 (commit 85b91bd, 43 experiments)

## Per-Image Breakdown (ep=1, gs=5111)
- Webkit: 0 penalty (SOLVED)
- kodim08: ~6.0 pen (-0.30 dB PSNR, -1.1% size) -- MAIN TARGET
- kodim13: ~3.4 pen (-0.17 dB PSNR, -2.1% size)
- kodim01: 0 pen (smaller + better PSNR)
- zoltan: 0 pen (smaller)

## Root Cause of Remaining Gap (PSNR only)
- 26% entropy coding overhead vs libjxl at same quantization level
- This forces us to use coarser quantization (rq=5 at gs=5111 vs rq=6 at gs=8813)
- Our AC step ~2x coarser: scale=0.390 vs libjxl 0.807
- libjxl can afford finer quant because its ANS coding is more efficient

## Key Remaining Optimization: ANS Shift Selection with RebalanceHistogram
libjxl's enc_ans.cc tries 7 different shift values (0,2,4,6,8,10,12) for each
ANS histogram table. Lower shifts reduce the number of extra bits per frequency
in the header, at the cost of slightly less precise ANS coding. The frequency
table must be "rebalanced" at each shift to maintain sum=4096 while staying on
the precision grid. libjxl's RebalanceHistogram uses entropy-optimal greedy
adjustment with allowed_counts tables.

Implementation effort: significant (need frequency rebalancing + full pipeline
integration where rebalanced distributions are used for both header AND stream
encoding). Estimated impact: 3-8% size reduction on ANS tables, enabling ep=2
which would reduce parity from 1.96 to ~1.7.

## Successfully Applied (keep these patterns)
- **Finer greedy cluster counts**: [1-32] found better context maps, 2.38->2.03
- **HF metadata spatial prediction**: multi-predictor modular encoding, 2.03->1.96
- **Modular predictors 2/4/5**: Top/Select/Gradient for lossless, 869->835 KB
- **Entropy-optimal ANS normalization**: greedy entropy-minimizing freq adjustment

## Tried and Failed (do NOT retry)
- quant_ac multiplier 1.0/1.15/1.22/1.25/1.30/1.35/1.40
- Dead zone thresholds (0.52-0.64)
- Bias-aware AC quantization
- CfL dist_mul 1e-9, CfL dm_multiplier, CfL TOWARDS_ZERO 1.0/4.0
- Remove/add softer_rq/finer_rq candidates
- Fix base_rq to match actual gs, matched-gs candidates
- HybridUint (4,1,2) or competitive config selection
- AQ with 0.79/d, gs from 0.6725/d, gs override to 8813
- Disable entropy merge (2x faster but 0.01 worse parity)
- x_qm_scale 3->2
- All 1..32 cluster counts (same as 14 selected)
- extra_dc_precision=2 (PSNR +0.70 but zoltan +0.45%, net worse)
- Competitive ep=1/2 (ep=1 always wins size)
- Content-adaptive ep (is_flat_graphic heuristic, zoltan too big at ep=2)
- ANS shift 12 vs 13 (identical output, both full precision)
- RD selection with lambda*SSE (SSE too small in opsin space, wrong metric)
- Lower greedy clustering threshold (24 vs 48, no effect)
- More HF metadata predictor candidates (no effect, same winner)
