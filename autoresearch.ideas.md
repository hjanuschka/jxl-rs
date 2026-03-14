# Autoresearch Ideas (deferred)

## Current Status: parity_score=1.96 (commit 85b91bd, 42 experiments)

## Per-Image Breakdown (ep=1, gs=5111, with greedy clustering + HF metadata prediction)
- Webkit: 0 penalty (SOLVED)
- kodim08: ~6.2 pen (-0.31 dB PSNR) -- MAIN TARGET (pure PSNR gap)
- kodim13: ~3.6 pen (-0.18 dB PSNR) -- second target
- kodim01: 0 pen (smaller + better PSNR now)
- zoltan: 0 pen (smaller now)

## Root Cause of Remaining Gap (PSNR only, size is now -10% vs libjxl)
- gs=5111 gives rq=5 (scale=0.390), libjxl gs=8813 gives rq=6 (scale=0.807)
- Our quantization step is ~2x coarser, losing fine detail
- Pure-size competition always picks coarsest rq
- Can't increase gs without AQ map recalibration
- All AQ/gs/rq parameter tuning has been exhausted (see Tried and Failed)

## Promising Ideas (untried)
- **LZ77 for modular streams**: libjxl uses LZ77 for DC and lossless data.
  Could significantly improve lossless compression (835 KB vs 484 KB gap).
  Complex to implement but high potential.
- **RD-aware candidate selection with decode-based PSNR**: encode each candidate,
  decode it, compute actual PSNR, pick best parity. Very expensive but would
  directly optimize the target metric. Could be effort-gated.
- **Custom block context map with AQ-based splitting**: use per-image AQ stats
  to split AC contexts by quantization level. Already implemented but disabled.
- **Per-channel predictor selection for HF metadata**: use multi-leaf tree for
  CfL-x/CfL-b/transform/EPF channels with independent predictors.
- **ANS shift selection**: libjxl tries 7 different shift values (0,2,4,6,8,10,12)
  for each ANS histogram table, picking cheapest total cost. We hardcode shift=13.
  Needs RebalanceHistogram implementation to adjust frequencies at lower shifts.
  This is likely the biggest remaining entropy coding gap (~26% overhead vs libjxl).
  Implementation effort: moderate (need frequency rebalancing + cost estimation).

## Recently Successful (keep these patterns)
- **Finer greedy cluster counts**: [1-32] found better context maps, parity 2.38->2.03
- **HF metadata spatial prediction**: multi-predictor modular encoding for CfL/EPF maps, parity 2.03->1.96
- **Modular predictors 2/4/5**: Top/Select/Gradient for lossless, 869->835 KB

## Tried and Failed (do NOT retry)
- quant_ac multiplier 1.0/1.15/1.22/1.25/1.30/1.35/1.40: no effect or massive regression
- Dead zone threshold 0.52/0.56/0.58/0.64: worse PSNR or massive regression
- Bias-aware AC quantization (AdjustQuantBias): no change at fine scale
- CfL dist_mul 1e-9 (both scalar+SIMD paths): no change, integer discretization dominates
- CfL weights with dm_multiplier: no change (x_dm=0.8 negligible, b_dm=1.0)
- CfL TOWARDS_ZERO 1.0 or 4.0: both worse than 2.6
- Remove softer_rq candidate: massive size regression
- Fix base_rq to match actual gs: PSNR improves but +8.9% size kills parity
- Add matched-gs rq candidates: never win size competition
- Add finer_rq=base+1 candidate: never wins size competition  
- HybridUint config (4,1,2) or competitive config selection: slight regression
- AQ with 0.79/d: AQ modulation miscalibrated, -1.46 dB
- gs from 0.6725/d: AQ/gs mismatch, -1.49 dB
- Override gs to 8813: +21.5% size, AQ calibrated for wrong gs
- Disable entropy merge: barely worse parity (2.39 vs 2.38), 2x faster
- x_qm_scale 3->2: regression
- All 1..32 cluster counts: same as 14 selected counts, just 20% slower
