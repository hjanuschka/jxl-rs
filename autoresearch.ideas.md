# Autoresearch Ideas (deferred)

## Current status
- Best parity_score: **1.595** (790 experiments)
- Improvement: 2.380 -> 1.595 (-33.0%)
- Best configuration:
  - `TOWARDS_ZERO = 2.965` (CfL shrink)
  - Merge multipliers: rect=4.0, 16x16=5.5, 32x32=7.0
  - `b_dm_multiplier = 0.94`
  - `x_dm_multiplier *= 0.97` (0.776 effective)
  - `epf_iters = 2`
  - Custom EPF weights: Y=6.0, X=5.5, B=2.0
  - `quant_lf = base + 1` for >=7000 blocks, +2 for <7000 blocks
  - `extra_dc_precision = 1`

## Remaining penalty
- kodim08: ~5.5 pen (-0.27 dB: B=-0.414, R=-0.198, G=-0.036)
- kodim13: ~2.5 pen (~-0.12 dB: B=-0.220, R=-0.054, G=+0.131)
- All penalty from B-channel PSNR on natural photos

## Recently tried and FAILED
- Custom LfQuantFactors (B=288): Webkit catastrophic -1.46 dB (F16 precision + small image)
- B-channel AC rounding bias (0.1): Keeps too many tiny coefficients, kills entropy
- Custom EPF sharp LUT: 16-byte header overhead offsets any PSNR benefit
- EPF sharpness 3/5: Too much/little smoothing vs default 4
- Per-channel DC predictor selection: All channels prefer same predictor

## Exhaustively confirmed DO NOT RETRY
### EPF weights (15+ experiments)
- Y: 6.0 (3-40 tested, plateau at 3-10)
- X: 5.5 (3.5-6.0 tested, 5.5 marginal best)
- B: 2.0 (1.0-3.5 tested, 2.0 optimal)
- zeroflush: no effect on output
- sigma quant_mul: 0.50/0.55 worse
- Custom sharp LUT: 16-byte overhead kills gains
### Dequant multipliers (8+ experiments)
- b_dm: 0.94 (0.91-0.95 tested, stable)
- x_dm*0.97 optimal (0.95 overshoots, 0.975-1.0 worse)
- Freq-dependent B: no effect (integer steps too coarse)
### Quantization (15+ experiments)
- Adaptive quant_lf: +2 for <7000 blocks, +1 for >=7000
- ep=2: always overshoots even with harder rq
- Dead zones: always worse
- AdjustQuantBias/rounding bias: always worse
- Custom LfQuantFactors: F16 precision kills small images
### CfL (8+ experiments)
- shrink=2.965 stable across all configs
- B reg 0.3: no effect
- Separate B/X shrink: worse
### Other (10+ experiments)
- Gab weight changes: catastrophic
- Custom coeff orders: no gain
- HybridUint configs: (4,2,0) optimal
- ANS shift optimization: no gain
- Per-channel DC predictors: no gain

## Remaining paths (from most to least promising)
1. **Trellis quantization for B-channel** -- jointly optimize small groups of B coefficients
   to minimize bitrate+distortion instead of independent rounding. Complex but targets root cause.
2. **Adaptive B dequant per-block** -- use different b_dm for different block content types.
   Blocks with high B energy need less aggressive multiplier (keep detail).
   Blocks with low B energy benefit from more aggressive (save bytes).
3. **EPF iteration count per-block** -- 3 iterations for chroma-heavy blocks, 2 for others.
   Requires epf_iters=3 in header (more smoothing overall).
4. **CfL factor optimization with RD cost** -- instead of L1-optimal CfL factors, 
   find factors that minimize joint rate-distortion (fewer nonzero B residuals).
5. **ANS RebalanceHistogram** -- marginal entropy improvement in distribution normalization.
