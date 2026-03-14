# Autoresearch Ideas (deferred)

## Current status
- Best parity_score: **1.600** (770 experiments)
- Improvement: 2.380 -> 1.600 (-32.8%)
- Best configuration:
  - `TOWARDS_ZERO = 2.965` (CfL shrink)
  - Merge multipliers: rect=4.0, 16x16=5.5, 32x32=7.0
  - `b_dm_multiplier = 0.94`
  - `x_dm_multiplier *= 0.97` (0.776 effective)
  - `epf_iters = 2`
  - Custom EPF weights: Y=5.0, X=5.0, B=2.0
  - `quant_lf = base + 1` (finer DC quantization)
  - `extra_dc_precision = 1`

## Key metrics
- mean_size: -7.9%, max_size: -0.03% (VERY tight!)
- mean_psnr_gap: +0.34 dB, worst: -0.27 dB (kodim08)

## Remaining penalty
- kodim08: ~5.5 pen (-0.27 dB: B=-0.414, R=-0.198, G=-0.036)
- kodim13: ~2.5 pen (~-0.10 dB: B=-0.220, R=-0.054, G=+0.131)

## Exhaustively confirmed DO NOT RETRY
### EPF weights (15 experiments)
- Y: 5.0 (3-40 tested, plateau at 3-10)
- X: 5.0 (3.5, 4.0, 4.5, 6.0 all worse)
- B: 2.0 (1.0-2.5 tested, 2.0 optimal)
- zeroflush: no effect
- sigma quant_mul: 0.50/0.55 worse
### Dequant multipliers (8 experiments)
- b_dm: 0.94 (0.91-0.95, stable across configs)
- x_dm*0.97 optimal (0.95 overshoots, 0.975/0.98/1.0 worse)
- Freq-dependent B: no effect (steps too coarse)
### Quantization (12 experiments)
- quant_lf+1 optimal (+2 overshoots one image)
- ep=2: always overshoots even with harder rq
- Dead zones: always worse
- AdjustQuantBias rounding: always worse
### CfL (8 experiments)
- shrink=2.965 stable
- B reg 0.3: no effect
- Separate B/X shrink: worse
### Other (6 experiments)
- Gab weight changes: catastrophic without inv-gab match
- Custom coeff orders: no gain
- HybridUint(4,1,2) for modular/AC: slightly worse than (4,2,0)
- ANS shift [0,3,6,9,12]: no gain

## Only remaining paths (ALL high effort, likely diminishing returns)
1. **LZ77 for modular DC** -- save ~0.5-1% on DC streams
2. **Fix EPF sharpness map** -- bug investigation
3. **ANS RebalanceHistogram** -- marginal entropy improvement
4. **Custom dequant weight tables** -- F16 writer exists but complex
