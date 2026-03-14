# Autoresearch Ideas (deferred)

## Current status
- Best parity_score: **1.462** (840 experiments)
- Improvement: 2.380 -> 1.462 (-38.6%)
- Config:
  - `TOWARDS_ZERO = 3.2` (CfL shrink)
  - Merge muls: rect=4.0, 16=5.5, 32=7.0
  - `b_dm = 0.94`, `x_dm *= 0.97`
  - EPF: iters=2, Y=4.0, X=5.5, B=1.5
  - quant_lf: +4 for <=7000 blocks, +1 for larger
  - ep: 2 for <=7000 blocks, 1 for larger

## Penalty breakdown (1.462 = mean of 5 image penalties)
- kodim01: pen=0.000
- kodim08: pen=~5.2 (-0.26 dB gap, B-channel dominated)
- kodim13: pen=~2.1 (-0.11 dB gap, B-channel)
- zoltan: pen=~0.02 (size +0.02%)
- Webkit: pen=0.000

## Exhaustively confirmed DO NOT RETRY (with current config)
- EPF Y: 4.0 (tested 3-40), X: 5.5 (tested 4.5-6.0), B: 1.5 (tested 1.0-2.0)
- b_dm: 0.94 (tested 0.88-0.95)
- x_dm*0.97 (tested 0.96-0.98)
- TOWARDS_ZERO: 3.2 (tested 2.5-3.4)
- quant_lf+4 for small, +1 for large (tested +1 through +6)
- ep=2 for small, ep=1 for large (ep=2 on zoltan overshoots by +1.5%)
- CfL regularization: zero to 5e-3, no effect (factor granularity dominates)
- Separate B/X CfL shrink: both directions worse
- Rounding bias for B AC: always worse
- Custom LfQuantFactors: Webkit catastrophic
- EPF iter=3: over-smooths
- EPF zeroflush: zero effect on output
- Custom sharp LUT: 16-byte overhead kills gains
- Gab weights: catastrophic
- Custom coeff orders: no gain

## Remaining paths
1. **Save bytes on zoltan to unlock ep=2 everywhere** - needs ~3KB savings.
   LZ77 for modular DC could save ~0.5-1%. Custom dequant tables unclear.
2. **Trellis quantization for B-channel** - jointly optimize coefficient groups.
   Complex but targets the root cause of B-channel PSNR gap.
3. **ANS RebalanceHistogram** - marginal entropy improvement.
4. **Investigate why EPF sigma_custom causes decode failure** - could unlock
   fine-grained sigma tuning per frame.
