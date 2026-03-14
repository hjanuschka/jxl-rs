# Autoresearch Ideas (deferred)

## Current status
- Best parity_score: **1.609** (756 experiments)
- Best configuration:
  - `TOWARDS_ZERO = 2.965` (CfL shrink)
  - Merge multipliers: rect=4.0, 16x16=5.5, 32x32=7.0
  - `b_dm_multiplier = 0.94` (retuned with EPF weights)
  - `x_dm_multiplier *= 0.97` (new, 0.776 effective)
  - `epf_iters = 2`
  - Custom EPF weights: Y=5.0, X=5.0, B=2.0 (NEW: reduced Y and B)
  - `extra_dc_precision = 1`

## Remaining penalty breakdown (estimated)
- kodim08: ~5.5 pen (-0.275 dB PSNR gap)
- kodim13: ~2.5 pen (~-0.12 dB PSNR gap)
- kodim01/zoltan/Webkit: 0 pen

## Exhaustively confirmed (DO NOT RETRY)
### EPF weights
- Y: 5.0 optimal (tested 3-40, flat below 10)
- X: 5.0 optimal (3.5 and 6.0 both worse)
- B: 2.0 optimal (1.0, 1.5, 1.75, 2.5, 3.5 all worse)
- zeroflush params: no effect
- EPF sigma quant_mul: 0.50 and 0.55 both worse (over-smoothing)
- EPF sharpness map: NO effect on djxl output (encoding bug?)

### Quantization
- b_dm_multiplier: 0.94 optimal with EPF weights (0.91-0.95 tested)
- x_dm_multiplier: 0.97 optimal (0.95 overshoots size, 0.975/1.0 worse parity)
- Dead zones: always worse (0.505, 0.535, 0.465 all bad)
- AdjustQuantBias-aware rounding: more nonzeros, worse parity
- quant_ac, rq candidates: no effect (uniform always wins)
- ep=2: great PSNR (+0.73 avg) but zoltan overshoots by +0.72%
- ep=2 + harder rq=7: still overshoots

### CfL / color
- CfL shrink: 2.965 optimal (tested 2.2-3.2, stable across EPF changes)
- Separate B vs X CfL shrink: always worse
- CfL dist_mul, B base-correlation: confirmed bad
- Gaborish weight mismatch: catastrophic (encoder/decoder must match)
- Inverse gab B-weight reduction: introduces systematic error

### Other
- ANS shift [0,3,6,9,12]: no gain
- Merge multipliers: saturated
- AQ map modifications: never selected (uniform wins)

## Only remaining paths
1. **LZ77 for modular DC streams** -- save ~1% bytes, could make ep=2 viable
   for zoltan without size overshoot
2. **Fix EPF sharpness map encoding bug** -- if fixed, per-block sharpness
   could help kodim08/13 specifically
3. **Custom dequant weight tables** -- signal different HF weight shapes to
   decoder (F16 writer now available). Could better match our quantization grid
4. **Dual-pass Y-roundtrip + CfL refit** -- very high effort but addresses
   root cause of chroma gap
