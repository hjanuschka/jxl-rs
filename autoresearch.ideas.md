# Autoresearch Ideas (deferred)

## Current status
- Best parity_score: **1.454** (887 experiments)
- Improvement: 2.380 -> 1.454 (-38.9%)
- Config:
  - `TOWARDS_ZERO = 3.2` (CfL shrink)
  - CfL formula: `84 * (sum_yb - sum_yy) / (sum_yy + reg)` (corrected)
  - CfL regularizer: `N * 84^2 * 1e-9 * 0.5` (matching libjxl)
  - `b_dm = 0.94`, `x_dm *= 0.97`
  - EPF: iters=2, Y=6.0, X=5.5, B=1.5
  - quant_lf: +4 for <=7000 blocks, +0 for larger
  - ep: 2 for <=7000 blocks, 1 for larger

## Per-image penalty breakdown
- kodim01: pen=0.000
- kodim08: pen=5.184 (B-chan -0.395 dB, R-chan -0.189 dB)
- kodim13: pen=2.088 (B-chan -0.189 dB)
- zoltan: pen=0.000
- Webkit: pen=0.000

## Exhaustively confirmed DO NOT RETRY
- All EPF weights (Y, X, B), b_dm, x_dm, TOWARDS_ZERO, quant_lf, ep
- Separate per-channel TZ: worse both directions
- Adaptive b_dm: worse
- CfL regularization: matches libjxl (essentially zero)
- Newton CfL solver: produces identical factors (linear regression = Newton)
- Y roundtrip (with and without AdjustQuantBias): worse (CfL factors mismatch)
- b_dm=1.0 matching decoder: worse (6.4% over-reconstruction bias helps)
- AQ scale: 1.25 confirmed (1.20 catastrophic)
- q_for_global_scale: 0.39 confirmed
- Extra quant candidates: harder_rq not selected, softer_rq2 catastrophic

## Root cause of remaining gap
1. **Coarser global_scale** (5111 vs 8813): ~1.7x larger quant steps
2. **No AdjustQuantBlockAC**: per-block quant adjustment heuristics
3. **b_dm mismatch**: encoder 0.94 vs decoder 1.0 (intentionally beneficial)

## Remaining structural paths (complex, high-effort)
1. **AdjustQuantBlockAC port** - per-block quant adjustment. Directly targets
   kodim08/kodim13 PSNR gap. Would need: flatness detection, quadrant HF analysis,
   activity-based quant reduction. ~200 lines of heuristic code.
2. **Higher global_scale (0.79/d)** - matches libjxl but requires complete
   recalibration of all parameters (dm, AQ, EPF, etc.)
3. **Trellis quantization** - joint coefficient RD optimization
4. **LZ77 for modular DC** - save bytes to unlock ep=2 for zoltan
