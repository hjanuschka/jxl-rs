# Autoresearch Ideas (deferred)

## Entropy Coding (DONE - ANS implemented)
- ~~Add ANS support for modular DC encoding~~ DONE: massive win (20.81->13.87)
- ~~ANS for HF metadata~~ DONE: mean_size 3.04%->0.89%
- Competitive HybridUint config selection for modular streams (try multiple configs like libjxl's kBest/kFast modes)
- Try ANS with multiple histogram clusters for modular DC (per-channel histograms via multi-leaf tree with separate context indices). Current implementation uses single histogram for all channels.

## Quantization / Global Scale
- The Webkit PSNR gap (-1.87 dB, 37.5 penalty) is caused by global_scale=5111 (from 0.39/d AQ path) making AC quantization trivially coarse. Even rq=1 produces scale=0.078/dw -- everything rounds to zero. libjxl Falcon path uses 0.79/d giving gs=8630 and preserves AC detail. Attempted separate Falcon encode path but quant calibration was wrong (produced -4.6 dB). This is the biggest remaining gap (37.5 of 50.86 total penalty).
- Need a properly calibrated combined global_scale: high enough for AC detail but balanced for DC. Maybe use 0.55/d as compromise?
- The softer_rq candidate (base-1) improves size by ~4.7% at cost of 0.18 dB PSNR -- net beneficial under current scoring.

## Per-Image Analysis
- Current best (parity_score=10.17): Webkit=37.48 pen, kodim08=7.10, kodim13=4.39, kodim01=1.87, zoltan=0.02
- Webkit dominates penalty (74% of total). All other images are near parity.
- For Webkit: AC coefficients are ALL zero. File is 18% smaller than libjxl. Penalty is 100% from PSNR.
- For photos: PSNR gaps (-0.06 to -0.36 dB) come from aggressive uniform rq=5 candidate winning over AQ.

## Transform / EPF
- EPF iters: tested matching libjxl (2->1 at d=1.0). Slight PSNR regression because our quant was tuned for 2 iters.
- Dead-zone removal: no effect (subsumed by round()).
- libjxl at Falcon only uses DCT8x8 (no merge heuristic). Our effort-7 uses complex merge logic.

## Promising unexplored
- Try intermediate global_scale (0.55/d instead of 0.39/d) for flat graphics only
- Try per-block adaptive quant using the AQ map with global_scale matched to its median
- Improve histogram clustering (entropy-based distance metrics matching enc_cluster.cc)
- Try weighted predictor (predictor 14 in JXL spec) for modular DC
