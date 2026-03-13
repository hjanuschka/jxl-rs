# Autoresearch Ideas (deferred)

## Webkit PSNR Gap (-1.87 dB, 37.5 penalty) -- ROOT CAUSE IDENTIFIED
- Both our encoder and libjxl produce DC-only output for Webkit (AC all zero)
- Our gs=5111/qlf=17 gives PSNR 42.90; libjxl gs=8813/qlf=10 gives 44.77
- When we use libjxl's exact gs=8813/qlf=10, our PSNR DROPS to 40.13!
- 4.64 dB gap between our encoder and libjxl at SAME quant params
- Per-block analysis shows systematic G channel +2.0 and B channel +0.9 offset
- This is NOT from global_scale, EPF, gaborish, or HybridUint configs
- Must be from: (1) XYB conversion difference, (2) DCT precision, (3) CfL implementation
- TODO: Add debug logging to dump raw XYB DC values for single block and compare
- TODO: Check if libjxl's OpsinAbsorbance premul_absorb differs from our forward matrix

## Entropy Coding (DONE - ANS + per-channel implemented)
- ~~Add ANS support for modular DC encoding~~ DONE
- ~~ANS for HF metadata~~ DONE
- ~~Per-channel ANS histograms for DC~~ DONE: parity 10.17->10.04
- Try per-channel ANS for HF metadata (4 channels with different sizes -- needs variable-size channel support)
- Try ANS with histogram clustering for modular DC (group similar channels)

## Quantization Improvements
- The softer_rq candidate (base-1) helps size but costs PSNR. Under parity scoring (20x PSNR weight), better to optimize for PSNR when size is already below libjxl.
- RD-optimized candidate selection: weight encoded_size + lambda*distortion instead of pure size
- extra_dc_precision=1: doubles DC granularity without changing global_scale. Need to plumb through quantize_vardct_blocks.

## Per-Image Analysis (current best: parity_score=10.04)
- Webkit: 37.48 pen (-1.87 dB PSNR, -18% size) -- fundamentally different encode pipeline issue
- kodim08: 7.10 pen (-0.35 dB PSNR)
- kodim13: 4.39 pen (-0.22 dB PSNR)
- kodim01: now <=0.64 pen (size parity or better after per-channel ANS)
- zoltan: 0.02 pen (at parity)

## Tried and Failed
- Competitive HybridUint config selection: no improvement, (4,2,0) already optimal
- x_qm_scale change (3->2): regression, photo PSNR worsened
- Falcon global_scale for flat graphics: regression (-4.64 dB) due to unknown encode pipeline issue
- New predictors (Select, Average): decoder mismatch (inverted left/right convention)
- Higher rq candidates: never win size competition
- EPF iters change (2->1): slight regression, quant tuned for 2 iters
- Dead-zone removal: no effect
- Disabling gaborish globally: massive regression (quant tuned for gab)
