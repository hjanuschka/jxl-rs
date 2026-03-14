# Autoresearch Ideas

## Current status
- Best parity_score: **1.4528** (940 experiments)  
- Improvement: 2.380 -> 1.4528 (-39.0%)

## KEY DISCOVERY: libjxl effort 3 (kFalcon) pipeline
- **NO CfL optimization** (zero ytox/ytob maps, base_b=1.0)
- **NO adaptive quantization** (uniform quant = 0.79/d)
- **NO AdjustQuantBlockAC** (only at effort >=5)
- **Dead-zone thresholds**: Y 0.56/0.62, X/B 0.58/0.62
- **Y roundtrip**: quantize Y with dead-zone, dequant with AdjustQuantBias
- **GS = 10354** (from q=0.79/d), raw_quant=5

## Why our encoder is different (and generally better!)
- We DO compute CfL (significant quality benefit)
- Our GS=5111, raw_quant=10 (effective scale 0.780 vs 0.790)
- We use standard round-to-nearest (no dead-zone)
- No Y roundtrip
- RD criterion selects uniform quant (adaptive never wins)

## Exhaustively confirmed DO NOT RETRY
(see git log for 940 experiments of parameter sweeps)
- All EPF, b_dm, x_dm, TOWARDS_ZERO, quant_lf, ep confirmed
- Y roundtrip (any variant): always worse with our CfL factors
- Dead-zone (any threshold): always worse with our GS
- Zero CfL: catastrophic (4.78)
- Zero CfL + dead-zone + Y roundtrip: 9.34
- CfL + dead-zone + Y roundtrip: 6.27
- Higher GS without recalibration: 11.3
- AQ candidate never selected (uniform always wins)

## Remaining gap
- kodim08: pen=5.184 (B-chan -0.394 dB)
- kodim13: pen=2.088 (B-chan -0.189 dB)

## Unexplored paths
1. **GS recalibration project**: Switch to GS=10354 and retune ALL params
2. **Trellis quantization**: Joint coeff RD optimization
3. **LZ77 for modular DC**: Save bytes
4. **Per-block b_dm based on B residual magnitude**: Target specific blocks
5. **Entropy coding improvements**: More efficient ANS could give size savings
