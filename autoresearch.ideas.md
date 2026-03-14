# Autoresearch Ideas

## Current status
- **PSNR PARITY ACHIEVED**: ALL 19 images PSNR >= libjxl
- Best worst_psnr_deficit: **0.000** (target met!)
- Mean size: +7.07% vs libjxl (secondary concern)
- Worst PSNR margin: +0.010 dB (kodim07)

## Key change that achieved parity
- **Removed adaptive quantization map**: The AQ map was being selected by size-RD
  for images like kodim13, producing smaller files but worse PSNR. libjxl kFalcon
  also uses uniform quant, so this matches the reference encoder.
- Combined with parameter retuning: b_dm=1.05, x_dm=0.90, qlf_boost=2, EPF Y=4.0

## Current config
- Uniform quant only (no adaptive map)
- b_dm = 1.05, x_dm = 0.90
- EPF iters=2, Y=4.0, X=5.5, B=1.5
- quant_lf_boost = 2 for <=7000 blocks, 0 for larger
- ep=2 for <=7000 blocks, 1 for larger
- TOWARDS_ZERO = 3.2
- AQ scale = 1.25 (only affects GS/qlf computation now)

## Size reduction opportunities (maintaining PSNR >= 0)
1. **Entropy coding improvements**: Better ANS/Huffman could save bytes
2. **LZ77 for modular DC**: Repetitive DC patterns compress better with LZ77
3. **Transform map optimization**: Better DCT size selection could save bytes
4. **Trellis quantization**: Joint coefficient RD for optimal bit allocation
5. **Per-image adaptive parameters**: Adjust b_dm/x_dm based on image content

## Exhaustively confirmed for new metric
- b_dm=1.05 optimal (1.10 fails, 1.0 works but less optimal)
- x_dm=0.90 optimal (0.95 barely fails, 0.875 works but larger)
- qlf_boost=2 optimal (0 fails, 4 works but larger)
- EPF Y=4.0 slightly better than 6.0
- Adaptive b_dm (0.06*ytob) not needed with uniform quant
- ep=1 uniform barely fails (1 image -0.007dB)
- epf_iters=1: worse; epf_iters=3: much worse; epf_iters=2 optimal
- gab=false: catastrophic
- Default EPF weights (Y=40): much worse than custom

## DO NOT RETRY
- AQ scale < 1.0: catastrophic (tested 0.85)
- Adaptive map + size-RD selection: root cause of PSNR deficits
- q_for_gs > 0.39: makes things worse (tested 0.50)
