# Encoder SIMD microbenchmark summary

Speedup target tracking:
- XYB target >= 1.5x: current sample 0.86x (not met).
- Transform target (DCT32) >= 1.8x: current sample 0.32x (not met).
- Quant/token target >= 1.3x: tracking enabled; paired scalar-vs-assisted stage benchmark row pending.

| Suite | Benchmark | Mean | 95% CI |
|---|---|---:|---:|
| encoder_end_to_end_corpus | encode_rgb_flat | 847.613 ms | 841.490 ms .. 854.222 ms |
| encoder_end_to_end_corpus | encode_rgb_photo | 294.784 ms | 291.986 ms .. 297.826 ms |
| encoder_end_to_end_corpus | encode_rgba_alpha | 1.246 s | 1.233 s .. 1.260 s |
| encoder_forward_dct_reference | dct2d_32_scalar/Avx512 | 210.442 us | 206.314 us .. 215.832 us |
| encoder_forward_dct_reference | dct2d_32_simd/Avx512 | 659.238 us | 652.670 us .. 666.481 us |
| encoder_idct8_kernel_reference | idct2d_8_8/Scalar | 66.8 ns | 66.6 ns .. 67.0 ns |
| encoder_quant_token_prep | huffman_from_token_histogram | 225.033 us | 220.108 us .. 232.410 us |
| encoder_quant_token_prep | quantize_coeffs | 185.295 us | 183.559 us .. 187.082 us |
| encoder_quant_token_prep | tokenize_pack_signed | 198.222 us | 195.360 us .. 201.385 us |
| encoder_xyb | srgb_u8_to_xyb/1024x768 | 41.509 ms | 41.002 ms .. 42.150 ms |
| encoder_xyb | srgb_u8_to_xyb_assisted/Avx512 | 47.904 ms | 47.041 ms .. 48.905 ms |
| encoder_xyb | srgb_u8_to_xyb_scalar/1024x768 | 41.376 ms | 40.807 ms .. 42.130 ms |
