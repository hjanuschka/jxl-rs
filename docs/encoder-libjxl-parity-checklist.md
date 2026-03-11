# jxl-rs pure Rust encoder parity checklist vs libjxl

## Goal

Reach practical 1:1 encoder parity with libjxl, with only one intentional difference:

- jxl-rs encoder path is 100% Rust and memory safe by default.

"Parity" here means:

1. Feature parity (same format capabilities).
2. Bitstream interoperability parity (outputs decode in libjxl and jxl-rs, including advanced modes).
3. API parity for integrators (equivalent control over quality, speed, metadata, animation, progressive, etc).
4. Performance and quality parity envelope (same order-of-magnitude speed and compression quality at similar settings).

## Status legend

- `[x]` implemented
- `[~]` partial bootstrap
- `[ ]` missing

## Snapshot (current branch state)

### Quantization and adaptive quality

- `[x]` **Exact libjxl Squirrel-speed AQ pipeline port**: Direct port of
  `enc_adaptive_quantization.cc` including `RatioOfDerivativesOfCubicRootToSimpleGamma`,
  `MaskingSqrt`, `ComputeMask`, `GammaModulation`, `HfModulation`, `BlueModulation`,
  `PerBlockModulations`, Laplacian diff computation, 4x downsample, and `FuzzyErosion`.
  All three XYB channels used for perceptual masking.
- `[x]` Quantization calibration mirrors libjxl's `Quantizer::ComputeGlobalScaleAndQuant`
  + `InitialQuantDC` exactly: `kAcQuant=0.765`, `global_scale` from
  `ComputeGlobalScaleAndQuant(quant_dc, 0.39/d, 0)` with `absd=0` (matching
  Squirrel default speed -- no iterative `FindBestQuantization`).
- `[x]` Per-tile CfL optimization: least-squares ytox/ytob regression per 64x64 tile.
- `[x]` Gradient DC prediction: auto-selects from Zero/Left/Top/Gradient predictors.

### Gaborish and inverse gaborish

- `[x]` Gaborish enabled in frame header (`gab=true`) for decoder-side smoothing.
- `[x]` Inverse Gaborish (5x5 symmetric sharpening) applied before DCT, matching
  libjxl's `enc_heuristics.cc` flow: AQ on original opsin -> `GaborishInverse` ->
  DCT on sharpened data. Uses mirror boundary conditions matching libjxl's
  `Symmetric5` convolution. Per-channel weight support (`apply_inverse_gaborish_weighted`).
  Source image PSNR jumped 26->34 dB.

### EPF (Edge-Preserving Filter)

- `[x]` EPF enabled with `epf_iters=2` and default sharpness=4 for all blocks.
  Frame header writes `epf_sharp_custom=false`, `epf_weight_custom=false`,
  `epf_sigma_custom=false`. Both single-group and multi-group paths populate
  EPF sharpness channel (ch3 in HF metadata).
- `[x]` Fixed LoopFilter extensions field in frame header serialization.

### Transform and block strategy

- `[x]` Non-DCT8 transform-family plumbing active end to end for ids `1..26`
  (quant table routing, tokenization order dispatch, transform-map support checks,
  forced-map decode tests).
- `[x]` 8x8 special-transform forward synthesis via decoder-basis inversion
  (`IDENTITY`, `DCT2X2`, `DCT4X4`, `DCT4X8`, `DCT8X4`, `AFV0..AFV3`).
- `[x]` Forward/inverse consistency tests for non-special square transforms
  (`DCT16`, `DCT32`, `DCT64`).
- `[~]` Linear forward-solver parity paths for selected non-8x8 families:
  square (`DCT16`, gated `DCT32`) and rectangular (`DCT16X8`, `DCT8X16`,
  `DCT32X8`, `DCT8X32`, gated `DCT32X16`/`DCT16X32`); larger families scalar fallback.
- `[x]` Non-DCT8 tokenization order selection shape-id aligned with decoder
  permutation semantics.
- `[x]` Forced-map end-to-end decode tests through `DCT256`, `DCT256X128`, `DCT128X256`.
- `[~]` **Entropy-based DCT16 merging**: for smooth 2x2 block groups at any distance,
  estimates whether DCT16x16 is cheaper using non-zero coefficient fraction heuristic.
  Candidate selection picks the best by actual encoded byte size. Saves ~4K bytes at d=1
  on photographic content (77K -> 73K).
- `[x]` Conservative small-special candidate generation (`DCT4X8`/`DCT8X4`,
  `IDENTITY`/`DCT2X2`/`DCT4X4`, mixed special maps) and sparse AFV candidates
  for high-distance modes, all gated by exact-byte winner selection.

### Entropy coding

- `[x]` ANS + Huffman AC coding with byte-size winner selection.
- `[x]` Histogram clustering via seed-based `FastClusterHistograms`-style algorithm.
- `[x]` HybridUint config search across 4 presets per entropy candidate.
- `[x]` Coefficient order infrastructure: data-driven 8x8 order + Lehmer permutation.
- `[x]` Total-encode budget (MAX_TOTAL_ENCODES=32) prevents combinatorial blowup.

### Writer foundation

- `[x]` Bit writer, u32/i32 helpers, TOC writer.
- `[~]` Headers/container writing (minimal path).
- `[~]` Modular stream generation (multi-predictor, multi-channel).
- `[~]` Entropy modeling: ANS/Huffman AC coding in active VarDCT path.

### Not yet implemented

- `[ ]` Advanced format features (progressive, animation, metadata boxes, JPEG reconstruction).
- `[ ]` Patches/splines/noise tools.
- `[ ]` Iterative `FindBestQuantization` butteraugli feedback loop (Kitten speed and slower).
- `[ ]` Full AC strategy heuristic (libjxl's `ProcessRectACS` with entropy estimation).
- `[ ]` `AdjustQuantField` for non-8x8 transforms (max/mean interpolation).
- `[ ]` Custom block entropy model (`FindBestBlockEntropyModel`).
- `[ ]` Modular transforms (palette, squeeze, RCT).

### Current benchmarks

| Image | Distance | jxl-rs bytes | jxl-rs dB | libjxl bytes | libjxl dB | Notes |
|-------|----------|-------------|-----------|-------------|-----------|-------|
| photo | d=1 | 73,304 | 28.79 | 65,444 | 28.62 | 12% bigger, +0.17 dB |
| photo | d=2 | 32,109 | 27.99 | 25,886 | 27.94 | 24% bigger, +0.05 dB |
| photo | d=3 | 7,463 | 27.81 | 13,342 | 27.63 | 44% smaller, +0.18 dB |
| source | d=1 | 299,030 | 31.57 | 356,273 | 34.79 | 16% smaller, -3.2 dB |

Key remaining gaps:
- **Photo d=1/d=2 size**: ~12-24% bigger, primarily from all-DCT8x8 encoding.
  libjxl uses larger transforms (DCT16/32) for smooth regions. Entropy-based
  DCT16 merging partially addresses this (saves ~4K at d=1).
- **Source d=1 PSNR**: -3.2 dB gap from lack of full AC strategy selection
  and iterative butteraugli feedback. Text/screenshot content needs per-block
  quality adaptation that our single-pass AQ can't provide.

## libjxl subsystem parity map

| ID | libjxl subsystem | Status | jxl-rs current files | Milestone |
|---|---|---|---|---|
| P01 | Bit writing primitives (`enc_bit_writer`) | [x] | `encode/bit_writer.rs`, `encode/encodings.rs` | M1 |
| P02 | Codestream header serialization | [~] | `encode/headers.rs` | M2 |
| P03 | Container writing | [~] | `encode/container.rs` | M2, M8 |
| P04 | Public encoder API | [~] | `encode/encoder.rs`, `encode/options.rs` | M4, M9 |
| P05 | Input pixel ingestion | [~] | `encode/headers.rs`, `encode/encoder.rs` | M4 |
| P06 | Context map writer | [~] | `encode/entropy/context_map.rs` | M3, M5 |
| P07 | Huffman table build | [~] | `encode/entropy/{huffman,huffman_encode}.rs` | M3 |
| P08 | HybridUint entropy paths | [~] | `encode/entropy/hybrid_uint.rs` | M3, M6 |
| P09 | ANS histogram + stream | [~] | `encode/entropy/ans.rs` | M3 |
| P10 | Modular tree building | [~] | `encode/modular.rs` | M4, M5 |
| P11 | Modular transforms | [ ] | none | M5 |
| P12 | Modular lossless frame | [~] | `encode/modular_encode.rs` | M4 |
| P13 | Near-lossless controls | [ ] | none | M5 |
| P14 | Fast lossless heuristics | [ ] | none | M5, M7 |
| P15 | VarDCT color path (XYB) | [x] | `encode/{xyb,vardct}.rs` | M6 |
| P16 | VarDCT block strategy | [~] | `encode/vardct.rs` | M6 |
| P17 | Quant field generation | [x] | `encode/vardct.rs` | M6, M7 |
| P18 | Coefficient tokenization | [~] | `encode/vardct.rs` | M6 |
| P19 | Progressive pass emission | [ ] | none | M8 |
| P20 | Animation and frame refs | [ ] | none | M8 |
| P21 | Patches/splines/noise | [ ] | none | M8 |
| P22 | Metadata boxes | [ ] | none | M8 |
| P23 | JPEG reconstruction | [ ] | none | M8 |
| P24 | Effort presets | [ ] | none | M7 |
| P25 | Parallel encode scheduling | [ ] | none | M5, M7 |
| P26 | Conformance test harness | [~] | unit tests + fuzz | M9 |
| P27 | CLI parity | [~] | `jxl_cli/src/bin/jxle.rs` | M4, M9 |

## Line-by-line implementation checklist

### A. Public API and integration

- [~] A01 `JxlEncoder::encode_image(...)` for raw buffers (RGB/Gray/RGBA, u8/u16/f32).
- [ ] A02 Per-frame settings type (lossless, distance, effort, modular vs vardct).
- [ ] A03 Streaming output callback API.
- [ ] A04 Stable documented API surface with semver.

### B. Header and container parity

- [ ] B01 Full `ImageMetadata` serialization.
- [ ] B02 Full `FrameHeader` serialization (crop/blending/references).
- [ ] B03 Container box policy for metadata ordering.
- [ ] B04 `jxlp` chunked codestream output.

### C. Entropy parity

- [x] C01 ANS writer with decoder roundtrip tests.
- [~] C02 Histogram building from symbol distributions.
- [x] C03 Histogram clustering and context map optimization.
- [x] C04 Full Huffman table construction.
- [x] C05 HybridUint config tuning.
- [~] C06 Non-simple context map encoding (multi-cluster AC contexts).

### D. Modular lossless parity

- [~] D01 Real channel residual pipeline.
- [ ] D02 Tree generation from statistics.
- [ ] D03 Modular transforms (palette, squeeze, RCT).
- [ ] D04 Near-lossless controls.
- [ ] D05 Fast-lossless path.
- [ ] D06 Lossless pixel exactness guarantee.

### E. VarDCT lossy parity

- [x] E01 Forward lossy color pipeline (sRGB -> XYB).
- [~] E02 Block strategy selection and transform dispatch.
  - Done: DCT16 entropy-based merging, special-transform candidates, AFV candidates.
  - Remaining: full `ProcessRectACS` entropy estimation, DCT16x8/8x16 merging,
    DCT32/64 merging, `FindBestFirstLevelDivisionForSquare`.
- [x] E03 Quant field generation pipeline (full libjxl Squirrel-speed AQ port).
- [x] E04 LF/HF coefficient tokenization and entropy coding.
- [~] E05 Quality tuning heuristics and effort tiers.
- [x] E06 Inverse Gaborish pre-sharpening (libjxl `GaborishInverse` port).
- [x] E07 EPF (Edge-Preserving Filter) enabled with default parameters.

### F. Advanced format parity

- [ ] F01 Progressive passes.
- [ ] F02 Animation frame sequence encode.
- [ ] F03 Extra channel parity.
- [ ] F04 Patches/splines/noise encoding.
- [ ] F05 JPEG reconstruction path.

### G. Performance and determinism

- [~] G00 SIMD acceleration plan (deferred until algorithmic parity stable).
- [ ] G01 Deterministic parallel scheduler.
- [ ] G02 Effort presets.
- [ ] G03 Benchmark dashboard.

### H. Conformance and testing

- [~] H01 Differential harness: jxl-rs encode -> djxl decode validation.
- [~] H02 Rate/distortion comparison vs libjxl.
- [ ] H03 Expanded fuzzing.
- [ ] H04 Determinism tests.

## Integrator readiness checklist

- [ ] R01 Safe API: no `unsafe` required for common encode operations.
- [ ] R02 Deterministic outputs documented and tested.
- [ ] R03 Resource limits (memory, threads, dimensions).
- [ ] R04 Panic-free error handling.
- [ ] R05 Stable long-term API.
