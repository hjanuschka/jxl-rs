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

Recent incremental VarDCT prototype progress (this branch):
- `[x]` Non-DCT8 transform-family plumbing is active end to end for ids `1..26`
  (quant table routing, tokenization order dispatch, transform-map support checks,
  forced-map decode tests).
- `[x]` 8x8 special-transform forward synthesis now uses decoder-basis inversion
  for parity by construction (`IDENTITY`, `DCT2X2`, `DCT4X4`, `DCT4X8`, `DCT8X4`,
  `AFV0..AFV3`), with roundtrip tests against `transform_to_pixels`.
- `[x]` Added forward/inverse consistency tests for non-special square transform
  families (`DCT16`, `DCT32`, `DCT64`) using decoder `transform_to_pixels` plus
  LF extraction from clamped 8x8 subblocks.
- `[~]` Added linear forward-solver parity paths for selected non-8x8 families:
  square (`DCT16`, gated `DCT32`) and rectangular (`DCT16X8`, `DCT8X16`,
  `DCT32X8`, `DCT8X32`, gated `DCT32X16`/`DCT16X32`) with transform-specific
  ignored LF coefficient index tests; larger families still use scalar fallback.
- `[x]` Non-DCT8 tokenization order selection is now shape-id aligned with decoder
  permutation semantics (canonical order per shape shared across transposed pairs),
  with lazy cached shape-order lookup in tokenization.
- `[x]` Forced-map end-to-end decode tests cover the large transform family through
  `DCT256`, `DCT256X128`, and `DCT128X256`.
- `[x]` Transform-map candidate search still chooses by final encoded byte size,
  with large-image guardrails and total-encode budget to avoid pathological
  search/runtime behavior.
- `[x]` Quantization calibration now mirrors libjxl's
  `Quantizer::ComputeGlobalScaleAndQuant` + `InitialQuantDC`: global_scale,
  quant_lf, and base_raw_quant all derived from butteraugli distance via libjxl's
  constants (kAcQuant=0.79, kDcQuant=1.096, kDcQuantPow=0.83, kQuantFieldTarget=5,
  kGlobalScaleDenom=65536). PSNR at d=1 now exceeds libjxl (29.18 vs 28.62 dB).
- `[x]` Gaborish enabled in frame header (gab=true) for free decoder-side smoothing,
  with inverse Gaborish encoder pre-filter available but deferred pending
  butteraugli-based adaptive quantization.
- `[x]` Histogram clustering via seed-based `FastClusterHistograms`-style algorithm
  (`build_greedy_clustered_context_map`), ~11% byte savings at d=1.
- `[x]` HybridUint config search across 4 presets per entropy candidate.
- `[x]` Total-encode budget (MAX_TOTAL_ENCODES=32) prevents combinatorial blowup.
- `[x]` Gradient DC prediction: modular DC now auto-selects from Zero/Left/Top/Gradient
  predictors by estimated cost. Massive savings at d=3 (-39% photo, -48% flat).
- `[x]` Coefficient order infrastructure: data-driven 8x8 order computation and
  Lehmer-code permutation encoder in place (activates when order differs from natural).
- `[x]` Inverse Gaborish 5x5 kernel implemented (currently deferred pending butteraugli).
- `[x]` Added conservative small-special candidate generation (`DCT4X8`/`DCT8X4`,
  `IDENTITY`/`DCT2X2`/`DCT4X4`, plus mixed special maps) and sparse AFV candidate
  generation for high-distance modes on moderate grids, all gated by exact-byte
  winner selection.
- `[x]` Per-tile CfL optimization: least-squares ytox/ytob regression per 64x64 tile,
  reducing chroma residuals. Skipped at near-lossless (d<0.5).
- `[x]` Pixel-domain HF activity adaptive quantization: libjxl-style `HfModulation`
  with 4-connected Y-channel gradient sums, continuous quant modulation via
  percentile normalization, and distance-dependent damping. Matches libjxl file
  size at d=1 (65424 vs 65444 bytes) with +0.22 dB better PSNR.

- `[x]` Writer foundation: bit writer, u32/i32 helpers, TOC writer.
- `[~]` Minimal headers/container writing.
- `[~]` Minimal entropy helpers (HybridUint config, fixed/simple Huffman/context map writer primitives).
- `[~]` Minimal modular decodable stream generation (constant tree leaf, synthetic output).
- `[~]` Early real image-to-bitstream path: RGB8 raw input into modular stream (group-aware bootstrap path, interleaved + strided API inputs) plus Gray8 API variants (expanded to RGB bootstrap path). Histogram-driven Huffman residual coding now available alongside fixed-token bootstrap.
- `[~]` Entropy modeling and ANS/Huffman AC coding in active VarDCT path
  (byte-size winner selection between ANS and Huffman).
- `[x]` VarDCT lossy encoding.
- `[ ]` Advanced format features (progressive, animation, metadata boxes, JPEG reconstruction).

## libjxl subsystem parity map

This maps libjxl encoder areas to jxl-rs files and milestone/issue buckets from `docs/jxl-encoder-issue-breakdown.md`.

| ID | libjxl subsystem | Status | jxl-rs current files | jxl-rs target files | Milestone mapping |
|---|---|---|---|---|---|
| P01 | Bit writing primitives (`enc_bit_writer`) | [x] | `jxl/src/encode/bit_writer.rs`, `jxl/src/encode/encodings.rs` | keep current | M1 |
| P02 | Codestream header field serialization (`enc_fields`, `enc_file`) | [~] | `jxl/src/encode/headers.rs` | split into `headers/file_header.rs`, `headers/frame_header.rs`, `headers/image_metadata.rs` under `encode/` | M2 |
| P03 | Container writing (`enc_file` container side, boxes) | [~] | `jxl/src/encode/container.rs` | add `encode/container_boxes.rs` for Exif/XMP/JUMBF/jxlp policy | M2, M8 |
| P04 | Public encoder API surface (`JxlEncoder`, frame settings) | [~] | `jxl/src/encode/encoder.rs`, `jxl/src/encode/options.rs`, `jxl/src/api/mod.rs` | add `jxl/src/encode/api.rs`, per-frame settings types | M4, M9 |
| P05 | Input pixel ingestion and normalization | [~] | `jxl/src/encode/headers.rs` (RGB8 single-group bootstrap path), `jxl/src/encode/encoder.rs` | `jxl/src/encode/input.rs`, `jxl/src/encode/color_pipeline.rs` | M4 |
| P06 | Context map writer and optimization (`enc_entropy_coder`) | [~] | `jxl/src/encode/entropy/context_map.rs` | add non-simple context map and optimization pass | M3, M5 |
| P07 | Huffman table build from symbol stats | [~] | `jxl/src/encode/entropy/{huffman,huffman_encode}.rs` (frequency-driven coding active in VarDCT) | keep improving clustering/modeling | M3 |
| P08 | HybridUint full usage in entropy paths | [~] | `jxl/src/encode/entropy/hybrid_uint.rs`, `jxl/src/encode/vardct.rs` | continue tuning per-token usage | M3, M4, M6 |
| P09 | ANS histogram + stream writing (`enc_ans`) | [~] | `jxl/src/encode/entropy/ans.rs` (+ VarDCT wiring in `encode/vardct.rs`) | expand to remaining paths | M3 |
| P10 | Modular tree building from image stats (`enc_modular_tree`) | [~] | `jxl/src/encode/modular.rs` (single-leaf constants) | `jxl/src/encode/modular/tree_builder.rs` | M4, M5 |
| P11 | Modular transforms (palette/squeeze/rct) | [ ] | none | `jxl/src/encode/modular/transforms.rs` | M5 |
| P12 | Real modular lossless frame encoding (`enc_modular`) | [~] | `jxl/src/encode/modular_encode.rs` (real image residual coding path exists; parity still incomplete) | `jxl/src/encode/modular/frame.rs`, `jxl/src/encode/modular/channel.rs` | M4 |
| P13 | Near-lossless controls | [ ] | none | add to modular pipeline and `JxlEncoderOptions` | M5 |
| P14 | Fast lossless heuristics (`enc_fast_lossless`) | [ ] | none | `jxl/src/encode/modular/fast_lossless.rs` | M5, M7 |
| P15 | VarDCT forward color path (`enc_xyb`, color transforms) | [x] | `jxl/src/encode/{xyb,vardct}.rs` | split into dedicated modules later | M6 |
| P16 | VarDCT block strategy and transforms (`enc_frame`, heuristics) | [~] | `jxl/src/encode/vardct.rs` (transform-map candidates + dispatch active) | `jxl/src/encode/vardct/block_strategy.rs`, `.../transform.rs` | M6 |
| P17 | Quant field generation and tuning (`enc_quant_weights`) | [~] | `jxl/src/encode/vardct.rs` (adaptive raw quant map + transform-aware quantization) | `jxl/src/encode/vardct/quant.rs` | M6, M7 |
| P18 | Coefficient tokenization + entropy coding | [~] | `jxl/src/encode/vardct.rs` (LF/HF tokenization + ANS/Huffman AC paths) | `jxl/src/encode/vardct/tokens.rs`, `.../entropy.rs` | M6 |
| P19 | Progressive pass emission (`enc_progressive_split`) | [ ] | none | `jxl/src/encode/progressive.rs` | M8 |
| P20 | Animation and frame references | [ ] | none | `jxl/src/encode/animation.rs` | M8 |
| P21 | Patches/splines/noise tools (`enc_patch_dictionary`, `enc_splines`, `enc_noise`) | [ ] | none | `jxl/src/encode/tools/{patches,splines,noise}.rs` | M8 |
| P22 | Metadata and container extras policy | [ ] | minimal `jxlc` only | extend `encode/container.rs` + metadata policy module | M8 |
| P23 | JPEG reconstruction (`jbrd`) support decision/implementation | [ ] | none | `jxl/src/encode/jpeg_reconstruction.rs` (if in scope) | M8 |
| P24 | Effort presets and encode tuning | [ ] | `JxlEncoderOptions` fields unused for tuning | `jxl/src/encode/effort.rs` | M7 |
| P25 | Parallel encode scheduling and determinism | [ ] | none | `jxl/src/encode/scheduler.rs` | M5, M7 |
| P26 | Differential and conformance test harness | [~] | unit tests + one fuzz target | `tools/encoder_diff/`, corpus harness in `ci/` | M9 |
| P27 | CLI parity with cjxl-like workflows | [~] | `jxl_cli/src/bin/jxle.rs` bootstrap modes | `jxl_cli/src/enc/` integration path | M4, M6, M9 |

## Line-by-line implementation checklist

### A. Public API and integration

- [~] A01 Add `JxlEncoder::encode_image(...)` for raw in-memory buffers (RGB/Gray/RGBA, u8/u16/f32).
  - Current: `encode_modular_u8_rgb_codestream/container` + `encode_image` for RGB8 interleaved/strided inputs and Gray8 variants, group-aware bootstrap path.
  - Files: `jxl/src/encode/encoder.rs`, new `jxl/src/encode/input.rs`.
  - Milestone: M4.
- [ ] A02 Add per-frame settings type mirroring libjxl style controls (lossless, distance, effort, modular vs vardct).
  - Files: `jxl/src/encode/options.rs`, new `jxl/src/encode/frame_settings.rs`.
  - Milestone: M4, M6.
- [ ] A03 Add streaming output callback API (push chunks instead of single `Vec<u8>`).
  - Files: new `jxl/src/encode/streaming.rs`.
  - Milestone: M9.
- [ ] A04 Stabilize and document API surface with semver guarantees.
  - Files: docs + public API modules.
  - Milestone: M9.

### B. Header and container parity

- [ ] B01 Serialize full `ImageMetadata` options, not only all-default path.
  - Files: split from `jxl/src/encode/headers.rs`.
  - Milestone: M2.
- [ ] B02 Serialize full `FrameHeader` options including crop/blending/reference details.
  - Files: split from `jxl/src/encode/headers.rs`.
  - Milestone: M2, M8.
- [ ] B03 Add container box policy for metadata boxes and ordering rules.
  - Files: `jxl/src/encode/container.rs`, new `.../container_boxes.rs`.
  - Milestone: M8.
- [ ] B04 Add `jxlp` chunked codestream output for streaming/large codestream support.
  - Files: `jxl/src/encode/container.rs`.
  - Milestone: M8.

### C. Entropy parity

- [x] C01 Implement ANS writer end to end with decoder roundtrip tests.
  - Files: `jxl/src/encode/entropy/ans.rs`.
  - Milestone: M3.
- [~] C02 Implement histogram building from symbol distributions.
  - Files: `jxl/src/encode/entropy/histograms.rs`.
  - Milestone: M3.
- [x] C03 Implement histogram clustering and context map optimization.
  - Files: `jxl/src/encode/vardct.rs` (`build_greedy_clustered_context_map` seed-based
    clustering of per-context histograms, integrated as candidates alongside preset maps).
  - Milestone: M3, M5.
- [x] C04 Implement full Huffman table construction path (not fixed-symbol-only).
  - Files: `jxl/src/encode/entropy/{huffman,huffman_encode}.rs`.
  - Milestone: M3.
- [x] C05 Hook HybridUint configs to real token distributions and tune choices.
  - Files: `jxl/src/encode/entropy/hybrid_uint.rs` + callers.
  - Milestone: M3, M4, M6.

### D. Modular lossless parity

- [~] D01 Build real channel residual pipeline from input image data.
  - Files: `jxl/src/encode/modular_encode.rs` (active path), future split to `modular/channel.rs`.
  - Milestone: M4.
- [ ] D02 Build tree generation from statistics (split properties, thresholds).
  - Files: new `jxl/src/encode/modular/tree_builder.rs`.
  - Milestone: M4, M5.
- [ ] D03 Implement modular transforms (palette, squeeze, rct variants).
  - Files: new `jxl/src/encode/modular/transforms.rs`.
  - Milestone: M5.
- [ ] D04 Implement near-lossless controls.
  - Files: modular pipeline + options.
  - Milestone: M5.
- [ ] D05 Add fast-lossless path and effort-based switching.
  - Files: new `jxl/src/encode/modular/fast_lossless.rs`.
  - Milestone: M5, M7.
- [ ] D06 Guarantee lossless pixel exactness across RGB/Gray/alpha cases.
  - Tests: corpus encode/decode equality.
  - Milestone: M4, M5.

### E. VarDCT lossy parity

- [x] E01 Implement forward lossy color pipeline (sRGB/linear and XYB path as needed).
  - Files: `jxl/src/encode/{xyb,vardct}.rs`.
  - Milestone: M6.
- [~] E02 Implement block strategy selection and transform dispatch.
  - Files: `jxl/src/encode/vardct.rs` (active implementation), future split to `vardct/block_strategy.rs`.
  - Milestone: M6.
- [x] E03 Implement quant field generation and quantization pipeline.
  - Files: `jxl/src/encode/vardct.rs` (libjxl-calibrated global_scale/quant_lf/base_raw_quant
    + pixel-domain HF activity adaptive quant with continuous modulation and per-tile CfL).
  - Remaining gap: butteraugli-based perceptual masking for truly adaptive allocation.
  - Milestone: M6, M7.
- [x] E04 Implement LF/HF coefficient tokenization and entropy coding.
  - Files: `jxl/src/encode/vardct.rs`.
  - Milestone: M6.
- [~] E05 Add quality tuning heuristics and effort tiers.
  - Files: `jxl/src/encode/vardct.rs`, `encode/effort.rs`.
  - Milestone: M7.

### F. Advanced format parity

- [ ] F01 Progressive passes.
  - Files: new `jxl/src/encode/progressive.rs`.
  - Milestone: M8.
- [ ] F02 Animation frame sequence encode with references and blending.
  - Files: new `jxl/src/encode/animation.rs`.
  - Milestone: M8.
- [ ] F03 Extra channel parity and per-channel controls.
  - Files: encoder headers + input pipeline.
  - Milestone: M8.
- [ ] F04 Patches/splines/noise tool encoding.
  - Files: new `jxl/src/encode/tools/*.rs`.
  - Milestone: M8.
- [ ] F05 JPEG reconstruction path (if accepted in scope).
  - Files: new `jxl/src/encode/jpeg_reconstruction.rs`.
  - Milestone: M8.

### G. Performance and determinism parity

- [~] G00 SIMD acceleration plan (deferred): keep scalar/reference math while
  algorithmic parity is still moving, then SIMD-accelerate hot paths
  (`encode/vardct` forward transforms, quantization loops, token scans) once
  transform/entropy decisions are stable.
- [ ] G01 Deterministic parallel scheduler with fixed ordering.
  - Files: new `jxl/src/encode/scheduler.rs`.
  - Milestone: M5, M7.
- [ ] G02 Effort presets mapped to predictable speed/quality behavior.
  - Files: `jxl/src/encode/effort.rs`, options.
  - Milestone: M7.
- [ ] G03 Benchmark dashboard for speed/ratio/quality vs cjxl baselines.
  - Files: `bench/` + `ci/` harness.
  - Milestone: M7.

### H. Conformance, diff testing, and hardening

- [~] H01 Differential harness: jxl-rs encode -> djxl decode validation on corpus.
  - Current: repeated manual corpus checks are in use; formal harness/CI integration still pending.
  - Files: `tools/encoder_diff/` and CI scripts.
  - Milestone: M9.
- [~] H02 Differential harness: compare rate/distortion vs libjxl (matched settings).
  - Current: ad-hoc bench/corpus comparisons are used during iteration; standardized harness pending.
  - Files: `bench/` + docs.
  - Milestone: M7, M9.
- [ ] H03 Expanded fuzzing for encode paths and encode/decode loops.
  - Files: `jxl/fuzz/fuzz_targets/*`.
  - Milestone: M9.
- [ ] H04 Determinism tests across seeds/platforms/toolchains for fixed inputs/options.
  - Files: `ci/` + test modules.
  - Milestone: M0, M9.

## Integrator readiness checklist (memory-safe-by-default promise)

- [ ] R01 Safe API first: no `unsafe` required by users for common encode operations.
- [ ] R02 Deterministic outputs for fixed input/options documented and tested.
- [ ] R03 Clear resource limits (memory budget, thread budget, max image dimensions).
- [ ] R04 Panic-free error handling for user-controlled inputs/options.
- [ ] R05 Stable long-term API with migration notes.

## Current branch anchors (already in place)

- Encoder API bootstrap: `jxl/src/encode/encoder.rs`
- Basic options type: `jxl/src/encode/options.rs`
- Bit writer and encodings: `jxl/src/encode/bit_writer.rs`, `jxl/src/encode/encodings.rs`
- Header writer bootstrap: `jxl/src/encode/headers.rs`
- Container wrapper bootstrap: `jxl/src/encode/container.rs`
- Entropy bootstrap: `jxl/src/encode/entropy/{context_map,huffman,histograms,hybrid_uint}.rs`
- Modular bootstrap: `jxl/src/encode/modular.rs`
- CLI bootstrap entrypoint: `jxl_cli/src/bin/jxle.rs`

## Recommended immediate next cut (to move from bootstrap to true encoder)

1. Implement A01 + D01 together (real pixel ingestion -> modular residuals).
2. Implement C02 + C04 for real histogram-driven Huffman in modular path.
3. Keep bitstream shape minimal, but make payload data-dependent and lossless for u8 RGB first.
4. Add corpus-level encode/decode equality tests for that first true path.

This yields the first production-meaningful parity step while keeping the pure Rust safety goal intact.
