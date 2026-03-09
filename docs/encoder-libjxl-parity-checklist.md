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

- `[x]` Writer foundation: bit writer, u32/i32 helpers, TOC writer.
- `[~]` Minimal headers/container writing.
- `[~]` Minimal entropy helpers (HybridUint config, fixed/simple Huffman/context map writer primitives).
- `[~]` Minimal modular decodable stream generation (constant tree leaf, synthetic output).
- `[ ]` Real image-to-bitstream encoding (no raw pixel ingestion pipeline yet).
- `[ ]` Full entropy modeling and ANS encoding.
- `[ ]` VarDCT lossy encoding.
- `[ ]` Advanced format features (progressive, animation, metadata boxes, JPEG reconstruction).

## libjxl subsystem parity map

This maps libjxl encoder areas to jxl-rs files and milestone/issue buckets from `docs/jxl-encoder-issue-breakdown.md`.

| ID | libjxl subsystem | Status | jxl-rs current files | jxl-rs target files | Milestone mapping |
|---|---|---|---|---|---|
| P01 | Bit writing primitives (`enc_bit_writer`) | [x] | `jxl/src/encode/bit_writer.rs`, `jxl/src/encode/encodings.rs` | keep current | M1 |
| P02 | Codestream header field serialization (`enc_fields`, `enc_file`) | [~] | `jxl/src/encode/headers.rs` | split into `headers/file_header.rs`, `headers/frame_header.rs`, `headers/image_metadata.rs` under `encode/` | M2 |
| P03 | Container writing (`enc_file` container side, boxes) | [~] | `jxl/src/encode/container.rs` | add `encode/container_boxes.rs` for Exif/XMP/JUMBF/jxlp policy | M2, M8 |
| P04 | Public encoder API surface (`JxlEncoder`, frame settings) | [~] | `jxl/src/encode/encoder.rs`, `jxl/src/encode/options.rs`, `jxl/src/api/mod.rs` | add `jxl/src/encode/api.rs`, per-frame settings types | M4, M9 |
| P05 | Input pixel ingestion and normalization | [ ] | none (bootstrap only) | `jxl/src/encode/input.rs`, `jxl/src/encode/color_pipeline.rs` | M4 |
| P06 | Context map writer and optimization (`enc_entropy_coder`) | [~] | `jxl/src/encode/entropy/context_map.rs` | add non-simple context map and optimization pass | M3, M5 |
| P07 | Huffman table build from symbol stats | [~] | `jxl/src/encode/entropy/huffman.rs` (fixed-symbol only) | add histogram-driven table builder | M3 |
| P08 | HybridUint full usage in entropy paths | [~] | `jxl/src/encode/entropy/hybrid_uint.rs` | wire into real modular/vardct token paths | M3, M4, M6 |
| P09 | ANS histogram + stream writing (`enc_ans`) | [ ] | none | `jxl/src/encode/entropy/ans.rs` | M3 |
| P10 | Modular tree building from image stats (`enc_modular_tree`) | [~] | `jxl/src/encode/modular.rs` (single-leaf constants) | `jxl/src/encode/modular/tree_builder.rs` | M4, M5 |
| P11 | Modular transforms (palette/squeeze/rct) | [ ] | none | `jxl/src/encode/modular/transforms.rs` | M5 |
| P12 | Real modular lossless frame encoding (`enc_modular`) | [ ] | minimal synthetic sections only | `jxl/src/encode/modular/frame.rs`, `jxl/src/encode/modular/channel.rs` | M4 |
| P13 | Near-lossless controls | [ ] | none | add to modular pipeline and `JxlEncoderOptions` | M5 |
| P14 | Fast lossless heuristics (`enc_fast_lossless`) | [ ] | none | `jxl/src/encode/modular/fast_lossless.rs` | M5, M7 |
| P15 | VarDCT forward color path (`enc_xyb`, color transforms) | [ ] | none | `jxl/src/encode/vardct/color.rs` | M6 |
| P16 | VarDCT block strategy and transforms (`enc_frame`, heuristics) | [ ] | none | `jxl/src/encode/vardct/block_strategy.rs`, `.../transform.rs` | M6 |
| P17 | Quant field generation and tuning (`enc_quant_weights`) | [ ] | none | `jxl/src/encode/vardct/quant.rs` | M6, M7 |
| P18 | Coefficient tokenization + entropy coding | [ ] | none | `jxl/src/encode/vardct/tokens.rs`, `.../entropy.rs` | M6 |
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

- [ ] A01 Add `JxlEncoder::encode_image(...)` for raw in-memory buffers (RGB/Gray/RGBA, u8/u16/f32).
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

- [ ] C01 Implement ANS writer end to end with decoder roundtrip tests.
  - Files: new `jxl/src/encode/entropy/ans.rs`.
  - Milestone: M3.
- [ ] C02 Implement histogram building from symbol distributions.
  - Files: `jxl/src/encode/entropy/histograms.rs`.
  - Milestone: M3.
- [ ] C03 Implement histogram clustering and context map optimization.
  - Files: `jxl/src/encode/entropy/{histograms,context_map}.rs`.
  - Milestone: M3, M5.
- [ ] C04 Implement full Huffman table construction path (not fixed-symbol-only).
  - Files: `jxl/src/encode/entropy/huffman.rs`.
  - Milestone: M3.
- [ ] C05 Hook HybridUint configs to real token distributions and tune choices.
  - Files: `jxl/src/encode/entropy/hybrid_uint.rs` + callers.
  - Milestone: M3, M4, M6.

### D. Modular lossless parity

- [ ] D01 Build real channel residual pipeline from input image data.
  - Files: new `jxl/src/encode/modular/channel.rs`.
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

- [ ] E01 Implement forward lossy color pipeline (sRGB/linear and XYB path as needed).
  - Files: new `jxl/src/encode/vardct/color.rs`.
  - Milestone: M6.
- [ ] E02 Implement block strategy selection and transform dispatch.
  - Files: new `jxl/src/encode/vardct/block_strategy.rs`.
  - Milestone: M6.
- [ ] E03 Implement quant field generation and quantization pipeline.
  - Files: new `jxl/src/encode/vardct/quant.rs`.
  - Milestone: M6, M7.
- [ ] E04 Implement LF/HF coefficient tokenization and entropy coding.
  - Files: new `jxl/src/encode/vardct/{tokens,entropy}.rs`.
  - Milestone: M6.
- [ ] E05 Add quality tuning heuristics and effort tiers.
  - Files: `jxl/src/encode/vardct/heuristics.rs`, `encode/effort.rs`.
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

- [ ] H01 Differential harness: jxl-rs encode -> djxl decode validation on corpus.
  - Files: `tools/encoder_diff/` and CI scripts.
  - Milestone: M9.
- [ ] H02 Differential harness: compare rate/distortion vs libjxl (matched settings).
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
