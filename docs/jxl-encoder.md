# JXL-rs Pure Rust Encoder Plan

## Mission

Build a **100% pure Rust** JPEG XL encoder inside jxl-rs, with a staged path from "valid files" to "competitive with cjxl".

## libjxl parity tracker

Detailed subsystem parity checklist is tracked in:

- `docs/encoder-libjxl-parity-checklist.md`

## Non-negotiable constraints

- [ ] No C or C++ FFI in the encoder path.
- [ ] No wrapping or shelling out to `cjxl` at runtime.
- [ ] Encoder dependency graph is pure Rust (no `*-sys` crates, no `links`-based native libs) for encoder builds.
- [ ] Encoder code path is deterministic across runs for the same options and input.
- [ ] Output is spec conformant and decodable by both `djxl` and `jxl-rs`.

---

## Scope strategy (do not try to do everything at once)

### Stage A (MVP)

- Still images only
- Modular lossless first
- RGB/Gray + optional alpha
- Bare codestream + container with `jxlc`

### Stage B (usable lossy)

- VarDCT baseline
- sRGB and linear-light workflows
- Single frame, no progressive passes initially

### Stage C (competitive)

- Advanced psychovisual heuristics
- Effort tiers and parallel tuning
- Progressive, animation, advanced features

---

## Proposed architecture in this repo

- [ ] Add `jxl/src/encode/` module tree (or `jxl_encode` crate if isolation is cleaner).
- [ ] Add `JxlEncoder` high-level API in `jxl::api` mirroring decoder ergonomics.
- [ ] Keep shared format structs in `headers/*` and add serialization paths (not decode-only).
- [ ] Add `BitWriter` symmetric to current `BitReader`.
- [ ] Add `container::write` symmetric to `container::parse`.
- [ ] Add entropy encoders mirroring `entropy_coding::decode` structure.

Suggested module layout:

- `encode/api.rs`
- `encode/options.rs`
- `encode/bit_writer.rs`
- `encode/container.rs`
- `encode/headers.rs`
- `encode/entropy/{hybrid_uint.rs, huffman.rs, ans.rs, context_map.rs}`
- `encode/modular/*`
- `encode/vardct/*`

---

## Milestone plan (checkable)

## M0 - Groundwork and guardrails

- [ ] Write `docs/encoder-rfc.md` with exact feature scope for v0/v1/v2.
- [ ] Decide crate/module strategy (`jxl::encode` vs separate `jxl_encode` crate).
- [ ] Add Cargo feature flags (example: `encoder`, `encoder-vardct`, `encoder-advanced`).
- [ ] Add CI job to verify encoder feature graph stays pure Rust.
- [ ] Add CI job for deterministic output snapshot tests.

Exit criteria:

- [ ] Team agrees on staged scope and API sketch.
- [ ] CI fails if native dependencies enter encoder feature path.

---

## M1 - Bitstream write foundation

- [ ] Implement `BitWriter` with bit packing, byte alignment, and bounds-safe growth.
- [ ] Implement "write then read" property tests against `BitReader`.
- [ ] Add writer helpers for U32 coders and small enums used by headers.
- [ ] Add byte boundary and padding validation tests.
- [ ] Add fuzz target for writer/readback roundtrip.

Exit criteria:

- [ ] Randomized tests: values written with `BitWriter` roundtrip exactly through `BitReader`.
- [ ] No panics or UB under fuzzing.

---

## M2 - Header and container serialization

- [ ] Add serialization traits parallel to `UnconditionalCoder` decode traits.
- [ ] Implement serialization for signature, size, metadata, frame header essentials.
- [ ] Implement container box writing (`jxlc`, optional basic `jxlp` later).
- [ ] Emit minimal valid codestream with a trivial frame payload stub.
- [ ] Add golden byte tests for known tiny examples.

Exit criteria:

- [ ] Generated files pass `djxl --info` and `jxl-rs --info`.
- [ ] Bare codestream and container output both validate.

---

## M3 - Entropy encoder stack

- [ ] Implement `HybridUint` encoding path.
- [ ] Implement Huffman table builder + writer.
- [ ] Implement ANS histogram builder + bitstream writer.
- [ ] Implement context map writer.
- [ ] Add property tests: decode(encode(symbols)) identity for each entropy backend.
- [ ] Add differential tests against small known streams.

Exit criteria:

- [ ] Entropy unit tests pass with randomized symbol streams.
- [ ] Encoded entropy streams decode correctly through existing Rust decoder path.

---

## M4 - Modular lossless MVP (first real encoder)

- [ ] Add input normalization API (`u8`, `u16`, `f32` accepted as explicit formats).
- [ ] Implement simple modular predictor mode (start with fixed predictor, then selectable).
- [ ] Implement modular tree emission (start with trivial tree).
- [ ] Encode one still frame with RGB/Gray, optional alpha.
- [ ] Support lossless mode end-to-end.
- [ ] Add CLI encoding command (example: `jxle input.png output.jxl`).
- [ ] Add roundtrip tests over corpus: encode -> decode -> pixel equality for lossless.

Exit criteria:

- [ ] At least 95% of test corpus encodes and roundtrips losslessly.
- [ ] All encoded files decode in `djxl` and `jxl-rs`.

---

## M5 - Modular quality and performance pass

- [ ] Implement better predictor selection heuristics.
- [ ] Implement modular transforms (palette/squeeze where beneficial).
- [ ] Add near-lossless controls.
- [ ] Add tiling/group parallelism with deterministic combine order.
- [ ] Add encode-time memory budget option.

Exit criteria:

- [ ] Lossless size is within agreed target versus PNG and cjxl-lossless baselines.
- [ ] Encode speed is stable and does not regress on benchmark corpus.

---

## M6 - VarDCT lossy baseline

- [ ] Add forward color pipeline needed for lossy path (focus on sRGB + linear first).
- [ ] Add forward transforms in `jxl_transforms` (DCT variants used by VarDCT path).
- [ ] Implement quant field generation and quantization.
- [ ] Implement block strategy selection (start simple, then adaptive).
- [ ] Implement LF/HF group tokenization and entropy coding for coefficients.
- [ ] Emit decodable single-frame lossy streams.

Exit criteria:

- [ ] Lossy outputs decode correctly in `djxl` and `jxl-rs` across corpus.
- [ ] Objective quality threshold met at selected distance settings.

---

## M7 - "Best-class" optimization track

- [ ] Port or reimplement advanced libjxl heuristics in idiomatic Rust.
- [ ] Add adaptive quantization and perceptual masking improvements.
- [ ] Add effort presets with clear speed/size tradeoff.
- [ ] Add multithread scaling for large images.
- [ ] Build regression dashboard for size, quality, and speed.

Exit criteria:

- [ ] At target quality, file size is within agreed delta of `cjxl` on representative corpus.
- [ ] Effort presets show monotonic behavior (higher effort -> better compression, slower encode).

---

## M8 - Advanced format features

- [ ] Progressive passes.
- [ ] Animation and frame references.
- [ ] Full extra channel handling parity.
- [ ] Container extras (metadata boxes policy and passthrough rules).
- [ ] Optional JPEG reconstruction support (if in scope).

Exit criteria:

- [ ] Feature matrix documented and validated with dedicated tests.

---

## M9 - Stabilization and release

- [ ] Public encoder API docs and examples.
- [ ] Fuzzing coverage for encode path and encode/decode loops.
- [ ] Semver policy and compatibility guarantees.
- [ ] Release checklist and changelog.

Exit criteria:

- [ ] Encoder shipped in stable release with documented limitations.

---

## Pure Rust compliance plan

- [ ] Define "pure Rust" precisely for CI enforcement (no native linking in encoder graph).
- [ ] Add script in `tools/` that inspects `cargo metadata` for encoder features and fails on native `links` crates.
- [ ] Ensure encoder does not depend on `jxl_cms`/`lcms2` path.
- [ ] Prefer Rust-native image I/O crates for CLI integration.
- [ ] Keep any optional native tools out of default CI gates for encoder correctness.

---

## Differential testing strategy

- [ ] Build corpus buckets: photographic, synthetic, alpha-heavy, HDR-ish, tiny images, huge images.
- [ ] For each sample, run:
  - [ ] `encode(jxl-rs) -> decode(jxl-rs)`
  - [ ] `encode(jxl-rs) -> decode(djxl)`
  - [ ] (later) quality comparison against `cjxl` at matched settings
- [ ] Add byte-level determinism snapshot tests for fixed seeds/options.
- [ ] Add property tests for metadata preservation behavior.

---

## Suggested first 8 PRs (small, reviewable)

1. [ ] PR1: Encoder RFC + feature flags + empty `JxlEncoder` API skeleton.
2. [ ] PR2: `BitWriter` + property tests.
3. [ ] PR3: Container/header writing for minimal stream.
4. [ ] PR4: Entropy encode primitives (HybridUint/Huffman/ANS core).
5. [ ] PR5: Modular lossless single-frame RGB path.
6. [ ] PR6: CLI command for encoding PNG -> JXL.
7. [ ] PR7: Predictor and transform heuristics for modular compression gains.
8. [ ] PR8: VarDCT baseline (single frame, basic options).

---

## Key risks and mitigations

- [ ] Risk: Full cjxl parity is very large scope.
  - Mitigation: lock phased scope and ship useful milestones early.
- [ ] Risk: Pure Rust color management complexity.
  - Mitigation: start with well-defined color inputs, add broader profile support incrementally.
- [ ] Risk: Entropy coding bugs are hard to debug.
  - Mitigation: strict property tests + differential decode checks + fuzzing.
- [ ] Risk: Performance regression from naive ports.
  - Mitigation: benchmark every milestone; optimize hot loops after correctness is stable.

---

## Definition of success

- [ ] Users can encode PNG/RAW-style buffers to valid JXL with a native Rust API.
- [ ] Lossless path is production-usable first.
- [ ] Lossy path reaches competitive size/quality and speed over time.
- [ ] Entire encoder pipeline remains pure Rust.
