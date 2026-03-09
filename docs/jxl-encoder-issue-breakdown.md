# JXL-rs Encoder - GitHub issue breakdown

This turns the roadmap in `jxl-encoder.md` into a trackable issue plan.

Detailed parity gaps vs libjxl are tracked in:

- `docs/encoder-libjxl-parity-checklist.md`

## Quick setup

- Create 1 tracking epic issue + 10 milestone issues (M0..M9).
- Add labels:
  - `area/encoder`
  - `type/epic`
  - `type/milestone`
  - `kind/infra`, `kind/correctness`, `kind/perf`, `kind/api`
  - `stage/m0` ... `stage/m9`
  - `priority/p0`, `priority/p1`, `priority/p2`
- Add all milestone issues to one project board with status columns: `Todo`, `In Progress`, `Blocked`, `Review`, `Done`.

## Dependency graph

| ID | Title | Depends on |
|---|---|---|
| EPIC | [encoder] Pure Rust JPEG XL encoder tracking issue | none |
| M0 | [encoder][M0] RFC + pure Rust guardrails + CI gates | none |
| M1 | [encoder][M1] BitWriter foundation + property tests | M0 |
| M2 | [encoder][M2] Header + container serialization | M1 |
| M3 | [encoder][M3] Entropy encoder stack (HybridUint/Huffman/ANS/context map) | M1 |
| M4 | [encoder][M4] Modular lossless MVP (single frame) | M2, M3 |
| M5 | [encoder][M5] Modular compression quality + perf pass | M4 |
| M6 | [encoder][M6] VarDCT lossy baseline | M2, M3, M4 |
| M7 | [encoder][M7] Competitive optimization track | M6 |
| M8 | [encoder][M8] Advanced format features (progressive/animation/etc) | M6 |
| M9 | [encoder][M9] Stabilization + release | M5, M7, M8 |

---

## EPIC issue (optional but strongly recommended)

### Title
`[encoder] Pure Rust JPEG XL encoder tracking issue`

### Body
```md
## Goal
Ship a 100% pure Rust JPEG XL encoder in jxl-rs.

## Non-negotiables
- [ ] No C/C++ FFI in encoder path
- [ ] No shelling out to cjxl
- [ ] Deterministic outputs for fixed input+options
- [ ] Outputs decode in djxl and jxl-rs

## Milestones
- [ ] M0 RFC + guardrails
- [ ] M1 BitWriter
- [ ] M2 Header/container write
- [ ] M3 Entropy stack
- [ ] M4 Modular lossless MVP
- [ ] M5 Modular quality/perf
- [ ] M6 VarDCT baseline
- [ ] M7 Competitive optimization
- [ ] M8 Advanced features
- [ ] M9 Stabilization/release

## Dependency policy
A milestone can start in parallel only if all blocking dependencies are complete.
```

---

## Milestone issues (copy/paste ready)

## M0 issue

### Title
`[encoder][M0] RFC + pure Rust guardrails + CI gates`

### Labels
`area/encoder`, `type/milestone`, `stage/m0`, `kind/infra`, `priority/p0`

### Body
```md
## Objective
Lock scope and enforce pure Rust constraints before implementation.

## Deliverables
- [ ] Add `docs/encoder-rfc.md` with v0/v1/v2 scope
- [ ] Decide module strategy (`jxl::encode` vs separate crate)
- [ ] Add Cargo feature flags for encoder stages
- [ ] Add CI check: encoder dependency graph must be pure Rust
- [ ] Add CI check: deterministic output snapshots for fixed seed/options

## Acceptance criteria
- [ ] Team signoff on staged scope and API direction
- [ ] CI fails if native deps enter encoder feature path
- [ ] CI deterministic snapshot check runs green

## Out of scope
- Actual encode output implementation
```

---

## M1 issue

### Title
`[encoder][M1] BitWriter foundation + property tests`

### Labels
`area/encoder`, `type/milestone`, `stage/m1`, `kind/correctness`, `priority/p0`

### Body
```md
## Objective
Build robust write-side bitstream primitives that roundtrip with existing BitReader.

## Deliverables
- [ ] Implement `BitWriter` with safe growth and bit packing
- [ ] Implement alignment/padding APIs
- [ ] Implement helpers for small enum/U32 coders used by headers
- [ ] Add randomized write->read property tests against `BitReader`
- [ ] Add fuzz target for writer/readback roundtrip

## Acceptance criteria
- [ ] Roundtrip tests pass for randomized vectors
- [ ] No panic in fuzz target over agreed run budget
- [ ] API docs added for `BitWriter`

## Dependencies
- M0 complete
```

---

## M2 issue

### Title
`[encoder][M2] Header + container serialization`

### Labels
`area/encoder`, `type/milestone`, `stage/m2`, `kind/correctness`, `priority/p0`

### Body
```md
## Objective
Write valid JPEG XL headers and container structures.

## Deliverables
- [ ] Add serialization traits parallel to decode coders
- [ ] Serialize signature, size, metadata, minimal frame header
- [ ] Implement container writer for bare codestream + `jxlc`
- [ ] Emit minimal valid codestream with stub frame payload
- [ ] Add golden tests for tiny examples

## Acceptance criteria
- [ ] `djxl --info` succeeds on generated files
- [ ] `jxl-rs --info` succeeds on generated files
- [ ] Bare codestream and container outputs both validated in tests

## Dependencies
- M1 complete
```

---

## M3 issue

### Title
`[encoder][M3] Entropy encoder stack (HybridUint/Huffman/ANS/context map)`

### Labels
`area/encoder`, `type/milestone`, `stage/m3`, `kind/correctness`, `priority/p0`

### Body
```md
## Objective
Implement core entropy coding needed by modular and VarDCT paths.

## Deliverables
- [ ] Implement HybridUint encoding
- [ ] Implement Huffman table builder and stream writer
- [ ] Implement ANS histogram build and stream writer
- [ ] Implement context map writer
- [ ] Add property tests: decode(encode(symbols)) == symbols
- [ ] Add differential tests against known small streams

## Acceptance criteria
- [ ] Randomized entropy tests pass
- [ ] Existing Rust decode path can decode encoded entropy streams
- [ ] Error handling is explicit and panic-free for malformed inputs

## Dependencies
- M1 complete
```

---

## M4 issue

### Title
`[encoder][M4] Modular lossless MVP (single frame)`

### Labels
`area/encoder`, `type/milestone`, `stage/m4`, `kind/api`, `kind/correctness`, `priority/p0`

### Body
```md
## Objective
Ship first usable encoder: still-image modular lossless.

## Deliverables
- [ ] Input normalization API (`u8`, `u16`, `f32` explicit formats)
- [ ] Simple modular predictor mode (fixed first, selectable second)
- [ ] Trivial modular tree emission
- [ ] Encode one still frame RGB/Gray with optional alpha
- [ ] Support lossless end-to-end
- [ ] Add CLI encoding command (`jxle input.png output.jxl` or equivalent)
- [ ] Add corpus roundtrip tests: encode->decode pixel equality

## Acceptance criteria
- [ ] >=95% corpus encodes and roundtrips losslessly
- [ ] Encoded files decode in both `djxl` and `jxl-rs`
- [ ] Public API is documented for MVP usage

## Dependencies
- M2 complete
- M3 complete
```

---

## M5 issue

### Title
`[encoder][M5] Modular compression quality + perf pass`

### Labels
`area/encoder`, `type/milestone`, `stage/m5`, `kind/perf`, `priority/p1`

### Body
```md
## Objective
Improve lossless size and speed while preserving determinism.

## Deliverables
- [ ] Better predictor selection heuristics
- [ ] Modular transforms where useful (palette/squeeze)
- [ ] Near-lossless controls
- [ ] Deterministic parallelism for tiling/groups
- [ ] Memory budget option for encoding
- [ ] Benchmark harness for compression ratio + throughput

## Acceptance criteria
- [ ] Size target vs PNG and cjxl-lossless baseline is met
- [ ] No benchmark regression beyond agreed tolerance
- [ ] Determinism checks still pass

## Dependencies
- M4 complete
```

---

## M6 issue

### Title
`[encoder][M6] VarDCT lossy baseline`

### Labels
`area/encoder`, `type/milestone`, `stage/m6`, `kind/correctness`, `kind/perf`, `priority/p0`

### Body
```md
## Objective
Deliver first decodable lossy VarDCT encoder path.

## Deliverables
- [ ] Forward color pipeline for lossy path (sRGB + linear first)
- [ ] Forward transforms in `jxl_transforms` as needed
- [ ] Quant field generation and quantization
- [ ] Initial block strategy selection (simple baseline)
- [ ] LF/HF tokenization and entropy coding for coefficients
- [ ] Emit decodable single-frame lossy streams

## Acceptance criteria
- [ ] Outputs decode in `djxl` and `jxl-rs` on corpus
- [ ] Objective quality threshold achieved at chosen distances
- [ ] CLI/API options documented for baseline lossy encode

## Dependencies
- M2 complete
- M3 complete
- M4 complete
```

---

## M7 issue

### Title
`[encoder][M7] Competitive optimization track (size/quality/speed)`

### Labels
`area/encoder`, `type/milestone`, `stage/m7`, `kind/perf`, `priority/p1`

### Body
```md
## Objective
Close the gap to cjxl with better heuristics and tuning.

## Deliverables
- [ ] Reimplement advanced heuristics in idiomatic Rust
- [ ] Adaptive quantization and masking improvements
- [ ] Effort presets with predictable tradeoffs
- [ ] Better multithread scaling for large images
- [ ] Regression dashboard for size/quality/speed

## Acceptance criteria
- [ ] At target quality, size is within agreed delta vs cjxl corpus baseline
- [ ] Effort presets are monotonic (higher effort => better compression, slower speed)
- [ ] Benchmark dashboard runs in CI or scheduled jobs

## Dependencies
- M6 complete
```

---

## M8 issue

### Title
`[encoder][M8] Advanced format features (progressive, animation, extras)`

### Labels
`area/encoder`, `type/milestone`, `stage/m8`, `kind/api`, `kind/correctness`, `priority/p2`

### Body
```md
## Objective
Expand feature parity beyond single-frame baseline.

## Deliverables
- [ ] Progressive pass emission
- [ ] Animation + frame references
- [ ] Full extra channel handling parity
- [ ] Container extras policy (metadata passthrough/write)
- [ ] Optional JPEG reconstruction support decision + implementation (if in scope)

## Acceptance criteria
- [ ] Feature matrix documented
- [ ] Dedicated tests per advanced feature
- [ ] Interop checks pass with djxl across advanced samples

## Dependencies
- M6 complete
```

---

## M9 issue

### Title
`[encoder][M9] Stabilization + release`

### Labels
`area/encoder`, `type/milestone`, `stage/m9`, `kind/infra`, `kind/api`, `priority/p0`

### Body
```md
## Objective
Harden, document, and release the encoder.

## Deliverables
- [ ] Public API docs and examples
- [ ] Encode path fuzzing + encode/decode loop fuzzing
- [ ] Semver and compatibility policy
- [ ] Release checklist and changelog
- [ ] Final known-limitations document

## Acceptance criteria
- [ ] Stable release published with documented limitations
- [ ] CI and fuzzing thresholds satisfied
- [ ] API docs are complete for common workflows

## Dependencies
- M5 complete
- M7 complete
- M8 complete
```

---

## Recommended creation order

1. EPIC
2. M0
3. M1
4. M2, M3 (parallel)
5. M4
6. M5, M6 (partially parallel after M4)
7. M7, M8
8. M9

## Recommended first assignment split

- Contributor A: M0 + M1
- Contributor B: M2
- Contributor C: M3
- Contributor D: M4 API + CLI integration

This keeps early work parallel while keeping hard blockers clear.
