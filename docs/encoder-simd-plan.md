# Encoder SIMD acceleration plan

This plan tracks SIMD work for the pure-Rust encoder path.

## Principles

- Keep bitstream output deterministic.
- Keep the encoder path pure Rust and memory-safe by default.
- Add SIMD only where algorithmic parity is already stable.
- Prefer a scalar reference path that remains easy to audit.

## Scope

Primary encode hotspots to accelerate:

- XYB forward color transform
- Forward transforms (DCT8/DCT16/DCT32 and rectangular variants)
- Quantization/dequantization loops
- Coefficient scan/token prepasses
- Adaptive-quantization inner loops

Out of scope for first wave:

- Changing coding decisions or heuristics
- Non-deterministic parallel reductions
- Decoder-side SIMD changes

## Rollout strategy

1. Benchmark first, optimize second.
2. Land SIMD behind explicit runtime/feature gating.
3. Keep scalar and SIMD outputs byte-identical where required.
4. Expand CI conformance before flipping defaults.

## Checkable execution list

Legend:

- [ ] not started
- [~] in progress
- [x] done

### S0 - Baseline and guardrails

- [ ] S001 Add encoder microbench crate/targets for XYB, DCT, quant, token prepasses.
- [ ] S002 Capture baseline timings on representative corpus (RGB photos, flat graphics, RGBA alpha sample).
- [ ] S003 Add benchmark report artifact in CI (summary table + trend file).
- [ ] S004 Add "scalar reference" benchmark mode pinning SIMD off.
- [ ] S005 Add determinism smoke: scalar run A vs scalar run B byte-identical.

### S1 - Portable SIMD foundations

- [ ] S101 Add SIMD capability detection and dispatch layer (no behavior change).
- [ ] S102 Add common SIMD math helpers (load/store, lane ops, clamped conversion).
- [ ] S103 Add feature-gated portable SIMD module with scalar fallback.
- [ ] S104 Add per-kernel test harness comparing SIMD vs scalar buffers.
- [ ] S105 Add fuzz target for SIMD/scalar equivalence on random small images.

### S2 - XYB acceleration

- [ ] S201 SIMD path for sRGB u8 -> linearized float conversion.
- [ ] S202 SIMD path for XYB matrix application and channel packing.
- [ ] S203 SIMD path for RGBA alpha-split compatible preprocessing.
- [ ] S204 Add exhaustive unit tests for edge values (0, 1, 254, 255).
- [ ] S205 Verify byte-identical codestream output vs scalar on corpus.
- [ ] S206 Record speedup target: >= 1.5x in XYB-heavy benchmark.

### S3 - Transform acceleration (jxl_transforms + encode/vardct integration)

- [ ] S301 SIMD DCT8 forward kernel.
- [ ] S302 SIMD DCT16 forward kernel.
- [ ] S303 SIMD rectangular DCT16x8 / DCT8x16 kernels.
- [ ] S304 SIMD DCT32 forward kernel (guarded for profitable sizes only).
- [ ] S305 Validate numerical tolerances and final quantized equivalence.
- [ ] S306 Record speedup target: >= 1.8x in transform-heavy benchmark.

### S4 - Quantization and token prepasses

- [ ] S401 SIMD quant field application loops.
- [ ] S402 SIMD dequant/weight multiply loops used in scoring.
- [ ] S403 SIMD coefficient zig-zag/order prepass helpers.
- [ ] S404 SIMD residual stats/histogram pre-accumulation where safe.
- [ ] S405 Verify no coding-decision drift at fixed settings.
- [ ] S406 Record speedup target: >= 1.3x for tokenization+quant stage.

### S5 - AQ and heuristic hot loops

- [ ] S501 SIMD masking and modulation loops from AQ path.
- [ ] S502 SIMD Laplacian/downsample inner loops.
- [ ] S503 Validate AQ field parity metrics (delta thresholds + visual checks).
- [ ] S504 Corpus-level RD regression check (no quality drop beyond threshold).

### S6 - Architecture-specific tuning

- [ ] S601 x86 tuned paths (SSE4.2/AVX2) behind runtime dispatch.
- [ ] S602 ARM NEON tuned paths behind runtime dispatch.
- [ ] S603 Add architecture matrix benchmarks (x86_64 + arm64 runners where available).
- [ ] S604 Keep portable SIMD path as shared baseline across architectures.

### S7 - CI and release gating

- [ ] S701 Add CI job: scalar vs SIMD codestream byte-equality corpus.
- [ ] S702 Add CI job: SIMD fuzz/equivalence smoke.
- [ ] S703 Add CI job: benchmark threshold checks (non-failing informational at first).
- [ ] S704 Promote stable SIMD kernels to default-on runtime dispatch.
- [ ] S705 Document user controls (`JXL_ENC_SIMD`, debug flags, benchmark mode).

### S8 - Deterministic multithreading follow-up (after SIMD)

- [ ] S801 Design deterministic per-group work scheduling.
- [ ] S802 Implement deterministic merge/reduction ordering.
- [ ] S803 Add threads=1 reference mode and threads>1 equivalence tests.
- [ ] S804 Add CI determinism checks across thread counts.

## Acceptance criteria for "SIMD wave 1 complete"

- [ ] A1 SIMD enabled by default through runtime dispatch on supported targets.
- [ ] A2 Scalar and SIMD produce byte-identical codestreams on conformance corpus.
- [ ] A3 No RD regressions outside agreed thresholds.
- [ ] A4 End-to-end encode speedup >= 1.5x on representative RGB corpus.
- [ ] A5 Docs and CI fully reflect supported SIMD behavior.

## Current status

- [x] Plan defined and expanded into executable checklist.
- [ ] Implementation in progress.
