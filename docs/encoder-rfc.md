# RFC: Pure Rust JPEG XL encoder for jxl-rs

Status: draft

## Summary

This RFC defines the initial encoder scope for jxl-rs and the rollout strategy.

Primary goals:

- Ship a 100% pure Rust encoder path.
- Keep decoder stability while encoder support is added incrementally.
- Land small, reviewable milestones with hard acceptance criteria.

## Non-negotiable constraints

- No C/C++ FFI in encoder code paths.
- No runtime delegation to `cjxl`.
- Deterministic output for fixed input and options.
- Interop: output must decode in `djxl` and in `jxl-rs`.

## Module strategy

Initial implementation is in `jxl::encode` behind feature flags.

Feature flags:

- `encoder`
- `encoder-vardct` (implies `encoder`)
- `encoder-advanced` (implies `encoder-vardct`)

This allows incremental landing and CI gating without impacting current decoder users.

## Versioned scope

### v0

- Writer infrastructure (bit writer, basic serialization primitives)
- Signature/header/container writing bootstrap
- Internal testing only

### v1

- Modular lossless still-image encoding
- RGB/gray with optional alpha
- Basic CLI entry point for encode path

### v2

- VarDCT lossy baseline
- Effort controls
- Expanded compatibility and performance targets

## Testing and validation

Required from early milestones:

- Property tests for write/read roundtrip
- Determinism snapshots for fixed corpus/options
- Differential decode checks in `djxl` and `jxl-rs`

## Pure Rust dependency enforcement

A repository tool (`tools/check_encoder_pure_rust.py`) validates that the encoder feature graph does not pull in crates with native `links` metadata.

## Out of scope for this RFC

- Full libjxl feature parity in one step
- Animation/progressive in the first encoder releases
- Bitstream-level parity with `cjxl`
