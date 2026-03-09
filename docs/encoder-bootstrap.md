# Encoder bootstrap status (local)

Current local `jxl-encoder` branch includes an early pure-Rust encoder bootstrap.

## Implemented

- `jxl::encode::BitWriter` with roundtrip tests against `BitReader`
- Minimal container writer (`JXL ` signature + `ftyp` + `jxlc`)
- U32/i32 encoding helpers for JPEG XL field coders
- Minimal codestream header emission (parses to `WithImageInfo`)
- Minimal single-frame metadata + TOC emission (parses to `WithFrameInfo`)
- Minimal decodable modular image stream (black output bootstrap, including multi-group images)
- RGB8 modular payload encoding path (raw interleaved RGB8 input, group-aware)
- Generic `encode_image` API entrypoint with RGB8 interleaved/strided buffer support
- `jxle` CLI helper binary for generating bootstrap streams
- Pure-Rust dependency guard script: `tools/check_encoder_pure_rust.py`

## What this does NOT do yet

- No general-purpose pixel section payload encoding yet (current RGB8 path is single-group and bootstrap-level)
- No production-quality lossless modular coding yet
- No VarDCT lossy coding yet
- Streams with frame info are metadata-only and not renderable

## Useful commands

```bash
# Run encoder-focused tests
cargo test -p jxl --no-default-features --features encoder encode::

# Verify pure-Rust dependency policy
./tools/check_encoder_pure_rust.py --manifest-path jxl/Cargo.toml --package jxl --features encoder

# Emit a minimal container stream with image info only
cargo run -p jxl_cli --no-default-features --bin jxle -- out.jxl --width 321 --height 123

# Emit a minimal stream that also includes frame metadata + TOC
cargo run -p jxl_cli --no-default-features --bin jxle -- out.jxl --width 321 --height 123 --with-frame-info

# Emit a minimal decodable modular image stream
cargo run -p jxl_cli --no-default-features --bin jxle -- out.jxl --width 128 --height 64 --modular-image

# Same stream shape, but with a constant modular offset
cargo run -p jxl_cli --no-default-features --bin jxle -- out.jxl --width 128 --height 64 --modular-image --modular-offset 12

# Use predictor=1 (West) to generate a ramp-like stream
cargo run -p jxl_cli --no-default-features --bin jxle -- out.jxl --width 128 --height 64 --modular-image --modular-offset 1 --modular-predictor 1

# Encode raw interleaved RGB8 bytes (width*height*3)
cargo run -p jxl_cli --no-default-features --bin jxle -- out.jxl --width 128 --height 64 --raw-rgb8-input frame.rgb
```
