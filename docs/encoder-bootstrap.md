# Encoder bootstrap status (local)

Current local `jxl-encoder` branch includes an early pure-Rust encoder bootstrap.

## Implemented

- `jxl::encode::BitWriter` with roundtrip tests against `BitReader`
- Minimal container writer (`JXL ` signature + `ftyp` + `jxlc`)
- U32/i32 encoding helpers for JPEG XL field coders
- Minimal codestream header emission (parses to `WithImageInfo`)
- Minimal single-frame metadata + TOC emission (parses to `WithFrameInfo`)
- Minimal decodable modular image stream for small images (`<= 256x256`)
- `jxle` CLI helper binary for generating bootstrap streams
- Pure-Rust dependency guard script: `tools/check_encoder_pure_rust.py`

## What this does NOT do yet

- No pixel section payload encoding yet
- No lossless modular image coding yet
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
```
