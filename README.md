# JPEG XL in Rust

This is a work-in-progress reimplementation of JPEG XL in Rust, aiming to be conforming, safe, and fast.

The current stable focus is decoding. Encoder work has started under the `encoder` Cargo feature.

Experimental bootstrap encoder helper:

```bash
cargo run -p jxl_cli --no-default-features --bin jxle -- out.jxl --width 320 --height 240
```

This currently emits metadata/bootstrap streams and is not yet a full image encoder.

We strive to decode all conformant JPEG XL bitstreams correctly. If you find an image that can be decoded with the reference
implementation `djxl` (from [`libjxl`](https://github.com/libjxl/libjxl)) but is decoded incorrectly or not at all by `jxl-rs`,
please report it by [opening an issue](https://github.com/libjxl/jxl-rs/issues/new).

For more information, including contributing instructions, refer to the [libjxl repository](https://github.com/libjxl/libjxl).
