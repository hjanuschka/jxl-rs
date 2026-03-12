# jxl-rs encoder API snapshot

This document summarizes the current high-level encoder API surface.

## Stable-in-practice surface (current)

- `encode::JxlEncoder`
- `encode::JxlEncoderOptions`
- `encode::JxlEncoderMode`
- `encode::JxlEncoderImageData`

Key methods:

- `encode_image_codestream(...)`
- `encode_image(...)`
- `encode_image_with_callback(...)`
- `encode_image_with_callback_chunked(...)`

## Current compatibility intent

- Keep source compatibility for common encode calls whenever possible.
- Additive expansion uses `#[non_exhaustive]` option types.
- Behavioral changes are documented in parity checklist updates.
- Resource guardrails are part of the high-level API (`max_width`, `max_height`, `max_pixels`, `max_output_bytes`, `threads`).

## Current limitations

- High-level per-frame control is still a bootstrap layer.
- Metadata box support is currently limited to raw Exif/XML/JUMBF payload insertion.
- Some advanced format controls are currently only exposed through lower-level helpers.
