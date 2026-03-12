// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// High-level encoder mode selection.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlEncoderMode {
    /// Modular encoding path.
    Modular,
    /// VarDCT encoding path.
    VarDct,
}

/// High-level encoder configuration.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JxlEncoderOptions {
    /// Lossless mode toggle.
    pub lossless: bool,
    /// Effort preset, where larger values typically trade speed for compression.
    pub effort: u8,
    /// Emit an ISOBMFF container (`jxlc`) instead of a bare codestream.
    pub container: bool,
    /// Encoder mode preference.
    pub mode: JxlEncoderMode,
    /// Target quality distance in milli-units (1000 = distance 1.0).
    pub distance_milli: u16,
    /// Near-lossless strength (0 = disabled).
    pub near_lossless: u8,
    /// Fast-lossless heuristic toggle.
    pub fast_lossless: bool,
    /// Maximum accepted width for encode API input.
    pub max_width: u32,
    /// Maximum accepted height for encode API input.
    pub max_height: u32,
    /// Maximum accepted total pixels for encode API input.
    pub max_pixels: u64,
    /// Optional `jxlp` chunk size when writing container output.
    /// If `None`, writes a single `jxlc` box.
    pub jxlp_chunk_size: Option<usize>,
    /// Optional hard limit for returned encoded bytes.
    pub max_output_bytes: Option<usize>,
    /// Encoder thread count. Current encoder path is deterministic and serial;
    /// only `1` is currently accepted.
    pub threads: usize,
    /// Optional Exif metadata payload written as an `Exif` box.
    pub exif: Option<Vec<u8>>,
    /// Optional XML/XMP payload written as an `xml ` box.
    pub xml: Option<Vec<u8>>,
    /// Optional JUMBF payload written as a `jumb` box.
    pub jumbf: Option<Vec<u8>>,
    /// Optional raw JPEG reconstruction payload written as a `jbrd` box.
    pub jpeg_reconstruction: Option<Vec<u8>>,
}

impl Default for JxlEncoderOptions {
    fn default() -> Self {
        Self {
            lossless: true,
            effort: 7,
            container: true,
            mode: JxlEncoderMode::Modular,
            distance_milli: 1000,
            near_lossless: 0,
            fast_lossless: false,
            max_width: 1 << 24,
            max_height: 1 << 24,
            max_pixels: 1u64 << 32,
            jxlp_chunk_size: None,
            max_output_bytes: None,
            threads: 1,
            exif: None,
            xml: None,
            jumbf: None,
            jpeg_reconstruction: None,
        }
    }
}
