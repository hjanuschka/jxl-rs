// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
}

impl Default for JxlEncoderOptions {
    fn default() -> Self {
        Self {
            lossless: true,
            effort: 7,
            container: true,
        }
    }
}
