// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Encoding-side scaffolding for patches/splines/noise tools.

/// Placeholder patch tool parameters.
#[derive(Clone, Debug, Default)]
pub struct PatchToolParams {}

/// Placeholder spline tool parameters.
#[derive(Clone, Debug, Default)]
pub struct SplineToolParams {}

/// Placeholder noise tool parameters.
#[derive(Clone, Debug, Default)]
pub struct NoiseToolParams {}

/// Placeholder tools bundle.
#[derive(Clone, Debug, Default)]
pub struct EncoderToolsConfig {
    pub patches: Option<PatchToolParams>,
    pub splines: Option<SplineToolParams>,
    pub noise: Option<NoiseToolParams>,
}
