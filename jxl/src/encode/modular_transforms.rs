// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Modular transform scaffolding for encoder parity work.
//!
//! This module provides bootstrap types used to stage palette/squeeze/RCT
//! integration into the modular encoder path.

use crate::error::Result;

/// Placeholder modular transform plan.
#[derive(Clone, Debug, Default)]
pub struct ModularTransformPlan {
    pub use_palette: bool,
    pub use_squeeze: bool,
    pub use_rct: bool,
}

/// Build a bootstrap transform plan from image characteristics.
pub fn build_bootstrap_plan(_width: usize, _height: usize, _channels: usize) -> ModularTransformPlan {
    ModularTransformPlan::default()
}

/// Apply transform plan to signed channel data (no-op scaffolding).
pub fn apply_plan_signed(
    _plan: &ModularTransformPlan,
    _width: usize,
    _height: usize,
    _channels: usize,
    data: &[i32],
) -> Result<Vec<i32>> {
    Ok(data.to_vec())
}
