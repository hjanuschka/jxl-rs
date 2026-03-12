// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG reconstruction scaffolding.
//!
//! This module is intentionally minimal for now.
//! It provides the public data container used by upcoming `jbrd` parsing work.

use crate::error::{Error, Result};

/// Parsed JPEG reconstruction payload from a `jbrd` box.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct JpegReconstructionData {
    /// Raw `jbrd` payload bytes.
    pub raw: Vec<u8>,
}

impl JpegReconstructionData {
    /// Parse raw `jbrd` payload bytes.
    pub fn parse(payload: &[u8]) -> Result<Self> {
        if payload.is_empty() {
            return Err(Error::InvalidBox);
        }
        Ok(Self {
            raw: payload.to_vec(),
        })
    }
}
