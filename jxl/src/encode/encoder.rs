// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{CODESTREAM_SIGNATURE, CONTAINER_SIGNATURE},
    error::Result,
};

use super::{BitWriter, JxlEncoderOptions, container};

/// Top-level bitstream flavor for encoder output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlEncoderBitstreamKind {
    Codestream,
    Container,
}

/// High-level encoder entry point.
pub struct JxlEncoder {
    options: JxlEncoderOptions,
}

impl Default for JxlEncoder {
    fn default() -> Self {
        Self::new(JxlEncoderOptions::default())
    }
}

impl JxlEncoder {
    pub fn new(options: JxlEncoderOptions) -> Self {
        Self { options }
    }

    pub fn options(&self) -> &JxlEncoderOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut JxlEncoderOptions {
        &mut self.options
    }

    /// Encodes a standalone JPEG XL signature blob.
    ///
    /// This is a tiny bootstrap API used while bringing up writer infrastructure.
    pub fn encode_signature(&self, kind: JxlEncoderBitstreamKind) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();
        match kind {
            JxlEncoderBitstreamKind::Codestream => {
                writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;
            }
            JxlEncoderBitstreamKind::Container => {
                writer.write_aligned_bytes(&CONTAINER_SIGNATURE)?;
            }
        }
        Ok(writer.finish())
    }

    /// Wraps a pre-encoded codestream in a minimal JXL container.
    pub fn wrap_codestream_in_container(&self, codestream: &[u8]) -> Result<Vec<u8>> {
        container::wrap_codestream(codestream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::{JxlSignatureType, ProcessingResult, check_signature},
        container::{BitstreamKind, ContainerParser},
    };

    #[test]
    fn test_encode_codestream_signature() {
        let enc = JxlEncoder::default();
        let sig = enc
            .encode_signature(JxlEncoderBitstreamKind::Codestream)
            .unwrap();
        assert_eq!(
            check_signature(&sig),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_container_signature() {
        let enc = JxlEncoder::default();
        let sig = enc
            .encode_signature(JxlEncoderBitstreamKind::Container)
            .unwrap();
        assert_eq!(
            check_signature(&sig),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
    }

    #[test]
    fn test_wrap_codestream_in_container() {
        let enc = JxlEncoder::default();
        let codestream = vec![0xFF, 0x0A, 0x01, 0x02, 0x03];
        let container = enc.wrap_codestream_in_container(&codestream).unwrap();

        let mut parser = ContainerParser::new();
        let mut out = Vec::new();
        for event in parser.process_bytes(&container) {
            match event.unwrap() {
                crate::container::ParseEvent::BitstreamKind(kind) => {
                    assert_eq!(kind, BitstreamKind::Container);
                }
                crate::container::ParseEvent::Codestream(buf) => out.extend_from_slice(buf),
            }
        }

        assert_eq!(out, codestream);
    }
}
