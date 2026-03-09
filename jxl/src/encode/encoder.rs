// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{CODESTREAM_SIGNATURE, CONTAINER_SIGNATURE},
    error::Result,
};

use super::{BitWriter, JxlEncoderOptions, container, headers};

/// Top-level bitstream flavor for encoder output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlEncoderBitstreamKind {
    Codestream,
    Container,
}

/// Supported input buffer layouts for bootstrap image encoding.
#[derive(Clone, Copy, Debug)]
pub enum JxlEncoderImageData<'a> {
    /// Interleaved RGB8 samples: `[r,g,b, r,g,b, ...]`.
    Rgb8Interleaved(&'a [u8]),
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

    /// Encodes an image buffer to a bare codestream.
    pub fn encode_image_codestream(
        &self,
        size: (u32, u32),
        image: JxlEncoderImageData<'_>,
    ) -> Result<Vec<u8>> {
        match image {
            JxlEncoderImageData::Rgb8Interleaved(rgb) => {
                headers::encode_modular_u8_rgb_image_codestream(size, rgb)
            }
        }
    }

    /// Encodes an image buffer using the container preference in options.
    pub fn encode_image(
        &self,
        size: (u32, u32),
        image: JxlEncoderImageData<'_>,
    ) -> Result<Vec<u8>> {
        let codestream = self.encode_image_codestream(size, image)?;
        if self.options.container {
            container::wrap_codestream(&codestream)
        } else {
            Ok(codestream)
        }
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

    /// Encodes a minimal header-only codestream.
    pub fn encode_minimal_codestream_header(&self, size: (u32, u32)) -> Result<Vec<u8>> {
        headers::encode_minimal_codestream_header(size)
    }

    /// Encodes a minimal single-frame codestream up to frame metadata + TOC.
    pub fn encode_minimal_single_frame_codestream(&self, size: (u32, u32)) -> Result<Vec<u8>> {
        headers::encode_minimal_single_frame_codestream(size)
    }

    /// Encodes a minimal fully decodable modular-image codestream.
    pub fn encode_minimal_modular_image_codestream(&self, size: (u32, u32)) -> Result<Vec<u8>> {
        headers::encode_minimal_modular_image_codestream(size)
    }

    /// Encodes a minimal fully decodable modular-image codestream with
    /// constant modular leaf parameters.
    pub fn encode_minimal_modular_image_codestream_with_params(
        &self,
        size: (u32, u32),
        offset: i32,
        predictor: u32,
    ) -> Result<Vec<u8>> {
        headers::encode_minimal_modular_image_codestream_with_params(size, offset, predictor)
    }

    /// Encodes a minimal fully decodable modular-image codestream with a
    /// constant modular leaf offset and predictor `Zero`.
    pub fn encode_minimal_modular_image_codestream_with_offset(
        &self,
        size: (u32, u32),
        offset: i32,
    ) -> Result<Vec<u8>> {
        headers::encode_minimal_modular_image_codestream_with_offset(size, offset)
    }

    /// Encodes an interleaved RGB8 buffer into a single-group modular codestream.
    pub fn encode_modular_u8_rgb_codestream(
        &self,
        size: (u32, u32),
        rgb: &[u8],
    ) -> Result<Vec<u8>> {
        headers::encode_modular_u8_rgb_image_codestream(size, rgb)
    }

    /// Encodes a minimal header-only stream wrapped in a JXL container.
    pub fn encode_minimal_container_header(&self, size: (u32, u32)) -> Result<Vec<u8>> {
        let codestream = headers::encode_minimal_codestream_header(size)?;
        container::wrap_codestream(&codestream)
    }

    /// Encodes a minimal single-frame stream wrapped in a JXL container.
    pub fn encode_minimal_single_frame_container(&self, size: (u32, u32)) -> Result<Vec<u8>> {
        let codestream = headers::encode_minimal_single_frame_codestream(size)?;
        container::wrap_codestream(&codestream)
    }

    /// Encodes a minimal decodable modular-image stream wrapped in a JXL container.
    pub fn encode_minimal_modular_image_container(&self, size: (u32, u32)) -> Result<Vec<u8>> {
        let codestream = headers::encode_minimal_modular_image_codestream(size)?;
        container::wrap_codestream(&codestream)
    }

    /// Encodes a minimal decodable modular-image stream with constant leaf
    /// parameters, wrapped in a JXL container.
    pub fn encode_minimal_modular_image_container_with_params(
        &self,
        size: (u32, u32),
        offset: i32,
        predictor: u32,
    ) -> Result<Vec<u8>> {
        let codestream =
            headers::encode_minimal_modular_image_codestream_with_params(size, offset, predictor)?;
        container::wrap_codestream(&codestream)
    }

    /// Encodes a minimal decodable modular-image stream with constant offset
    /// and predictor `Zero`, wrapped in a JXL container.
    pub fn encode_minimal_modular_image_container_with_offset(
        &self,
        size: (u32, u32),
        offset: i32,
    ) -> Result<Vec<u8>> {
        let codestream =
            headers::encode_minimal_modular_image_codestream_with_offset(size, offset)?;
        container::wrap_codestream(&codestream)
    }

    /// Encodes an interleaved RGB8 buffer into a single-group modular stream,
    /// wrapped in a JXL container.
    pub fn encode_modular_u8_rgb_container(&self, size: (u32, u32), rgb: &[u8]) -> Result<Vec<u8>> {
        let codestream = headers::encode_modular_u8_rgb_image_codestream(size, rgb)?;
        container::wrap_codestream(&codestream)
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

    #[test]
    fn test_encode_image_codestream_rgb8_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let rgb = vec![0u8; 8 * 4 * 3];
        let bytes = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_image_respects_container_option() {
        let rgb = vec![0u8; 8 * 4 * 3];

        let mut enc = JxlEncoder::new(JxlEncoderOptions::default());
        enc.options_mut().container = true;
        let bytes = enc
            .encode_image((8, 4), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );

        enc.options_mut().container = false;
        let bytes = enc
            .encode_image((8, 4), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_minimal_codestream_header_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc.encode_minimal_codestream_header((64, 32)).unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_minimal_container_header_has_container_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc.encode_minimal_container_header((64, 32)).unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
    }

    #[test]
    fn test_encode_minimal_single_frame_codestream_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc
            .encode_minimal_single_frame_codestream((64, 32))
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_minimal_single_frame_container_has_container_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc.encode_minimal_single_frame_container((64, 32)).unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
    }

    #[test]
    fn test_encode_minimal_modular_image_codestream_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc
            .encode_minimal_modular_image_codestream((64, 32))
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_minimal_modular_image_container_has_container_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc
            .encode_minimal_modular_image_container((64, 32))
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
    }

    #[test]
    fn test_encode_minimal_modular_image_codestream_with_offset_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc
            .encode_minimal_modular_image_codestream_with_offset((64, 32), 7)
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_minimal_modular_image_container_with_offset_has_container_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc
            .encode_minimal_modular_image_container_with_offset((64, 32), 7)
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
    }

    #[test]
    fn test_encode_minimal_modular_image_codestream_with_params_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc
            .encode_minimal_modular_image_codestream_with_params((64, 32), 7, 1)
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_minimal_modular_image_container_with_params_has_container_signature() {
        let enc = JxlEncoder::default();
        let bytes = enc
            .encode_minimal_modular_image_container_with_params((64, 32), 7, 1)
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
    }

    #[test]
    fn test_encode_modular_u8_rgb_codestream_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let rgb = vec![0u8; 16 * 8 * 3];
        let bytes = enc.encode_modular_u8_rgb_codestream((16, 8), &rgb).unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_modular_u8_rgb_container_has_container_signature() {
        let enc = JxlEncoder::default();
        let rgb = vec![0u8; 16 * 8 * 3];
        let bytes = enc.encode_modular_u8_rgb_container((16, 8), &rgb).unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
    }
}
