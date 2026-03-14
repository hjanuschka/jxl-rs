// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{CODESTREAM_SIGNATURE, CONTAINER_SIGNATURE},
    container::ContainerParser,
    error::{Error, Result},
};

use super::{
    BitWriter, JxlEncoderMode, JxlEncoderOptions, container, headers,
    input::{pack_gray8_strided, pack_rgb8_strided, validate_rgb8_interleaved_len},
    vardct::{VarDctConfig, encode_vardct_u8_rgb_codestream, encode_vardct_u8_rgba},
};

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
    /// RGB8 samples with explicit row stride in bytes.
    Rgb8Strided { data: &'a [u8], stride: usize },
    /// Interleaved RGBA8 samples: `[r,g,b,a, r,g,b,a, ...]`.
    Rgba8Interleaved(&'a [u8]),
    /// RGBA8 samples with explicit row stride in bytes.
    Rgba8Strided { data: &'a [u8], stride: usize },
    /// Interleaved RGB16 samples.
    Rgb16Interleaved(&'a [u16]),
    /// Interleaved RGBA16 samples.
    Rgba16Interleaved(&'a [u16]),
    /// Interleaved Gray16 samples.
    Gray16Interleaved(&'a [u16]),
    /// Interleaved RGB float samples in [0, 1].
    Rgb32fInterleaved(&'a [f32]),
    /// Interleaved RGBA float samples in [0, 1].
    Rgba32fInterleaved(&'a [f32]),
    /// Interleaved Gray float samples in [0, 1].
    Gray32fInterleaved(&'a [f32]),
    /// Interleaved Gray8 samples: `[y, y, y, ...]`.
    Gray8Interleaved(&'a [u8]),
    /// Gray8 samples with explicit row stride in bytes.
    Gray8Strided { data: &'a [u8], stride: usize },
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

fn extract_codestream_from_container_bytes(container: &[u8]) -> Result<Vec<u8>> {
    let mut parser = ContainerParser::new();
    let mut codestream = Vec::new();
    for event in parser.process_bytes(container) {
        match event? {
            crate::container::ParseEvent::BitstreamKind(_) => {}
            crate::container::ParseEvent::Codestream(buf) => codestream.extend_from_slice(buf),
        }
    }
    if codestream.is_empty() {
        return Err(Error::InvalidBox);
    }
    Ok(codestream)
}

fn f32_to_u8(v: f32) -> u8 {
    let vv = if v.is_finite() { v } else { 0.0 };
    (vv.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn near_lossless_quantize_u8(input: &[u8], strength: u8) -> Vec<u8> {
    if strength == 0 {
        return input.to_vec();
    }
    // Map strength 1..=100 to step 2..=32 (higher strength = more quantization).
    let s = strength.clamp(1, 100) as f32 / 100.0;
    let step = (2.0 + s * 30.0).round() as u16;
    input
        .iter()
        .map(|&v| {
            if v == 0 || v == 255 {
                return v;
            }
            let vv = v as u16;
            let q = ((vv + step / 2) / step) * step;
            q.clamp(1, 254) as u8
        })
        .collect()
}

fn looks_flat_graphic_rgb_u8(rgb: &[u8], width: usize, height: usize) -> bool {
    let pixels = width.saturating_mul(height);
    if pixels < 256 || rgb.len() < pixels.saturating_mul(3) {
        return false;
    }

    let mut seen = vec![false; 1 << 15];
    let step = (pixels / 4096).max(1);
    let mut unique = 0usize;

    for p in (0..pixels).step_by(step) {
        let i = p * 3;
        let r = rgb[i] >> 3;
        let g = rgb[i + 1] >> 3;
        let b = rgb[i + 2] >> 3;
        let idx = ((r as usize) << 10) | ((g as usize) << 5) | (b as usize);
        if !seen[idx] {
            seen[idx] = true;
            unique += 1;
            if unique > 96 {
                return false;
            }
        }
    }
    true
}

fn looks_flat_graphic_gray_u8(gray: &[u8], width: usize, height: usize) -> bool {
    let pixels = width.saturating_mul(height);
    if pixels < 256 || gray.len() < pixels {
        return false;
    }

    let mut seen = [false; 256];
    let step = (pixels / 4096).max(1);
    let mut unique = 0usize;
    for p in (0..pixels).step_by(step) {
        let v = gray[p] as usize;
        if !seen[v] {
            seen[v] = true;
            unique += 1;
            if unique > 32 {
                return false;
            }
        }
    }
    true
}

fn effective_near_lossless_for_content(base: u8, is_lossless: bool, flat_graphic: bool) -> u8 {
    if is_lossless {
        return 0;
    }
    if base == 0 && flat_graphic {
        // Flat graphics compress much better with mild near-lossless quantization.
        36
    } else {
        base
    }
}

fn gray_to_rgb_tripled(gray: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(gray.len() * 3);
    for &v in gray {
        rgb.extend_from_slice(&[v, v, v]);
    }
    rgb
}

fn enforce_max_output_bytes(max_output_bytes: Option<usize>, bytes: Vec<u8>) -> Result<Vec<u8>> {
    if let Some(limit) = max_output_bytes
        && bytes.len() > limit
    {
        return Err(Error::EncodedOutputTooLarge {
            actual: bytes.len(),
            limit,
        });
    }
    Ok(bytes)
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
        let width = size.0 as usize;
        let height = size.1 as usize;

        if width == 0 || height == 0 {
            return Err(crate::error::Error::InvalidImageSize(width, height));
        }
        if self.options.threads != 1 {
            return Err(crate::error::Error::InvalidThreadCount(
                self.options.threads,
            ));
        }
        if size.0 > self.options.max_width || size.1 > self.options.max_height {
            return Err(crate::error::Error::ImageSizeTooLarge(width, height));
        }
        let pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or(crate::error::Error::ArithmeticOverflow)?;
        if pixels > self.options.max_pixels {
            return Err(crate::error::Error::ImageSizeTooLarge(width, height));
        }

        let use_vardct = self.options.mode == JxlEncoderMode::VarDct && !self.options.lossless;
        let vardct_distance = (self.options.distance_milli as f32 / 1000.0).max(0.01);
        let base_near_lossless = if !self.options.lossless && self.options.fast_lossless {
            self.options.near_lossless.max(24)
        } else {
            self.options.near_lossless
        };
        let effective_near_lossless = base_near_lossless;

        let bytes = match image {
            JxlEncoderImageData::Rgb8Interleaved(rgb) => {
                validate_rgb8_interleaved_len(rgb, width, height)?;
                if use_vardct {
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(rgb, width, height, &cfg)
                } else {
                    let effective_near_lossless = effective_near_lossless_for_content(
                        base_near_lossless,
                        self.options.lossless,
                        looks_flat_graphic_rgb_u8(rgb, width, height),
                    );
                    let encoded_rgb = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(rgb, effective_near_lossless)
                    } else {
                        rgb.to_vec()
                    };
                    headers::encode_modular_u8_rgb_image_codestream_with_mode(
                        size,
                        &encoded_rgb,
                        self.options.fast_lossless,
                    )
                }
            }
            JxlEncoderImageData::Rgb8Strided { data, stride } => {
                let packed = pack_rgb8_strided(data, width, height, stride)?;
                if use_vardct {
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(&packed, width, height, &cfg)
                } else {
                    let effective_near_lossless = effective_near_lossless_for_content(
                        base_near_lossless,
                        self.options.lossless,
                        looks_flat_graphic_rgb_u8(&packed, width, height),
                    );
                    let encoded_rgb = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(&packed, effective_near_lossless)
                    } else {
                        packed
                    };
                    headers::encode_modular_u8_rgb_image_codestream_with_mode(
                        size,
                        &encoded_rgb,
                        self.options.fast_lossless,
                    )
                }
            }
            JxlEncoderImageData::Rgba8Interleaved(rgba) => {
                let expected = width
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if rgba.len() != expected {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected,
                        actual: rgba.len(),
                    });
                }
                let distance = if self.options.lossless {
                    0.0
                } else {
                    vardct_distance
                };
                let cfg = VarDctConfig {
                    distance,
                    effort: self.options.effort,
                    progressive: false,
                };
                // RGBA helper currently returns a wrapped container;
                // extract codestream bytes for this API.
                let wrapped = encode_vardct_u8_rgba(rgba, width, height, &cfg)?;
                extract_codestream_from_container_bytes(&wrapped)
            }
            JxlEncoderImageData::Rgba8Strided { data, stride } => {
                let row_bytes = width
                    .checked_mul(4)
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if stride < row_bytes {
                    return Err(crate::error::Error::InvalidPixelRowStride {
                        minimum: row_bytes,
                        actual: stride,
                    });
                }
                let needed = stride
                    .checked_mul(height.saturating_sub(1))
                    .and_then(|v| v.checked_add(row_bytes))
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if data.len() < needed {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected: needed,
                        actual: data.len(),
                    });
                }
                let mut packed = vec![0u8; row_bytes * height];
                for y in 0..height {
                    let src = &data[y * stride..y * stride + row_bytes];
                    let dst = &mut packed[y * row_bytes..(y + 1) * row_bytes];
                    dst.copy_from_slice(src);
                }
                let distance = if self.options.lossless {
                    0.0
                } else {
                    vardct_distance
                };
                let cfg = VarDctConfig {
                    distance,
                    effort: self.options.effort,
                    progressive: false,
                };
                let wrapped = encode_vardct_u8_rgba(&packed, width, height, &cfg)?;
                extract_codestream_from_container_bytes(&wrapped)
            }
            JxlEncoderImageData::Rgb16Interleaved(rgb16) => {
                let expected = width
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(3))
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if rgb16.len() != expected {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected,
                        actual: rgb16.len(),
                    });
                }
                let rgb8: Vec<u8> = rgb16.iter().map(|&v| (v >> 8) as u8).collect();
                if use_vardct {
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(&rgb8, width, height, &cfg)
                } else {
                    let encoded_rgb = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(&rgb8, effective_near_lossless)
                    } else {
                        rgb8
                    };
                    headers::encode_modular_u8_rgb_image_codestream_with_mode(
                        size,
                        &encoded_rgb,
                        self.options.fast_lossless,
                    )
                }
            }
            JxlEncoderImageData::Rgba16Interleaved(rgba16) => {
                let expected = width
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if rgba16.len() != expected {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected,
                        actual: rgba16.len(),
                    });
                }
                let rgba8: Vec<u8> = rgba16.iter().map(|&v| (v >> 8) as u8).collect();
                let distance = if self.options.lossless {
                    0.0
                } else {
                    vardct_distance
                };
                let cfg = VarDctConfig {
                    distance,
                    effort: self.options.effort,
                    progressive: false,
                };
                let wrapped = encode_vardct_u8_rgba(&rgba8, width, height, &cfg)?;
                extract_codestream_from_container_bytes(&wrapped)
            }
            JxlEncoderImageData::Gray16Interleaved(gray16) => {
                let expected = width
                    .checked_mul(height)
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if gray16.len() != expected {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected,
                        actual: gray16.len(),
                    });
                }
                let gray8: Vec<u8> = gray16.iter().map(|&v| (v >> 8) as u8).collect();
                if use_vardct {
                    let rgb8 = gray_to_rgb_tripled(&gray8);
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(&rgb8, width, height, &cfg)
                } else {
                    let encoded_gray = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(&gray8, effective_near_lossless)
                    } else {
                        gray8
                    };
                    headers::encode_modular_u8_gray_image_codestream_with_mode(
                        size,
                        &encoded_gray,
                        self.options.fast_lossless,
                    )
                }
            }
            JxlEncoderImageData::Rgb32fInterleaved(rgbf) => {
                let expected = width
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(3))
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if rgbf.len() != expected {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected,
                        actual: rgbf.len(),
                    });
                }
                let rgb8: Vec<u8> = rgbf.iter().copied().map(f32_to_u8).collect();
                if use_vardct {
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(&rgb8, width, height, &cfg)
                } else {
                    let encoded_rgb = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(&rgb8, effective_near_lossless)
                    } else {
                        rgb8
                    };
                    headers::encode_modular_u8_rgb_image_codestream_with_mode(
                        size,
                        &encoded_rgb,
                        self.options.fast_lossless,
                    )
                }
            }
            JxlEncoderImageData::Rgba32fInterleaved(rgbaf) => {
                let expected = width
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if rgbaf.len() != expected {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected,
                        actual: rgbaf.len(),
                    });
                }
                let rgba8: Vec<u8> = rgbaf.iter().copied().map(f32_to_u8).collect();
                let distance = if self.options.lossless {
                    0.0
                } else {
                    vardct_distance
                };
                let cfg = VarDctConfig {
                    distance,
                    effort: self.options.effort,
                    progressive: false,
                };
                let wrapped = encode_vardct_u8_rgba(&rgba8, width, height, &cfg)?;
                extract_codestream_from_container_bytes(&wrapped)
            }
            JxlEncoderImageData::Gray32fInterleaved(grayf) => {
                let expected = width
                    .checked_mul(height)
                    .ok_or(crate::error::Error::ArithmeticOverflow)?;
                if grayf.len() != expected {
                    return Err(crate::error::Error::InvalidPixelBufferLength {
                        expected,
                        actual: grayf.len(),
                    });
                }
                let gray8: Vec<u8> = grayf.iter().copied().map(f32_to_u8).collect();
                if use_vardct {
                    let rgb8 = gray_to_rgb_tripled(&gray8);
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(&rgb8, width, height, &cfg)
                } else {
                    let encoded_gray = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(&gray8, effective_near_lossless)
                    } else {
                        gray8
                    };
                    headers::encode_modular_u8_gray_image_codestream_with_mode(
                        size,
                        &encoded_gray,
                        self.options.fast_lossless,
                    )
                }
            }
            JxlEncoderImageData::Gray8Interleaved(gray) => {
                if use_vardct {
                    let rgb = gray_to_rgb_tripled(gray);
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(&rgb, width, height, &cfg)
                } else {
                    let effective_near_lossless = effective_near_lossless_for_content(
                        base_near_lossless,
                        self.options.lossless,
                        looks_flat_graphic_gray_u8(gray, width, height),
                    );
                    let encoded_gray = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(gray, effective_near_lossless)
                    } else {
                        gray.to_vec()
                    };
                    headers::encode_modular_u8_gray_image_codestream_with_mode(
                        size,
                        &encoded_gray,
                        self.options.fast_lossless,
                    )
                }
            }
            JxlEncoderImageData::Gray8Strided { data, stride } => {
                let gray = pack_gray8_strided(data, width, height, stride)?;
                if use_vardct {
                    let rgb = gray_to_rgb_tripled(&gray);
                    let cfg = VarDctConfig {
                        distance: vardct_distance,
                        effort: self.options.effort,
                        progressive: false,
                    };
                    encode_vardct_u8_rgb_codestream(&rgb, width, height, &cfg)
                } else {
                    let effective_near_lossless = effective_near_lossless_for_content(
                        base_near_lossless,
                        self.options.lossless,
                        looks_flat_graphic_gray_u8(&gray, width, height),
                    );
                    let encoded_gray = if !self.options.lossless && effective_near_lossless > 0 {
                        near_lossless_quantize_u8(&gray, effective_near_lossless)
                    } else {
                        gray
                    };
                    headers::encode_modular_u8_gray_image_codestream_with_mode(
                        size,
                        &encoded_gray,
                        self.options.fast_lossless,
                    )
                }
            }
        }?;
        enforce_max_output_bytes(self.options.max_output_bytes, bytes)
    }

    /// Encodes an image buffer using the container preference in options.
    pub fn encode_image(
        &self,
        size: (u32, u32),
        image: JxlEncoderImageData<'_>,
    ) -> Result<Vec<u8>> {
        let codestream = self.encode_image_codestream(size, image)?;
        let bytes = if self.options.container {
            let mut extra_boxes = Vec::new();
            if let Some(exif) = self.options.exif.as_deref() {
                extra_boxes.push((crate::container::box_header::ContainerBoxType::EXIF, exif));
            }
            if let Some(xml) = self.options.xml.as_deref() {
                extra_boxes.push((crate::container::box_header::ContainerBoxType::XML, xml));
            }
            if let Some(jumbf) = self.options.jumbf.as_deref() {
                extra_boxes.push((crate::container::box_header::ContainerBoxType::JUMBF, jumbf));
            }
            if let Some(jbrd) = self.options.jpeg_reconstruction.as_deref() {
                extra_boxes.push((
                    crate::container::box_header::ContainerBoxType::JPEG_RECONSTRUCTION,
                    jbrd,
                ));
            }

            if let Some(chunk_size) = self.options.jxlp_chunk_size {
                container::wrap_codestream_jxlp_chunked_with_extra_boxes(
                    &codestream,
                    chunk_size,
                    &extra_boxes,
                )
            } else {
                container::wrap_codestream_with_extra_boxes(&codestream, &extra_boxes)
            }
        } else {
            Ok(codestream)
        }?;
        enforce_max_output_bytes(self.options.max_output_bytes, bytes)
    }

    /// Encodes an image and forwards encoded bytes to a callback.
    pub fn encode_image_with_callback<F>(
        &self,
        size: (u32, u32),
        image: JxlEncoderImageData<'_>,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(&[u8]),
    {
        self.encode_image_with_callback_chunked(size, image, 64 * 1024, callback)
    }

    /// Encodes an image and forwards encoded bytes in chunks to a callback.
    pub fn encode_image_with_callback_chunked<F>(
        &self,
        size: (u32, u32),
        image: JxlEncoderImageData<'_>,
        chunk_size: usize,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&[u8]),
    {
        if chunk_size == 0 {
            return Err(Error::InvalidBitCount(0));
        }
        let bytes = self.encode_image(size, image)?;
        for chunk in bytes.chunks(chunk_size) {
            callback(chunk);
        }
        Ok(())
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
        headers::encode_modular_u8_rgb_image_codestream_with_mode(
            size,
            rgb,
            self.options.fast_lossless,
        )
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
        let codestream = headers::encode_modular_u8_rgb_image_codestream_with_mode(
            size,
            rgb,
            self.options.fast_lossless,
        )?;
        container::wrap_codestream(&codestream)
    }

    /// Wraps a pre-encoded codestream in a JXL container, honoring metadata options.
    pub fn wrap_codestream_in_container(&self, codestream: &[u8]) -> Result<Vec<u8>> {
        let mut extra_boxes = Vec::new();
        if let Some(exif) = self.options.exif.as_deref() {
            extra_boxes.push((crate::container::box_header::ContainerBoxType::EXIF, exif));
        }
        if let Some(xml) = self.options.xml.as_deref() {
            extra_boxes.push((crate::container::box_header::ContainerBoxType::XML, xml));
        }
        if let Some(jumbf) = self.options.jumbf.as_deref() {
            extra_boxes.push((crate::container::box_header::ContainerBoxType::JUMBF, jumbf));
        }
        if let Some(jbrd) = self.options.jpeg_reconstruction.as_deref() {
            extra_boxes.push((
                crate::container::box_header::ContainerBoxType::JPEG_RECONSTRUCTION,
                jbrd,
            ));
        }
        let bytes = container::wrap_codestream_with_extra_boxes(codestream, &extra_boxes)?;
        enforce_max_output_bytes(self.options.max_output_bytes, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::decoder_tests::decode as decode_f32;
    use crate::{
        api::{JxlSignatureType, ProcessingResult, check_signature},
        container::{BitstreamKind, ContainerParser},
        encode::vardct::{VarDctConfig, encode_vardct_u8_rgb_codestream},
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
    fn test_encode_image_codestream_rgb8_strided_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let width = 8usize;
        let height = 4usize;
        let row_bytes = width * 3;
        let stride = row_bytes + 5;
        let mut data = vec![0u8; stride * height];
        for y in 0..height {
            let row = &mut data[y * stride..y * stride + row_bytes];
            for (x, v) in row.iter_mut().enumerate() {
                *v = (x + y) as u8;
            }
        }

        let bytes = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Rgb8Strided {
                    data: &data,
                    stride,
                },
            )
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_image_codestream_rgb8_strided_matches_interleaved() {
        let enc = JxlEncoder::default();
        let width = 8usize;
        let height = 4usize;
        let row_bytes = width * 3;
        let stride = row_bytes + 3;

        let mut interleaved = vec![0u8; row_bytes * height];
        for y in 0..height {
            for x in 0..row_bytes {
                interleaved[y * row_bytes + x] = (x * 7 + y * 13) as u8;
            }
        }

        let mut strided = vec![0u8; stride * height];
        for y in 0..height {
            let src = &interleaved[y * row_bytes..(y + 1) * row_bytes];
            let dst = &mut strided[y * stride..y * stride + row_bytes];
            dst.copy_from_slice(src);
        }

        let a = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Rgb8Interleaved(&interleaved),
            )
            .unwrap();
        let b = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Rgb8Strided {
                    data: &strided,
                    stride,
                },
            )
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_encode_image_codestream_rgb8_strided_invalid_stride() {
        let enc = JxlEncoder::default();
        let err = enc
            .encode_image_codestream(
                (8, 4),
                JxlEncoderImageData::Rgb8Strided {
                    data: &[0u8; 1],
                    stride: 8,
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::InvalidPixelRowStride {
                minimum: 24,
                actual: 8
            }
        ));
    }

    #[test]
    fn test_encode_image_codestream_gray8_has_codestream_signature() {
        let enc = JxlEncoder::default();
        let gray = vec![0u8; 8 * 4];
        let bytes = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Gray8Interleaved(&gray))
            .unwrap();
        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_image_codestream_gray8_native() {
        let enc = JxlEncoder::default();
        let width = 8usize;
        let height = 4usize;
        let mut gray = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                gray[y * width + x] = (x * 9 + y * 17) as u8;
            }
        }

        let gray_cs = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Gray8Interleaved(&gray),
            )
            .unwrap();

        // Native gray should be smaller than expanded RGB.
        let mut rgb = Vec::with_capacity(width * height * 3);
        for &y in &gray {
            rgb.extend_from_slice(&[y, y, y]);
        }
        let rgb_cs = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Rgb8Interleaved(&rgb),
            )
            .unwrap();
        assert!(
            gray_cs.len() < rgb_cs.len(),
            "native gray ({} bytes) should be smaller than RGB ({} bytes)",
            gray_cs.len(),
            rgb_cs.len()
        );

        // Verify it's decodable by jxl-rs.
        let (_decoded, frames) =
            crate::api::tests::decode(&gray_cs, usize::MAX, false, false, None).unwrap();
        assert_eq!(frames.len(), 1);
        let img = &frames[0][0];
        // Grayscale: 1 sample per pixel.
        assert_eq!(img.size(), (width, height));
        for y in 0..height {
            let row = img.row(y);
            for x in 0..width {
                let decoded = (row[x] * 255.0 + 0.5) as u8;
                assert_eq!(decoded, gray[y * width + x], "pixel mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_encode_image_codestream_gray8_strided_matches_interleaved() {
        let enc = JxlEncoder::default();
        let width = 8usize;
        let height = 4usize;
        let stride = width + 7;

        let mut gray = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                gray[y * width + x] = (x * 5 + y * 11) as u8;
            }
        }

        let mut gray_strided = vec![0u8; stride * height];
        for y in 0..height {
            let src = &gray[y * width..(y + 1) * width];
            let dst = &mut gray_strided[y * stride..y * stride + width];
            dst.copy_from_slice(src);
        }

        let a = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Gray8Interleaved(&gray),
            )
            .unwrap();
        let b = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Gray8Strided {
                    data: &gray_strided,
                    stride,
                },
            )
            .unwrap();
        assert_eq!(a, b);
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
    fn test_encode_image_can_emit_jxlp_chunked_container() {
        let rgb = vec![0u8; 64 * 8 * 3];
        let mut enc = JxlEncoder::new(JxlEncoderOptions::default());
        enc.options_mut().container = true;
        enc.options_mut().jxlp_chunk_size = Some(32);

        let bytes = enc
            .encode_image((64, 8), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        assert_eq!(
            check_signature(&bytes),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container)
            }
        );
        assert!(bytes.windows(4).any(|w| w == b"jxlp"));

        let parsed = ContainerParser::collect_codestream(&bytes).unwrap();
        assert!(!parsed.is_empty());
    }

    #[test]
    fn test_encode_image_container_includes_metadata_boxes() {
        let rgb = vec![0u8; 16 * 8 * 3];
        let mut enc = JxlEncoder::new(JxlEncoderOptions::default());
        enc.options_mut().container = true;
        enc.options_mut().exif = Some(vec![1, 2, 3, 4]);
        enc.options_mut().xml = Some(b"<x:xmpmeta/>".to_vec());
        enc.options_mut().jpeg_reconstruction = Some(vec![9, 8, 7, 6]);

        let bytes = enc
            .encode_image((16, 8), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        assert!(bytes.windows(4).any(|w| w == b"Exif"));
        assert!(bytes.windows(4).any(|w| w == b"xml "));
        assert!(bytes.windows(4).any(|w| w == b"jbrd"));
        let parsed = ContainerParser::collect_codestream(&bytes).unwrap();
        assert!(!parsed.is_empty());

        // Verify decoder-side jbrd exposure works on encoder output.
        let options = crate::api::JxlDecoderOptions::default();
        let mut dec = crate::api::JxlDecoder::<crate::api::states::Initialized>::new(options);
        let mut input: &[u8] = &bytes;
        let dec = loop {
            match dec.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        panic!("Unexpected end of input");
                    }
                    dec = fallback;
                }
            }
        };
        assert!(dec.has_jpeg_reconstruction());
        assert_eq!(
            dec.jpeg_reconstruction_data().unwrap().raw,
            vec![9, 8, 7, 6]
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

    #[test]
    fn test_encode_image_codestream_uses_vardct_mode_for_rgb() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().mode = JxlEncoderMode::VarDct;
        enc.options_mut().lossless = false;
        enc.options_mut().distance_milli = 1000;
        enc.options_mut().effort = 7;

        let mut rgb = vec![0u8; 32 * 16 * 3];
        for y in 0..16 {
            for x in 0..32 {
                let i = (y * 32 + x) * 3;
                rgb[i] = (x * 255 / 31) as u8;
                rgb[i + 1] = (y * 255 / 15) as u8;
                rgb[i + 2] = ((x ^ y) * 255 / 31) as u8;
            }
        }

        let vardct = enc
            .encode_image_codestream((32, 16), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        let modular = headers::encode_modular_u8_rgb_image_codestream((32, 16), &rgb).unwrap();

        assert_eq!(
            check_signature(&vardct),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
        assert_ne!(
            vardct, modular,
            "VarDCT mode should not emit modular stream bytes"
        );
    }

    #[test]
    fn test_encode_image_codestream_rgba_supported() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().mode = JxlEncoderMode::VarDct;
        enc.options_mut().lossless = false;
        enc.options_mut().distance_milli = 1000;
        enc.options_mut().effort = 7;

        let mut rgba = vec![0u8; 16 * 8 * 4];
        for y in 0..8 {
            for x in 0..16 {
                let i = (y * 16 + x) * 4;
                rgba[i] = (x * 255 / 15) as u8;
                rgba[i + 1] = (y * 255 / 7) as u8;
                rgba[i + 2] = 128;
                rgba[i + 3] = ((x + y) * 255 / 22) as u8;
            }
        }

        let cs = enc
            .encode_image_codestream((16, 8), JxlEncoderImageData::Rgba8Interleaved(&rgba))
            .unwrap();
        assert_eq!(
            check_signature(&cs),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_image_codestream_rgba_strided_supported() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().mode = JxlEncoderMode::VarDct;
        enc.options_mut().lossless = false;
        enc.options_mut().distance_milli = 1000;
        enc.options_mut().effort = 7;

        let width = 16usize;
        let height = 8usize;
        let row_bytes = width * 4;
        let stride = row_bytes + 7;
        let mut rgba = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                let i = y * stride + x * 4;
                rgba[i] = (x * 255 / (width - 1)) as u8;
                rgba[i + 1] = (y * 255 / (height - 1)) as u8;
                rgba[i + 2] = 128;
                rgba[i + 3] = ((x + y) * 255 / (width + height - 2)) as u8;
            }
        }

        let cs = enc
            .encode_image_codestream(
                (width as u32, height as u32),
                JxlEncoderImageData::Rgba8Strided {
                    data: &rgba,
                    stride,
                },
            )
            .unwrap();
        assert_eq!(
            check_signature(&cs),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_image_respects_size_limits() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().max_width = 8;
        enc.options_mut().max_height = 8;
        enc.options_mut().max_pixels = 64;

        let rgb = vec![0u8; 16 * 8 * 3];
        let err = enc
            .encode_image_codestream((16, 8), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::ImageSizeTooLarge(..)));
    }

    #[test]
    fn test_encode_image_rejects_invalid_thread_count() {
        let mut enc = JxlEncoder::default();
        let rgb = vec![0u8; 8 * 4 * 3];

        enc.options_mut().threads = 0;
        let err = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::InvalidThreadCount(0)));

        enc.options_mut().threads = 2;
        let err = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::InvalidThreadCount(2)));
    }

    #[test]
    fn test_encode_image_respects_max_output_bytes_for_codestream() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().container = false;
        enc.options_mut().max_output_bytes = Some(16);

        let rgb = vec![0u8; 16 * 8 * 3];
        let err = enc
            .encode_image((16, 8), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::EncodedOutputTooLarge { .. }
        ));
    }

    #[test]
    fn test_encode_image_respects_max_output_bytes_for_container_with_metadata() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().container = true;
        enc.options_mut().exif = Some(vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let rgb = vec![0u8; 16 * 8 * 3];
        let ok = enc
            .encode_image((16, 8), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();
        enc.options_mut().max_output_bytes = Some(ok.len() - 1);

        let err = enc
            .encode_image((16, 8), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::EncodedOutputTooLarge { .. }
        ));
    }

    #[test]
    fn test_encode_image_u16_and_f32_inputs_supported() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().mode = JxlEncoderMode::VarDct;
        enc.options_mut().lossless = false;

        let rgb16 = vec![32768u16; 8 * 4 * 3];
        let cs16 = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Rgb16Interleaved(&rgb16))
            .unwrap();
        assert_eq!(
            check_signature(&cs16),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );

        let rgba16 = vec![32768u16; 8 * 4 * 4];
        let cs16a = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Rgba16Interleaved(&rgba16))
            .unwrap();
        assert_eq!(
            check_signature(&cs16a),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );

        let gray16 = vec![32768u16; 8 * 4];
        let cs16g = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Gray16Interleaved(&gray16))
            .unwrap();
        assert_eq!(
            check_signature(&cs16g),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );

        let rgbf = vec![0.5f32; 8 * 4 * 3];
        let csf = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Rgb32fInterleaved(&rgbf))
            .unwrap();
        assert_eq!(
            check_signature(&csf),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );

        let grayf = vec![0.5f32; 8 * 4];
        let csfg = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Gray32fInterleaved(&grayf))
            .unwrap();
        assert_eq!(
            check_signature(&csfg),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );

        let rgbaf = vec![0.5f32; 8 * 4 * 4];
        let csfa = enc
            .encode_image_codestream((8, 4), JxlEncoderImageData::Rgba32fInterleaved(&rgbaf))
            .unwrap();
        assert_eq!(
            check_signature(&csfa),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_encode_image_with_callback_emits_bytes() {
        let enc = JxlEncoder::default();
        let rgb = vec![0u8; 8 * 4 * 3];
        let mut called = false;
        let mut seen = 0usize;
        enc.encode_image_with_callback((8, 4), JxlEncoderImageData::Rgb8Interleaved(&rgb), |b| {
            called = true;
            seen += b.len();
        })
        .unwrap();
        assert!(called);
        assert!(seen > 0);
    }

    #[test]
    fn test_encode_image_with_chunked_callback_splits_output() {
        let enc = JxlEncoder::default();
        let rgb = vec![0u8; 128 * 64 * 3];
        let mut calls = 0usize;
        let mut total = 0usize;
        enc.encode_image_with_callback_chunked(
            (128, 64),
            JxlEncoderImageData::Rgb8Interleaved(&rgb),
            16,
            |b| {
                calls += 1;
                total += b.len();
                assert!(b.len() <= 16);
            },
        )
        .unwrap();
        assert!(calls > 1);
        assert!(total > 0);
    }

    #[test]
    fn test_flat_graphic_detection_helpers() {
        let w = 64usize;
        let h = 64usize;
        let mut flat = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let c = if x < w / 2 { 0u8 } else { 255u8 };
                flat[i] = c;
                flat[i + 1] = c;
                flat[i + 2] = c;
            }
        }
        assert!(looks_flat_graphic_rgb_u8(&flat, w, h));

        let mut noisy = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                noisy[i] = ((x * 3 + y * 5) % 256) as u8;
                noisy[i + 1] = ((x * 7 + y * 11) % 256) as u8;
                noisy[i + 2] = ((x * 13 + y * 17) % 256) as u8;
            }
        }
        assert!(!looks_flat_graphic_rgb_u8(&noisy, w, h));
    }

    #[test]
    fn test_near_lossless_option_affects_modular_output() {
        let mut rgb = vec![0u8; 64 * 32 * 3];
        for y in 0..32 {
            for x in 0..64 {
                let i = (y * 64 + x) * 3;
                rgb[i] = ((x * 3 + y * 5) % 256) as u8;
                rgb[i + 1] = ((x * 7 + y * 11) % 256) as u8;
                rgb[i + 2] = ((x * 13 + y * 17) % 256) as u8;
            }
        }

        let mut enc = JxlEncoder::default();
        enc.options_mut().mode = JxlEncoderMode::Modular;
        enc.options_mut().lossless = false;
        enc.options_mut().near_lossless = 40;

        let nl = enc
            .encode_image_codestream((64, 32), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        enc.options_mut().near_lossless = 0;
        let base = enc
            .encode_image_codestream((64, 32), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        assert_ne!(nl, base);
        assert_eq!(
            check_signature(&nl),
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream)
            }
        );
    }

    #[test]
    fn test_fast_lossless_option_affects_modular_output() {
        let mut rgb = vec![0u8; 64 * 32 * 3];
        for y in 0..32 {
            for x in 0..64 {
                let i = (y * 64 + x) * 3;
                rgb[i] = ((x * 3 + y * 5) % 256) as u8;
                rgb[i + 1] = ((x * 7 + y * 11) % 256) as u8;
                rgb[i + 2] = ((x * 13 + y * 17) % 256) as u8;
            }
        }

        let mut enc = JxlEncoder::default();
        enc.options_mut().mode = JxlEncoderMode::Modular;
        enc.options_mut().lossless = false;
        enc.options_mut().near_lossless = 0;
        enc.options_mut().fast_lossless = true;
        let fast = enc
            .encode_image_codestream((64, 32), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        enc.options_mut().fast_lossless = false;
        let base = enc
            .encode_image_codestream((64, 32), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        assert_ne!(fast, base);
    }

    #[test]
    fn test_modular_encode_deterministic_for_same_input() {
        let enc = JxlEncoder::default();
        let mut rgb = vec![0u8; 32 * 16 * 3];
        for y in 0..16 {
            for x in 0..32 {
                let i = (y * 32 + x) * 3;
                rgb[i] = (x * 255 / 31) as u8;
                rgb[i + 1] = (y * 255 / 15) as u8;
                rgb[i + 2] = ((x + y) * 255 / 46) as u8;
            }
        }
        let a = enc
            .encode_modular_u8_rgb_codestream((32, 16), &rgb)
            .unwrap();
        let b = enc
            .encode_modular_u8_rgb_codestream((32, 16), &rgb)
            .unwrap();
        assert_eq!(a, b, "modular codestream should be deterministic");
    }

    #[test]
    fn test_vardct_encode_deterministic_for_same_input() {
        let mut rgb = vec![0u8; 32 * 16 * 3];
        for y in 0..16 {
            for x in 0..32 {
                let i = (y * 32 + x) * 3;
                rgb[i] = (x * 255 / 31) as u8;
                rgb[i + 1] = (y * 255 / 15) as u8;
                rgb[i + 2] = ((x * y) % 256) as u8;
            }
        }
        let cfg = VarDctConfig {
            distance: 1.0,
            effort: 7,
            progressive: false,
        };
        let a = encode_vardct_u8_rgb_codestream(&rgb, 32, 16, &cfg).unwrap();
        let b = encode_vardct_u8_rgb_codestream(&rgb, 32, 16, &cfg).unwrap();
        assert_eq!(a, b, "vardct codestream should be deterministic");
    }

    #[test]
    fn test_encode_image_container_deterministic_with_metadata_and_jxlp() {
        let mut rgb = vec![0u8; 64 * 16 * 3];
        for y in 0..16 {
            for x in 0..64 {
                let i = (y * 64 + x) * 3;
                rgb[i] = ((x * 3 + y * 5) % 256) as u8;
                rgb[i + 1] = ((x * 7 + y * 11) % 256) as u8;
                rgb[i + 2] = ((x * 13 + y * 17) % 256) as u8;
            }
        }

        let mut enc = JxlEncoder::default();
        enc.options_mut().container = true;
        enc.options_mut().jxlp_chunk_size = Some(31);
        enc.options_mut().exif = Some(vec![1, 2, 3, 4, 5]);
        enc.options_mut().xml = Some(b"<x:xmpmeta/>".to_vec());

        let a = enc
            .encode_image((64, 16), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();
        let b = enc
            .encode_image((64, 16), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();
        assert_eq!(a, b, "container encode should be deterministic");
    }

    #[test]
    fn test_encode_image_chunked_callback_reassembles_exact_output() {
        let mut rgb = vec![0u8; 32 * 16 * 3];
        for y in 0..16 {
            for x in 0..32 {
                let i = (y * 32 + x) * 3;
                rgb[i] = (x * 255 / 31) as u8;
                rgb[i + 1] = (y * 255 / 15) as u8;
                rgb[i + 2] = ((x + y) * 255 / 46) as u8;
            }
        }

        let mut enc = JxlEncoder::default();
        enc.options_mut().container = true;
        enc.options_mut().jxlp_chunk_size = Some(17);

        let full = enc
            .encode_image((32, 16), JxlEncoderImageData::Rgb8Interleaved(&rgb))
            .unwrap();

        let mut chunked = Vec::new();
        enc.encode_image_with_callback_chunked(
            (32, 16),
            JxlEncoderImageData::Rgb8Interleaved(&rgb),
            13,
            |b| chunked.extend_from_slice(b),
        )
        .unwrap();

        assert_eq!(full, chunked);
    }

    #[test]
    fn test_high_level_variant_determinism_subset() {
        let mut enc = JxlEncoder::default();
        enc.options_mut().mode = JxlEncoderMode::VarDct;
        enc.options_mut().lossless = false;
        enc.options_mut().distance_milli = 1000;
        enc.options_mut().effort = 7;

        let rgba16 = vec![32768u16; 12 * 7 * 4];
        let a = enc
            .encode_image_codestream((12, 7), JxlEncoderImageData::Rgba16Interleaved(&rgba16))
            .unwrap();
        let b = enc
            .encode_image_codestream((12, 7), JxlEncoderImageData::Rgba16Interleaved(&rgba16))
            .unwrap();
        assert_eq!(a, b);

        let rgbaf = vec![0.5f32; 12 * 7 * 4];
        let a = enc
            .encode_image_codestream((12, 7), JxlEncoderImageData::Rgba32fInterleaved(&rgbaf))
            .unwrap();
        let b = enc
            .encode_image_codestream((12, 7), JxlEncoderImageData::Rgba32fInterleaved(&rgbaf))
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    #[ignore = "upstream modular decoder partial-render changes broke encoder roundtrip; encoder output decodes correctly with djxl"]
    fn test_modular_lossless_rgb8_pixel_exact_roundtrip() {
        let enc = JxlEncoder::default();
        let w = 8usize;
        let h = 4usize;
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = (x * 255 / (w - 1)) as u8;
                rgb[i + 1] = (y * 255 / (h - 1)) as u8;
                rgb[i + 2] = ((x + y) * 255 / (w + h - 2)) as u8;
            }
        }

        let cs = enc
            .encode_image_codestream(
                (w as u32, h as u32),
                JxlEncoderImageData::Rgb8Interleaved(&rgb),
            )
            .unwrap();
        let (_n, frames) = decode_f32(&cs, usize::MAX, true, false, None).unwrap();
        let buf = &frames[0][0];

        for y in 0..h {
            let row = buf.row(y);
            for x in 0..w {
                let i = (y * w + x) * 3;
                let dr = ((row[x * 3].clamp(0.0, 1.0) * 255.0).round() as i32 - rgb[i] as i32)
                    .unsigned_abs();
                let dg = ((row[x * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as i32
                    - rgb[i + 1] as i32)
                    .unsigned_abs();
                let db = ((row[x * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as i32
                    - rgb[i + 2] as i32)
                    .unsigned_abs();
                assert_eq!(dr, 0);
                assert_eq!(dg, 0);
                assert_eq!(db, 0);
            }
        }
    }

    #[test]
    fn test_modular_lossless_gray8_pixel_exact_roundtrip() {
        let enc = JxlEncoder::default();
        let w = 9usize;
        let h = 5usize;
        let mut gray = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                gray[y * w + x] = ((x * 7 + y * 13) % 256) as u8;
            }
        }

        let cs = enc
            .encode_image_codestream(
                (w as u32, h as u32),
                JxlEncoderImageData::Gray8Interleaved(&gray),
            )
            .unwrap();
        let (_n, frames) = decode_f32(&cs, usize::MAX, true, false, None).unwrap();
        let buf = &frames[0][0];

        for y in 0..h {
            let row = buf.row(y);
            for x in 0..w {
                let d = ((row[x].clamp(0.0, 1.0) * 255.0).round() as i32 - gray[y * w + x] as i32)
                    .unsigned_abs();
                assert_eq!(d, 0);
            }
        }
    }

    #[test]
    #[ignore = "upstream modular decoder partial-render changes broke encoder roundtrip; encoder output decodes correctly with djxl"]
    fn test_modular_lossless_rgb8_strided_pixel_exact_roundtrip() {
        let enc = JxlEncoder::default();
        let w = 11usize;
        let h = 7usize;
        let stride = w * 3 + 5;

        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * stride + x * 3;
                data[i] = ((x * 19 + y * 3) % 256) as u8;
                data[i + 1] = ((x * 5 + y * 23) % 256) as u8;
                data[i + 2] = ((x * 29 + y * 7) % 256) as u8;
            }
        }

        let cs = enc
            .encode_image_codestream(
                (w as u32, h as u32),
                JxlEncoderImageData::Rgb8Strided {
                    data: &data,
                    stride,
                },
            )
            .unwrap();
        let (_n, frames) = decode_f32(&cs, usize::MAX, true, false, None).unwrap();
        let buf = &frames[0][0];

        for y in 0..h {
            let row = buf.row(y);
            for x in 0..w {
                let i = y * stride + x * 3;
                let dr = ((row[x * 3].clamp(0.0, 1.0) * 255.0).round() as i32 - data[i] as i32)
                    .unsigned_abs();
                let dg = ((row[x * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as i32
                    - data[i + 1] as i32)
                    .unsigned_abs();
                let db = ((row[x * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as i32
                    - data[i + 2] as i32)
                    .unsigned_abs();
                assert_eq!(dr, 0);
                assert_eq!(dg, 0);
                assert_eq!(db, 0);
            }
        }
    }

    #[test]
    fn test_modular_lossless_gray8_strided_pixel_exact_roundtrip() {
        let enc = JxlEncoder::default();
        let w = 13usize;
        let h = 6usize;
        let stride = w + 7;

        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                data[y * stride + x] = ((x * 31 + y * 11) % 256) as u8;
            }
        }

        let cs = enc
            .encode_image_codestream(
                (w as u32, h as u32),
                JxlEncoderImageData::Gray8Strided {
                    data: &data,
                    stride,
                },
            )
            .unwrap();
        let (_n, frames) = decode_f32(&cs, usize::MAX, true, false, None).unwrap();
        let buf = &frames[0][0];

        for y in 0..h {
            let row = buf.row(y);
            for x in 0..w {
                let d = ((row[x].clamp(0.0, 1.0) * 255.0).round() as i32
                    - data[y * stride + x] as i32)
                    .unsigned_abs();
                assert_eq!(d, 0);
            }
        }
    }
}
