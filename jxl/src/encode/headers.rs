// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::CODESTREAM_SIGNATURE,
    error::{Error, Result},
    headers::encodings::{U32, U32Coder},
};

use super::{
    BitWriter, HybridUintConfig,
    modular_encode::{write_hf_group_section_huffman, write_lf_global_section_huffman},
    write_minimal_modular_global_data_with_params,
    write_minimal_modular_lf_global_section_with_params, write_toc, write_u32,
};

fn large_size_coder() -> U32Coder {
    U32Coder::Select(
        U32::BitsOffset { n: 9, off: 1 },
        U32::BitsOffset { n: 13, off: 1 },
        U32::BitsOffset { n: 18, off: 1 },
        U32::BitsOffset { n: 30, off: 1 },
    )
}

fn validate_size(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidImageSize(width as usize, height as usize));
    }

    let max_large_dim = 1u32 << 30;
    if width > max_large_dim || height > max_large_dim {
        return Err(Error::ImageDimensionTooLarge(width.max(height) as u64));
    }
    Ok(())
}

fn validate_rgb_u8_buffer(width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    let px_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Error::ArithmeticOverflow)?;
    let expected = px_count.checked_mul(3).ok_or(Error::ArithmeticOverflow)?;
    if rgb.len() != expected {
        return Err(Error::InvalidPixelBufferLength {
            expected,
            actual: rgb.len(),
        });
    }
    Ok(())
}

fn write_size(writer: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    validate_size(width, height)?;

    // Encode in "large" mode (small=false), with ratio = Unknown.
    writer.write(1, 0)?;
    let size_coder = large_size_coder();
    write_u32(writer, &size_coder, height)?;

    // AspectRatio::Unknown == 0, coded with Bits(3).
    writer.write(3, 0)?;

    write_u32(writer, &size_coder, width)?;
    Ok(())
}

fn bit_depth_int_coder() -> U32Coder {
    U32Coder::Select(
        U32::Val(8),
        U32::Val(10),
        U32::Val(12),
        U32::BitsOffset { n: 6, off: 1 },
    )
}

fn extra_channel_count_coder() -> U32Coder {
    U32Coder::Select(
        U32::Val(0),
        U32::Val(1),
        U32::BitsOffset { n: 4, off: 2 },
        U32::BitsOffset { n: 12, off: 1 },
    )
}

/// Writes image metadata for non-XYB encoding.
/// `grayscale`: if true, writes color_space=Gray instead of RGB.
pub(crate) fn write_image_metadata(
    writer: &mut BitWriter,
    xyb_encoded: bool,
    grayscale: bool,
) -> Result<()> {
    if xyb_encoded {
        // ImageMetadata::all_default = true.
        writer.write(1, 1)?;
        return Ok(());
    }

    // ImageMetadata::all_default = false.
    writer.write(1, 0)?;

    // extra_fields = false.
    writer.write(1, 0)?;

    // bit_depth: 8-bit integer default.
    writer.write(1, 0)?; // floating_point_sample = false
    write_u32(writer, &bit_depth_int_coder(), 8)?;

    // modular_16bit_sufficient = true
    writer.write(1, 1)?;

    // extra_channel_info len = 0
    write_u32(writer, &extra_channel_count_coder(), 0)?;

    // xyb_encoded = false
    writer.write(1, 0)?;

    if grayscale {
        // color_encoding.all_default = false (default is RGB)
        writer.write(1, 0)?;

        // want_icc = false
        writer.write(1, 0)?;

        // color_space = Gray (enum value 1, u2S selector 1 = bits 01)
        writer.write(2, 1)?;

        // white_point = D65 (value=1)
        // u2S(Val(0), Val(1), Bits(4)+2, Bits(6)+18): D65=1 → selector 01
        writer.write(2, 1)?;

        // primaries: NOT present for Gray (condition excludes it)

        // CustomTransferFunction (nested struct, no all_default annotation):
        // have_gamma = false (condition: color_space != XYB, which is true for Gray)
        writer.write(1, 0)?;
        // transfer_function = SRGB (value=13)
        // u2S(Val(0), Val(1), Bits(4)+2, Bits(6)+18): 13-2=11 → selector 10 + Bits(4)=11
        writer.write(2, 2)?;
        writer.write(4, 11)?;

        // rendering_intent = Relative (value=1)
        // u2S(Val(0), Val(1), ...): Relative=1 → selector 01
        writer.write(2, 1)?;
    } else {
        // color_encoding.all_default = true (defaults to sRGB).
        writer.write(1, 1)?;
    }

    // extensions selector = 0 (u64 coding)
    writer.write(2, 0)?;

    Ok(())
}

#[allow(dead_code)]
fn write_minimal_image_metadata_fields(writer: &mut BitWriter, xyb_encoded: bool) -> Result<()> {
    write_image_metadata(writer, xyb_encoded, false)
}

/// Write image metadata for animated JXL: VarDCT, XYB-encoded, with animation header.
/// `tps_numerator` and `tps_denominator` define ticks per second (e.g., 20/1 = 20fps).
/// `num_loops` = 0 means infinite loop.
pub(crate) fn write_image_metadata_animated(
    writer: &mut BitWriter,
    tps_numerator: u32,
    tps_denominator: u32,
    num_loops: u32,
) -> Result<()> {
    // ImageMetadata::all_default = false (need animation)
    writer.write(1, 0)?;

    // extra_fields = true (needed for have_animation)
    writer.write(1, 1)?;

    // orientation = Identity (1), Bits(3)+1, value=1 => write 0
    writer.write(3, 0)?;

    // have_intrinsic_size = false
    writer.write(1, 0)?;

    // have_preview = false
    writer.write(1, 0)?;

    // have_animation = true
    writer.write(1, 1)?;

    // Animation struct:
    //   tps_numerator: u2S(100, 1000, Bits(10)+1, Bits(30)+1)
    write_u32(writer, &tps_numerator_coder(), tps_numerator)?;
    //   tps_denominator: u2S(1, 1001, Bits(8)+1, Bits(10)+1)
    write_u32(writer, &tps_denominator_coder(), tps_denominator)?;
    //   num_loops: u2S(0, Bits(3), Bits(16), Bits(32))
    write_u32(writer, &num_loops_coder(), num_loops)?;
    //   have_timecodes = false
    writer.write(1, 0)?;

    // bit_depth: default 8-bit int
    writer.write(1, 0)?; // floating_point_sample = false
    write_u32(writer, &bit_depth_int_coder(), 8)?;

    // modular_16bit_sufficient = true
    writer.write(1, 1)?;

    // extra_channel_info len = 0
    write_u32(writer, &extra_channel_count_coder(), 0)?;

    // xyb_encoded = true (VarDCT default)
    writer.write(1, 1)?;

    // color_encoding: all_default = true (sRGB)
    writer.write(1, 1)?;

    // tone_mapping: all_default = true (conditioned on extra_fields)
    writer.write(1, 1)?;

    // extensions = 0 (u64 coding: selector 00)
    writer.write(2, 0)?;

    Ok(())
}

/// Write image metadata for VarDCT, XYB-encoded, with 1 alpha extra channel.
pub(crate) fn write_image_metadata_with_alpha(writer: &mut BitWriter) -> Result<()> {
    // ImageMetadata::all_default = false
    writer.write(1, 0)?;

    // extra_fields = false (no orientation/animation/tone_mapping needed)
    writer.write(1, 0)?;

    // bit_depth: 8-bit integer default
    writer.write(1, 0)?; // floating_point_sample = false
    write_u32(writer, &bit_depth_int_coder(), 8)?;

    // modular_16bit_sufficient = true
    writer.write(1, 1)?;

    // extra_channel_info: 1 channel
    write_u32(writer, &extra_channel_count_coder(), 1)?;

    // ExtraChannelInfo[0]: Alpha, all_default = true
    // all_default = true means: type=Alpha, bit_depth=8bit int, dim_shift=0,
    // name="", alpha_associated=false
    writer.write(1, 1)?;

    // xyb_encoded = true
    writer.write(1, 1)?;

    // color_encoding: all_default = true (sRGB)
    writer.write(1, 1)?;

    // extensions = 0
    writer.write(2, 0)?;

    Ok(())
}

/// Write image metadata for animated JXL with alpha.
pub(crate) fn write_image_metadata_animated_with_alpha(
    writer: &mut BitWriter,
    tps_numerator: u32,
    tps_denominator: u32,
    num_loops: u32,
) -> Result<()> {
    // ImageMetadata::all_default = false
    writer.write(1, 0)?;

    // extra_fields = true (needed for have_animation)
    writer.write(1, 1)?;

    // orientation = Identity (1), Bits(3)+1
    writer.write(3, 0)?;

    // have_intrinsic_size = false
    writer.write(1, 0)?;

    // have_preview = false
    writer.write(1, 0)?;

    // have_animation = true
    writer.write(1, 1)?;
    write_u32(writer, &tps_numerator_coder(), tps_numerator)?;
    write_u32(writer, &tps_denominator_coder(), tps_denominator)?;
    write_u32(writer, &num_loops_coder(), num_loops)?;
    writer.write(1, 0)?; // have_timecodes = false

    // bit_depth: 8-bit integer
    writer.write(1, 0)?;
    write_u32(writer, &bit_depth_int_coder(), 8)?;

    // modular_16bit_sufficient = true
    writer.write(1, 1)?;

    // extra_channel_info: 1 channel (alpha)
    write_u32(writer, &extra_channel_count_coder(), 1)?;
    // ExtraChannelInfo[0]: all_default = true (Alpha, 8bit)
    writer.write(1, 1)?;

    // xyb_encoded = true
    writer.write(1, 1)?;

    // color_encoding: all_default = true (sRGB)
    writer.write(1, 1)?;

    // tone_mapping: all_default = true (conditioned on extra_fields)
    writer.write(1, 1)?;

    // extensions = 0
    writer.write(2, 0)?;

    Ok(())
}

/// Write file header with alpha channel (no animation).
pub(crate) fn write_file_header_with_alpha(
    writer: &mut BitWriter,
    width: u32,
    height: u32,
) -> Result<()> {
    writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;
    write_size(writer, width, height)?;
    write_image_metadata_with_alpha(writer)?;
    // CustomTransformData::all_default = true.
    writer.write(1, 1)?;
    Ok(())
}

/// Write file header with alpha channel + animation.
pub(crate) fn write_file_header_animated_with_alpha(
    writer: &mut BitWriter,
    width: u32,
    height: u32,
    tps_numerator: u32,
    tps_denominator: u32,
    num_loops: u32,
) -> Result<()> {
    writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;
    write_size(writer, width, height)?;
    write_image_metadata_animated_with_alpha(writer, tps_numerator, tps_denominator, num_loops)?;
    // CustomTransformData::all_default = true.
    writer.write(1, 1)?;
    Ok(())
}

/// Write file header (signature + size + metadata + CustomTransformData) for animation.
pub(crate) fn write_file_header_animated(
    writer: &mut BitWriter,
    width: u32,
    height: u32,
    tps_numerator: u32,
    tps_denominator: u32,
    num_loops: u32,
) -> Result<()> {
    writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;
    write_size(writer, width, height)?;
    write_image_metadata_animated(writer, tps_numerator, tps_denominator, num_loops)?;
    // CustomTransformData::all_default = true.
    writer.write(1, 1)?;
    Ok(())
}

fn tps_numerator_coder() -> U32Coder {
    // u2S(100, 1000, Bits(10)+1, Bits(30)+1)
    U32Coder::Select(
        U32::Val(100),
        U32::Val(1000),
        U32::BitsOffset { n: 10, off: 1 },
        U32::BitsOffset { n: 30, off: 1 },
    )
}

fn tps_denominator_coder() -> U32Coder {
    // u2S(1, 1001, Bits(8)+1, Bits(10)+1)
    U32Coder::Select(
        U32::Val(1),
        U32::Val(1001),
        U32::BitsOffset { n: 8, off: 1 },
        U32::BitsOffset { n: 10, off: 1 },
    )
}

fn num_loops_coder() -> U32Coder {
    // u2S(0, Bits(3), Bits(16), Bits(32))
    U32Coder::Select(U32::Val(0), U32::Bits(3), U32::Bits(16), U32::Bits(32))
}

pub(crate) fn write_file_header(
    writer: &mut BitWriter,
    width: u32,
    height: u32,
    xyb_encoded: bool,
    grayscale: bool,
) -> Result<()> {
    writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;

    write_size(writer, width, height)?;

    write_image_metadata(writer, xyb_encoded, grayscale)?;

    // CustomTransformData::all_default = true.
    writer.write(1, 1)?;

    Ok(())
}

fn write_minimal_file_header_fields_with_metadata(
    writer: &mut BitWriter,
    width: u32,
    height: u32,
    xyb_encoded: bool,
) -> Result<()> {
    write_file_header(writer, width, height, xyb_encoded, false)
}

fn write_minimal_file_header_fields(writer: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    write_minimal_file_header_fields_with_metadata(writer, width, height, true)
}

fn write_default_modular_frame_header_with_xyb(
    writer: &mut BitWriter,
    xyb_encoded: bool,
) -> Result<()> {
    // FrameHeader::all_default = false so we can set encoding = Modular.
    writer.write(1, 0)?;

    // frame_type = RegularFrame (0)
    writer.write(2, 0)?;

    // encoding = Modular (1)
    writer.write(1, 1)?;

    // flags = 0 (u64 coding)
    writer.write(2, 0)?;

    if !xyb_encoded {
        // do_ycbcr = false (present only for non-XYB metadata).
        writer.write(1, 0)?;
    }

    // upsampling = 1, via u2S(1,2,4,8)
    write_u32(
        writer,
        &U32Coder::Select(U32::Val(1), U32::Val(2), U32::Val(4), U32::Val(8)),
        1,
    )?;

    // group_size_shift = 1
    writer.write(2, 1)?;

    // passes.num_passes = 1 via u2S(1,2,3,Bits(3)+4)
    write_u32(
        writer,
        &U32Coder::Select(
            U32::Val(1),
            U32::Val(2),
            U32::Val(3),
            U32::BitsOffset { n: 3, off: 4 },
        ),
        1,
    )?;

    // have_crop = false
    writer.write(1, 0)?;

    // blending_info.mode = Replace (0)
    write_u32(
        writer,
        &U32Coder::Select(
            U32::Val(0),
            U32::Val(1),
            U32::Val(2),
            U32::BitsOffset { n: 2, off: 3 },
        ),
        0,
    )?;

    // is_last = true
    writer.write(1, 1)?;

    // name = "" (String length 0)
    write_u32(
        writer,
        &U32Coder::Select(
            U32::Val(0),
            U32::Bits(4),
            U32::BitsOffset { n: 5, off: 16 },
            U32::BitsOffset { n: 10, off: 48 },
        ),
        0,
    )?;

    // restoration_filter.all_default = false
    // (defaults enable Gaborish + EPF which corrupt lossless modular data)
    writer.write(1, 0)?;

    // gab = false (disable Gaborish smoothing)
    writer.write(1, 0)?;

    // epf_iters = 0 (disable edge-preserving filter)
    // Coded as Bits(2), value 0.
    writer.write(2, 0)?;

    // extensions selector = 0 (u64 coding)
    writer.write(2, 0)?;

    Ok(())
}

fn write_default_modular_frame_header(writer: &mut BitWriter) -> Result<()> {
    write_default_modular_frame_header_with_xyb(writer, true)
}

fn default_num_groups(width: u32, height: u32) -> u32 {
    // FrameHeader defaults for modular path: group_dim = 256.
    width.div_ceil(256) * height.div_ceil(256)
}

fn default_num_lf_groups(width: u32, height: u32) -> u32 {
    // With default frame header parameters, LF groups operate on size_blocks()
    // where each block is 8x8, then grouped by group_dim=256 blocks.
    width.div_ceil(2048) * height.div_ceil(2048)
}

fn default_num_toc_entries(width: u32, height: u32) -> u32 {
    let num_groups = default_num_groups(width, height);

    if num_groups == 1 {
        return 1;
    }

    let num_lf_groups = default_num_lf_groups(width, height);
    2 + num_lf_groups + num_groups
}

/// Encodes a minimal JPEG XL codestream header that can be parsed up to
/// `WithImageInfo` by the decoder.
///
/// Current bootstrap format choices:
/// - default image metadata (`all_default = true`)
/// - default custom transform data (`all_default = true`)
/// - no frame data (header-only codestream)
pub fn encode_minimal_codestream_header(size: (u32, u32)) -> Result<Vec<u8>> {
    let (width, height) = size;
    let mut writer = BitWriter::new();
    write_minimal_file_header_fields(&mut writer, width, height)?;
    Ok(writer.finish())
}

/// Encodes a minimal single-frame codestream up to frame metadata + TOC.
///
/// Notes:
/// - Frame sections are present with zero lengths.
/// - This is useful for parser/bootstrap testing and frame-header plumbing.
pub fn encode_minimal_single_frame_codestream(size: (u32, u32)) -> Result<Vec<u8>> {
    let (width, height) = size;
    validate_size(width, height)?;

    let mut writer = BitWriter::new();
    write_minimal_file_header_fields(&mut writer, width, height)?;

    // The decoder byte-aligns before frame-header parsing.
    writer.byte_align_zero_pad()?;

    // FrameHeader::all_default = true.
    writer.write(1, 1)?;

    // TOC (num_entries based on default frame geometry).
    let entries = vec![0u32; default_num_toc_entries(width, height) as usize];
    write_toc(&mut writer, &entries)?;

    Ok(writer.finish())
}

/// Encodes a minimal fully decodable modular single-frame codestream.
///
/// The produced image uses a constant modular leaf offset/predictor pair.
pub fn encode_minimal_modular_image_codestream_with_params(
    size: (u32, u32),
    offset: i32,
    predictor: u32,
) -> Result<Vec<u8>> {
    let (width, height) = size;
    validate_size(width, height)?;

    let num_groups = default_num_groups(width, height);
    let num_lf_groups = default_num_lf_groups(width, height);

    let mut lf_global_writer = BitWriter::new();
    write_minimal_modular_lf_global_section_with_params(&mut lf_global_writer, offset, predictor)?;
    let lf_global_section = lf_global_writer.finish();

    let mut hf_group_writer = BitWriter::new();
    write_minimal_modular_global_data_with_params(&mut hf_group_writer, offset, predictor)?;
    let hf_group_section = hf_group_writer.finish();

    let mut writer = BitWriter::new();
    write_minimal_file_header_fields(&mut writer, width, height)?;

    // The decoder byte-aligns before frame-header parsing.
    writer.byte_align_zero_pad()?;

    write_default_modular_frame_header(&mut writer)?;

    if num_groups == 1 {
        // Single-group special case: one combined section.
        write_toc(&mut writer, &[lf_global_section.len() as u32])?;
        writer.write_aligned_bytes(&lf_global_section)?;
        return Ok(writer.finish());
    }

    // General case layout:
    // - LF global
    // - LF groups (all empty for this bootstrap stream)
    // - HF global (empty)
    // - HF groups (one minimal modular sub-bitstream per group)
    let mut toc_entries = Vec::new();
    toc_entries.push(lf_global_section.len() as u32);
    toc_entries.extend(std::iter::repeat_n(0u32, num_lf_groups as usize));
    toc_entries.push(0); // HF global
    toc_entries.extend(std::iter::repeat_n(
        hf_group_section.len() as u32,
        num_groups as usize,
    ));

    write_toc(&mut writer, &toc_entries)?;

    writer.write_aligned_bytes(&lf_global_section)?;
    for _ in 0..num_groups {
        writer.write_aligned_bytes(&hf_group_section)?;
    }

    Ok(writer.finish())
}

/// Encodes a minimal fully decodable modular single-frame codestream with
/// predictor `Zero`.
pub fn encode_minimal_modular_image_codestream_with_offset(
    size: (u32, u32),
    offset: i32,
) -> Result<Vec<u8>> {
    encode_minimal_modular_image_codestream_with_params(size, offset, 0)
}

/// Encodes a minimal fully decodable modular single-frame codestream.
pub fn encode_minimal_modular_image_codestream(size: (u32, u32)) -> Result<Vec<u8>> {
    encode_minimal_modular_image_codestream_with_params(size, 0, 0)
}

/// Encodes an interleaved RGB8 buffer into a modular codestream.
///
/// Uses histogram-driven Huffman encoding for proper compression.
/// The encoding uses the West (left) predictor with offset 0 for good
/// compression of smooth images; channel-major, row-major order.
pub fn encode_modular_u8_rgb_image_codestream(size: (u32, u32), rgb: &[u8]) -> Result<Vec<u8>> {
    encode_modular_u8_rgb_image_codestream_with_mode(size, rgb, false)
}

pub(crate) fn encode_modular_u8_rgb_image_codestream_with_mode(
    size: (u32, u32),
    rgb: &[u8],
    fast_lossless: bool,
) -> Result<Vec<u8>> {
    let (width, height) = size;
    validate_size(width, height)?;
    validate_rgb_u8_buffer(width, height, rgb)?;

    let uint_config = if fast_lossless {
        HybridUintConfig::new(3, 1, 0)
    } else {
        HybridUintConfig::new(4, 2, 0)
    };

    let width_usize = width as usize;
    let num_groups = default_num_groups(width, height);
    let num_lf_groups = default_num_lf_groups(width, height);
    let num_groups_x = width.div_ceil(256);

    /// Collect signed residuals for a group region using a modular predictor.
    fn collect_group_residuals_predictor(
        rgb: &[u8],
        width_usize: usize,
        origin: (u32, u32),
        group_size: (u32, u32),
        offset: i32,
        predictor: u32,
    ) -> Vec<i32> {
        let (ox, oy) = origin;
        let (gw, gh) = group_size;
        let gw_usize = gw as usize;
        let mut residuals = Vec::with_capacity(gw_usize * (gh as usize) * 3);

        // Channel-major, row-major within each channel.
        for channel in 0..3 {
            // Buffer previous row for predictors that use N/NW
            let mut prev_row = vec![0i32; gw_usize];

            for y in oy..(oy + gh) {
                let local_y = (y - oy) as usize;
                let mut curr_row = Vec::with_capacity(gw_usize);

                for lx in 0..gw_usize {
                    let x = ox as usize + lx;
                    let pixel_index = y as usize * width_usize + x;
                    let sample = i32::from(rgb[pixel_index * 3 + channel]);

                    let w = if lx > 0 { curr_row[lx - 1] } else { if local_y > 0 { prev_row[0] } else { 0 } };
                    let n = if local_y > 0 { prev_row[lx] } else { w };
                    let nw = if local_y > 0 && lx > 0 { prev_row[lx - 1] } else { w };

                    let pred = match predictor {
                        1 => w,                           // Left
                        2 => n,                           // Top
                        3 => (w + n) / 2,                 // Average(W, N)
                        4 => {                            // Select (median-like)
                            let p = w + n - nw;
                            if (p - w).abs() < (p - n).abs() { w } else { n }
                        }
                        5 => w + n - nw,                   // Gradient
                        _ => 0,                           // Zero
                    };
                    residuals.push(sample - (pred + offset));
                    curr_row.push(sample);
                }
                prev_row = curr_row;
            }
        }
        residuals
    }

    fn residual_score(residuals: &[i32]) -> u64 {
        residuals
            .iter()
            .map(|&v| (v as i64).unsigned_abs())
            .sum::<u64>()
    }

    // Single-group path: LF global section contains tree + residuals.
    if num_groups == 1 {
        let offset = 0i32;
        let mut best_predictor = if fast_lossless { 1u32 } else { 0u32 };
        let mut best_residuals = collect_group_residuals_predictor(
            rgb,
            width_usize,
            (0, 0),
            (width, height),
            offset,
            best_predictor,
        );
        let mut best_score = residual_score(&best_residuals);

        if !fast_lossless {
            for predictor in [1u32, 2, 4, 5] {
                let residuals = collect_group_residuals_predictor(
                    rgb,
                    width_usize,
                    (0, 0),
                    (width, height),
                    offset,
                    predictor,
                );
                let score = residual_score(&residuals);
                if score < best_score {
                    best_score = score;
                    best_predictor = predictor;
                    best_residuals = residuals;
                }
            }
        }

        let mut lf_global_writer = BitWriter::new();
        write_lf_global_section_huffman(
            &mut lf_global_writer,
            offset,
            best_predictor,
            Some(&best_residuals),
            uint_config,
        )?;
        let lf_global_section = lf_global_writer.finish();

        let mut writer = BitWriter::new();
        write_minimal_file_header_fields_with_metadata(&mut writer, width, height, false)?;
        writer.byte_align_zero_pad()?;
        write_default_modular_frame_header_with_xyb(&mut writer, false)?;
        write_toc(&mut writer, &[lf_global_section.len() as u32])?;
        writer.write_aligned_bytes(&lf_global_section)?;

        return Ok(writer.finish());
    }

    // Multi-group path:
    // - section 0: LF global (tree + empty residual histograms)
    // - section 1..N: LF groups (empty)
    // - section N+1: HF global (empty)
    // - section N+2..: HF groups (local tree + residual data)
    let mut lf_global_writer = BitWriter::new();
    write_lf_global_section_huffman(&mut lf_global_writer, 0, 0, None, uint_config)?;
    let lf_global_section = lf_global_writer.finish();

    let mut hf_group_sections = Vec::with_capacity(num_groups as usize);
    for group in 0..num_groups {
        let gx = group % num_groups_x;
        let gy = group / num_groups_x;
        let ox = gx * 256;
        let oy = gy * 256;
        let gw = (width - ox).min(256);
        let gh = (height - oy).min(256);

        let offset = 0i32;
        let mut best_predictor = if fast_lossless { 1u32 } else { 0u32 };
        let mut best_residuals = collect_group_residuals_predictor(
            rgb,
            width_usize,
            (ox, oy),
            (gw, gh),
            offset,
            best_predictor,
        );
        let mut best_score = residual_score(&best_residuals);

        if !fast_lossless {
            for predictor in [1u32, 2, 4, 5] {
                let residuals = collect_group_residuals_predictor(
                    rgb,
                    width_usize,
                    (ox, oy),
                    (gw, gh),
                    offset,
                    predictor,
                );
                let score = residual_score(&residuals);
                if score < best_score {
                    best_score = score;
                    best_predictor = predictor;
                    best_residuals = residuals;
                }
            }
        }

        let mut group_writer = BitWriter::new();
        write_hf_group_section_huffman(
            &mut group_writer,
            offset,
            best_predictor,
            &best_residuals,
            uint_config,
        )?;
        hf_group_sections.push(group_writer.finish());
    }

    let mut writer = BitWriter::new();
    write_minimal_file_header_fields_with_metadata(&mut writer, width, height, false)?;
    writer.byte_align_zero_pad()?;
    write_default_modular_frame_header_with_xyb(&mut writer, false)?;

    let mut toc_entries = Vec::new();
    toc_entries.push(lf_global_section.len() as u32);
    toc_entries.extend(std::iter::repeat_n(0u32, num_lf_groups as usize));
    toc_entries.push(0); // HF global
    toc_entries.extend(hf_group_sections.iter().map(|s| s.len() as u32));

    write_toc(&mut writer, &toc_entries)?;
    writer.write_aligned_bytes(&lf_global_section)?;
    for section in hf_group_sections {
        writer.write_aligned_bytes(&section)?;
    }

    Ok(writer.finish())
}

/// Encodes an interleaved Gray8 buffer into a native grayscale modular codestream.
///
/// Unlike `encode_modular_u8_rgb_image_codestream`, this produces a single-channel
/// modular image with `color_space=Gray`, avoiding the 3x data expansion of
/// converting gray to RGB.
pub fn encode_modular_u8_gray_image_codestream(size: (u32, u32), gray: &[u8]) -> Result<Vec<u8>> {
    encode_modular_u8_gray_image_codestream_with_mode(size, gray, false)
}

pub(crate) fn encode_modular_u8_gray_image_codestream_with_mode(
    size: (u32, u32),
    gray: &[u8],
    fast_lossless: bool,
) -> Result<Vec<u8>> {
    let (width, height) = size;
    validate_size(width, height)?;
    let expected_len = (width as usize) * (height as usize);
    if gray.len() != expected_len {
        return Err(Error::InvalidPixelBufferLength {
            expected: expected_len,
            actual: gray.len(),
        });
    }

    let uint_config = if fast_lossless {
        HybridUintConfig::new(3, 1, 0)
    } else {
        HybridUintConfig::new(4, 2, 0)
    };

    let width_usize = width as usize;
    let num_groups = default_num_groups(width, height);
    let num_lf_groups = default_num_lf_groups(width, height);
    let num_groups_x = width.div_ceil(256);

    /// Collect signed residuals for a gray group region using a predictor.
    fn collect_gray_residuals_predictor(
        gray: &[u8],
        width_usize: usize,
        origin: (u32, u32),
        group_size: (u32, u32),
        predictor: u32,
    ) -> Vec<i32> {
        let (ox, oy) = origin;
        let (gw, gh) = group_size;
        let mut residuals = Vec::with_capacity((gw as usize) * (gh as usize));

        // Single channel, row-major.
        let mut prev_row_first = 0i32;
        for y in oy..(oy + gh) {
            let local_y = (y - oy) as usize;
            let first_pred = match predictor {
                1 => {
                    if local_y > 0 {
                        prev_row_first
                    } else {
                        0
                    }
                }
                _ => 0,
            };
            let first_idx = y as usize * width_usize + ox as usize;
            let first_sample = i32::from(gray[first_idx]);
            residuals.push(first_sample - first_pred);
            prev_row_first = first_sample;

            let mut prev = first_sample;
            for x in (ox + 1)..(ox + gw) {
                let idx = y as usize * width_usize + x as usize;
                let sample = i32::from(gray[idx]);
                let pred = match predictor {
                    1 => prev,
                    _ => 0,
                };
                residuals.push(sample - pred);
                prev = sample;
            }
        }
        residuals
    }

    fn residual_score(residuals: &[i32]) -> u64 {
        residuals
            .iter()
            .map(|&v| (v as i64).unsigned_abs())
            .sum::<u64>()
    }

    // Single-group path
    if num_groups == 1 {
        let mut best_predictor = if fast_lossless { 1u32 } else { 0u32 };
        let mut best_residuals = collect_gray_residuals_predictor(
            gray,
            width_usize,
            (0, 0),
            (width, height),
            best_predictor,
        );
        let mut best_score = residual_score(&best_residuals);
        if !fast_lossless {
            for predictor in [1u32, 2, 4, 5] {
                let residuals = collect_gray_residuals_predictor(
                    gray,
                    width_usize,
                    (0, 0),
                    (width, height),
                    predictor,
                );
                let score = residual_score(&residuals);
                if score < best_score {
                    best_score = score;
                    best_predictor = predictor;
                    best_residuals = residuals;
                }
            }
        }

        let mut lf_global_writer = BitWriter::new();
        write_lf_global_section_huffman(
            &mut lf_global_writer,
            0,
            best_predictor,
            Some(&best_residuals),
            uint_config,
        )?;
        let lf_global_section = lf_global_writer.finish();

        let mut writer = BitWriter::new();
        write_file_header(&mut writer, width, height, false, true)?;
        writer.byte_align_zero_pad()?;
        write_default_modular_frame_header_with_xyb(&mut writer, false)?;
        write_toc(&mut writer, &[lf_global_section.len() as u32])?;
        writer.write_aligned_bytes(&lf_global_section)?;
        return Ok(writer.finish());
    }

    // Multi-group path
    let mut lf_global_writer = BitWriter::new();
    write_lf_global_section_huffman(&mut lf_global_writer, 0, 0, None, uint_config)?;
    let lf_global_section = lf_global_writer.finish();

    let mut hf_group_sections = Vec::with_capacity(num_groups as usize);
    for group in 0..num_groups {
        let gx = group % num_groups_x;
        let gy = group / num_groups_x;
        let ox = gx * 256;
        let oy = gy * 256;
        let gw = (width - ox).min(256);
        let gh = (height - oy).min(256);

        let mut best_predictor = if fast_lossless { 1u32 } else { 0u32 };
        let mut best_residuals =
            collect_gray_residuals_predictor(gray, width_usize, (ox, oy), (gw, gh), best_predictor);
        let mut best_score = residual_score(&best_residuals);
        if !fast_lossless {
            for predictor in [1u32, 2, 4, 5] {
                let residuals = collect_gray_residuals_predictor(
                    gray,
                    width_usize,
                    (ox, oy),
                    (gw, gh),
                    predictor,
                );
                let score = residual_score(&residuals);
                if score < best_score {
                    best_score = score;
                    best_predictor = predictor;
                    best_residuals = residuals;
                }
            }
        }

        let mut group_writer = BitWriter::new();
        write_hf_group_section_huffman(
            &mut group_writer,
            0,
            best_predictor,
            &best_residuals,
            uint_config,
        )?;
        hf_group_sections.push(group_writer.finish());
    }

    let mut writer = BitWriter::new();
    write_file_header(&mut writer, width, height, false, true)?;
    writer.byte_align_zero_pad()?;
    write_default_modular_frame_header_with_xyb(&mut writer, false)?;

    let mut toc_entries = Vec::new();
    toc_entries.push(lf_global_section.len() as u32);
    toc_entries.extend(std::iter::repeat_n(0u32, num_lf_groups as usize));
    toc_entries.push(0); // HF global
    toc_entries.extend(hf_group_sections.iter().map(|s| s.len() as u32));

    write_toc(&mut writer, &toc_entries)?;
    writer.write_aligned_bytes(&lf_global_section)?;
    for section in hf_group_sections {
        writer.write_aligned_bytes(&section)?;
    }

    Ok(writer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::{JxlBitDepth, JxlDecoder, JxlDecoderOptions, ProcessingResult, states},
        bit_reader::BitReader,
        encode::{container, modular_encode},
        headers::{
            FileHeader, JxlHeader,
            encodings::{Empty, UnconditionalCoder},
            image_metadata::ImageMetadata,
            size::Size,
            transform_data::{CustomTransformData, CustomTransformDataNonserialized},
        },
    };

    fn decode_to_image_info(mut input: &[u8]) -> (JxlDecoder<states::WithImageInfo>, &[u8]) {
        let mut dec = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut prev_len = input.len().saturating_add(1);

        for _ in 0..32 {
            match dec.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => return (result, input),
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    assert!(
                        input.len() < prev_len,
                        "decoder made no forward progress while asking for more input"
                    );
                    prev_len = input.len();
                    assert!(
                        !input.is_empty(),
                        "unexpected EOF while parsing minimal header"
                    );
                    dec = fallback;
                }
            }
        }

        panic!("decoder did not reach image-info state within expected iterations");
    }

    fn decode_to_frame_info(input: &[u8]) -> JxlDecoder<states::WithFrameInfo> {
        let (mut dec, mut input) = decode_to_image_info(input);
        let mut prev_len = input.len().saturating_add(1);

        for _ in 0..32 {
            match dec.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => return result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    assert!(
                        input.len() < prev_len,
                        "decoder made no forward progress while asking for frame input"
                    );
                    prev_len = input.len();
                    assert!(!input.is_empty(), "unexpected EOF while parsing frame info");
                    dec = fallback;
                }
            }
        }

        panic!("decoder did not reach frame-info state within expected iterations");
    }

    #[test]
    fn test_encode_minimal_header_components_roundtrip() {
        let codestream = encode_minimal_codestream_header((321, 123)).unwrap();

        let mut br = BitReader::new(&codestream);

        // Signature (0xFF 0x0A)
        assert_eq!(br.read(8).unwrap(), 0xFF);
        assert_eq!(br.read(8).unwrap(), 0x0A);

        let size = Size::read_unconditional(&(), &mut br, &Empty {}).unwrap();
        assert_eq!(size.xsize(), 321);
        assert_eq!(size.ysize(), 123);

        let metadata = ImageMetadata::read_unconditional(&(), &mut br, &Empty {}).unwrap();
        assert!(metadata.xyb_encoded);

        let nonserialized = CustomTransformDataNonserialized {
            xyb_encoded: metadata.xyb_encoded,
        };
        let _transform =
            CustomTransformData::read_unconditional(&(), &mut br, &nonserialized).unwrap();
    }

    #[test]
    fn test_encode_minimal_file_header_roundtrip() {
        let codestream = encode_minimal_codestream_header((321, 123)).unwrap();

        let mut br = BitReader::new(&codestream);
        let header = FileHeader::read(&mut br).unwrap();

        assert_eq!(header.size.xsize(), 321);
        assert_eq!(header.size.ysize(), 123);
    }

    #[test]
    fn test_encode_minimal_codestream_header_parses_info() {
        let codestream = encode_minimal_codestream_header((321, 123)).unwrap();
        let (dec, _remaining) = decode_to_image_info(&codestream);

        let info = dec.basic_info();
        assert_eq!(info.size, (321, 123));
        assert_eq!(info.bit_depth, JxlBitDepth::Int { bits_per_sample: 8 });
        assert!(info.extra_channels.is_empty());
    }

    #[test]
    fn test_encode_minimal_header_in_container_parses_info() {
        let codestream = encode_minimal_codestream_header((77, 66)).unwrap();
        let container_stream = container::wrap_codestream(&codestream).unwrap();
        let (dec, _remaining) = decode_to_image_info(&container_stream);

        assert_eq!(dec.basic_info().size, (77, 66));
    }

    #[test]
    fn test_encode_minimal_single_frame_codestream_parses_frame_info() {
        let codestream = encode_minimal_single_frame_codestream((321, 123)).unwrap();
        let frame_decoder = decode_to_frame_info(&codestream);
        assert_eq!(frame_decoder.frame_header().size, (321, 123));
    }

    #[test]
    fn test_encode_minimal_single_frame_container_parses_frame_info() {
        let codestream = encode_minimal_single_frame_codestream((321, 123)).unwrap();
        let container_stream = container::wrap_codestream(&codestream).unwrap();
        let frame_decoder = decode_to_frame_info(&container_stream);
        assert_eq!(frame_decoder.frame_header().size, (321, 123));
    }

    #[test]
    fn test_encode_minimal_modular_image_codestream_parses_frame_info() {
        let codestream = encode_minimal_modular_image_codestream((17, 9)).unwrap();
        let frame_decoder = decode_to_frame_info(&codestream);
        assert_eq!(frame_decoder.frame_header().size, (17, 9));
    }

    #[test]
    fn test_encode_minimal_modular_image_container_parses_frame_info() {
        let codestream = encode_minimal_modular_image_codestream((17, 9)).unwrap();
        let container_stream = container::wrap_codestream(&codestream).unwrap();
        let frame_decoder = decode_to_frame_info(&container_stream);
        assert_eq!(frame_decoder.frame_header().size, (17, 9));
    }

    #[test]
    fn test_encode_minimal_modular_image_decodes_one_frame() {
        let codestream = encode_minimal_modular_image_codestream((1, 1)).unwrap();
        let (decoded_frames, frames) =
            crate::api::tests::decode(&codestream, usize::MAX, false, false, None).unwrap();

        assert_eq!(decoded_frames, 1);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), 1);
        assert_eq!(frames[0][0].size(), (3, 1));

        let row = frames[0][0].row(0);
        for &v in row {
            assert!((v - 0.0).abs() < 1e-6, "expected black pixel, got {v}");
        }
    }

    #[test]
    fn test_encode_minimal_modular_image_container_decodes_one_frame() {
        let codestream = encode_minimal_modular_image_codestream((1, 1)).unwrap();
        let container_stream = container::wrap_codestream(&codestream).unwrap();
        let (decoded_frames, frames) =
            crate::api::tests::decode(&container_stream, usize::MAX, false, false, None).unwrap();

        assert_eq!(decoded_frames, 1);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0][0].size(), (3, 1));
    }

    #[test]
    fn test_encode_minimal_modular_image_multigroup_decodes_one_frame() {
        let codestream = encode_minimal_modular_image_codestream((257, 1)).unwrap();
        let (decoded_frames, frames) =
            crate::api::tests::decode(&codestream, usize::MAX, false, false, None).unwrap();

        assert_eq!(decoded_frames, 1);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0][0].size(), (257 * 3, 1));

        let row = frames[0][0].row(0);
        for &v in row {
            assert!((v - 0.0).abs() < 1e-6, "expected black pixel, got {v}");
        }
    }

    #[test]
    fn test_encode_minimal_modular_image_with_offset_decodes_non_black() {
        let codestream = encode_minimal_modular_image_codestream_with_offset((8, 4), 12).unwrap();
        let (decoded_frames, frames) =
            crate::api::tests::decode(&codestream, usize::MAX, false, false, None).unwrap();

        assert_eq!(decoded_frames, 1);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0][0].size(), (8 * 3, 4));

        let mut any_non_zero = false;
        for y in 0..4 {
            for &v in frames[0][0].row(y) {
                if v.abs() > 1e-6 {
                    any_non_zero = true;
                    break;
                }
            }
            if any_non_zero {
                break;
            }
        }
        assert!(
            any_non_zero,
            "expected non-black pixels for non-zero offset"
        );
    }

    #[test]
    fn test_encode_minimal_modular_image_with_west_predictor_varies_pixels() {
        let codestream = encode_minimal_modular_image_codestream_with_params((8, 1), 1, 1).unwrap();
        let (decoded_frames, frames) =
            crate::api::tests::decode(&codestream, usize::MAX, false, false, None).unwrap();

        assert_eq!(decoded_frames, 1);
        let row = frames[0][0].row(0);
        assert!(row.len() >= 6);

        let first_pixel_r = row[0];
        let second_pixel_r = row[3];
        assert!(
            (second_pixel_r - first_pixel_r).abs() > 1e-6,
            "expected varying pixels with west predictor"
        );
    }

    #[test]
    fn test_encode_modular_u8_rgb_image_codestream_invalid_len() {
        let err = encode_modular_u8_rgb_image_codestream((4, 4), &[0u8; 3]).unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidPixelBufferLength {
                expected: 48,
                actual: 3
            }
        ));
    }

    #[test]
    fn test_encode_modular_u8_rgb_image_codestream_has_non_xyb_metadata() {
        let size = (8, 4);
        let rgb = vec![0u8; (size.0 * size.1 * 3) as usize];
        let codestream = encode_modular_u8_rgb_image_codestream(size, &rgb).unwrap();

        let mut br = BitReader::new(&codestream);
        let header = FileHeader::read(&mut br).unwrap();
        assert!(!header.image_metadata.xyb_encoded);
    }

    #[test]
    fn test_encode_modular_u8_rgb_image_codestream_decodes_and_changes_with_input() {
        let size = (8, 4);
        let rgb_a = vec![0u8; (size.0 * size.1 * 3) as usize];
        let mut rgb_b = rgb_a.clone();
        rgb_b[0] = 255;

        let cs_a = encode_modular_u8_rgb_image_codestream(size, &rgb_a).unwrap();
        let cs_b = encode_modular_u8_rgb_image_codestream(size, &rgb_b).unwrap();
        assert_ne!(cs_a, cs_b);

        let (decoded_a, frames_a) =
            crate::api::tests::decode(&cs_a, usize::MAX, false, false, None).unwrap();
        let (decoded_b, frames_b) =
            crate::api::tests::decode(&cs_b, usize::MAX, false, false, None).unwrap();

        assert_eq!(decoded_a, 1);
        assert_eq!(decoded_b, 1);
        assert_eq!(
            frames_a[0][0].size(),
            ((size.0 * 3) as usize, size.1 as usize)
        );
        assert_eq!(
            frames_b[0][0].size(),
            ((size.0 * 3) as usize, size.1 as usize)
        );

        let row_a0 = frames_a[0][0].row(0);
        for &v in row_a0 {
            assert!((v - 0.0).abs() < 1e-6, "expected zero sample, got {v}");
        }

        assert_ne!(frames_a[0][0].row(0)[0], frames_b[0][0].row(0)[0]);

        // Determinism for fixed input.
        let cs_a2 = encode_modular_u8_rgb_image_codestream(size, &rgb_a).unwrap();
        assert_eq!(cs_a, cs_a2);
    }

    #[test]
    fn test_encode_multigroup_lf_global_section_parses() {
        // Test that the LF global section for multi-group encodes parses correctly.
        let uint_config = HybridUintConfig::new(4, 2, 0);
        let mut lf_writer = BitWriter::new();
        modular_encode::write_lf_global_section_huffman(&mut lf_writer, 0, 0, None, uint_config)
            .unwrap();
        let lf_bytes = lf_writer.finish();

        use crate::bit_reader::BitReader;
        let mut br = BitReader::new(&lf_bytes);

        // LfQuantFactors: all_default
        let all_default = br.read(1).unwrap();
        assert_eq!(all_default, 1);

        // No global tree
        let global_tree = br.read(1).unwrap();
        assert_eq!(global_tree, 0);

        // Parse group header
        let header = crate::headers::modular::GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);

        // Parse tree (should read tree histograms + data + residual histograms)
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        eprintln!(
            "LF global tree: {} nodes, bit_pos={}",
            tree.nodes.len(),
            br.total_bits_read()
        );
        assert_eq!(tree.nodes.len(), 1);
    }

    #[test]
    fn test_encode_multigroup_hf_group_section_parses() {
        // Test that an HF group section decodes correctly.
        let uint_config = HybridUintConfig::new(4, 2, 0);
        let residuals: Vec<i32> = vec![1, 2, 3, 4, 5];

        let mut group_writer = BitWriter::new();
        modular_encode::write_hf_group_section_huffman(
            &mut group_writer,
            5,
            0,
            &residuals,
            uint_config,
        )
        .unwrap();
        let group_bytes = group_writer.finish();

        use crate::bit_reader::BitReader;
        let mut br = BitReader::new(&group_bytes);

        // Parse group header
        let header = crate::headers::modular::GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);

        // Parse tree
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);

        // Read residuals
        use crate::entropy_coding::decode::SymbolReader;
        let mut reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for &expected in &residuals {
            let got = reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(got, expected);
        }
        reader.check_final_state(&tree.histograms, &mut br).unwrap();
    }

    #[test]
    fn test_encode_modular_u8_rgb_image_codestream_multigroup_all_zeros() {
        // Simple all-zeros image, 300x3 = 2 groups.
        let size = (300, 3);
        let rgb = vec![0u8; (size.0 * size.1 * 3) as usize];

        let cs = encode_modular_u8_rgb_image_codestream(size, &rgb).unwrap();
        let (decoded, _frames) =
            crate::api::tests::decode(&cs, usize::MAX, false, false, None).unwrap();
        assert_eq!(decoded, 1);
    }

    #[test]
    fn test_encode_modular_u8_rgb_image_codestream_multigroup_decodes() {
        // 300x3 = 2 groups: [0..256) and [256..300).
        let size = (300, 3);
        let mut rgb = vec![0u8; (size.0 * size.1 * 3) as usize];
        for y in 0..size.1 as usize {
            for x in 0..size.0 as usize {
                let idx = (y * size.0 as usize + x) * 3;
                rgb[idx] = (x % 256) as u8;
                rgb[idx + 1] = (y * 40) as u8;
                rgb[idx + 2] = ((x + y) % 256) as u8;
            }
        }

        let cs = encode_modular_u8_rgb_image_codestream(size, &rgb).unwrap();
        let (decoded, frames) =
            crate::api::tests::decode(&cs, usize::MAX, false, false, None).unwrap();

        assert_eq!(decoded, 1);
        assert_eq!(
            frames[0][0].size(),
            ((size.0 * 3) as usize, size.1 as usize)
        );
    }

    #[test]
    fn test_encode_pixel_roundtrip_constant() {
        // 2x1 constant image: (50,50,50) and (50,50,50).
        let size = (2u32, 1u32);
        let rgb: Vec<u8> = vec![50, 50, 50, 50, 50, 50];

        let cs = encode_modular_u8_rgb_image_codestream(size, &rgb).unwrap();
        let (_decoded, frames) =
            crate::api::tests::decode(&cs, usize::MAX, false, false, None).unwrap();

        let img = &frames[0][0];
        let row = img.row(0);
        for x in 0..size.0 as usize {
            let r = (row[x * 3] * 255.0 + 0.5) as u8;
            let g = (row[x * 3 + 1] * 255.0 + 0.5) as u8;
            let b = (row[x * 3 + 2] * 255.0 + 0.5) as u8;
            eprintln!("constant pixel ({x},0): decoded ({r},{g},{b}), expected (50,50,50)");
            assert_eq!(r, 50, "R mismatch at x={x}");
            assert_eq!(g, 50, "G mismatch at x={x}");
            assert_eq!(b, 50, "B mismatch at x={x}");
        }
    }

    #[test]
    fn test_encode_pixel_roundtrip_2vals() {
        // 2x1 image with two distinct pixel values.
        let size = (2u32, 1u32);
        let rgb: Vec<u8> = vec![50, 50, 50, 100, 100, 100];
        let cs = encode_modular_u8_rgb_image_codestream(size, &rgb).unwrap();

        // Test with both pipelines
        for use_simple in [true, false] {
            let (_decoded, frames) =
                crate::api::tests::decode(&cs, usize::MAX, use_simple, false, None).unwrap();
            let img = &frames[0][0];
            let row = img.row(0);
            for x in 0..size.0 as usize {
                let r = (row[x * 3] * 255.0 + 0.5) as u8;
                let g = (row[x * 3 + 1] * 255.0 + 0.5) as u8;
                let b = (row[x * 3 + 2] * 255.0 + 0.5) as u8;
                assert_eq!(r, rgb[x * 3], "R mismatch at x={x}");
                assert_eq!(g, rgb[x * 3 + 1], "G mismatch at x={x}");
                assert_eq!(b, rgb[x * 3 + 2], "B mismatch at x={x}");
            }
        }
    }

    #[test]
    fn test_encode_pixel_roundtrip_simple() {
        // 2x1 image with different R values, G=B=0.
        let size = (2u32, 1u32);
        let rgb: Vec<u8> = vec![50, 0, 0, 100, 0, 0];
        let cs = encode_modular_u8_rgb_image_codestream(size, &rgb).unwrap();
        let (_decoded, frames) =
            crate::api::tests::decode(&cs, usize::MAX, false, false, None).unwrap();
        let img = &frames[0][0];
        let row = img.row(0);
        for x in 0..size.0 as usize {
            let r = (row[x * 3] * 255.0 + 0.5) as u8;
            let g = (row[x * 3 + 1] * 255.0 + 0.5) as u8;
            let b = (row[x * 3 + 2] * 255.0 + 0.5) as u8;
            assert_eq!(r, rgb[x * 3], "R mismatch at x={x}");
            assert_eq!(g, rgb[x * 3 + 1], "G mismatch at x={x}");
            assert_eq!(b, rgb[x * 3 + 2], "B mismatch at x={x}");
        }
    }

    #[test]
    fn test_encode_modular_u8_rgb_image_container_decodes() {
        let size = (16, 8);
        let rgb = vec![0u8; (size.0 * size.1 * 3) as usize];
        let cs = encode_modular_u8_rgb_image_codestream(size, &rgb).unwrap();
        let container_stream = container::wrap_codestream(&cs).unwrap();

        let (decoded, frames) =
            crate::api::tests::decode(&container_stream, usize::MAX, false, false, None).unwrap();
        assert_eq!(decoded, 1);
        assert_eq!(
            frames[0][0].size(),
            ((size.0 * 3) as usize, size.1 as usize)
        );
    }

    #[test]
    fn test_minimal_header_snapshot_1x1() {
        let codestream = encode_minimal_codestream_header((1, 1)).unwrap();
        assert_eq!(codestream, vec![0xFF, 0x0A, 0x00, 0x00, 0x00, 0x0C]);
    }

    #[test]
    fn test_minimal_header_deterministic_output() {
        for size in [(1, 1), (17, 9), (321, 123), (4096, 2048)] {
            let a = encode_minimal_codestream_header(size).unwrap();
            let b = encode_minimal_codestream_header(size).unwrap();
            assert_eq!(a, b, "non-deterministic output for size {size:?}");

            let a = encode_minimal_single_frame_codestream(size).unwrap();
            let b = encode_minimal_single_frame_codestream(size).unwrap();
            assert_eq!(
                a, b,
                "non-deterministic single-frame output for size {size:?}"
            );
        }

        for size in [
            (1, 1),
            (17, 9),
            (128, 128),
            (256, 256),
            (257, 1),
            (257, 257),
        ] {
            let a = encode_minimal_modular_image_codestream(size).unwrap();
            let b = encode_minimal_modular_image_codestream(size).unwrap();
            assert_eq!(a, b, "non-deterministic modular output for size {size:?}");

            let a = encode_minimal_modular_image_codestream_with_offset(size, 7).unwrap();
            let b = encode_minimal_modular_image_codestream_with_offset(size, 7).unwrap();
            assert_eq!(
                a, b,
                "non-deterministic modular output with offset for size {size:?}"
            );

            let a = encode_minimal_modular_image_codestream_with_params(size, 7, 1).unwrap();
            let b = encode_minimal_modular_image_codestream_with_params(size, 7, 1).unwrap();
            assert_eq!(
                a, b,
                "non-deterministic modular output with params for size {size:?}"
            );
        }
    }

    /// Helper: encode RGB8 image, decode with jxl-rs, verify pixel-perfect match.
    fn assert_pixel_perfect_roundtrip(width: u32, height: u32, rgb: &[u8]) {
        assert_eq!(
            rgb.len(),
            (width as usize) * (height as usize) * 3,
            "wrong buffer size"
        );
        let cs = encode_modular_u8_rgb_image_codestream((width, height), rgb).unwrap();
        let (_decoded, frames) =
            crate::api::tests::decode(&cs, usize::MAX, false, false, None).unwrap();
        let img = &frames[0][0];
        assert_eq!(
            img.size(),
            ((width * 3) as usize, height as usize),
            "decoded image size mismatch"
        );
        for y in 0..height as usize {
            let row = img.row(y);
            for x in 0..width as usize {
                let r = (row[x * 3] * 255.0 + 0.5) as u8;
                let g = (row[x * 3 + 1] * 255.0 + 0.5) as u8;
                let b = (row[x * 3 + 2] * 255.0 + 0.5) as u8;
                let idx = (y * width as usize + x) * 3;
                assert_eq!(
                    (r, g, b),
                    (rgb[idx], rgb[idx + 1], rgb[idx + 2]),
                    "pixel mismatch at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_roundtrip_corpus() {
        // Deterministic PRNG for reproducibility.
        fn simple_rng(seed: u64) -> impl Iterator<Item = u8> {
            let mut state = seed;
            std::iter::from_fn(move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Some((state >> 33) as u8)
            })
        }

        // 1) All-black
        assert_pixel_perfect_roundtrip(4, 4, &vec![0u8; 4 * 4 * 3]);

        // 2) All-white
        assert_pixel_perfect_roundtrip(4, 4, &vec![255u8; 4 * 4 * 3]);

        // 3) Single pixel
        assert_pixel_perfect_roundtrip(1, 1, &[42, 128, 200]);

        // 4) Gradient (1x256)
        let grad: Vec<u8> = (0..=255u8).flat_map(|v| [v, v, v]).collect();
        assert_pixel_perfect_roundtrip(256, 1, &grad);

        // 5) Random small image
        let rgb5: Vec<u8> = simple_rng(1).take(16 * 16 * 3).collect();
        assert_pixel_perfect_roundtrip(16, 16, &rgb5);

        // 6) Random medium image (single-group boundary)
        let rgb6: Vec<u8> = simple_rng(2).take(100 * 100 * 3).collect();
        assert_pixel_perfect_roundtrip(100, 100, &rgb6);

        // 7) Multi-group image (>256 in one dimension)
        let rgb7: Vec<u8> = simple_rng(3).take(300 * 2 * 3).collect();
        assert_pixel_perfect_roundtrip(300, 2, &rgb7);

        // 8) Multi-group image (>256 in both dimensions)
        let rgb8: Vec<u8> = simple_rng(4).take(257 * 257 * 3).collect();
        assert_pixel_perfect_roundtrip(257, 257, &rgb8);

        // 9) Narrow tall image
        let rgb9: Vec<u8> = simple_rng(5).take(1 * 300 * 3).collect();
        assert_pixel_perfect_roundtrip(1, 300, &rgb9);

        // 10) Wide short image
        let rgb10: Vec<u8> = simple_rng(6).take(500 * 1 * 3).collect();
        assert_pixel_perfect_roundtrip(500, 1, &rgb10);

        // 11) Checkerboard pattern
        let mut checker = vec![0u8; 64 * 64 * 3];
        for y in 0..64 {
            for x in 0..64 {
                let v = if (x + y) % 2 == 0 { 0 } else { 255 };
                let idx = (y * 64 + x) * 3;
                checker[idx] = v;
                checker[idx + 1] = v;
                checker[idx + 2] = v;
            }
        }
        assert_pixel_perfect_roundtrip(64, 64, &checker);

        // 12) Edge values: min/max per channel
        assert_pixel_perfect_roundtrip(2, 1, &[0, 0, 0, 255, 255, 255]);
        assert_pixel_perfect_roundtrip(2, 1, &[0, 128, 255, 255, 128, 0]);
    }

    /// Helper: encode Gray8 image, decode with jxl-rs, verify pixel-perfect match.
    fn assert_gray_roundtrip(width: u32, height: u32, gray: &[u8]) {
        assert_eq!(
            gray.len(),
            (width as usize) * (height as usize),
            "wrong buffer size"
        );
        let cs = encode_modular_u8_gray_image_codestream((width, height), gray).unwrap();
        let (_decoded, frames) =
            crate::api::tests::decode(&cs, usize::MAX, false, false, None).unwrap();
        let img = &frames[0][0];
        assert_eq!(
            img.size(),
            (width as usize, height as usize),
            "decoded gray image size mismatch"
        );
        for y in 0..height as usize {
            let row = img.row(y);
            for x in 0..width as usize {
                let decoded = (row[x] * 255.0 + 0.5) as u8;
                assert_eq!(
                    decoded,
                    gray[y * width as usize + x],
                    "gray pixel mismatch at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_roundtrip_corpus_gray() {
        fn simple_rng(seed: u64) -> impl Iterator<Item = u8> {
            let mut state = seed;
            std::iter::from_fn(move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Some((state >> 33) as u8)
            })
        }

        // 1) All-black
        assert_gray_roundtrip(4, 4, &vec![0u8; 16]);

        // 2) All-white
        assert_gray_roundtrip(4, 4, &vec![255u8; 16]);

        // 3) Single pixel
        assert_gray_roundtrip(1, 1, &[128]);

        // 4) Gradient
        let grad: Vec<u8> = (0..=255u8).collect();
        assert_gray_roundtrip(256, 1, &grad);

        // 5) Random
        let gray5: Vec<u8> = simple_rng(10).take(100 * 100).collect();
        assert_gray_roundtrip(100, 100, &gray5);

        // 6) Multi-group
        let gray6: Vec<u8> = simple_rng(11).take(300 * 300).collect();
        assert_gray_roundtrip(300, 300, &gray6);

        // 7) Edge values
        assert_gray_roundtrip(2, 1, &[0, 255]);
    }
}
