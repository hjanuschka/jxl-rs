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
    BitWriter, HybridUintConfig, write_minimal_modular_global_data_with_entropy_params,
    write_minimal_modular_global_data_with_params,
    write_minimal_modular_lf_global_section_with_entropy_params,
    write_minimal_modular_lf_global_section_with_params, write_split0_fixed_token_signed_stream,
    write_toc, write_u32,
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

fn write_minimal_image_metadata_fields(writer: &mut BitWriter, xyb_encoded: bool) -> Result<()> {
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

    // color_encoding.all_default = true (defaults to RGB/sRGB)
    writer.write(1, 1)?;

    // extensions selector = 0 (u64 coding)
    writer.write(2, 0)?;

    Ok(())
}

fn write_minimal_file_header_fields_with_metadata(
    writer: &mut BitWriter,
    width: u32,
    height: u32,
    xyb_encoded: bool,
) -> Result<()> {
    writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;
    write_size(writer, width, height)?;
    write_minimal_image_metadata_fields(writer, xyb_encoded)?;

    // CustomTransformData::all_default = true.
    writer.write(1, 1)?;

    Ok(())
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

    // restoration_filter.all_default = true
    writer.write(1, 1)?;

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
/// Current constraints:
/// - input is tightly packed interleaved RGB8 (`[r,g,b, r,g,b, ...]`)
/// - uses default file metadata/bootstrap frame header
/// - residual coding uses fixed split0 token bootstrap coding
pub fn encode_modular_u8_rgb_image_codestream(size: (u32, u32), rgb: &[u8]) -> Result<Vec<u8>> {
    let (width, height) = size;
    validate_size(width, height)?;
    validate_rgb_u8_buffer(width, height, rgb)?;

    let uint_config = HybridUintConfig::new(0, 0, 0);

    let width_usize = width as usize;
    let num_groups = default_num_groups(width, height);
    let num_lf_groups = default_num_lf_groups(width, height);
    let num_groups_x = width.div_ceil(256);

    let choose_group_entropy_params = |origin: (u32, u32), size: (u32, u32)| -> (i32, u16) {
        let (ox, oy) = origin;
        let (gw, gh) = size;

        let mut min_sample = u8::MAX;
        let mut max_sample = u8::MIN;
        for y in oy..(oy + gh) {
            for x in ox..(ox + gw) {
                let pixel_index = y as usize * width_usize + x as usize;
                let base = pixel_index * 3;
                for c in 0..3 {
                    let sample = rgb[base + c];
                    min_sample = min_sample.min(sample);
                    max_sample = max_sample.max(sample);
                }
            }
        }

        if min_sample == max_sample {
            return (i32::from(min_sample), 0);
        }

        let range = u32::from(max_sample - min_sample);
        let values = range + 1;
        let ceil_log2_values = (u32::BITS - (values - 1).leading_zeros()) as u32;
        let token = (ceil_log2_values + 2) as u16;
        let low = 1i32 << (u32::from(token) - 2);
        let offset = i32::from(min_sample) - low;

        (offset, token)
    };

    let write_group_samples = |writer: &mut BitWriter,
                               origin: (u32, u32),
                               size: (u32, u32),
                               tree_offset: i32,
                               token: u16|
     -> Result<()> {
        let (ox, oy) = origin;
        let (gw, gh) = size;
        let mut signed_residuals = Vec::with_capacity((gw as usize) * (gh as usize) * 3);

        // Decoder channel order is channel-major, then row-major for each channel.
        for channel in 0..3 {
            for y in oy..(oy + gh) {
                for x in ox..(ox + gw) {
                    let pixel_index = y as usize * width_usize + x as usize;
                    let sample = i32::from(rgb[pixel_index * 3 + channel]);
                    signed_residuals.push(sample - tree_offset);
                }
            }
        }

        write_split0_fixed_token_signed_stream(writer, u32::from(token), &signed_residuals)
    };

    // Single-group path stores all channel samples in section 0.
    if num_groups == 1 {
        let (tree_offset, token) = choose_group_entropy_params((0, 0), (width, height));

        let mut lf_global_writer = BitWriter::new();
        write_minimal_modular_lf_global_section_with_entropy_params(
            &mut lf_global_writer,
            tree_offset,
            0,
            token,
            uint_config,
        )?;
        write_group_samples(
            &mut lf_global_writer,
            (0, 0),
            (width, height),
            tree_offset,
            token,
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
    // - section 0: LF global only
    // - section 1 (LF groups): empty
    // - section 2 (HF global): empty
    // - section 3..: one per image group with local tree + residual bits
    let mut lf_global_writer = BitWriter::new();
    write_minimal_modular_lf_global_section_with_entropy_params(
        &mut lf_global_writer,
        0,
        0,
        0,
        uint_config,
    )?;
    let lf_global_section = lf_global_writer.finish();

    let mut hf_group_sections = Vec::with_capacity(num_groups as usize);
    for group in 0..num_groups {
        let gx = group % num_groups_x;
        let gy = group / num_groups_x;
        let ox = gx * 256;
        let oy = gy * 256;
        let gw = (width - ox).min(256);
        let gh = (height - oy).min(256);

        let (tree_offset, token) = choose_group_entropy_params((ox, oy), (gw, gh));

        let mut group_writer = BitWriter::new();
        write_minimal_modular_global_data_with_entropy_params(
            &mut group_writer,
            tree_offset,
            0,
            token,
            uint_config,
        )?;
        write_group_samples(&mut group_writer, (ox, oy), (gw, gh), tree_offset, token)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::{JxlBitDepth, JxlDecoder, JxlDecoderOptions, ProcessingResult, states},
        bit_reader::BitReader,
        encode::container,
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
    fn test_encode_modular_u8_rgb_image_codestream_multigroup_decodes() {
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
}
