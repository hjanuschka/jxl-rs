// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::CODESTREAM_SIGNATURE,
    error::{Error, Result},
    headers::encodings::{U32, U32Coder},
};

use super::{BitWriter, write_u32};

fn large_size_coder() -> U32Coder {
    U32Coder::Select(
        U32::BitsOffset { n: 9, off: 1 },
        U32::BitsOffset { n: 13, off: 1 },
        U32::BitsOffset { n: 18, off: 1 },
        U32::BitsOffset { n: 30, off: 1 },
    )
}

fn toc_entry_coder() -> U32Coder {
    U32Coder::Select(
        U32::Bits(10),
        U32::BitsOffset { n: 14, off: 1024 },
        U32::BitsOffset { n: 22, off: 17408 },
        U32::BitsOffset {
            n: 30,
            off: 4211712,
        },
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

fn write_minimal_file_header_fields(writer: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;
    write_size(writer, width, height)?;

    // ImageMetadata::all_default = true.
    writer.write(1, 1)?;

    // CustomTransformData::all_default = true.
    writer.write(1, 1)?;

    Ok(())
}

fn default_num_toc_entries(width: u32, height: u32) -> u32 {
    // These numbers follow FrameHeader defaults:
    // - group_dim = 256
    // - lf_group_dim = 2048
    // - num_passes = 1
    let num_groups_x = width.div_ceil(256);
    let num_groups_y = height.div_ceil(256);
    let num_groups = num_groups_x * num_groups_y;

    if num_groups == 1 {
        return 1;
    }

    let num_lf_groups_x = width.div_ceil(2048);
    let num_lf_groups_y = height.div_ceil(2048);
    let num_lf_groups = num_lf_groups_x * num_lf_groups_y;

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
    // Toc::permuted = false, then permutation reader aligns to next byte.
    writer.write(1, 0)?;
    writer.byte_align_zero_pad()?;

    let entry_coder = toc_entry_coder();
    for _ in 0..default_num_toc_entries(width, height) {
        write_u32(&mut writer, &entry_coder, 0)?;
    }

    // TOC reader aligns to byte boundary after entries.
    writer.byte_align_zero_pad()?;

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
    }
}
