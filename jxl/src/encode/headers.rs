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

fn write_size(writer: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidImageSize(width as usize, height as usize));
    }

    let max_large_dim = 1u32 << 30;
    if width > max_large_dim || height > max_large_dim {
        return Err(Error::ImageDimensionTooLarge(width.max(height) as u64));
    }

    // Encode in "large" mode (small=false), with ratio = Unknown.
    writer.write(1, 0)?;
    let size_coder = large_size_coder();
    write_u32(writer, &size_coder, height)?;

    // AspectRatio::Unknown == 0, coded with Bits(3).
    writer.write(3, 0)?;

    write_u32(writer, &size_coder, width)?;
    Ok(())
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

    writer.write_aligned_bytes(&CODESTREAM_SIGNATURE)?;
    write_size(&mut writer, width, height)?;

    // ImageMetadata::all_default = true.
    writer.write(1, 1)?;

    // CustomTransformData::all_default = true.
    writer.write(1, 1)?;

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

    fn decode_to_image_info(mut input: &[u8]) -> JxlDecoder<states::WithImageInfo> {
        let mut dec = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut prev_len = input.len().saturating_add(1);

        for _ in 0..32 {
            match dec.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => return result,
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
        let dec = decode_to_image_info(&codestream);

        let info = dec.basic_info();
        assert_eq!(info.size, (321, 123));
        assert_eq!(info.bit_depth, JxlBitDepth::Int { bits_per_sample: 8 });
        assert!(info.extra_channels.is_empty());
    }

    #[test]
    fn test_encode_minimal_header_in_container_parses_info() {
        let codestream = encode_minimal_codestream_header((77, 66)).unwrap();
        let container_stream = container::wrap_codestream(&codestream).unwrap();
        let dec = decode_to_image_info(&container_stream);

        assert_eq!(dec.basic_info().size, (77, 66));
    }

    #[test]
    fn test_minimal_header_snapshot_1x1() {
        let codestream = encode_minimal_codestream_header((1, 1)).unwrap();
        assert_eq!(codestream, vec![0xFF, 0x0A, 0x00, 0x00, 0x00, 0x0C]);
    }
}
