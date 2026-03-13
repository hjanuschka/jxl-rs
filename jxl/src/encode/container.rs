// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::CONTAINER_SIGNATURE,
    container::box_header::ContainerBoxType,
    error::{Error, Result},
};

use super::BitWriter;

/// Additional user metadata box to be written before codestream boxes.
pub type ContainerExtraBox<'a> = (ContainerBoxType, &'a [u8]);

/// Writes the fixed JPEG XL container signature and `ftyp` box.
pub fn write_container_prefix(writer: &mut BitWriter) -> Result<()> {
    writer.write_aligned_bytes(&CONTAINER_SIGNATURE)?;

    // Required file type box.
    writer.write_aligned_bytes(&20u32.to_be_bytes())?;
    writer.write_aligned_bytes(b"ftyp")?;
    writer.write_aligned_bytes(b"jxl ")?;
    writer.write_aligned_bytes(&0u32.to_be_bytes())?;
    writer.write_aligned_bytes(b"jxl ")?;

    Ok(())
}

fn validate_extra_box_type(box_type: ContainerBoxType) -> Result<()> {
    match box_type {
        ContainerBoxType::EXIF
        | ContainerBoxType::XML
        | ContainerBoxType::JUMBF
        | ContainerBoxType::JPEG_RECONSTRUCTION => Ok(()),
        _ => Err(Error::InvalidBox),
    }
}

fn write_extra_boxes(writer: &mut BitWriter, extra_boxes: &[ContainerExtraBox<'_>]) -> Result<()> {
    for (box_type, payload) in extra_boxes {
        validate_extra_box_type(*box_type)?;
        write_box_header(writer, *box_type, payload.len() as u64)?;
        writer.write_aligned_bytes(payload)?;
    }
    Ok(())
}

/// Writes a box header for a known-size payload.
pub fn write_box_header(
    writer: &mut BitWriter,
    box_type: ContainerBoxType,
    payload_size: u64,
) -> Result<()> {
    let small_box_total_size = payload_size.checked_add(8).ok_or(Error::SizeOverflow)?;

    if small_box_total_size <= u64::from(u32::MAX) {
        writer.write_aligned_bytes(&(small_box_total_size as u32).to_be_bytes())?;
        writer.write_aligned_bytes(&box_type.0)?;
        return Ok(());
    }

    let large_box_total_size = payload_size.checked_add(16).ok_or(Error::SizeOverflow)?;
    writer.write_aligned_bytes(&1u32.to_be_bytes())?;
    writer.write_aligned_bytes(&box_type.0)?;
    writer.write_aligned_bytes(&large_box_total_size.to_be_bytes())?;
    Ok(())
}

/// Wraps an already encoded JPEG XL codestream (`0xFF 0x0A ...`) in a
/// minimal container (`JXL ` signature, `ftyp`, single `jxlc` box).
pub fn wrap_codestream(codestream: &[u8]) -> Result<Vec<u8>> {
    wrap_codestream_with_extra_boxes(codestream, &[])
}

/// Wraps codestream in a container and inserts optional metadata boxes before codestream.
///
/// Allowed box types in `extra_boxes`: `Exif`, `xml `, `jumb`, `jbrd`.
pub fn wrap_codestream_with_extra_boxes(
    codestream: &[u8],
    extra_boxes: &[ContainerExtraBox<'_>],
) -> Result<Vec<u8>> {
    let mut estimated = CONTAINER_SIGNATURE.len() + 20 + 8 + codestream.len();
    for (_, payload) in extra_boxes {
        estimated += 8 + payload.len();
    }

    let mut writer = BitWriter::with_capacity(estimated);
    write_container_prefix(&mut writer)?;
    write_extra_boxes(&mut writer, extra_boxes)?;
    write_box_header(
        &mut writer,
        ContainerBoxType::CODESTREAM,
        codestream.len() as u64,
    )?;
    writer.write_aligned_bytes(codestream)?;
    Ok(writer.finish())
}

/// Wraps a codestream using chunked `jxlp` boxes.
///
/// `chunk_size` is the number of codestream bytes per `jxlp` chunk.
pub fn wrap_codestream_jxlp_chunked(codestream: &[u8], chunk_size: usize) -> Result<Vec<u8>> {
    wrap_codestream_jxlp_chunked_with_extra_boxes(codestream, chunk_size, &[])
}

/// Wraps a codestream using chunked `jxlp` boxes and optional metadata boxes.
///
/// Allowed box types in `extra_boxes`: `Exif`, `xml `, `jumb`, `jbrd`.
pub fn wrap_codestream_jxlp_chunked_with_extra_boxes(
    codestream: &[u8],
    chunk_size: usize,
    extra_boxes: &[ContainerExtraBox<'_>],
) -> Result<Vec<u8>> {
    if chunk_size == 0 {
        return Err(Error::InvalidBitCount(0));
    }

    // Each jxlp box payload starts with a 4-byte index/flags field.
    let num_chunks = codestream.len().div_ceil(chunk_size).max(1);
    let mut estimated = CONTAINER_SIGNATURE.len() + 20;
    for (_, payload) in extra_boxes {
        estimated += 8 + payload.len();
    }
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(codestream.len());
        let payload = 4 + end.saturating_sub(start);
        estimated += 8 + payload;
    }

    let mut writer = BitWriter::with_capacity(estimated);
    write_container_prefix(&mut writer)?;
    write_extra_boxes(&mut writer, extra_boxes)?;

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(codestream.len());
        let chunk = &codestream[start..end];
        let is_last = i + 1 == num_chunks;

        let mut index = i as u32;
        if is_last {
            index |= 0x8000_0000;
        }

        let payload_size = (4 + chunk.len()) as u64;
        write_box_header(
            &mut writer,
            ContainerBoxType::PARTIAL_CODESTREAM,
            payload_size,
        )?;
        writer.write_aligned_bytes(&index.to_be_bytes())?;
        writer.write_aligned_bytes(chunk)?;
    }

    Ok(writer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ContainerParser;

    #[test]
    fn test_wrap_codestream_roundtrip() {
        let codestream = vec![0xFF, 0x0A, 0x11, 0x22, 0x33, 0x44];
        let container = wrap_codestream(&codestream).unwrap();
        let parsed = ContainerParser::collect_codestream(&container).unwrap();
        assert_eq!(parsed, codestream);
    }

    #[test]
    fn test_write_box_header_uses_64_bit_box_for_large_payload() {
        let mut writer = BitWriter::new();
        write_box_header(
            &mut writer,
            ContainerBoxType::CODESTREAM,
            u64::from(u32::MAX),
        )
        .unwrap();
        let bytes = writer.finish();

        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], &[0, 0, 0, 1]);
        assert_eq!(&bytes[4..8], b"jxlc");

        let xlbox = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(xlbox, u64::from(u32::MAX) + 16);
    }

    #[test]
    fn test_wrap_codestream_jxlp_chunked_roundtrip() {
        let codestream = vec![0xFF, 0x0A, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
        let container = wrap_codestream_jxlp_chunked(&codestream, 3).unwrap();
        let parsed = ContainerParser::collect_codestream(&container).unwrap();
        assert_eq!(parsed, codestream);
    }

    #[test]
    fn test_wrap_codestream_jxlp_chunked_rejects_zero_chunk_size() {
        let codestream = vec![0xFF, 0x0A];
        let err = wrap_codestream_jxlp_chunked(&codestream, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidBitCount(0)));
    }

    #[test]
    fn test_wrap_codestream_with_extra_boxes_roundtrip() {
        let codestream = vec![0xFF, 0x0A, 0x11, 0x22, 0x33, 0x44];
        let exif = [1u8, 2, 3, 4];
        let xml = b"<x:xmpmeta/>";
        let container = wrap_codestream_with_extra_boxes(
            &codestream,
            &[
                (ContainerBoxType::EXIF, &exif),
                (ContainerBoxType::XML, xml),
            ],
        )
        .unwrap();
        assert!(container.windows(4).any(|w| w == b"Exif"));
        assert!(container.windows(4).any(|w| w == b"xml "));
        let parsed = ContainerParser::collect_codestream(&container).unwrap();
        assert_eq!(parsed, codestream);
    }

    #[test]
    fn test_wrap_codestream_with_extra_boxes_rejects_invalid_box_type() {
        let codestream = vec![0xFF, 0x0A];
        let err = wrap_codestream_with_extra_boxes(
            &codestream,
            &[(ContainerBoxType::CODESTREAM, b"invalid")],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidBox));
    }

    #[test]
    fn test_wrap_codestream_with_extra_boxes_preserves_box_order() {
        let codestream = vec![0xFF, 0x0A, 0x11, 0x22];
        let exif = [1u8, 2, 3];
        let xml = b"<x/>";
        let jumbf = [9u8, 8, 7, 6];
        let container = wrap_codestream_with_extra_boxes(
            &codestream,
            &[
                (ContainerBoxType::EXIF, &exif),
                (ContainerBoxType::XML, xml),
                (ContainerBoxType::JUMBF, &jumbf),
            ],
        )
        .unwrap();

        let pos_exif = container
            .windows(4)
            .position(|w| w == b"Exif")
            .expect("Exif box missing");
        let pos_xml = container
            .windows(4)
            .position(|w| w == b"xml ")
            .expect("xml box missing");
        let pos_jumb = container
            .windows(4)
            .position(|w| w == b"jumb")
            .expect("jumb box missing");
        let pos_jxlc = container
            .windows(4)
            .position(|w| w == b"jxlc")
            .expect("jxlc box missing");

        assert!(pos_exif < pos_xml);
        assert!(pos_xml < pos_jumb);
        assert!(pos_jumb < pos_jxlc);
    }

    #[test]
    fn test_wrap_codestream_jxlp_chunked_with_extra_boxes_order() {
        let codestream = vec![0xFF, 0x0A, 0x11, 0x22, 0x33, 0x44, 0x55];
        let exif = [1u8, 2, 3, 4];
        let xml = b"<xmp/>";
        let container = wrap_codestream_jxlp_chunked_with_extra_boxes(
            &codestream,
            3,
            &[
                (ContainerBoxType::EXIF, &exif),
                (ContainerBoxType::XML, xml),
            ],
        )
        .unwrap();

        let pos_exif = container
            .windows(4)
            .position(|w| w == b"Exif")
            .expect("Exif box missing");
        let pos_xml = container
            .windows(4)
            .position(|w| w == b"xml ")
            .expect("xml box missing");
        let pos_jxlp = container
            .windows(4)
            .position(|w| w == b"jxlp")
            .expect("jxlp box missing");

        assert!(pos_exif < pos_xml);
        assert!(pos_xml < pos_jxlp);
    }
}
