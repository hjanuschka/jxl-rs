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
    let mut writer =
        BitWriter::with_capacity(CONTAINER_SIGNATURE.len() + 20 + 8 + codestream.len());
    write_container_prefix(&mut writer)?;
    write_box_header(
        &mut writer,
        ContainerBoxType::CODESTREAM,
        codestream.len() as u64,
    )?;
    writer.write_aligned_bytes(codestream)?;
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
}
