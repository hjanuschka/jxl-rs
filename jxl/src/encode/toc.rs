// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    headers::encodings::{U32, U32Coder},
};

use super::{BitWriter, write_u32};

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

/// Writes TOC data in non-permuted form.
///
/// Layout:
/// - `permuted = false`
/// - permutation payload (empty for non-permuted) + byte alignment
/// - one encoded entry length per section
/// - byte alignment
pub fn write_toc(writer: &mut BitWriter, entries: &[u32]) -> Result<()> {
    // Toc::permuted = false.
    writer.write(1, 0)?;

    // Permutation::read_unconditional always jumps to byte boundary.
    writer.byte_align_zero_pad()?;

    let coder = toc_entry_coder();
    for &entry in entries {
        write_u32(writer, &coder, entry)?;
    }

    // Non-section parser jumps to byte boundary after reading TOC entries.
    writer.byte_align_zero_pad()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        headers::{
            encodings::UnconditionalCoder,
            toc::{Toc, TocNonserialized},
        },
    };

    #[test]
    fn test_write_toc_roundtrip_single_entry() {
        let mut writer = BitWriter::new();
        write_toc(&mut writer, &[0]).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let toc =
            Toc::read_unconditional(&(), &mut br, &TocNonserialized { num_entries: 1 }).unwrap();

        assert!(!toc.permuted);
        assert_eq!(toc.entries, vec![0]);
    }

    #[test]
    fn test_write_toc_roundtrip_multi_entries() {
        let entries = vec![0, 1, 1023, 1024, 2048, 200_000];

        let mut writer = BitWriter::new();
        write_toc(&mut writer, &entries).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let toc = Toc::read_unconditional(
            &(),
            &mut br,
            &TocNonserialized {
                num_entries: entries.len() as u32,
            },
        )
        .unwrap();

        assert!(!toc.permuted);
        assert_eq!(toc.entries, entries);

        // Reader should be on a byte boundary after TOC decode.
        br.jump_to_byte_boundary().unwrap();
        assert!(matches!(
            br.read(1),
            Err(crate::error::Error::OutOfBounds(_))
        ));
    }

    #[test]
    fn test_write_toc_empty_entries() {
        let mut writer = BitWriter::new();
        write_toc(&mut writer, &[]).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let toc =
            Toc::read_unconditional(&(), &mut br, &TocNonserialized { num_entries: 0 }).unwrap();
        assert!(toc.entries.is_empty());
    }
}
