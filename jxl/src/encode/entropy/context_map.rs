// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    util::CeilLog2,
};

use super::super::BitWriter;

/// Writes a simple context map (`is_simple = true`).
///
/// The caller provides one histogram id per context.
pub fn write_simple_context_map(writer: &mut BitWriter, context_map: &[u8]) -> Result<()> {
    // is_simple = true
    writer.write(1, 1)?;

    if context_map.is_empty() {
        // bits_per_entry = 0
        writer.write(2, 0)?;
        return Ok(());
    }

    let max_symbol = context_map.iter().copied().max().unwrap_or(0);
    let bits_per_entry = if max_symbol == 0 {
        0
    } else {
        (usize::from(max_symbol) + 1).ceil_log2()
    };

    // Simple context map coding has only 2 bits for bits_per_entry => max 3.
    if bits_per_entry > 3 {
        return Err(Error::InvalidContextMap(max_symbol as u32));
    }

    writer.write(2, bits_per_entry as u64)?;
    if bits_per_entry > 0 {
        for &ctx in context_map {
            writer.write(bits_per_entry, ctx as u64)?;
        }
    }
    Ok(())
}

/// Writes a simple all-zero context map with `num_contexts` entries.
pub fn write_simple_zero_context_map(writer: &mut BitWriter, num_contexts: usize) -> Result<()> {
    let zeros = vec![0u8; num_contexts];
    write_simple_context_map(writer, &zeros)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bit_reader::BitReader, entropy_coding::context_map::decode_context_map};

    fn decode_map(bytes: &[u8], num_contexts: usize) -> Vec<u8> {
        let mut br = BitReader::new(bytes);
        decode_context_map(num_contexts, &mut br).unwrap()
    }

    #[test]
    fn test_write_simple_zero_context_map_roundtrip() {
        let mut writer = BitWriter::new();
        write_simple_zero_context_map(&mut writer, 8).unwrap();
        let bytes = writer.finish();

        let got = decode_map(&bytes, 8);
        assert_eq!(got, vec![0u8; 8]);
    }

    #[test]
    fn test_write_simple_context_map_roundtrip() {
        let map = vec![0, 1, 2, 3, 1, 0, 2, 3, 0, 1];

        let mut writer = BitWriter::new();
        write_simple_context_map(&mut writer, &map).unwrap();
        let bytes = writer.finish();

        let got = decode_map(&bytes, map.len());
        assert_eq!(got, map);
    }

    #[test]
    fn test_write_simple_context_map_too_large_symbol() {
        let mut writer = BitWriter::new();
        let err = write_simple_context_map(&mut writer, &[0, 8]).unwrap_err();
        assert!(matches!(err, Error::InvalidContextMap(8)));
    }
}
