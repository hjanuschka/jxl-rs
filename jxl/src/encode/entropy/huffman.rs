// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    util::CeilLog2,
};

use crate::encode::BitWriter;

/// Writes a value in the varint16 format used by entropy headers.
pub fn write_varint16(writer: &mut BitWriter, value: u16) -> Result<()> {
    if value == 0 {
        writer.write(1, 0)?;
        return Ok(());
    }

    writer.write(1, 1)?;

    if value == 1 {
        writer.write(4, 0)?;
        return Ok(());
    }

    let nbits = (usize::from(value).ilog2() as usize).min(15);
    let base = 1u16 << nbits;
    let extra = value - base;

    writer.write(4, nbits as u64)?;
    writer.write(nbits, extra as u64)?;
    Ok(())
}

/// Writes a Huffman table that decodes every symbol read as `symbol`.
///
/// This emits the "simple table" coding when `alphabet_size > 1`.
pub fn write_single_symbol_huffman_table(
    writer: &mut BitWriter,
    alphabet_size: usize,
    symbol: u16,
) -> Result<()> {
    if alphabet_size == 0 {
        return Err(Error::InvalidHuffman);
    }

    if usize::from(symbol) >= alphabet_size {
        return Err(Error::InvalidHuffman);
    }

    if alphabet_size == 1 {
        // Table::decode(al_size == 1) consumes no bits.
        return Ok(());
    }

    // simple_code_or_skip = 1
    writer.write(2, 1)?;

    // num_symbols = 1 => encoded as 0
    writer.write(2, 0)?;

    let max_bits = alphabet_size.ceil_log2();
    writer.write(max_bits, symbol as u64)?;
    Ok(())
}

/// Writes one or more single-symbol Huffman tables with a per-table symbol.
///
/// The stream layout matches `HuffmanCodes::decode`:
/// 1) all alphabet-size varint16 values
/// 2) all table payloads
pub fn write_single_symbol_huffman_codes_with_symbols(
    writer: &mut BitWriter,
    alphabet_sizes: &[usize],
    symbols: &[u16],
) -> Result<()> {
    if alphabet_sizes.len() != symbols.len() {
        return Err(Error::InvalidHuffman);
    }

    for &sz in alphabet_sizes {
        if sz == 0 {
            return Err(Error::InvalidHuffman);
        }

        // HuffmanCodes::decode expects varint16 = alphabet_size - 1.
        write_varint16(writer, (sz - 1) as u16)?;
    }

    for (&sz, &symbol) in alphabet_sizes.iter().zip(symbols.iter()) {
        write_single_symbol_huffman_table(writer, sz, symbol)?;
    }

    Ok(())
}

/// Writes one or more single-symbol Huffman tables with the same symbol value.
pub fn write_single_symbol_huffman_codes(
    writer: &mut BitWriter,
    alphabet_sizes: &[usize],
    symbol: u16,
) -> Result<()> {
    let symbols = vec![symbol; alphabet_sizes.len()];
    write_single_symbol_huffman_codes_with_symbols(writer, alphabet_sizes, &symbols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        entropy_coding::{decode::decode_varint16, huffman::HuffmanCodes},
    };

    #[test]
    fn test_write_varint16_roundtrip() {
        let values = [0u16, 1, 2, 3, 7, 8, 15, 16, 255, 256, 1023, 4096, 16384];
        for v in values {
            let mut writer = BitWriter::new();
            write_varint16(&mut writer, v).unwrap();
            let bytes = writer.finish();

            let mut br = BitReader::new(&bytes);
            let got = decode_varint16(&mut br).unwrap();
            assert_eq!(got, v);
        }
    }

    #[test]
    fn test_write_single_symbol_huffman_codes_roundtrip() {
        let mut writer = BitWriter::new();
        write_single_symbol_huffman_codes(&mut writer, &[8], 5).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();

        for _ in 0..16 {
            assert_eq!(codes.read(&mut br, 0), 5);
        }
    }

    #[test]
    fn test_write_single_symbol_huffman_table_alphabet_one() {
        let mut writer = BitWriter::new();
        write_single_symbol_huffman_codes(&mut writer, &[1], 0).unwrap();
        let bytes = writer.finish();

        // Only varint16(0) should be present => one bit 0 => one zero-padded byte.
        assert_eq!(bytes, vec![0x00]);

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();
        assert_eq!(codes.read(&mut br, 0), 0);
    }

    #[test]
    fn test_write_single_symbol_huffman_codes_with_symbols_roundtrip() {
        let mut writer = BitWriter::new();
        write_single_symbol_huffman_codes_with_symbols(&mut writer, &[1, 8, 1], &[0, 7, 0])
            .unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(3, &mut br).unwrap();
        assert_eq!(codes.read(&mut br, 0), 0);
        assert_eq!(codes.read(&mut br, 1), 7);
        assert_eq!(codes.read(&mut br, 2), 0);
    }
}
