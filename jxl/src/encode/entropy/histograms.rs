// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;

use super::{
    HybridUintConfig, write_simple_context_map, write_single_symbol_huffman_codes_with_symbols,
};
use crate::encode::BitWriter;

/// Writes prefix/Huffman histograms where each histogram table decodes to a
/// single fixed symbol.
///
/// - `context_map`: one histogram-id per context
/// - `histogram_symbols`: one fixed decoded symbol per histogram-id
pub fn write_fixed_symbol_huffman_histograms(
    writer: &mut BitWriter,
    context_map: &[u8],
    histogram_symbols: &[u16],
) -> Result<()> {
    // LZ77 disabled.
    writer.write(1, 0)?;

    if context_map.len() > 1 {
        write_simple_context_map(writer, context_map)?;
    }

    // use_prefix_code = true => Huffman path, log_alpha_size = 15.
    writer.write(1, 1)?;

    for _ in histogram_symbols {
        HybridUintConfig::new(15, 0, 0).write(writer, 15)?;
    }

    let alphabet_sizes: Vec<usize> = histogram_symbols
        .iter()
        .map(|&sym| usize::from(sym).saturating_add(1).max(1))
        .collect();

    write_single_symbol_huffman_codes_with_symbols(writer, &alphabet_sizes, histogram_symbols)?;

    Ok(())
}

/// Writes a minimal histogram stream that decodes to:
/// - lz77 disabled
/// - one histogram cluster (all contexts map to 0)
/// - prefix/Huffman codes
/// - alphabet size 1 (single symbol 0)
///
/// The resulting histogram stream decodes every symbol request as `0`.
pub fn write_single_symbol_huffman_histograms(
    writer: &mut BitWriter,
    num_contexts: usize,
) -> Result<()> {
    let context_map = vec![0u8; num_contexts.max(1)];
    write_fixed_symbol_huffman_histograms(writer, &context_map, &[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        entropy_coding::decode::{Histograms, SymbolReader},
    };

    #[test]
    fn test_single_symbol_huffman_histograms_roundtrip_single_context() {
        let mut writer = BitWriter::new();
        write_single_symbol_huffman_histograms(&mut writer, 1).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let hist = Histograms::decode(1, &mut br, true).unwrap();
        let mut reader = SymbolReader::new(&hist, &mut br, None).unwrap();

        for _ in 0..16 {
            assert_eq!(reader.read_unsigned(&hist, &mut br, 0), 0);
        }

        reader.check_final_state(&hist, &mut br).unwrap();
    }

    #[test]
    fn test_single_symbol_huffman_histograms_roundtrip_multi_context() {
        let mut writer = BitWriter::new();
        write_single_symbol_huffman_histograms(&mut writer, 6).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let hist = Histograms::decode(6, &mut br, true).unwrap();

        assert_eq!(hist.map_context_to_cluster(0), 0);
        assert_eq!(hist.map_context_to_cluster(5), 0);
    }

    #[test]
    fn test_fixed_symbol_huffman_histograms_roundtrip() {
        let mut writer = BitWriter::new();
        let context_map = [0u8, 1, 2, 3, 4, 5];
        let histogram_symbols = [0u16, 0, 0, 7, 0, 0];
        write_fixed_symbol_huffman_histograms(&mut writer, &context_map, &histogram_symbols)
            .unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let hist = Histograms::decode(6, &mut br, true).unwrap();
        let mut reader = SymbolReader::new(&hist, &mut br, None).unwrap();

        for _ in 0..8 {
            assert_eq!(reader.read_unsigned(&hist, &mut br, 3), 7);
            assert_eq!(reader.read_unsigned(&hist, &mut br, 1), 0);
        }

        reader.check_final_state(&hist, &mut br).unwrap();
    }
}
