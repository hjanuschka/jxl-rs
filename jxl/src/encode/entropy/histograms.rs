// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;

use super::{HybridUintConfig, write_simple_zero_context_map};
use crate::encode::BitWriter;

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
    // LZ77 disabled.
    writer.write(1, 0)?;

    if num_contexts > 1 {
        write_simple_zero_context_map(writer, num_contexts)?;
    }

    // use_prefix_code = true => Huffman path, log_alpha_size = 15.
    writer.write(1, 1)?;

    // One HybridUint config for one histogram cluster.
    HybridUintConfig::new(15, 0, 0).write(writer, 15)?;

    // One Huffman table with alphabet size 1:
    // decode_varint16 = 0 => alphabet_size = 1.
    writer.write(1, 0)?;

    Ok(())
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
}
