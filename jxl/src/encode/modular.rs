// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    headers::encodings::{U32, U32Coder},
};

use super::{
    BitWriter, pack_signed, write_fixed_symbol_huffman_histograms,
    write_single_symbol_huffman_histograms, write_u32,
};

fn transform_count_coder() -> U32Coder {
    U32Coder::Select(
        U32::Val(0),
        U32::Val(1),
        U32::BitsOffset { n: 4, off: 2 },
        U32::BitsOffset { n: 8, off: 18 },
    )
}

/// Writes a minimal `GroupHeader`:
/// - configurable `use_global_tree`
/// - default weighted predictor header (`all_default = true`)
/// - no transforms
pub fn write_minimal_group_header(writer: &mut BitWriter, use_global_tree: bool) -> Result<()> {
    writer.write(1, u64::from(use_global_tree))?;

    // WeightedHeader::all_default = true.
    writer.write(1, 1)?;

    // transforms length = 0.
    write_u32(writer, &transform_count_coder(), 0)?;

    Ok(())
}

/// Writes a minimal modular tree payload with a single leaf.
///
/// The resulting leaf uses:
/// - predictor: `Zero`
/// - offset: `offset`
/// - multiplier: `1`
pub fn write_single_leaf_tree_with_offset(writer: &mut BitWriter, offset: i32) -> Result<()> {
    let offset_symbol = pack_signed(offset) as u32;
    let offset_symbol = u16::try_from(offset_symbol).map_err(|_| Error::InvalidHuffman)?;

    // Tree-building histograms across the 6 tree contexts:
    // [split_val, property, predictor, offset, multiplier_log, multiplier_bits]
    let tree_context_map = [0u8, 1, 2, 3, 4, 5];
    let tree_symbols = [0u16, 0, 0, offset_symbol, 0, 0];
    write_fixed_symbol_huffman_histograms(writer, &tree_context_map, &tree_symbols)?;

    // Entropy histograms for decoded channel residual symbols.
    write_single_symbol_huffman_histograms(writer, 1)?;
    Ok(())
}

/// Writes a minimal modular tree payload (single leaf, offset 0).
pub fn write_single_leaf_tree(writer: &mut BitWriter) -> Result<()> {
    write_single_leaf_tree_with_offset(writer, 0)
}

/// Writes minimal modular global subbitstream payload for tiny channels.
///
/// The payload uses:
/// - local tree (`use_global_tree = false`)
/// - default weighted predictor config
/// - no modular transforms
/// - single-leaf local tree with single-symbol histograms
pub fn write_minimal_modular_global_data(writer: &mut BitWriter) -> Result<()> {
    write_minimal_group_header(writer, /*use_global_tree=*/ false)?;
    write_single_leaf_tree(writer)?;
    Ok(())
}

/// Writes a minimal LF global section prefix for a modular frame:
/// - default LF quant factors
/// - no global modular tree
/// - minimal modular global data payload
pub fn write_minimal_modular_lf_global_section(writer: &mut BitWriter) -> Result<()> {
    // LfQuantFactors::new => default path.
    writer.write(1, 1)?;

    // decode_lf_global tree flag: no global tree.
    writer.write(1, 0)?;

    write_minimal_modular_global_data(writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        frame::quantizer::LfQuantFactors,
        headers::{JxlHeader, modular::GroupHeader},
    };

    #[test]
    fn test_write_minimal_group_header_roundtrip() {
        let mut writer = BitWriter::new();
        write_minimal_group_header(&mut writer, false).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let header = GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);
        assert!(header.transforms.is_empty());
    }

    #[test]
    fn test_write_single_leaf_tree_roundtrip() {
        let mut writer = BitWriter::new();
        write_single_leaf_tree(&mut writer).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();

        assert_eq!(tree.nodes.len(), 1);
        let node_debug = format!("{:?}", tree.nodes[0]);
        assert!(node_debug.contains("Leaf"));
        assert!(node_debug.contains("Zero"));
        assert!(node_debug.contains("offset: 0"));
        assert!(node_debug.contains("multiplier: 1"));
    }

    #[test]
    fn test_write_single_leaf_tree_with_offset_roundtrip() {
        let mut writer = BitWriter::new();
        write_single_leaf_tree_with_offset(&mut writer, 7).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();

        assert_eq!(tree.nodes.len(), 1);
        let node_debug = format!("{:?}", tree.nodes[0]);
        assert!(node_debug.contains("Leaf"));
        assert!(node_debug.contains("Zero"));
        assert!(node_debug.contains("offset: 7"));
        assert!(node_debug.contains("multiplier: 1"));
    }

    #[test]
    fn test_write_minimal_modular_lf_global_section_prefix_roundtrip() {
        let mut writer = BitWriter::new();
        write_minimal_modular_lf_global_section(&mut writer).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let lf = LfQuantFactors::new(&mut br).unwrap();
        assert!(lf.quant_factors.iter().all(|q| *q > 0.0));

        // global tree present flag
        assert_eq!(br.read(1).unwrap(), 0);

        // GroupHeader + local tree should parse.
        let _header = GroupHeader::read(&mut br).unwrap();
        let _tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
    }
}
