// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    headers::encodings::{U32, U32Coder},
};

use super::{
    BitWriter, HybridUintConfig, pack_signed, write_fixed_symbol_huffman_histograms_with_configs,
    write_u32,
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
/// - predictor: `predictor`
/// - offset: `offset`
/// - multiplier: `1`
/// - residual histogram token/config: `symbol_token` + `uint_config`
pub fn write_single_leaf_tree_with_entropy_params(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    symbol_token: u16,
    uint_config: HybridUintConfig,
) -> Result<()> {
    if predictor > 13 {
        return Err(Error::InvalidPredictor(predictor));
    }

    let predictor_symbol =
        u16::try_from(predictor).map_err(|_| Error::InvalidPredictor(predictor))?;
    let offset_symbol = pack_signed(offset) as u32;
    let offset_symbol = u16::try_from(offset_symbol).map_err(|_| Error::InvalidHuffman)?;

    // Tree-building histograms across the 6 tree contexts:
    // [split_val, property, predictor, offset, multiplier_log, multiplier_bits]
    let tree_context_map = [0u8, 1, 2, 3, 4, 5];
    let tree_symbols = [0u16, 0, predictor_symbol, offset_symbol, 0, 0];
    let tree_uint_configs = [HybridUintConfig::new(15, 0, 0); 6];
    write_fixed_symbol_huffman_histograms_with_configs(
        writer,
        &tree_context_map,
        &tree_symbols,
        &tree_uint_configs,
    )?;

    // Entropy histograms for decoded channel residual symbols.
    let residual_context_map = [0u8];
    write_fixed_symbol_huffman_histograms_with_configs(
        writer,
        &residual_context_map,
        &[symbol_token],
        &[uint_config],
    )?;
    Ok(())
}

/// Writes a minimal modular tree payload with default residual token/config.
pub fn write_single_leaf_tree_with_params(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
) -> Result<()> {
    write_single_leaf_tree_with_entropy_params(
        writer,
        offset,
        predictor,
        0,
        HybridUintConfig::new(15, 0, 0),
    )
}

/// Writes a minimal modular tree payload with predictor `Zero`.
pub fn write_single_leaf_tree_with_offset(writer: &mut BitWriter, offset: i32) -> Result<()> {
    write_single_leaf_tree_with_params(writer, offset, 0)
}

/// Writes a minimal modular tree payload (single leaf, offset 0, predictor `Zero`).
pub fn write_single_leaf_tree(writer: &mut BitWriter) -> Result<()> {
    write_single_leaf_tree_with_params(writer, 0, 0)
}

/// Writes modular residual bits for split0/msb0/lsb0 fixed-token streams.
///
/// This inverts HybridUint(read) for the specific config used by
/// `HybridUintConfig::new(0, 0, 0)`.
pub fn write_split0_fixed_token_signed_stream(
    writer: &mut BitWriter,
    token: u32,
    signed_values: &[i32],
) -> Result<()> {
    if token == 0 {
        for &signed in signed_values {
            if signed != 0 {
                return Err(Error::InvalidFixedTokenValue {
                    token,
                    value: pack_signed(signed),
                });
            }
        }
        return Ok(());
    }

    let nbits = token.saturating_sub(1) as usize;
    if nbits >= 32 {
        return Err(Error::InvalidFixedTokenValue { token, value: 0 });
    }

    let base = 1u32 << nbits;
    let max = base + ((1u32 << nbits) - 1);

    for &signed in signed_values {
        let value = pack_signed(signed);
        if value < base || value > max {
            return Err(Error::InvalidFixedTokenValue { token, value });
        }
        writer.write(nbits, u64::from(value - base))?;
    }

    Ok(())
}

/// Writes minimal modular global subbitstream payload.
///
/// The payload uses:
/// - local tree (`use_global_tree = false`)
/// - default weighted predictor config
/// - no modular transforms
/// - single-leaf local tree with constant params and residual entropy params
pub fn write_minimal_modular_global_data_with_entropy_params(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    symbol_token: u16,
    uint_config: HybridUintConfig,
) -> Result<()> {
    write_minimal_group_header(writer, /*use_global_tree=*/ false)?;
    write_single_leaf_tree_with_entropy_params(
        writer,
        offset,
        predictor,
        symbol_token,
        uint_config,
    )?;
    Ok(())
}

/// Writes minimal modular global subbitstream payload.
///
/// The payload uses:
/// - local tree (`use_global_tree = false`)
/// - default weighted predictor config
/// - no modular transforms
/// - single-leaf local tree with constant `offset` and `predictor`
pub fn write_minimal_modular_global_data_with_params(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
) -> Result<()> {
    write_minimal_modular_global_data_with_entropy_params(
        writer,
        offset,
        predictor,
        0,
        HybridUintConfig::new(15, 0, 0),
    )
}

/// Writes minimal modular global subbitstream payload with predictor `Zero`.
pub fn write_minimal_modular_global_data_with_offset(
    writer: &mut BitWriter,
    offset: i32,
) -> Result<()> {
    write_minimal_modular_global_data_with_params(writer, offset, 0)
}

/// Writes minimal modular global subbitstream payload with offset 0.
pub fn write_minimal_modular_global_data(writer: &mut BitWriter) -> Result<()> {
    write_minimal_modular_global_data_with_params(writer, 0, 0)
}

/// Writes a minimal LF global section prefix for a modular frame:
/// - default LF quant factors
/// - no global modular tree
/// - minimal modular global data payload
pub fn write_minimal_modular_lf_global_section_with_entropy_params(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    symbol_token: u16,
    uint_config: HybridUintConfig,
) -> Result<()> {
    // LfQuantFactors::new => default path.
    writer.write(1, 1)?;

    // decode_lf_global tree flag: no global tree.
    writer.write(1, 0)?;

    write_minimal_modular_global_data_with_entropy_params(
        writer,
        offset,
        predictor,
        symbol_token,
        uint_config,
    )
}

/// Writes a minimal LF global section prefix for a modular frame:
/// - default LF quant factors
/// - no global modular tree
/// - minimal modular global data payload
pub fn write_minimal_modular_lf_global_section_with_params(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
) -> Result<()> {
    write_minimal_modular_lf_global_section_with_entropy_params(
        writer,
        offset,
        predictor,
        0,
        HybridUintConfig::new(15, 0, 0),
    )
}

/// Writes a minimal LF global section prefix with predictor `Zero`.
pub fn write_minimal_modular_lf_global_section_with_offset(
    writer: &mut BitWriter,
    offset: i32,
) -> Result<()> {
    write_minimal_modular_lf_global_section_with_params(writer, offset, 0)
}

/// Writes a minimal LF global section prefix with offset 0.
pub fn write_minimal_modular_lf_global_section(writer: &mut BitWriter) -> Result<()> {
    write_minimal_modular_lf_global_section_with_params(writer, 0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        entropy_coding::decode::{Histograms, SymbolReader},
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
    fn test_write_single_leaf_tree_with_params_roundtrip() {
        let mut writer = BitWriter::new();
        write_single_leaf_tree_with_params(&mut writer, 3, 1).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();

        assert_eq!(tree.nodes.len(), 1);
        let node_debug = format!("{:?}", tree.nodes[0]);
        assert!(node_debug.contains("Leaf"));
        assert!(node_debug.contains("West"));
        assert!(node_debug.contains("offset: 3"));
    }

    #[test]
    fn test_write_single_leaf_tree_with_params_invalid_predictor() {
        let mut writer = BitWriter::new();
        let err = write_single_leaf_tree_with_params(&mut writer, 0, 99).unwrap_err();
        assert!(matches!(err, crate::error::Error::InvalidPredictor(99)));
    }

    #[test]
    fn test_write_split0_fixed_token_signed_stream_roundtrip() {
        let mut writer = BitWriter::new();
        write_fixed_symbol_huffman_histograms_with_configs(
            &mut writer,
            &[0],
            &[10],
            &[HybridUintConfig::new(0, 0, 0)],
        )
        .unwrap();
        write_split0_fixed_token_signed_stream(&mut writer, 10, &[256, 300, 511]).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let hist = Histograms::decode(1, &mut br, true).unwrap();
        let mut reader = SymbolReader::new(&hist, &mut br, None).unwrap();
        assert_eq!(reader.read_signed(&hist, &mut br, 0), 256);
        assert_eq!(reader.read_signed(&hist, &mut br, 0), 300);
        assert_eq!(reader.read_signed(&hist, &mut br, 0), 511);
        reader.check_final_state(&hist, &mut br).unwrap();
    }

    #[test]
    fn test_write_split0_fixed_token_signed_stream_token0_only_zero() {
        let mut writer = BitWriter::new();
        write_split0_fixed_token_signed_stream(&mut writer, 0, &[0, 0, 0]).unwrap();
        assert!(writer.finish().is_empty());

        let mut writer = BitWriter::new();
        let err = write_split0_fixed_token_signed_stream(&mut writer, 0, &[1]).unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::InvalidFixedTokenValue { token: 0, value: 2 }
        ));
    }

    #[test]
    fn test_write_split0_fixed_token_signed_stream_invalid_value() {
        let mut writer = BitWriter::new();
        let err = write_split0_fixed_token_signed_stream(&mut writer, 10, &[255]).unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::InvalidFixedTokenValue {
                token: 10,
                value: 510
            }
        ));
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
