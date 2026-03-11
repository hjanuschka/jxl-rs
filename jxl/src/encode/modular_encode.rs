// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Histogram-driven modular residual encoder.
//!
//! Replaces the bootstrap fixed-token path with proper frequency-based
//! Huffman encoding for real compression of modular channel residuals.

use crate::error::{Error, Result};

use super::{
    BitWriter,
    entropy::{
        HybridUintConfig, HybridUintEncoded,
        huffman_encode::{
            HuffmanCode, build_huffman_code, write_huffman_histograms, write_huffman_symbol,
        },
    },
    modular::write_minimal_group_header,
    pack_signed,
};

/// Precomputed token stream for a group of residual values.
pub struct TokenStream {
    /// HybridUint-encoded tokens (symbol + extra bits).
    pub(crate) tokens: Vec<HybridUintEncoded>,
    /// The Huffman code built from frequencies.
    pub(crate) code: HuffmanCode,
    /// The HybridUint config used.
    pub(crate) uint_config: HybridUintConfig,
}

/// Encode residual values into a token stream with histogram.
fn build_token_stream(
    signed_residuals: &[i32],
    uint_config: HybridUintConfig,
) -> Result<TokenStream> {
    let mut max_token = 0u32;
    let mut tokens = Vec::with_capacity(signed_residuals.len());

    for &sr in signed_residuals {
        let unsigned = pack_signed(sr);
        let enc = uint_config.encode(unsigned);
        max_token = max_token.max(enc.token);
        tokens.push(enc);
    }

    // Build frequency distribution.
    let alphabet_size = (max_token as usize + 1).max(1);
    let mut frequencies = vec![0u64; alphabet_size];
    for enc in &tokens {
        frequencies[enc.token as usize] += 1;
    }

    let code = build_huffman_code(&frequencies).ok_or(Error::InvalidHuffman)?;

    Ok(TokenStream {
        tokens,
        code,
        uint_config,
    })
}

/// Write the token stream (Huffman symbols + extra bits) to the writer.
fn write_token_stream(writer: &mut BitWriter, stream: &TokenStream) -> Result<()> {
    for enc in &stream.tokens {
        write_huffman_symbol(writer, &stream.code, enc.token as usize)?;
        if enc.nbits > 0 {
            writer.write(enc.nbits as usize, u64::from(enc.extra_bits))?;
        }
    }
    Ok(())
}

/// Build the modular tree token stream (6 tree contexts) for a single leaf node.
///
/// The tree decoder reads symbols in this order for a leaf:
///   1. property (context 1) = 0 -> indicates leaf
///   2. predictor (context 2)
///   3. offset (context 3)
///   4. multiplier_log (context 4) = 0
///   5. multiplier_bits (context 5) = 0
///
/// Context 0 (split_val) is never read for leaf nodes.
fn build_tree_token_stream(offset: i32, predictor: u32) -> Result<TokenStream> {
    if predictor > 13 {
        return Err(Error::InvalidPredictor(predictor));
    }

    let tree_config = HybridUintConfig::new(15, 0, 0);
    let offset_unsigned = pack_signed(offset);

    // Symbols in the order the decoder reads them for a single leaf.
    let tree_values = [
        0u32,            // property = 0 (indicates leaf)
        predictor,       // predictor
        offset_unsigned, // offset (unsigned via pack_signed)
        0,               // multiplier_log = 0 (multiplier = 1)
        0,               // multiplier_bits = 0
    ];

    let mut max_token = 0u32;
    let mut tokens = Vec::with_capacity(tree_values.len());
    for &val in &tree_values {
        let enc = tree_config.encode(val);
        max_token = max_token.max(enc.token);
        tokens.push(enc);
    }

    let alphabet_size = (max_token as usize + 1).max(1);
    let mut frequencies = vec![0u64; alphabet_size];
    for enc in &tokens {
        frequencies[enc.token as usize] += 1;
    }

    let code = build_huffman_code(&frequencies).ok_or(Error::InvalidHuffman)?;

    Ok(TokenStream {
        tokens,
        code,
        uint_config: tree_config,
    })
}

/// Write a complete modular section with histogram-driven Huffman encoding.
///
/// Writes:
/// 1. GroupHeader (use_global_tree, weighted header, transforms)
/// 2. Tree (single leaf with given params)
/// 3. Residual histograms
/// 4. Residual data (Huffman symbols + extra bits)
pub fn write_modular_section_huffman(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    signed_residuals: &[i32],
    uint_config: HybridUintConfig,
    use_global_tree: bool,
) -> Result<()> {
    write_minimal_group_header(writer, use_global_tree)?;

    // Write tree using histogram-driven coding.
    let tree_stream = build_tree_token_stream(offset, predictor)?;

    // Tree histograms: 6 contexts, all map to histogram 0.
    let tree_context_map = [0u8; 6];
    write_huffman_histograms(
        writer,
        &tree_context_map,
        &[tree_stream.uint_config],
        &[tree_stream.code.clone()],
    )?;
    write_token_stream(writer, &tree_stream)?;

    // Residual histograms: 1 context.
    let residual_stream = build_token_stream(signed_residuals, uint_config)?;
    let residual_context_map = [0u8];
    write_huffman_histograms(
        writer,
        &residual_context_map,
        &[residual_stream.uint_config],
        &[residual_stream.code.clone()],
    )?;
    write_token_stream(writer, &residual_stream)?;

    Ok(())
}

/// Write LF global section with histogram-driven tree + residual encoding.
///
/// For single-group images, also includes the residual data.
/// For multi-group images, only writes tree + histograms (no residuals).
pub fn write_lf_global_section_huffman(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    signed_residuals: Option<&[i32]>,
    uint_config: HybridUintConfig,
) -> Result<()> {
    // LfQuantFactors: all_default = true.
    writer.write(1, 1)?;

    // Global tree present: no.
    writer.write(1, 0)?;

    if let Some(residuals) = signed_residuals {
        // Single-group: write full section with residuals.
        write_modular_section_huffman(writer, offset, predictor, residuals, uint_config, false)?;
    } else {
        // Multi-group: write group header + tree + empty residual histograms.
        write_minimal_group_header(writer, false)?;

        let tree_stream = build_tree_token_stream(offset, predictor)?;
        let tree_context_map = [0u8; 6];
        write_huffman_histograms(
            writer,
            &tree_context_map,
            &[tree_stream.uint_config],
            &[tree_stream.code.clone()],
        )?;
        write_token_stream(writer, &tree_stream)?;

        // Empty residual histograms (single symbol 0).
        let empty_code = build_huffman_code(&[1]).ok_or(Error::InvalidHuffman)?;
        write_huffman_histograms(writer, &[0u8], &[uint_config], &[empty_code])?;
    }

    Ok(())
}

/// Write an HF group section with histogram-driven encoding.
pub fn write_hf_group_section_huffman(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    signed_residuals: &[i32],
    uint_config: HybridUintConfig,
) -> Result<()> {
    write_modular_section_huffman(
        writer,
        offset,
        predictor,
        signed_residuals,
        uint_config,
        false,
    )
}

/// Build tree token stream for a zero-predictor, zero-offset tree.
/// Exposed for VarDCT HF metadata encoding.
pub fn build_tree_tokens_zero() -> Result<TokenStream> {
    build_tree_token_stream(0, 0) // predictor=0 (Zero), offset=0
}

/// Write tree token stream to a BitWriter.
/// Exposed for VarDCT HF metadata encoding.
pub fn write_tree_token_stream(writer: &mut BitWriter, stream: &TokenStream) -> Result<()> {
    write_token_stream(writer, stream)
}

/// Compute gradient prediction residuals for a 2D channel.
///
/// Gradient (ClampedGradient): pred = left + top - topleft, clamped to [min(left,top), max(left,top)].
/// This matches JXL predictor 5.
fn compute_gradient_residuals(data: &[i32], width: usize, height: usize) -> Vec<i32> {
    let mut residuals = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let val = data[y * width + x];
            let left = if x > 0 { data[y * width + x - 1] } else { 0 };
            let top = if y > 0 { data[(y - 1) * width + x] } else { 0 };
            let topleft = if x > 0 && y > 0 {
                data[(y - 1) * width + x - 1]
            } else {
                0
            };
            // ClampedGradient: gradient clamped to [min(left,top), max(left,top)]
            let grad = left as i64 + top as i64 - topleft as i64;
            let lo = left.min(top) as i64;
            let hi = left.max(top) as i64;
            let pred = grad.clamp(lo, hi) as i32;
            residuals.push(val - pred);
        }
    }
    residuals
}

/// Encode a multi-channel signed integer modular stream.
///
/// `data` contains `num_channels` channels stored sequentially,
/// each of size `width * height`.
///
/// Tries both Zero predictor and Gradient predictor, picks the smaller output.
/// Channels are encoded in order: all pixels of channel 0, then channel 1, etc.
/// Encode a multi-channel signed integer modular stream.
///
/// `data` contains `num_channels` channels stored sequentially,
/// each of size `width * height`.
///
/// Tries both Zero predictor and Gradient predictor, picks the smaller output.
/// Channels are encoded in order: all pixels of channel 0, then channel 1, etc.
pub fn encode_modular_signed_stream(
    writer: &mut BitWriter,
    width: usize,
    height: usize,
    num_channels: usize,
    data: &[i32],
) -> Result<()> {
    assert_eq!(data.len(), width * height * num_channels);

    let uint_config = HybridUintConfig::new(4, 1, 2);

    // Estimate cost with multiple predictors and pick the best.
    // We use sum of pack_signed(residual) as a proxy for entropy.
    let channel_size = width * height;
    let zero_cost: u64 = data.iter().map(|&v| pack_signed(v) as u64).sum();

    // Gradient predictor (5): ClampedGradient
    let mut grad_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        grad_residuals.extend(compute_gradient_residuals(ch, width, height));
    }
    let grad_cost: u64 = grad_residuals
        .iter()
        .map(|&v| pack_signed(v) as u64)
        .sum();

    // Left predictor (1)
    let mut left_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        for y in 0..height {
            for x in 0..width {
                let val = ch[y * width + x];
                let left = if x > 0 { ch[y * width + x - 1] } else { 0 };
                left_residuals.push(val - left);
            }
        }
    }
    let left_cost: u64 = left_residuals
        .iter()
        .map(|&v| pack_signed(v) as u64)
        .sum();

    // Top predictor (2)
    let mut top_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        for y in 0..height {
            for x in 0..width {
                let val = ch[y * width + x];
                let top = if y > 0 { ch[(y - 1) * width + x] } else { 0 };
                top_residuals.push(val - top);
            }
        }
    }
    let top_cost: u64 = top_residuals
        .iter()
        .map(|&v| pack_signed(v) as u64)
        .sum();

    // Pick the best
    let mut best_cost = zero_cost;
    let mut best_predictor = 0u32;
    let mut best_residuals: Option<Vec<i32>> = None;

    if grad_cost < best_cost {
        best_cost = grad_cost;
        best_predictor = 5;
        best_residuals = Some(grad_residuals);
    }
    if left_cost < best_cost {
        best_cost = left_cost;
        best_predictor = 1;
        best_residuals = Some(left_residuals);
    }
    if top_cost < best_cost {
        best_cost = top_cost;
        best_predictor = 2;
        best_residuals = Some(top_residuals);
    }
    let _ = best_cost;

    if let Some(residuals) = best_residuals {
        write_modular_section_huffman(writer, 0, best_predictor, &residuals, uint_config, false)
    } else {
        write_modular_section_huffman(writer, 0, 0, data, uint_config, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        entropy_coding::decode::SymbolReader,
        frame::quantizer::LfQuantFactors,
        headers::{JxlHeader, modular::GroupHeader},
    };

    fn finish_with_padding(writer: BitWriter) -> Vec<u8> {
        let mut w = writer;
        // Pad with zero bytes so BitReader has enough lookahead.
        w.byte_align_zero_pad().unwrap();
        for _ in 0..16 {
            w.write(8, 0).unwrap();
        }
        w.finish()
    }

    #[test]
    fn test_tree_encoding_zero_offset_roundtrip() {
        // Just test tree encoding for the all-zero case.
        let uint_config = HybridUintConfig::new(4, 2, 0);

        let mut writer = BitWriter::new();
        write_minimal_group_header(&mut writer, false).unwrap();

        // Write tree
        let tree_stream = super::build_tree_token_stream(0, 0).unwrap();
        super::write_huffman_histograms(
            &mut writer,
            &[0u8; 6],
            &[tree_stream.uint_config],
            &[tree_stream.code.clone()],
        )
        .unwrap();
        super::write_token_stream(&mut writer, &tree_stream).unwrap();

        // Write minimal residual histograms
        let empty_code = super::build_huffman_code(&[1])
            .ok_or(crate::error::Error::InvalidHuffman)
            .unwrap();
        super::write_huffman_histograms(&mut writer, &[0u8], &[uint_config], &[empty_code])
            .unwrap();

        let bytes = finish_with_padding(writer);

        let mut br = BitReader::new(&bytes);
        let header = GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);

        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);
        let node_debug = format!("{:?}", tree.nodes[0]);
        assert!(
            node_debug.contains("Zero"),
            "Expected Zero predictor, got: {}",
            node_debug
        );
        assert!(
            node_debug.contains("offset: 0"),
            "Expected offset 0, got: {}",
            node_debug
        );
    }

    #[test]
    fn test_tree_then_residual_roundtrip() {
        // Mimic write_modular_section_huffman but decode step-by-step.
        let residuals: Vec<i32> = vec![1, 2, 3];
        let uint_config = HybridUintConfig::new(4, 2, 0);

        let mut writer = BitWriter::new();
        // Step 1: Group header.
        write_minimal_group_header(&mut writer, false).unwrap();

        // Step 2: Tree histograms + data.
        let tree_stream = super::build_tree_token_stream(0, 0).unwrap();
        super::write_huffman_histograms(
            &mut writer,
            &[0u8; 6],
            &[tree_stream.uint_config],
            &[tree_stream.code.clone()],
        )
        .unwrap();
        super::write_token_stream(&mut writer, &tree_stream).unwrap();

        // Step 3: Residual histograms + data.
        let residual_stream = super::build_token_stream(&residuals, uint_config).unwrap();
        super::write_huffman_histograms(
            &mut writer,
            &[0u8],
            &[residual_stream.uint_config],
            &[residual_stream.code.clone()],
        )
        .unwrap();
        super::write_token_stream(&mut writer, &residual_stream).unwrap();

        let bytes = finish_with_padding(writer);

        // Decode.
        let mut br = BitReader::new(&bytes);
        let _header = GroupHeader::read(&mut br).unwrap();
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);

        let mut res_reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for &expected in &residuals {
            let got = res_reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(got, expected);
        }
        res_reader
            .check_final_state(&tree.histograms, &mut br)
            .unwrap();
    }

    #[test]
    fn test_2x1_image_standalone_section_bytes() {
        // Reproduce the exact section bytes for 2x1 (50,0,0) + (100,0,0)
        let residuals: Vec<i32> = vec![0, 50, -50, -50, -50, -50];
        let uint_config = HybridUintConfig::new(4, 2, 0);
        let offset = 50;

        let mut writer = BitWriter::new();
        super::write_lf_global_section_huffman(
            &mut writer,
            offset,
            0,
            Some(&residuals),
            uint_config,
        )
        .unwrap();
        let bytes = writer.finish();
        eprintln!(
            "Standalone section ({} bytes): {}",
            bytes.len(),
            bytes
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    #[test]
    fn test_residual_stream_neg25_pos25_roundtrip() {
        // Test encoding [-25, 25] roundtrip at the raw stream level.
        let uint_config = HybridUintConfig::new(4, 2, 0);
        let signed_residuals: Vec<i32> = vec![-25, 25, -25, 25, -25, 25];

        // Build token stream
        let stream = super::build_token_stream(&signed_residuals, uint_config).unwrap();

        // Check tokens
        for (i, enc) in stream.tokens.iter().enumerate() {
            eprintln!(
                "Token {}: token={}, nbits={}, extra={}",
                i, enc.token, enc.nbits, enc.extra_bits
            );
        }

        // Write just the Huffman table + token stream
        let mut writer = BitWriter::new();
        let context_map = [0u8];
        super::write_huffman_histograms(
            &mut writer,
            &context_map,
            &[stream.uint_config],
            &[stream.code.clone()],
        )
        .unwrap();
        super::write_token_stream(&mut writer, &stream).unwrap();
        let bytes = finish_with_padding(writer);

        // Decode: read Huffman histograms then read tokens
        let mut br = BitReader::new(&bytes);
        // HuffmanCodes::decode expects the number of contexts and reads
        // all alphabet sizes then all tables.
        let hists = crate::entropy_coding::decode::Histograms::decode(1, &mut br, false).unwrap();

        let mut reader = SymbolReader::new(&hists, &mut br, None).unwrap();
        for (i, &expected) in signed_residuals.iter().enumerate() {
            let got = reader.read_signed(&hists, &mut br, 0);
            assert_eq!(
                got, expected,
                "Residual {i}: expected {expected}, got {got}"
            );
        }
        reader.check_final_state(&hists, &mut br).unwrap();
    }

    #[test]
    fn test_2x1_image_roundtrip() {
        // 2x1 image: (50,0,0) and (100,0,0). Offset=50, residuals channel-major.
        // R: [50-50, 100-50] = [0, 50]
        // G: [0-50, 0-50] = [-50, -50]
        // B: [0-50, 0-50] = [-50, -50]
        let residuals: Vec<i32> = vec![0, 50, -50, -50, -50, -50];
        let uint_config = HybridUintConfig::new(4, 2, 0);
        let offset = 50;

        let mut writer = BitWriter::new();
        super::write_lf_global_section_huffman(
            &mut writer,
            offset,
            0,
            Some(&residuals),
            uint_config,
        )
        .unwrap();
        let bytes = finish_with_padding(writer);

        // Decode via full modular path.
        let mut br = BitReader::new(&bytes);
        // LfQuantFactors
        assert_eq!(br.read(1).unwrap(), 1); // all_default
        // Global tree
        assert_eq!(br.read(1).unwrap(), 0); // no global tree
        // GroupHeader
        let _header = GroupHeader::read(&mut br).unwrap();
        // Tree
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);
        // Tree nodes are not directly accessible (private), so just check via decode.

        let mut reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for (i, &expected) in residuals.iter().enumerate() {
            let got = reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(
                got, expected,
                "residual mismatch at index {}: got {} expected {}",
                i, got, expected
            );
        }
        reader.check_final_state(&tree.histograms, &mut br).unwrap();
    }

    #[test]
    fn test_hf_group_wide_range_roundtrip() {
        // Test with -5..=8 which triggers the failure in the full encoder.
        let residuals: Vec<i32> = (-5..=8).collect();
        let uint_config = HybridUintConfig::new(4, 2, 0);

        let mut writer = BitWriter::new();
        super::write_hf_group_section_huffman(&mut writer, 0, 0, &residuals, uint_config).unwrap();
        let bytes = finish_with_padding(writer);

        let mut br = BitReader::new(&bytes);
        let _header = GroupHeader::read(&mut br).unwrap();
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();

        let mut reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for (i, &expected) in residuals.iter().enumerate() {
            let got = reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(
                got, expected,
                "mismatch at index {}: got {} expected {}",
                i, got, expected
            );
        }
        reader.check_final_state(&tree.histograms, &mut br).unwrap();
    }

    #[test]
    fn test_residual_histogram_only_roundtrip() {
        // Test just writing+reading residual histograms + data, no tree.
        let residuals: Vec<i32> = vec![1, 2, 3];
        let uint_config = HybridUintConfig::new(4, 2, 0);
        let stream = super::build_token_stream(&residuals, uint_config).unwrap();

        let mut writer = BitWriter::new();
        super::write_huffman_histograms(
            &mut writer,
            &[0u8],
            &[stream.uint_config],
            &[stream.code.clone()],
        )
        .unwrap();
        super::write_token_stream(&mut writer, &stream).unwrap();
        let bytes = finish_with_padding(writer);

        let mut br = BitReader::new(&bytes);
        let hist = crate::entropy_coding::decode::Histograms::decode(1, &mut br, true).unwrap();
        let mut reader = SymbolReader::new(&hist, &mut br, None).unwrap();
        for &expected in &residuals {
            let got = reader.read_signed(&hist, &mut br, 0);
            assert_eq!(got, expected, "mismatch: got {} expected {}", got, expected);
        }
        reader.check_final_state(&hist, &mut br).unwrap();
    }

    #[test]
    fn test_write_modular_section_huffman_simple_roundtrip() {
        // Minimal: 3 positive residuals
        let residuals: Vec<i32> = vec![1, 2, 3];
        // Use config (0,0,0) to make tokens == pack_signed values (simplest path).
        let uint_config = HybridUintConfig::new(0, 0, 0);

        let mut writer = BitWriter::new();
        write_modular_section_huffman(&mut writer, 0, 0, &residuals, uint_config, false).unwrap();
        let bytes = finish_with_padding(writer);

        let mut br = BitReader::new(&bytes);
        let header = GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);

        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);

        let mut res_reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for &expected in &residuals {
            let got = res_reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(
                got, expected,
                "residual mismatch: got {} expected {}",
                got, expected
            );
        }
        res_reader
            .check_final_state(&tree.histograms, &mut br)
            .unwrap();
    }

    #[test]
    fn test_write_modular_section_huffman_roundtrip() {
        let residuals: Vec<i32> = (0..100).map(|i| (i % 10) - 5).collect();
        let uint_config = HybridUintConfig::new(4, 2, 0);

        let mut writer = BitWriter::new();
        write_modular_section_huffman(&mut writer, 0, 0, &residuals, uint_config, false).unwrap();
        let bytes = finish_with_padding(writer);

        // Verify we can parse the group header and tree.
        let mut br = BitReader::new(&bytes);
        let header = GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);
        assert!(header.transforms.is_empty());

        // Parse tree using the real decoder.
        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);

        // The Tree::read also reads the residual histograms.
        // Now read residual data using the tree's histograms.
        let mut res_reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for &expected in &residuals {
            let got = res_reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(got, expected, "residual mismatch");
        }
        res_reader
            .check_final_state(&tree.histograms, &mut br)
            .unwrap();
    }

    #[test]
    fn test_write_lf_global_section_huffman_single_group() {
        let residuals: Vec<i32> = vec![0, 1, -1, 2, -2, 3, -3];
        let uint_config = HybridUintConfig::new(4, 2, 0);

        let mut writer = BitWriter::new();
        write_lf_global_section_huffman(&mut writer, 0, 0, Some(&residuals), uint_config).unwrap();
        let bytes = finish_with_padding(writer);

        let mut br = BitReader::new(&bytes);
        let _lf = LfQuantFactors::new(&mut br).unwrap();
        assert_eq!(br.read(1).unwrap(), 0); // no global tree

        let header = GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);

        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);

        let mut res_reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for &expected in &residuals {
            let got = res_reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(got, expected);
        }
        res_reader
            .check_final_state(&tree.histograms, &mut br)
            .unwrap();
    }

    #[test]
    fn test_write_hf_group_section_huffman_roundtrip() {
        let residuals: Vec<i32> = (0..50).map(|i| i * 2 - 49).collect();
        let uint_config = HybridUintConfig::new(4, 2, 0);

        let mut writer = BitWriter::new();
        write_hf_group_section_huffman(&mut writer, 5, 1, &residuals, uint_config).unwrap();
        let bytes = finish_with_padding(writer);

        let mut br = BitReader::new(&bytes);
        let header = GroupHeader::read(&mut br).unwrap();
        assert!(!header.use_global_tree);

        let tree = crate::frame::modular::Tree::read(&mut br, 1024).unwrap();
        assert_eq!(tree.nodes.len(), 1);

        let mut res_reader = SymbolReader::new(&tree.histograms, &mut br, None).unwrap();
        for &expected in &residuals {
            let got = res_reader.read_signed(&tree.histograms, &mut br, 0);
            assert_eq!(got, expected);
        }
        res_reader
            .check_final_state(&tree.histograms, &mut br)
            .unwrap();
    }
}
