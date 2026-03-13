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
        ans::{AnsDistribution, AnsToken, write_ans_histograms, write_ans_stream},
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
/// Build a single-leaf tree with the given offset and predictor.
fn build_tree_token_stream(offset: i32, predictor: u32) -> Result<TokenStream> {
    let (stream, _map) = build_tree_token_stream_multi(&[(offset, predictor)])?;
    Ok(stream)
}

/// Build a tree from leaf specifications, returning (token_stream, leaf_id_to_channel_map).
///
/// If `leaves.len() == 1`, creates a single-leaf tree. Map is [0].
/// If `leaves.len() > 1`, creates a channel-split tree.
/// Each leaf has (offset, predictor).
///
/// JXL tree convention: LEFT child = property > splitval, RIGHT child = property <= splitval.
/// Tree is encoded in BFS order (breadth-first).
fn build_tree_token_stream_multi(leaves: &[(i32, u32)]) -> Result<(TokenStream, Vec<usize>)> {
    for &(_, predictor) in leaves {
        if predictor > 13 {
            return Err(Error::InvalidPredictor(predictor));
        }
    }

    let tree_config = HybridUintConfig::new(15, 0, 0);
    let mut tree_values: Vec<u32> = Vec::new();
    // Maps leaf_id (BFS order) to channel index
    let mut leaf_id_to_channel: Vec<usize> = Vec::new();

    if leaves.len() == 1 {
        tree_values.extend_from_slice(&[
            0,                                   // property = 0 (leaf)
            leaves[0].1,                         // predictor
            pack_signed(leaves[0].0),            // offset
            0,                                   // multiplier_log = 0
            0,                                   // multiplier_bits = 0
        ]);
        leaf_id_to_channel.push(0);
    } else {
        // Multi-leaf tree: chain of splits by channel property (property 0).
        // Encoded in BFS order.
        //
        // For N channels, we build a right-leaning chain:
        //   Node 0: split(channel, val=0)
        //     LEFT (ch > 0): Node 1 (continuation)
        //     RIGHT (ch <= 0): Leaf for channel 0
        //   Node 1: split(channel, val=1)
        //     LEFT (ch > 1): Node 2 (continuation) or Leaf for last channel
        //     RIGHT (ch <= 1): Leaf for channel 1
        //   ...
        //
        // BFS visits: all split nodes first, then leaves in order:
        //   split_0, split_1, ..., split_{N-2}, leaf_ch0, leaf_ch_{N-1}, leaf_ch1, ...
        //
        // Actually BFS interleaves: each split's children are queued left then right.
        // Let me build it properly with a BFS queue.

        let n = leaves.len();

        // Build tree structure first, then serialize in BFS order.
        // Each node is either Split{val, left_idx, right_idx} or Leaf{channel}.
        enum Node {
            Split { val: i32, left: usize, right: usize },
            Leaf { channel: usize },
        }

        let mut nodes: Vec<Node> = Vec::new();

        // Build chain: channel 0 splits off to the right, then channel 1, etc.
        // For channel i (0..n-2): split on val=i, RIGHT=leaf(ch_i), LEFT=next_split
        // Last split: LEFT=leaf(ch_{n-1}), RIGHT=leaf(ch_{n-2})

        // Build bottom-up: start with the last two channels
        let last_leaf = nodes.len();
        nodes.push(Node::Leaf { channel: n - 1 });
        let second_last_leaf = nodes.len();
        nodes.push(Node::Leaf { channel: n - 2 });

        // Last split node: split on val=n-2
        let mut current_split = nodes.len();
        nodes.push(Node::Split {
            val: (n - 2) as i32 - 1, // split on val = n-3... no
            left: last_leaf,
            right: second_last_leaf,
        });

        // Actually this approach is getting complicated. Let me use a recursive build.
        // Build a function that creates a subtree for channels [lo..hi]
        fn build_subtree(
            nodes: &mut Vec<Node>,
            leaves_data: &[(i32, u32)],
            lo: usize,
            hi: usize, // exclusive
        ) -> usize {
            if hi - lo == 1 {
                let idx = nodes.len();
                nodes.push(Node::Leaf { channel: lo });
                return idx;
            }
            // Split: channel <= lo goes RIGHT (leaf for channel lo)
            // channel > lo goes LEFT (subtree for channels lo+1..hi)
            let idx = nodes.len();
            nodes.push(Node::Split {
                val: lo as i32, // placeholder, will be overwritten
                left: 0,
                right: 0,
            });
            let left = build_subtree(nodes, leaves_data, lo + 1, hi);
            let right_idx = nodes.len();
            nodes.push(Node::Leaf { channel: lo });
            nodes[idx] = Node::Split {
                val: lo as i32,
                left,
                right: right_idx,
            };
            idx
        }

        nodes.clear();
        let root = build_subtree(&mut nodes, &leaves, 0, n);
        assert_eq!(root, 0);

        // Now serialize in BFS order
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0usize);
        while let Some(node_idx) = queue.pop_front() {
            match &nodes[node_idx] {
                Node::Split { val, left, right } => {
                    // Property 0 (channel) encoded as 0+1=1
                    tree_values.push(1);
                    tree_values.push(pack_signed(*val));
                    queue.push_back(*left);
                    queue.push_back(*right);
                }
                Node::Leaf { channel } => {
                    let (offset, predictor) = leaves[*channel];
                    tree_values.push(0); // leaf
                    tree_values.push(predictor);
                    tree_values.push(pack_signed(offset));
                    tree_values.push(0); // multiplier_log
                    tree_values.push(0); // multiplier_bits
                    leaf_id_to_channel.push(*channel);
                }
            }
        }
    }

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

    Ok((
        TokenStream {
            tokens,
            code,
            uint_config: tree_config,
        },
        leaf_id_to_channel,
    ))
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

/// Write a complete modular section using ANS entropy coding.
///
/// Same structure as write_modular_section_huffman but uses ANS instead of Huffman
/// for the residual data. Tree is still Huffman (tiny, overhead negligible).
/// Public wrapper for ANS modular section encoding, used by vardct HF metadata.
pub fn write_modular_section_ans_pub(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    signed_residuals: &[i32],
    uint_config: HybridUintConfig,
    use_global_tree: bool,
) -> Result<()> {
    write_modular_section_ans(writer, offset, predictor, signed_residuals, uint_config, use_global_tree)
}

fn write_modular_section_ans(
    writer: &mut BitWriter,
    offset: i32,
    predictor: u32,
    signed_residuals: &[i32],
    uint_config: HybridUintConfig,
    use_global_tree: bool,
) -> Result<()> {
    write_minimal_group_header(writer, use_global_tree)?;

    // Tree: use Huffman (tree is tiny, ANS overhead would be larger)
    let tree_stream = build_tree_token_stream(offset, predictor)?;
    let tree_context_map = [0u8; 6];
    write_huffman_histograms(
        writer,
        &tree_context_map,
        &[tree_stream.uint_config],
        &[tree_stream.code.clone()],
    )?;
    write_token_stream(writer, &tree_stream)?;

    // Residual data: use ANS
    let residual_encoded: Vec<HybridUintEncoded> = signed_residuals
        .iter()
        .map(|&v| uint_config.encode(pack_signed(v)))
        .collect();

    let max_token = residual_encoded
        .iter()
        .map(|e| e.token)
        .max()
        .unwrap_or(0);
    let alphabet_size = (max_token as usize + 1).max(1);
    let mut frequencies = vec![0u64; alphabet_size];
    for enc in &residual_encoded {
        frequencies[enc.token as usize] += 1;
    }

    let dist = AnsDistribution::from_frequencies(&frequencies)
        .ok_or(Error::InvalidAnsHistogram)?;

    let ans_tokens: Vec<AnsToken> = residual_encoded
        .iter()
        .map(|enc| AnsToken {
            symbol: enc.token,
            cluster: 0,
            extra_bits: enc.extra_bits,
            extra_nbits: enc.nbits as usize,
        })
        .collect();

    let residual_context_map = [0u8];
    write_ans_histograms(
        writer,
        &residual_context_map,
        &[uint_config],
        &[dist.clone()],
    )?;
    write_ans_stream(writer, &[dist], &ans_tokens)?;

    Ok(())
}

/// Write a modular section using ANS with per-channel histograms.
///
/// Uses a multi-leaf tree that splits by channel, giving each channel its own
/// entropy context. This can be more efficient when channels have different
/// distributions (e.g., Y vs X vs B in DC data).
pub fn write_modular_section_ans_perchannel(
    writer: &mut BitWriter,
    predictor: u32,
    signed_residuals: &[i32],
    num_channels: usize,
    channel_size: usize,
    uint_config: HybridUintConfig,
    use_global_tree: bool,
) -> Result<()> {
    if num_channels < 2 {
        // Fall back to single-histogram ANS for single channel
        return write_modular_section_ans(
            writer,
            0,
            predictor,
            signed_residuals,
            uint_config,
            use_global_tree,
        );
    }

    write_minimal_group_header(writer, use_global_tree)?;

    // Multi-leaf tree: one leaf per channel, all with same predictor
    let leaves: Vec<(i32, u32)> = (0..num_channels).map(|_| (0i32, predictor)).collect();
    let (tree_stream, leaf_id_to_channel) = build_tree_token_stream_multi(&leaves)?;
    let tree_context_map = [0u8; 6];
    write_huffman_histograms(
        writer,
        &tree_context_map,
        &[tree_stream.uint_config],
        &[tree_stream.code.clone()],
    )?;
    write_token_stream(writer, &tree_stream)?;

    // Build channel_to_leaf_id mapping (inverse of leaf_id_to_channel)
    let mut channel_to_leaf_id = vec![0usize; num_channels];
    for (leaf_id, &ch) in leaf_id_to_channel.iter().enumerate() {
        channel_to_leaf_id[ch] = leaf_id;
    }

    // Build per-leaf ANS distributions (indexed by leaf_id)
    let mut leaf_dists: Vec<AnsDistribution> = Vec::with_capacity(num_channels);
    let mut all_tokens: Vec<AnsToken> = Vec::new();

    // First pass: build distributions per leaf_id
    let mut leaf_freqs: Vec<Vec<u64>> = vec![Vec::new(); num_channels];
    let mut leaf_encoded: Vec<Vec<HybridUintEncoded>> = vec![Vec::new(); num_channels];

    for ch in 0..num_channels {
        let leaf_id = channel_to_leaf_id[ch];
        let ch_data = &signed_residuals[ch * channel_size..(ch + 1) * channel_size];
        let encoded: Vec<HybridUintEncoded> = ch_data
            .iter()
            .map(|&v| uint_config.encode(pack_signed(v)))
            .collect();

        let max_token = encoded.iter().map(|e| e.token).max().unwrap_or(0);
        let alphabet_size = (max_token as usize + 1).max(1);
        if leaf_freqs[leaf_id].len() < alphabet_size {
            leaf_freqs[leaf_id].resize(alphabet_size, 0);
        }
        for enc in &encoded {
            leaf_freqs[leaf_id][enc.token as usize] += 1;
        }
        leaf_encoded[leaf_id] = encoded;
    }

    for leaf_id in 0..num_channels {
        let dist = AnsDistribution::from_frequencies(&leaf_freqs[leaf_id])
            .ok_or(Error::InvalidAnsHistogram)?;
        leaf_dists.push(dist);
    }

    // Build token stream: data is emitted channel-by-channel (ch0, ch1, ch2)
    // Each channel's tokens use the leaf_id corresponding to that channel
    for ch in 0..num_channels {
        let leaf_id = channel_to_leaf_id[ch];
        for enc in &leaf_encoded[leaf_id] {
            all_tokens.push(AnsToken {
                symbol: enc.token,
                cluster: leaf_id,
                extra_bits: enc.extra_bits,
                extra_nbits: enc.nbits as usize,
            });
        }
    }

    // Context map: leaf_id -> histogram leaf_id (identity mapping)
    let context_map: Vec<u8> = (0..num_channels as u8).collect();
    let uint_configs_vec: Vec<HybridUintConfig> = vec![uint_config; num_channels];
    write_ans_histograms(writer, &context_map, &uint_configs_vec, &leaf_dists)?;
    write_ans_stream(writer, &leaf_dists, &all_tokens)?;

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

    // Match libjxl's default modular HybridUint config for VarDCT DC data.
    let uint_config = HybridUintConfig::new(4, 2, 0);

    // Try multiple predictors and pick the best by actual encoded byte size.
    let channel_size = width * height;

    // Helper: compute left/top/topleft matching the decoder's fallback chain.
    // The decoder (predict.rs PredictionData::get_with_neighbors) uses:
    //   left: x>0 -> row[y][x-1]; else y>0 -> row[y-1][0]; else 0
    //   top:  y>0 -> row[y-1][x]; else -> left
    //   topleft: ... (complex, see get_with_neighbors)
    let get_left = |ch: &[i32], x: usize, y: usize| -> i32 {
        if x > 0 {
            ch[y * width + x - 1]
        } else if y > 0 {
            ch[(y - 1) * width + 0]
        } else {
            0
        }
    };
    let get_top = |ch: &[i32], x: usize, y: usize| -> i32 {
        if y > 0 {
            ch[(y - 1) * width + x]
        } else {
            get_left(ch, x, y)
        }
    };
    let get_topleft = |ch: &[i32], x: usize, y: usize| -> i32 {
        if x > 0 {
            if y > 0 {
                ch[(y - 1) * width + x - 1]
            } else {
                get_left(ch, x, y)
            }
        } else if y > 0 {
            get_left(ch, x, y)
        } else {
            0
        }
    };

    // Gradient predictor (5): ClampedGradient
    let mut grad_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        for y in 0..height {
            for x in 0..width {
                let val = ch[y * width + x];
                let left = get_left(ch, x, y) as i64;
                let top = get_top(ch, x, y) as i64;
                let topleft = get_topleft(ch, x, y) as i64;
                let min = left.min(top);
                let max = left.max(top);
                let grad = left + top - topleft;
                let grad_clamp_max = if topleft < min { max } else { grad };
                let pred = if topleft > max { min } else { grad_clamp_max };
                grad_residuals.push(val - pred as i32);
            }
        }
    }
    // Left predictor (1)
    let mut left_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        for y in 0..height {
            for x in 0..width {
                let val = ch[y * width + x];
                let left = get_left(ch, x, y);
                left_residuals.push(val - left);
            }
        }
    }
    // Top predictor (2)
    let mut top_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        for y in 0..height {
            for x in 0..width {
                let val = ch[y * width + x];
                let top = get_top(ch, x, y);
                top_residuals.push(val - top);
            }
        }
    }
    // Select predictor (4): matches decoder's predict.rs select() exactly
    // p = left + top - topleft; if |p - left| < |p - top|: left else: top
    let mut select_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        for y in 0..height {
            for x in 0..width {
                let val = ch[y * width + x];
                let left = get_left(ch, x, y) as i64;
                let top = get_top(ch, x, y) as i64;
                let topleft = get_topleft(ch, x, y) as i64;
                let p = left + top - topleft;
                let pred = if (p - left).abs() < (p - top).abs() {
                    left
                } else {
                    top
                };
                select_residuals.push(val - pred as i32);
            }
        }
    }

    // AverageWestAndNorth predictor (3): (left + top) / 2 (integer truncation towards zero)
    let mut avg_wn_residuals = Vec::with_capacity(data.len());
    for c in 0..num_channels {
        let ch = &data[c * channel_size..(c + 1) * channel_size];
        for y in 0..height {
            for x in 0..width {
                let val = ch[y * width + x];
                let left = get_left(ch, x, y) as i64;
                let top = get_top(ch, x, y) as i64;
                let pred = (left + top) / 2;
                avg_wn_residuals.push(val - pred as i32);
            }
        }
    }

    // Pick predictor by exact encoded size (regression-safe).
    let candidates: [(u32, &[i32]); 6] = [
        (0, data),
        (5, &grad_residuals),
        (1, &left_residuals),
        (2, &top_residuals),
        (4, &select_residuals),
        (3, &avg_wn_residuals),
    ];

    let mut best_predictor = 0u32;
    let mut best_idx = 0usize;
    let mut best_size = usize::MAX;
    let mut best_use_ans = false;
    let mut best_perchannel = false;

    for (idx, (predictor, values)) in candidates.iter().enumerate() {
        // Try Huffman
        let mut tmp = BitWriter::new();
        write_modular_section_huffman(&mut tmp, 0, *predictor, values, uint_config, false)?;
        let sz = tmp.finish().len();
        if sz < best_size {
            best_size = sz;
            best_predictor = *predictor;
            best_idx = idx;
            best_use_ans = false;
            best_perchannel = false;
        }

        // Try ANS (only worthwhile for larger streams where ANS overhead pays off)
        if data.len() >= 256 {
            let mut tmp_ans = BitWriter::new();
            if write_modular_section_ans(&mut tmp_ans, 0, *predictor, values, uint_config, false)
                .is_ok()
            {
                let sz_ans = tmp_ans.finish().len();
                if sz_ans < best_size {
                    best_size = sz_ans;
                    best_predictor = *predictor;
                    best_idx = idx;
                    best_use_ans = true;
                    best_perchannel = false;
                }
            }

            // Try per-channel ANS (separate histogram per channel)
            if num_channels > 1 {
                let mut tmp_pcans = BitWriter::new();
                if write_modular_section_ans_perchannel(
                    &mut tmp_pcans, *predictor, values, num_channels, channel_size,
                    uint_config, false,
                ).is_ok() {
                    let sz_pc = tmp_pcans.finish().len();
                    if sz_pc < best_size {
                        best_size = sz_pc;
                        best_predictor = *predictor;
                        best_idx = idx;
                        best_use_ans = true;
                        best_perchannel = true;
                    }
                }
            }
        }
    }

    if best_use_ans && best_perchannel {
        write_modular_section_ans_perchannel(
            writer,
            best_predictor,
            candidates[best_idx].1,
            num_channels,
            channel_size,
            uint_config,
            false,
        )
    } else if best_use_ans {
        write_modular_section_ans(
            writer,
            0,
            best_predictor,
            candidates[best_idx].1,
            uint_config,
            false,
        )
    } else {
        write_modular_section_huffman(
            writer,
            0,
            best_predictor,
            candidates[best_idx].1,
            uint_config,
            false,
        )
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
