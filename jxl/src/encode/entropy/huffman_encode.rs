// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Multi-symbol Huffman table encoder.
//!
//! Supports:
//! - Simple tables (1-4 symbols, using the decoder's "simple" format)
//! - Complex tables (arbitrary symbol count, using code-length encoding)

use crate::error::{Error, Result};

use super::super::BitWriter;
use super::huffman::write_varint16;

const MAX_BITS: usize = 15;

const CODE_LENGTH_CODE_ORDER: [u8; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// A Huffman code assignment: symbol -> (code_length, bit_pattern).
#[derive(Clone, Debug)]
pub struct HuffmanCode {
    pub alphabet_size: usize,
    pub code_lengths: Vec<u8>,
    pub codes: Vec<u32>,
}

/// Build a canonical Huffman code from symbol frequencies.
pub fn build_huffman_code(frequencies: &[u64]) -> Option<HuffmanCode> {
    let alphabet_size = frequencies.len();
    if alphabet_size == 0 {
        return None;
    }

    let nonzero: Vec<(usize, u64)> = frequencies
        .iter()
        .enumerate()
        .filter(|&(_, f)| *f > 0)
        .map(|(i, f)| (i, *f))
        .collect();

    if nonzero.is_empty() {
        return None;
    }

    if nonzero.len() == 1 {
        let mut code_lengths = vec![0u8; alphabet_size];
        code_lengths[nonzero[0].0] = 1;
        let mut codes = vec![0u32; alphabet_size];
        codes[nonzero[0].0] = 0;
        return Some(HuffmanCode {
            alphabet_size,
            code_lengths,
            codes,
        });
    }

    let code_lengths = build_code_lengths(frequencies, MAX_BITS);
    let codes = canonical_codes(&code_lengths);

    Some(HuffmanCode {
        alphabet_size,
        code_lengths,
        codes,
    })
}

/// Build length-limited Huffman code lengths from frequencies.
fn build_code_lengths(frequencies: &[u64], max_bits: usize) -> Vec<u8> {
    let n = frequencies.len();
    let mut lengths = vec![0u8; n];

    let nonzero: Vec<(usize, u64)> = frequencies
        .iter()
        .enumerate()
        .filter(|&(_, f)| *f > 0)
        .map(|(i, f)| (i, *f))
        .collect();

    if nonzero.len() <= 1 {
        if let Some(&(idx, _)) = nonzero.first() {
            lengths[idx] = 1;
        }
        return lengths;
    }

    if nonzero.len() == 2 {
        lengths[nonzero[0].0] = 1;
        lengths[nonzero[1].0] = 1;
        return lengths;
    }

    // Two-queue Huffman tree construction using leaf indices (0..num_nz).
    let num_nz = nonzero.len();
    // Sort leaves by frequency ascending, break ties by symbol index.
    let mut leaves: Vec<(u64, usize)> = nonzero
        .iter()
        .enumerate()
        .map(|(li, &(_, f))| (f, li))
        .collect();
    leaves.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut queue1: std::collections::VecDeque<(u64, usize)> = leaves.into_iter().collect();
    let mut queue2: std::collections::VecDeque<(u64, usize)> = std::collections::VecDeque::new();

    // Tree has num_nz leaves (indices 0..num_nz) and up to num_nz-1 internal nodes.
    let mut parent = vec![0usize; 2 * num_nz];
    let mut depth = vec![0u8; 2 * num_nz];
    let mut next_node = num_nz;

    let pop_min = |q1: &mut std::collections::VecDeque<(u64, usize)>,
                   q2: &mut std::collections::VecDeque<(u64, usize)>|
     -> (u64, usize) {
        match (q1.front(), q2.front()) {
            (Some(&a), Some(&b)) => {
                if a.0 <= b.0 {
                    q1.pop_front().unwrap()
                } else {
                    q2.pop_front().unwrap()
                }
            }
            (Some(_), None) => q1.pop_front().unwrap(),
            (None, Some(_)) => q2.pop_front().unwrap(),
            (None, None) => unreachable!(),
        }
    };

    while queue1.len() + queue2.len() > 1 {
        let (f1, n1) = pop_min(&mut queue1, &mut queue2);
        let (f2, n2) = pop_min(&mut queue1, &mut queue2);
        parent[n1] = next_node;
        parent[n2] = next_node;
        queue2.push_back((f1.saturating_add(f2), next_node));
        next_node += 1;
    }

    let root = next_node - 1;
    depth[root] = 0;
    for i in (0..root).rev() {
        depth[i] = depth[parent[i]] + 1;
    }

    // Map leaf indices back to symbol indices.
    for (leaf_idx, &(sym, _)) in nonzero.iter().enumerate() {
        lengths[sym] = depth[leaf_idx];
    }

    cap_code_lengths(&mut lengths, max_bits);
    lengths
}

/// Cap code lengths at max_bits by redistributing.
fn cap_code_lengths(lengths: &mut [u8], max_bits: usize) {
    let max_bits_u8 = max_bits as u8;
    let max_len = *lengths.iter().max().unwrap_or(&0);
    if max_len <= max_bits_u8 {
        return;
    }

    let mut nz: Vec<(usize, u8)> = lengths
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l > 0)
        .map(|(i, l)| (i, *l))
        .collect();

    for item in &mut nz {
        if item.1 > max_bits_u8 {
            item.1 = max_bits_u8;
        }
    }

    nz.sort_by(|a, b| b.1.cmp(&a.1));

    loop {
        let kraft_sum: u64 = nz
            .iter()
            .map(|&(_, l)| 1u64 << (max_bits - l as usize))
            .sum();
        let target = 1u64 << max_bits;

        if kraft_sum == target {
            break;
        }

        if kraft_sum > target {
            if let Some(item) = nz.iter_mut().find(|item| item.1 > 1) {
                item.1 -= 1;
            } else {
                break;
            }
        } else if let Some(item) = nz
            .iter_mut()
            .rev()
            .find(|item| (item.1 as usize) < max_bits)
        {
            item.1 += 1;
        } else {
            break;
        }

        nz.sort_by(|a, b| b.1.cmp(&a.1));
    }

    for &(sym, len) in &nz {
        lengths[sym] = len;
    }
}

/// Build canonical Huffman bit patterns from code lengths.
fn canonical_codes(code_lengths: &[u8]) -> Vec<u32> {
    let n = code_lengths.len();
    let mut codes = vec![0u32; n];

    let max_len = *code_lengths.iter().max().unwrap_or(&0) as usize;
    if max_len == 0 {
        return codes;
    }

    let mut bl_count = vec![0u32; max_len + 1];
    for &len in code_lengths {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    let mut next_code = vec![0u32; max_len + 1];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    for (sym, &len) in code_lengths.iter().enumerate() {
        if len > 0 {
            codes[sym] = bit_reverse(next_code[len as usize], len as usize);
            next_code[len as usize] += 1;
        }
    }

    codes
}

fn bit_reverse(val: u32, nbits: usize) -> u32 {
    let mut result = 0u32;
    let mut v = val;
    for _ in 0..nbits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Counts the number of distinct nonzero-frequency symbols.
pub fn count_nonzero(frequencies: &[u64]) -> usize {
    frequencies.iter().filter(|f| **f > 0).count()
}

/// Write a Huffman table for a single histogram.
///
/// When `alphabet_size == 1`, writes nothing (decoder handles this
/// as a special case without reading any table bits).
pub fn write_huffman_table(writer: &mut BitWriter, code: &HuffmanCode) -> Result<()> {
    let al_size = code.alphabet_size;

    // Decoder special case: al_size==1 means the only possible symbol is 0,
    // no table data is read from the bitstream.
    if al_size == 1 {
        return Ok(());
    }

    let nonzero: Vec<(usize, u8)> = code
        .code_lengths
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l > 0)
        .map(|(i, l)| (i, *l))
        .collect();

    let num_nonzero = nonzero.len();

    if num_nonzero <= 1 {
        write_simple_table_1(
            writer,
            al_size,
            nonzero.first().map(|&(s, _)| s as u16).unwrap_or(0),
        )?;
        return Ok(());
    }

    if num_nonzero <= 4 {
        write_simple_table(writer, al_size, &nonzero)?;
        return Ok(());
    }

    write_complex_table(writer, code)?;
    Ok(())
}

fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let bits = usize::BITS - (n - 1).leading_zeros();
    bits as usize
}

fn write_simple_table_1(writer: &mut BitWriter, al_size: usize, symbol: u16) -> Result<()> {
    let max_bits = ceil_log2(al_size);

    // simple_code_or_skip = 1
    writer.write(2, 1)?;
    // nsym - 1 = 0
    writer.write(2, 0)?;
    writer.write(max_bits, u64::from(symbol))?;
    Ok(())
}

fn write_simple_table(
    writer: &mut BitWriter,
    al_size: usize,
    nonzero: &[(usize, u8)],
) -> Result<()> {
    let max_bits = ceil_log2(al_size);
    let num_symbols = nonzero.len();
    assert!((2..=4).contains(&num_symbols));

    // simple_code_or_skip = 1
    writer.write(2, 1)?;
    // nsym - 1
    writer.write(2, (num_symbols - 1) as u64)?;

    match num_symbols {
        2 => {
            let mut symbols: Vec<u16> = nonzero.iter().map(|&(s, _)| s as u16).collect();
            symbols.sort_unstable();
            writer.write(max_bits, u64::from(symbols[0]))?;
            writer.write(max_bits, u64::from(symbols[1]))?;
        }
        3 => {
            // Symbol with length 1 first, then sorted length-2 symbols.
            let mut by_len: Vec<(u16, u8)> = nonzero.iter().map(|&(s, l)| (s as u16, l)).collect();
            by_len.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
            let sym0 = by_len[0].0;
            let mut rest: Vec<u16> = by_len[1..].iter().map(|&(s, _)| s).collect();
            rest.sort_unstable();

            writer.write(max_bits, u64::from(sym0))?;
            writer.write(max_bits, u64::from(rest[0]))?;
            writer.write(max_bits, u64::from(rest[1]))?;
        }
        4 => {
            let mut by_len: Vec<(u16, u8)> = nonzero.iter().map(|&(s, l)| (s as u16, l)).collect();
            by_len.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

            let all_same_len = by_len.iter().all(|&(_, l)| l == by_len[0].1);

            if all_same_len {
                // tree_select = false, all length 2, sorted.
                let mut symbols: Vec<u16> = nonzero.iter().map(|&(s, _)| s as u16).collect();
                symbols.sort_unstable();
                for &s in &symbols {
                    writer.write(max_bits, u64::from(s))?;
                }
                writer.write(1, 0)?;
            } else {
                // tree_select = true: lengths 1, 2, 3, 3
                let sym0 = by_len[0].0;
                let sym1 = by_len[1].0;
                let mut rest: Vec<u16> = by_len[2..].iter().map(|&(s, _)| s).collect();
                rest.sort_unstable();

                writer.write(max_bits, u64::from(sym0))?;
                writer.write(max_bits, u64::from(sym1))?;
                writer.write(max_bits, u64::from(rest[0]))?;
                writer.write(max_bits, u64::from(rest[1]))?;
                writer.write(1, 1)?;
            }
        }
        _ => unreachable!(),
    }

    Ok(())
}

/// Code-length symbol with optional extra bits.
#[derive(Clone, Debug, Copy)]
struct CodeLengthSymbol {
    symbol: u8, // 0..15 = literal code length, 16 = repeat, 17 = zero fill
    extra: u8,  // extra bits value (for 16 and 17)
}

/// Encode code lengths into a sequence of code-length symbols.
/// Encode code lengths into a sequence of code-length symbols.
///
/// Uses the decoder's accumulative repeat semantics:
/// - Symbol 16 (repeat previous): extra_bits = code_len - 14 = 2 bits.
///   First use: repeat = extra_value + 3 (3..6).
///   Subsequent: repeat = (repeat - 2) << 2 + extra_value + 3.
/// - Symbol 17 (zero fill): extra_bits = 3 bits.
///   First use: repeat = extra_value + 3 (3..10).
///   Subsequent: repeat = (repeat - 2) << 3 + extra_value + 3.
///
/// For simplicity and correctness, we emit at most one repeat code per run
/// (for runs of 3+), falling back to individual literals for shorter runs.
fn encode_code_lengths(code_lengths: &[u8]) -> Vec<CodeLengthSymbol> {
    let mut symbols = Vec::new();
    let mut i = 0;

    while i < code_lengths.len() {
        let len = code_lengths[i];

        if len == 0 {
            // Count zero run.
            let mut run = 1;
            while i + run < code_lengths.len() && code_lengths[i + run] == 0 {
                run += 1;
            }

            if run >= 3 && run <= 10 {
                // Single zero-fill code (symbol 17, 3 extra bits, encodes 3..10).
                symbols.push(CodeLengthSymbol {
                    symbol: 17,
                    extra: (run - 3) as u8,
                });
                i += run;
            } else {
                // Individual zeros.
                for _ in 0..run {
                    symbols.push(CodeLengthSymbol {
                        symbol: 0,
                        extra: 0,
                    });
                    i += 1;
                }
            }
        } else {
            // Emit first literal.
            symbols.push(CodeLengthSymbol {
                symbol: len,
                extra: 0,
            });
            i += 1;

            // Count subsequent identical values.
            let mut run = 0;
            while i + run < code_lengths.len() && code_lengths[i + run] == len {
                run += 1;
            }

            if run >= 3 && run <= 6 {
                // Single repeat code (symbol 16, 2 extra bits, encodes 3..6).
                symbols.push(CodeLengthSymbol {
                    symbol: 16,
                    extra: (run - 3) as u8,
                });
                i += run;
            } else {
                // Individual literals.
                for _ in 0..run {
                    symbols.push(CodeLengthSymbol {
                        symbol: len,
                        extra: 0,
                    });
                    i += 1;
                }
            }
        }
    }

    symbols
}

/// Write a complex Huffman table using code-length encoding.
fn write_complex_table(writer: &mut BitWriter, code: &HuffmanCode) -> Result<()> {
    let cl_symbols = encode_code_lengths(&code.code_lengths[..code.alphabet_size]);

    let mut cl_freqs = [0u64; 18];
    for sym in &cl_symbols {
        cl_freqs[sym.symbol as usize] += 1;
    }

    let cl_code = build_huffman_code(&cl_freqs).ok_or(Error::InvalidHuffman)?;

    let mut cl_code_lengths = [0u8; 18];
    for (i, &len) in cl_code.code_lengths.iter().enumerate() {
        cl_code_lengths[i] = len;
    }

    // Determine skip prefix: how many leading entries in CODE_LENGTH_CODE_ORDER have length 0.
    let mut num_cl_codes = 18;
    while num_cl_codes > 0
        && cl_code_lengths[CODE_LENGTH_CODE_ORDER[num_cl_codes - 1] as usize] == 0
    {
        num_cl_codes -= 1;
    }

    // simple_code_or_skip: 0 = start from 0, 2 = skip 2, 3 = skip 3
    // (1 = simple table, not used for complex)
    let skip_count = {
        let mut skip = 0usize;
        for i in 0..18 {
            if cl_code_lengths[CODE_LENGTH_CODE_ORDER[i] as usize] == 0 {
                skip += 1;
            } else {
                break;
            }
        }
        skip
    };

    let simple_code_or_skip: u64 = match skip_count {
        0 | 1 => 0,
        2 => 2,
        _ => 3,
    };
    let start_pos = simple_code_or_skip as usize;

    writer.write(2, simple_code_or_skip)?;

    // Write code-length code lengths using static 4-bit Huffman table.
    let mut space = 32i32;
    for i in start_pos..18 {
        if space <= 0 {
            break;
        }
        let cl_len = cl_code_lengths[CODE_LENGTH_CODE_ORDER[i] as usize];
        write_static_huffman_symbol(writer, cl_len)?;
        if cl_len != 0 {
            space -= 32i32 >> cl_len;
        }
    }

    // Write code-length symbols.
    for sym in &cl_symbols {
        let s = sym.symbol as usize;
        let len = cl_code.code_lengths[s] as usize;
        if len == 0 && cl_code.alphabet_size == 1 {
            // Single-symbol CL code: don't write any bits.
            // (The decoder knows the only symbol.)
        } else {
            let bits = cl_code.codes[s];
            writer.write(len, u64::from(bits))?;
        }

        if sym.symbol == 16 {
            writer.write(2, u64::from(sym.extra))?;
        } else if sym.symbol == 17 {
            writer.write(3, u64::from(sym.extra))?;
        }
    }

    Ok(())
}

/// Write a value using the static 4-bit Huffman table for code-length code lengths.
///
/// Decoder table:
///   BITS: [2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4]
///   VALS: [0, 4, 3, 2, 0, 4, 3, 1, 0, 4, 3, 2, 0, 4, 3, 5]
fn write_static_huffman_symbol(writer: &mut BitWriter, value: u8) -> Result<()> {
    match value {
        0 => writer.write(2, 0b00)?,
        4 => writer.write(2, 0b01)?,
        3 => writer.write(2, 0b10)?,
        2 => writer.write(3, 0b011)?,
        1 => writer.write(4, 0b0111)?,
        5 => writer.write(4, 0b1111)?,
        _ => return Err(Error::InvalidHuffman),
    }
    Ok(())
}

/// Write a symbol using an already-built Huffman code.
///
/// When `alphabet_size == 1`, writes 0 bits (the decoder knows the only
/// possible symbol and doesn't read any bits).
pub fn write_huffman_symbol(
    writer: &mut BitWriter,
    code: &HuffmanCode,
    symbol: usize,
) -> Result<()> {
    if symbol >= code.alphabet_size {
        return Err(Error::InvalidHuffman);
    }
    // Single-symbol alphabet: decoder returns symbol 0 without reading bits.
    if code.alphabet_size == 1 {
        return Ok(());
    }
    let len = code.code_lengths[symbol] as usize;
    if len == 0 {
        return Ok(());
    }
    writer.write(len, u64::from(code.codes[symbol]))?;
    Ok(())
}

/// Write a complete set of Huffman histograms for multiple contexts.
pub fn write_huffman_histograms(
    writer: &mut BitWriter,
    context_map: &[u8],
    uint_configs: &[super::HybridUintConfig],
    huffman_codes: &[HuffmanCode],
) -> Result<()> {
    let num_contexts = context_map.len();
    let num_histograms = huffman_codes.len();

    if num_histograms == 0 || uint_configs.len() != num_histograms {
        return Err(Error::InvalidHuffman);
    }

    // LZ77 params: disabled.
    writer.write(1, 0)?;

    // Context map.
    if num_contexts > 1 {
        super::context_map::write_simple_context_map(writer, context_map)?;
    }

    // use_prefix_code = true (Huffman).
    writer.write(1, 1)?;

    // HybridUint configs.
    for config in uint_configs {
        config.write(writer, MAX_BITS)?;
    }

    // Alphabet sizes.
    for code in huffman_codes {
        write_varint16(writer, (code.alphabet_size - 1) as u16)?;
    }

    // Table payloads.
    for code in huffman_codes {
        write_huffman_table(writer, code)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        encode::BitWriter,
        entropy_coding::{
            decode::{Histograms, SymbolReader},
            huffman::HuffmanCodes,
        },
    };

    #[test]
    fn test_build_huffman_code_single_symbol() {
        let freqs = [0, 0, 5, 0];
        let code = build_huffman_code(&freqs).unwrap();
        assert_eq!(code.code_lengths[2], 1);
        assert_eq!(count_nonzero(&freqs), 1);
    }

    #[test]
    fn test_build_huffman_code_two_symbols() {
        let freqs = [10, 0, 5, 0];
        let code = build_huffman_code(&freqs).unwrap();
        assert_eq!(code.code_lengths[0], 1);
        assert_eq!(code.code_lengths[2], 1);
    }

    #[test]
    fn test_build_huffman_code_four_symbols() {
        let freqs = [10, 20, 5, 15];
        let code = build_huffman_code(&freqs).unwrap();
        for (i, &f) in freqs.iter().enumerate() {
            if f > 0 {
                assert!(code.code_lengths[i] > 0);
            }
        }
        assert!(code.code_lengths[1] <= code.code_lengths[2]);
    }

    #[test]
    fn test_canonical_codes_kraft_inequality() {
        let freqs = [100, 50, 25, 12, 6, 3, 1, 1];
        let code = build_huffman_code(&freqs).unwrap();

        let kraft: f64 = code
            .code_lengths
            .iter()
            .filter(|l| **l > 0)
            .map(|&l| 2f64.powi(-(l as i32)))
            .sum();
        assert!((kraft - 1.0).abs() < 1e-10, "Kraft sum = {}", kraft);
    }

    #[test]
    fn test_write_simple_table_1_roundtrip() {
        let mut freqs = [0u64; 256];
        freqs[42] = 100;

        let code = build_huffman_code(&freqs).unwrap();

        let mut writer = BitWriter::new();
        write_varint16(&mut writer, (code.alphabet_size - 1) as u16).unwrap();
        write_huffman_table(&mut writer, &code).unwrap();
        writer.write(32, 0).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();
        assert_eq!(codes.read(&mut br, 0), 42);
    }

    #[test]
    fn test_write_simple_table_2_roundtrip() {
        let mut freqs = [0u64; 16];
        freqs[3] = 50;
        freqs[7] = 50;

        let code = build_huffman_code(&freqs).unwrap();

        let mut writer = BitWriter::new();
        write_varint16(&mut writer, (code.alphabet_size - 1) as u16).unwrap();
        write_huffman_table(&mut writer, &code).unwrap();
        write_huffman_symbol(&mut writer, &code, 3).unwrap();
        write_huffman_symbol(&mut writer, &code, 7).unwrap();
        writer.write(32, 0).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();
        assert_eq!(codes.read(&mut br, 0), 3);
        assert_eq!(codes.read(&mut br, 0), 7);
    }

    #[test]
    fn test_write_simple_table_3_roundtrip() {
        let mut freqs = [0u64; 16];
        freqs[1] = 100;
        freqs[5] = 30;
        freqs[9] = 30;

        let code = build_huffman_code(&freqs).unwrap();

        let mut writer = BitWriter::new();
        write_varint16(&mut writer, (code.alphabet_size - 1) as u16).unwrap();
        write_huffman_table(&mut writer, &code).unwrap();
        write_huffman_symbol(&mut writer, &code, 1).unwrap();
        write_huffman_symbol(&mut writer, &code, 5).unwrap();
        write_huffman_symbol(&mut writer, &code, 9).unwrap();
        writer.write(32, 0).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();
        assert_eq!(codes.read(&mut br, 0), 1);
        assert_eq!(codes.read(&mut br, 0), 5);
        assert_eq!(codes.read(&mut br, 0), 9);
    }

    #[test]
    fn test_write_simple_table_4_equal_roundtrip() {
        let mut freqs = [0u64; 32];
        freqs[2] = 25;
        freqs[8] = 25;
        freqs[14] = 25;
        freqs[20] = 25;

        let code = build_huffman_code(&freqs).unwrap();

        let mut writer = BitWriter::new();
        write_varint16(&mut writer, (code.alphabet_size - 1) as u16).unwrap();
        write_huffman_table(&mut writer, &code).unwrap();
        for &s in &[2, 8, 14, 20] {
            write_huffman_symbol(&mut writer, &code, s).unwrap();
        }
        writer.write(32, 0).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();
        assert_eq!(codes.read(&mut br, 0), 2);
        assert_eq!(codes.read(&mut br, 0), 8);
        assert_eq!(codes.read(&mut br, 0), 14);
        assert_eq!(codes.read(&mut br, 0), 20);
    }

    #[test]
    fn test_write_complex_table_sparse_17_roundtrip() {
        // 17-symbol alphabet with gaps (symbols 11, 13, 15 have freq 0).
        // Exercises complex table writing with zero-frequency gaps.
        let mut freqs = vec![0u64; 17];
        for &s in &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16] {
            freqs[s] = 1;
        }
        let code = build_huffman_code(&freqs).unwrap();

        let mut writer = BitWriter::new();
        write_varint16(&mut writer, (code.alphabet_size - 1) as u16).unwrap();
        write_huffman_table(&mut writer, &code).unwrap();
        // Write all non-zero symbols.
        for &s in &[0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16] {
            write_huffman_symbol(&mut writer, &code, s).unwrap();
        }
        writer.write(32, 0).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();
        for &s in &[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16] {
            let got = codes.read(&mut br, 0);
            assert_eq!(got, s, "expected symbol {} got {}", s, got);
        }
    }

    #[test]
    fn test_write_complex_table_roundtrip() {
        let freqs = [100u64, 80, 60, 40, 20, 10, 5, 3];
        let code = build_huffman_code(&freqs).unwrap();

        let mut writer = BitWriter::new();
        write_varint16(&mut writer, (code.alphabet_size - 1) as u16).unwrap();
        write_huffman_table(&mut writer, &code).unwrap();
        for s in 0..8 {
            write_huffman_symbol(&mut writer, &code, s).unwrap();
        }
        writer.write(32, 0).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let codes = HuffmanCodes::decode(1, &mut br).unwrap();
        for s in 0..8u32 {
            assert_eq!(codes.read(&mut br, 0), s);
        }
    }

    #[test]
    fn test_write_huffman_histograms_roundtrip() {
        let freqs = [50u64, 30, 15, 5];
        let code = build_huffman_code(&freqs).unwrap();
        let uint_config = super::super::HybridUintConfig::new(4, 0, 0);

        let mut writer = BitWriter::new();
        write_huffman_histograms(&mut writer, &[0], &[uint_config], &[code.clone()]).unwrap();
        for &s in &[0usize, 1, 2, 3, 0, 1] {
            write_huffman_symbol(&mut writer, &code, s).unwrap();
        }
        writer.write(32, 0).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let hist = Histograms::decode(1, &mut br, true).unwrap();
        let mut reader = SymbolReader::new(&hist, &mut br, None).unwrap();
        for &expected in &[0u32, 1, 2, 3, 0, 1] {
            let got = reader.read_unsigned(&hist, &mut br, 0);
            assert_eq!(got, expected);
        }
        reader.check_final_state(&hist, &mut br).unwrap();
    }
}
