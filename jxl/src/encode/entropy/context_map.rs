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

fn mtf_forward(context_map: &[u8]) -> Vec<u8> {
    let mut mtf = [0u8; 256];
    for (i, v) in mtf.iter_mut().enumerate() {
        *v = i as u8;
    }

    let mut out = Vec::with_capacity(context_map.len());
    for &sym in context_map {
        let mut idx = 0usize;
        while idx < mtf.len() && mtf[idx] != sym {
            idx += 1;
        }
        debug_assert!(idx < mtf.len());
        out.push(idx as u8);

        if idx != 0 {
            let val = mtf[idx];
            for i in (1..=idx).rev() {
                mtf[i] = mtf[i - 1];
            }
            mtf[0] = val;
        }
    }
    out
}

fn write_compressed_context_map(writer: &mut BitWriter, context_map: &[u8]) -> Result<()> {
    use super::huffman_encode::{
        build_huffman_code, write_huffman_histograms, write_huffman_symbol,
    };

    // is_simple = false
    writer.write(1, 0)?;

    // use_mtf = true (helps with long runs / few symbols)
    writer.write(1, 1)?;

    let mtf_stream = mtf_forward(context_map);

    let max_symbol = mtf_stream.iter().copied().max().unwrap_or(0) as usize;
    let alphabet_size = (max_symbol + 1).max(1);
    let mut freqs = vec![0u64; alphabet_size];
    for &s in &mtf_stream {
        freqs[s as usize] += 1;
    }

    let code = if freqs.iter().all(|&f| f == 0) {
        build_huffman_code(&[1]).ok_or(Error::InvalidHuffman)?
    } else {
        build_huffman_code(&freqs).ok_or(Error::InvalidHuffman)?
    };

    // Context map symbols are <= 255, encode directly as token=value (< split_token).
    let uint_config = super::HybridUintConfig::new(8, 0, 0);

    // Inner histogram stream for context-map symbols.
    // num_contexts = 1 so this won't recursively write another context map.
    write_huffman_histograms(writer, &[0u8], &[uint_config], &[code.clone()])?;

    for &s in &mtf_stream {
        write_huffman_symbol(writer, &code, s as usize)?;
    }

    Ok(())
}

/// Writes a context map with adaptive strategy:
/// - all-zero maps -> simple encoding (3 bits)
/// - small maps with few contexts -> simple encoding
/// - larger/non-zero maps -> compressed non-simple encoding with MTF + Huffman
pub fn write_context_map(writer: &mut BitWriter, context_map: &[u8]) -> Result<()> {
    let max_symbol = context_map.iter().copied().max().unwrap_or(0);
    if context_map.len() <= 32 || max_symbol == 0 {
        return write_simple_context_map(writer, context_map);
    }

    // Non-simple context maps are validated by the decoder and must not have
    // histogram-id holes (i.e. ids must be contiguous 0..max).
    let mut seen = [false; 256];
    let mut distinct = 0usize;
    for &v in context_map {
        let idx = v as usize;
        if !seen[idx] {
            seen[idx] = true;
            distinct += 1;
        }
    }
    let has_holes = distinct != max_symbol as usize + 1;
    if has_holes {
        return write_simple_context_map(writer, context_map);
    }

    write_compressed_context_map(writer, context_map)
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
    fn test_write_context_map_compressed_roundtrip() {
        // Long map with multiple histogram ids and long runs.
        let mut map = vec![0u8; 555];
        map.extend(vec![1u8; 1200]);
        map.extend(vec![2u8; 1200]);
        map.extend(vec![3u8; 1200]);
        map.extend(vec![4u8; 1200]);

        let mut writer = BitWriter::new();
        write_context_map(&mut writer, &map).unwrap();
        writer.write(32, 0).unwrap(); // padding for safe peeks
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
