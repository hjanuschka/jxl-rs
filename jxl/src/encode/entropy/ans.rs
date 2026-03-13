// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! rANS entropy encoder matching the JXL ANS decoder.
//!
//! Produces bitstreams decodable by `AnsCodes::decode` + `AnsReader`.
//! Uses 12-bit precision (LOG_SUM_PROBS=12, SUM_PROBS=4096).

use crate::encode::bit_writer::BitWriter;
use crate::error::{Error, Result};

const LOG_SUM_PROBS: usize = 12;
const SUM_PROBS: u32 = 1 << LOG_SUM_PROBS;

/// Expected initial/final ANS state (must match decoder's AnsReader::CHECKSUM).
const ANS_CHECKSUM: u32 = 0x130000;

/// A normalized ANS distribution for one histogram.
#[derive(Clone, Debug)]
pub struct AnsDistribution {
    /// Frequency for each symbol. Sum must equal SUM_PROBS (4096).
    pub freqs: Vec<u16>,
    /// Cumulative frequencies. cumul[i] = sum(freqs[0..i]).
    pub cumul: Vec<u16>,
    /// Number of symbols with freq > 0.
    pub alphabet_size: usize,
}

impl AnsDistribution {
    /// Build a distribution from raw (unnormalized) frequency counts.
    ///
    /// Normalizes so that frequencies sum to exactly 4096.
    /// Symbols with freq=0 stay at 0.
    pub fn from_frequencies(raw_freqs: &[u64]) -> Option<Self> {
        let nonzero_count = raw_freqs.iter().filter(|&&f| f > 0).count();
        if nonzero_count == 0 {
            return None;
        }

        let alphabet_size = raw_freqs.len();
        let total: u64 = raw_freqs.iter().sum();
        if total == 0 {
            return None;
        }

        // Normalize to SUM_PROBS
        let mut freqs = vec![0u16; alphabet_size];
        let mut assigned = 0u32;

        // First pass: proportional assignment, ensuring nonzero symbols get >= 1
        for (i, &f) in raw_freqs.iter().enumerate() {
            if f > 0 {
                let proportional = ((f as u128 * SUM_PROBS as u128) / total as u128) as u16;
                freqs[i] = proportional.max(1);
                assigned += freqs[i] as u32;
            }
        }

        // Adjust to hit exactly SUM_PROBS
        while assigned != SUM_PROBS {
            if assigned > SUM_PROBS {
                // Find the symbol with the largest frequency and reduce it
                let max_idx = freqs
                    .iter()
                    .enumerate()
                    .filter(|&(_, &f)| f > 1)
                    .max_by_key(|&(_, &f)| f)
                    .map(|(i, _)| i);
                if let Some(idx) = max_idx {
                    freqs[idx] -= 1;
                    assigned -= 1;
                } else {
                    break; // Can't reduce further
                }
            } else {
                // Find the symbol with the largest raw frequency and increase it
                let best_idx = raw_freqs
                    .iter()
                    .enumerate()
                    .filter(|&(_, &f)| f > 0)
                    .max_by_key(|&(_, &f)| f)
                    .map(|(i, _)| i);
                if let Some(idx) = best_idx {
                    freqs[idx] += 1;
                    assigned += 1;
                } else {
                    break;
                }
            }
        }

        debug_assert_eq!(freqs.iter().map(|&f| f as u32).sum::<u32>(), SUM_PROBS);

        // Build cumulative
        let mut cumul = vec![0u16; alphabet_size + 1];
        for i in 0..alphabet_size {
            cumul[i + 1] = cumul[i] + freqs[i];
        }

        Some(Self {
            freqs,
            cumul,
            alphabet_size,
        })
    }

    /// Build a single-symbol distribution.
    pub fn single_symbol(symbol: usize) -> Self {
        let alphabet_size = symbol + 1;
        let mut freqs = vec![0u16; alphabet_size];
        freqs[symbol] = SUM_PROBS as u16;
        let mut cumul = vec![0u16; alphabet_size + 1];
        for i in 0..alphabet_size {
            cumul[i + 1] = cumul[i] + freqs[i];
        }
        Self {
            freqs,
            cumul,
            alphabet_size,
        }
    }
}

// ==================== Distribution encoding ====================

/// Write an ANS distribution to the bitstream.
///
/// Chooses the most compact encoding format:
/// - Single symbol: 1 symbol has all probability
/// - Two symbols: exactly 2 non-zero symbols
/// - Evenly distributed: all symbols have ~equal probability
/// - Complex: general distribution
pub fn write_ans_distribution(
    w: &mut BitWriter,
    dist: &AnsDistribution,
    log_alpha_size: usize,
) -> Result<()> {
    let nonzero: Vec<(usize, u16)> = dist
        .freqs
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 0)
        .map(|(i, &f)| (i, f))
        .collect();

    match nonzero.len() {
        0 => {
            // Should not happen; write single symbol 0
            w.write(1, 1)?; // simple
            w.write(1, 0)?; // single symbol
            write_ans_u8(w, 0)?;
            Ok(())
        }
        1 => {
            // Single symbol
            let (sym, _) = nonzero[0];
            w.write(1, 1)?; // simple
            w.write(1, 0)?; // single symbol
            write_ans_u8(w, sym as u8)?;
            Ok(())
        }
        2 => {
            // Two symbols
            let (s0, f0) = nonzero[0];
            let (s1, _f1) = nonzero[1];
            w.write(1, 1)?; // simple
            w.write(1, 1)?; // two symbols
            write_ans_u8(w, s0 as u8)?;
            write_ans_u8(w, s1 as u8)?;
            w.write(LOG_SUM_PROBS, f0 as u64)?;
            Ok(())
        }
        _ => {
            // Always use complex encoding for now (simpler, always correct).
            // The evenly distributed format requires exact match with decoder's
            // distribution layout, which is fragile.
            write_ans_distribution_complex(w, dist, log_alpha_size)
        }
    }
}

/// Write the "complex" ANS distribution format.
///
/// Uses shift=13 for maximum precision so that all frequencies up to 4096
/// roundtrip exactly through encode/decode.
fn write_ans_distribution_complex(
    w: &mut BitWriter,
    dist: &AnsDistribution,
    _log_alpha_size: usize,
) -> Result<()> {
    w.write(1, 0)?; // not simple
    w.write(1, 0)?; // not evenly distributed

    let alphabet_size = dist.alphabet_size;
    assert!(alphabet_size >= 3);

    // Use max shift for full precision
    let shift: i16 = 13;
    // Encode shift: variable-length code
    // Decode formula: read unary len (0..3), then shift = (1<<len) - 1 + read(len)
    // shift=13 -> len=3, rem = 13 - 7 = 6
    let (len, rem) = if shift == 0 {
        (0i16, 0i16)
    } else if shift <= 2 {
        (1, shift - 1)
    } else if shift <= 6 {
        (2, shift - 3)
    } else {
        (3, shift - 7)
    };
    for _ in 0..len {
        w.write(1, 1)?;
    }
    if len < 3 {
        w.write(1, 0)?;
    }
    if len > 0 {
        w.write(len as usize, rem as u64)?;
    }

    // Write alphabet_size - 3
    write_ans_u8(w, (alphabet_size - 3) as u8)?;

    // Compute log code for each frequency
    // code=0 -> freq=0, code=1 -> freq=1, code=k (k>=2) -> freq in [2^(k-1), 2^k)
    let log_code = |freq: u16| -> u16 {
        if freq == 0 {
            0
        } else if freq == 1 {
            1
        } else {
            (32 - (freq as u32).leading_zeros()) as u16
        } // = floor(log2(freq)) + 1
    };

    // Find omit_pos: decoder uses strict '>' so picks the FIRST symbol
    // with the largest log code value (ties broken by first occurrence).
    // The decoder initializes omit_data = None, then for each symbol:
    //   if first nonzero: set omit_data = (code, idx)
    //   else if code > current_max: update omit_data
    let mut omit_pos = 0usize;
    let mut omit_log = 0u16;
    let mut first = true;
    for i in 0..alphabet_size {
        let code = log_code(dist.freqs[i]);
        if code == 0 {
            continue;
        } // skip zero-freq symbols
        if first {
            omit_pos = i;
            omit_log = code;
            first = false;
        } else if code > omit_log {
            omit_log = code;
            omit_pos = i;
        }
    }

    // Pass 1: write all prefix codes (decoder reads all of these first)
    let mut codes = vec![0u16; alphabet_size];
    for i in 0..alphabet_size {
        let code = log_code(dist.freqs[i]);
        codes[i] = code;
        write_ans_prefix(w, code)?;
    }

    // Pass 2: write extra bits (decoder reads these in a second pass)
    for i in 0..alphabet_size {
        let code = codes[i];
        if code <= 1 || i == omit_pos {
            continue;
        }

        let zeros = code as i16 - 1;
        let bitcount = (shift - ((LOG_SUM_PROBS as i16 - zeros) >> 1)).clamp(0, zeros);
        if bitcount > 0 {
            let base = 1u32 << zeros;
            let extra = dist.freqs[i] as u32 - base;
            // Decoder reconstructs: base + (read << (zeros - bitcount)).
            let written = extra >> (zeros - bitcount);
            w.write(bitcount as usize, written as u64)?;
        }
    }

    Ok(())
}

/// Write ANS u8 encoding: if 0, write 0-bit. Otherwise write 1-bit + 3-bit len + len bits.
fn write_ans_u8(w: &mut BitWriter, val: u8) -> Result<()> {
    if val == 0 {
        w.write(1, 0)?;
    } else {
        w.write(1, 1)?;
        let n = 32 - (val as u32).leading_zeros(); // number of bits needed (1..8)
        let n = n - 1; // remove the leading 1 bit
        w.write(3, n as u64)?;
        let extra = val as u64 - (1 << n);
        w.write(n as usize, extra)?;
    }
    Ok(())
}

/// Write a prefix code symbol (0-13) using the fixed 7-bit prefix table.
///
/// This is the inverse of `AnsHistogram::read_prefix` in the decoder.
fn write_ans_prefix(w: &mut BitWriter, val: u16) -> Result<()> {
    // The decoder uses a fixed 7-bit lookup table. The code assignments are:
    //   val  code     bits
    //    0   00000     5
    //    1   0001      4
    //    2   0010      4
    //    3   011       3
    //    4   0100      4
    //    5   0101      4
    //    6   100       3
    //    7   101       3
    //    8   110       3
    //    9   111       3
    //   10   000       3
    //   11   010000    6
    //   12   01000001  7  (but decoder only peeks 7 bits, so this is 7 bits max)
    //   13   11000001  7
    //
    // Actually let me derive from the decoder table:
    // STATIC_TABLE[(val, bits)]: indexed by 7-bit peek value
    // val 6: bits=3, code=100 -> write 3 bits: 0b100
    // val 7: bits=3, code=101 -> write 3 bits: 0b101
    // etc.
    //
    // The decoder reads LSB-first. Let me extract the codes from the table:
    //
    // Looking at the TABLE in the decoder (128 entries, 7-bit peek):
    //   idx=0b0000000 (0):  val=10, bits=3 -> code for 10 is bottom 3 bits of 0 = 000
    //   idx=0b0000001 (1):  val=12, bits=7 -> code for 12 is 0000001
    //   idx=0b0000010 (2):  val= 0, bits=5 -> code for 0 is bottom 5 bits of 2 = 00010
    //   ...
    //
    // Let me just hardcode the (code, bits) pairs:
    static CODES: [(u8, u8); 14] = [
        (0b00010, 5),   // 0
        (0b01110, 5),   // 1: wait, let me re-derive...
        (0b01100, 5),   // 2
        (0b00011, 4),   // 3: wait...
        (0b01001, 5),   // 4
        (0b00101, 5),   // 5
        (0b00100, 3),   // 6: wrong...
        (0b00101, 3),   // 7
        (0b00110, 3),   // 8
        (0b00111, 3),   // 9
        (0b00000, 3),   // 10
        (0b0100000, 7), // 11
        (0b0000001, 7), // 12
        (0b1000001, 7), // 13
    ];

    // Actually, deriving this from the decoder table is error-prone.
    // Let me use a different approach: build the inverse of the decoder table.
    // I'll derive codes directly from the decoder's TABLE constant.
    let _ = CODES; // suppress unused warning

    // From the decoder TABLE (128 entries, indexed by 7-bit peek, LSB-first):
    //   (value, bits)
    // I need to find the canonical bit pattern for each value.
    // For each value v, find the smallest index i where TABLE[i] = (v, bits_v).
    // The code is the bottom bits_v bits of i.

    #[rustfmt::skip]
    const TABLE: [(u8, u8); 128] = [
        (10, 3), (12, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        (10, 3), (13, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
        (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
    ];

    let v = val as usize;
    assert!(v <= 13, "ANS prefix symbol must be 0-13, got {v}");

    // Find the code for value v: smallest table index with value v
    let mut code: u8 = 0;
    let mut nbits: u8 = 0;
    for (idx, &(tv, tb)) in TABLE.iter().enumerate() {
        if tv as u16 == val {
            code = (idx as u8) & ((1 << tb) - 1);
            nbits = tb;
            break;
        }
    }
    assert!(nbits > 0, "value {v} not found in prefix table");

    w.write(nbits as usize, code as u64)?;
    Ok(())
}

// ==================== rANS stream encoder ====================

/// Bit payload associated with one token in forward decoding order.
/// Uses inline storage for refills (most tokens need 0-2 refills).
#[derive(Debug, Default)]
struct AnsTokenBits {
    /// 16-bit rANS normalization words consumed by `AnsHistogram::read`.
    /// Inline up to 4 refills to avoid heap allocation for most tokens.
    refills_inline: [u16; 4],
    refills_len: u8,
    refills_overflow: Option<Vec<u16>>,
    /// HybridUint extra bits consumed after reading the token.
    extra_bits: u32,
    extra_nbits: usize,
}

impl AnsTokenBits {
    #[inline]
    fn push_refill(&mut self, val: u16) {
        let len = self.refills_len as usize;
        if len < 4 {
            self.refills_inline[len] = val;
            self.refills_len += 1;
        } else {
            self.refills_overflow
                .get_or_insert_with(Vec::new)
                .push(val);
        }
    }

    #[inline]
    fn refills(&self) -> impl Iterator<Item = &u16> {
        let inline_len = (self.refills_len as usize).min(4);
        self.refills_inline[..inline_len].iter().chain(
            self.refills_overflow
                .as_ref()
                .map(|v| v.as_slice())
                .unwrap_or(&[])
                .iter(),
        )
    }
}

/// Per-cluster reverse mapping from (symbol, offset) -> 12-bit ANS index.
#[derive(Debug)]
struct AnsEncodeLookup {
    by_symbol_offset: Vec<Vec<u16>>,
}

/// Build the reverse ANS mapping for one distribution by decoding it with the
/// same `AnsHistogram` implementation and inverting the alias table mapping.
fn build_ans_encode_lookup(
    dist: &AnsDistribution,
    log_alpha_size: usize,
) -> Result<AnsEncodeLookup> {
    // Decode the distribution through the real decoder path so the encoder uses
    // exactly the same alias mapping as the decoder.
    let mut tmp_w = BitWriter::new();
    write_ans_distribution(&mut tmp_w, dist, log_alpha_size)?;
    tmp_w.write(32, 0)?; // safety padding for BitReader peeks
    let tmp_bytes = tmp_w.finish();
    let mut tmp_br = crate::bit_reader::BitReader::new(&tmp_bytes);
    let hist = crate::entropy_coding::ans::AnsHistogram::decode(&mut tmp_br, log_alpha_size)?;

    let mut by_symbol_offset: Vec<Vec<u16>> = dist
        .freqs
        .iter()
        .map(|&f| vec![u16::MAX; f as usize])
        .collect();

    let bucket_mask = if hist.log_bucket_size == 0 {
        0
    } else {
        (1u32 << hist.log_bucket_size) - 1
    };

    for idx in 0..SUM_PROBS {
        let i = (idx >> hist.log_bucket_size) as usize;
        let pos = idx & bucket_mask;

        let bucket = hist.buckets[i];
        let use_alias = pos >= bucket.alias_cutoff as u32;

        let symbol = if use_alias {
            bucket.alias_symbol as usize
        } else {
            i
        };

        if symbol >= by_symbol_offset.len() {
            continue;
        }

        let offset = if use_alias {
            bucket.alias_offset as u32 + pos
        } else {
            pos
        } as usize;

        if offset >= by_symbol_offset[symbol].len() {
            continue;
        }

        by_symbol_offset[symbol][offset] = idx as u16;
    }

    // Validate completeness
    for (sym, offsets) in by_symbol_offset.iter().enumerate() {
        if offsets.iter().any(|&v| v == u16::MAX) {
            return Err(Error::InvalidAnsHistogram);
        }
        if !offsets.is_empty() {
            debug_assert_eq!(offsets.len(), dist.freqs[sym] as usize);
        }
    }

    Ok(AnsEncodeLookup { by_symbol_offset })
}

/// Token ready for ANS encoding.
pub struct AnsToken {
    /// Symbol index (after HybridUint tokenization).
    pub symbol: u32,
    /// Which histogram cluster this token belongs to.
    pub cluster: usize,
    /// Extra bits value (from HybridUint).
    pub extra_bits: u32,
    /// Number of extra bits.
    pub extra_nbits: usize,
}

/// Encode a sequence of tokens using rANS and write to a BitWriter.
///
/// `distributions` maps cluster index -> AnsDistribution.
/// Tokens are processed in reverse to produce a forward-readable ANS stream.
///
/// The output includes the 32-bit initial state followed by interleaved
/// rANS normalization data and HybridUint extra bits.
pub fn write_ans_stream(
    w: &mut BitWriter,
    distributions: &[AnsDistribution],
    tokens: &[AnsToken],
) -> Result<()> {
    // Must match write_ans_histograms() choice.
    let max_alpha = distributions
        .iter()
        .map(|d| d.alphabet_size)
        .max()
        .unwrap_or(1);
    let log_alpha_size = if max_alpha <= 32 {
        5
    } else if max_alpha <= 64 {
        6
    } else if max_alpha <= 128 {
        7
    } else {
        8
    };

    // Build reverse lookup tables once per cluster.
    let mut lookups = Vec::with_capacity(distributions.len());
    for dist in distributions {
        lookups.push(build_ans_encode_lookup(dist, log_alpha_size)?);
    }

    let mut state: u32 = ANS_CHECKSUM;
    // Per-token payloads collected in reverse token order.
    let mut token_bits_rev: Vec<AnsTokenBits> = Vec::with_capacity(tokens.len());

    // Process tokens in REVERSE order
    for token in tokens.iter().rev() {
        let mut bits = AnsTokenBits {
            extra_bits: token.extra_bits,
            extra_nbits: token.extra_nbits,
            ..Default::default()
        };

        let dist = &distributions[token.cluster];
        let freq = *dist
            .freqs
            .get(token.symbol as usize)
            .ok_or(Error::InvalidAnsHistogram)? as u32;
        if freq == 0 {
            return Err(Error::InvalidAnsHistogram); // can't encode symbol with 0 frequency
        }

        // Renormalize (matches libjxl ANSCoder::PutSymbol):
        // while (state >> (32 - LOG_SUM_PROBS)) >= freq, emit 16 LSB bits.
        while (state >> (32 - LOG_SUM_PROBS)) >= freq {
            bits.push_refill((state & 0xFFFF) as u16);
            state >>= 16;
        }

        // Inverse mapping: (symbol, offset) -> 12-bit idx.
        let q = state / freq;
        let offset = (state % freq) as usize;
        let idx = *lookups[token.cluster]
            .by_symbol_offset
            .get(token.symbol as usize)
            .and_then(|v| v.get(offset))
            .ok_or(Error::InvalidAnsHistogram)? as u32;

        // Core rANS encode
        state = q * SUM_PROBS + idx;

        token_bits_rev.push(bits);
    }

    // Write initial state, then per-token payloads in forward symbol order.
    w.write(32, state as u64)?;
    for bits in token_bits_rev.iter().rev() {
        for &r in bits.refills() {
            w.write(16, r as u64)?;
        }
        if bits.extra_nbits > 0 {
            w.write(bits.extra_nbits, bits.extra_bits as u64)?;
        }
    }

    Ok(())
}

/// Write the full ANS Histograms header (matching Histograms::decode format).
///
/// Format:
/// - LZ77 disabled (1 bit)
/// - Context map
/// - prefix_code = 0 (ANS mode, 1 bit)
/// - log_alpha_size (2 bits)
/// - HybridUint configs (one per histogram)
/// - ANS distributions (one per histogram)
pub fn write_ans_histograms(
    w: &mut BitWriter,
    context_map: &[u8],
    uint_configs: &[super::HybridUintConfig],
    distributions: &[AnsDistribution],
) -> Result<()> {
    let num_histograms = distributions.len();
    assert_eq!(uint_configs.len(), num_histograms);

    // LZ77: disabled
    w.write(1, 0)?;

    // Context map
    if context_map.len() > 1 {
        super::context_map::write_context_map(w, context_map)?;
    }

    // prefix_code = 0 (ANS)
    w.write(1, 0)?;

    // log_alpha_size = 5 + u(2)
    // Choose based on max alphabet size needed
    let max_alpha = distributions
        .iter()
        .map(|d| d.alphabet_size)
        .max()
        .unwrap_or(1);
    let log_alpha_size = if max_alpha <= 32 {
        5
    } else if max_alpha <= 64 {
        6
    } else if max_alpha <= 128 {
        7
    } else {
        8
    };
    w.write(2, (log_alpha_size - 5) as u64)?;

    // HybridUint configs
    for cfg in uint_configs {
        cfg.write(w, log_alpha_size)?;
    }

    // ANS distributions
    for dist in distributions {
        write_ans_distribution(w, dist, log_alpha_size)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_reader::BitReader;

    #[test]
    fn test_ans_distribution_normalize() {
        // Simple 3-symbol distribution
        let dist = AnsDistribution::from_frequencies(&[100, 200, 300]).unwrap();
        assert_eq!(dist.freqs.iter().map(|&f| f as u32).sum::<u32>(), SUM_PROBS);
        assert!(dist.freqs[0] > 0);
        assert!(dist.freqs[1] > 0);
        assert!(dist.freqs[2] > 0);
        // Relative ordering preserved
        assert!(dist.freqs[0] < dist.freqs[1]);
        assert!(dist.freqs[1] < dist.freqs[2]);
    }

    #[test]
    fn test_ans_distribution_single_symbol() {
        let dist = AnsDistribution::single_symbol(5);
        assert_eq!(dist.freqs[5], SUM_PROBS as u16);
        assert_eq!(dist.cumul[5], 0);
        assert_eq!(dist.cumul[6], SUM_PROBS as u16);
    }

    #[test]
    fn test_ans_u8_roundtrip() {
        for val in 0..=255u8 {
            let mut w = BitWriter::new();
            write_ans_u8(&mut w, val).unwrap();
            w.write(32, 0).unwrap(); // padding
            let bytes = w.finish();
            let mut br = BitReader::new(&bytes);
            // Decode using same logic as AnsHistogram::read_u8
            let decoded = if br.read(1).unwrap() != 0 {
                let n = br.read(3).unwrap();
                ((1 << n) + br.read(n as usize).unwrap()) as u8
            } else {
                0
            };
            assert_eq!(decoded, val, "u8 roundtrip failed for {val}");
        }
    }

    #[test]
    fn test_ans_prefix_roundtrip() {
        // Read the TABLE from the decoder to verify our encoding
        #[rustfmt::skip]
        const TABLE: [(u8, u8); 128] = [
            (10, 3), (12, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (13, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        ];

        for val in 0..=13u16 {
            let mut w = BitWriter::new();
            write_ans_prefix(&mut w, val).unwrap();
            w.write(32, 0).unwrap(); // padding
            let bytes = w.finish();
            let mut br = BitReader::new(&bytes);
            let idx = br.peek(7) as usize;
            let (decoded_val, decoded_bits) = TABLE[idx];
            br.consume(decoded_bits as usize).unwrap();
            assert_eq!(
                decoded_val as u16, val,
                "prefix roundtrip failed for val={val}"
            );
        }
    }

    #[test]
    fn test_ans_single_symbol_roundtrip() {
        // Encode a single-symbol distribution, then decode with the real decoder
        let dist = AnsDistribution::single_symbol(7);
        let mut w = BitWriter::new();
        write_ans_distribution(&mut w, &dist, 5).unwrap();
        w.write(32, 0).unwrap();
        let bytes = w.finish();

        let mut br = BitReader::new(&bytes);
        let decoded = crate::entropy_coding::ans::AnsHistogram::decode(&mut br, 5);
        assert!(
            decoded.is_ok(),
            "single symbol decode failed: {:?}",
            decoded.err()
        );
    }

    #[test]
    fn test_ans_two_symbol_roundtrip() {
        let dist = AnsDistribution::from_frequencies(&[100, 0, 0, 0, 0, 300]).unwrap();
        let mut w = BitWriter::new();
        write_ans_distribution(&mut w, &dist, 5).unwrap();
        w.write(32, 0).unwrap();
        let bytes = w.finish();

        let mut br = BitReader::new(&bytes);
        let decoded = crate::entropy_coding::ans::AnsHistogram::decode(&mut br, 5);
        assert!(
            decoded.is_ok(),
            "two symbol decode failed: {:?}",
            decoded.err()
        );
    }

    #[test]
    fn test_ans_complex_distribution_roundtrip() {
        // Test that the complex distribution encoding roundtrips correctly
        let test_cases: Vec<Vec<u64>> = vec![
            vec![500, 300, 200], // 3 symbols, different freqs
            vec![100, 100, 100], // 3 symbols, equal freqs
            vec![1000, 1, 1, 1], // 4 symbols, skewed
        ];

        for (tc_idx, raw_freqs) in test_cases.iter().enumerate() {
            let dist = AnsDistribution::from_frequencies(raw_freqs).unwrap();

            let mut w = BitWriter::new();
            write_ans_distribution(&mut w, &dist, 8).unwrap();
            w.write(32, 0).unwrap();
            w.write(32, 0).unwrap();
            let bytes = w.finish();

            let mut br = BitReader::new(&bytes);
            let decoded_hist =
                crate::entropy_coding::ans::AnsHistogram::decode(&mut br, 8).unwrap();

            // Verify total
            let dec_total: u32 = (0..dist.alphabet_size)
                .map(|s| decoded_hist.buckets[s].dist as u32)
                .sum();
            assert_eq!(
                dec_total, SUM_PROBS,
                "TC{tc_idx}: decoded total != {SUM_PROBS}"
            );

            // Verify each symbol matches
            for sym in 0..dist.alphabet_size {
                assert_eq!(
                    dist.freqs[sym], decoded_hist.buckets[sym].dist,
                    "TC{tc_idx} sym[{sym}]: freq mismatch"
                );
            }
        }
    }

    #[test]
    fn test_ans_stream_roundtrip() {
        // 3-symbol distribution
        let dist = AnsDistribution::from_frequencies(&[500, 300, 200]).unwrap();
        // split_exponent=5 (log_alpha_size), msb=0, lsb=0 => identity for symbols < 32.
        let uint_config = super::super::HybridUintConfig::new(5, 0, 0);

        // Write histograms header
        let mut w = BitWriter::new();
        write_ans_histograms(
            &mut w,
            &[0u8], // single context -> histogram 0
            &[uint_config],
            &[dist.clone()],
        )
        .unwrap();

        // Encode some symbols
        let symbols = [0u32, 1, 2, 0, 0, 1, 2, 2, 0, 1];
        let tokens: Vec<AnsToken> = symbols
            .iter()
            .map(|&s| AnsToken {
                symbol: s,
                cluster: 0,
                extra_bits: 0,
                extra_nbits: 0,
            })
            .collect();

        write_ans_stream(&mut w, &[dist], &tokens).unwrap();
        w.write(32, 0).unwrap(); // padding
        let bytes = w.finish();

        // Decode with the real decoder
        let mut br = BitReader::new(&bytes);
        let histograms =
            crate::entropy_coding::decode::Histograms::decode(1, &mut br, false).unwrap();
        let mut reader =
            crate::entropy_coding::decode::SymbolReader::new(&histograms, &mut br, None).unwrap();

        for &expected in &symbols {
            let got = reader.read_unsigned(&histograms, &mut br, 0);
            assert_eq!(got, expected, "ANS stream symbol mismatch");
        }

        reader
            .check_final_state(&histograms, &mut br)
            .expect("ANS final state mismatch");
    }
}
