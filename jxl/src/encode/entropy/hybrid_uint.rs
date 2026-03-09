// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    util::CeilLog2,
};

use super::super::BitWriter;

/// Serializable HybridUint configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridUintConfig {
    pub split_exponent: u32,
    pub msb_in_token: u32,
    pub lsb_in_token: u32,
}

/// Result of encoding a value through HybridUint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridUintEncoded {
    /// The token (symbol sent to the entropy coder).
    pub token: u32,
    /// Extra bits to write after the token.
    pub extra_bits: u32,
    /// Number of extra bits.
    pub nbits: u32,
}

impl HybridUintConfig {
    pub fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        Self {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }

    /// Encode a value into a HybridUint token + extra bits.
    ///
    /// This is the inverse of `HybridUint::read`.
    pub fn encode(&self, value: u32) -> HybridUintEncoded {
        let split_token = 1u32 << self.split_exponent;

        if value < split_token {
            return HybridUintEncoded {
                token: value,
                extra_bits: 0,
                nbits: 0,
            };
        }

        let bits_in_token = self.lsb_in_token + self.msb_in_token;
        let nbits_total = 32 - value.leading_zeros(); // ceil(log2(value+1))
        let nbits = nbits_total.saturating_sub(1); // floor(log2(value))

        // The number of extra bits to read from the bitstream.
        let extra_nbits = nbits.saturating_sub(bits_in_token);

        // Extract the parts.
        let low = value & ((1 << self.lsb_in_token) - 1);
        let value_nolow = value >> self.lsb_in_token;
        let msb = (value_nolow >> extra_nbits) & ((1 << self.msb_in_token) - 1);
        let extra = value_nolow & ((1 << extra_nbits) - 1);

        // Reconstruct the token.
        let token_base =
            split_token + ((extra_nbits - (self.split_exponent - bits_in_token)) << bits_in_token);
        let token = token_base + (msb << self.lsb_in_token) + low;

        HybridUintEncoded {
            token,
            extra_bits: extra,
            nbits: extra_nbits,
        }
    }

    pub fn validate_for_alphabet(&self, log_alpha_size: usize) -> Result<()> {
        if self.split_exponent > log_alpha_size as u32 {
            return Err(Error::InvalidUintConfig(
                self.split_exponent,
                self.msb_in_token,
                Some(self.lsb_in_token),
            ));
        }
        if self.msb_in_token > self.split_exponent {
            return Err(Error::InvalidUintConfig(
                self.split_exponent,
                self.msb_in_token,
                Some(self.lsb_in_token),
            ));
        }
        if self.msb_in_token + self.lsb_in_token > self.split_exponent {
            return Err(Error::InvalidUintConfig(
                self.split_exponent,
                self.msb_in_token,
                Some(self.lsb_in_token),
            ));
        }
        Ok(())
    }

    /// Writes this configuration in the exact bit layout expected by
    /// `entropy_coding::hybrid_uint::HybridUint::decode`.
    pub fn write(self, writer: &mut BitWriter, log_alpha_size: usize) -> Result<()> {
        self.validate_for_alphabet(log_alpha_size)?;

        let split_nbits = (log_alpha_size + 1).ceil_log2();
        writer.write(split_nbits, self.split_exponent as u64)?;

        if self.split_exponent != log_alpha_size as u32 {
            let msb_nbits = (self.split_exponent + 1).ceil_log2() as usize;
            writer.write(msb_nbits, self.msb_in_token as u64)?;

            let lsb_nbits = (self.split_exponent - self.msb_in_token + 1).ceil_log2() as usize;
            writer.write(lsb_nbits, self.lsb_in_token as u64)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bit_reader::BitReader, entropy_coding::hybrid_uint::HybridUint};

    #[test]
    fn test_hybrid_uint_encode_below_split() {
        let cfg = HybridUintConfig::new(4, 2, 0);
        for v in 0..16 {
            let enc = cfg.encode(v);
            assert_eq!(enc.token, v);
            assert_eq!(enc.nbits, 0);
        }
    }

    #[test]
    fn test_hybrid_uint_encode_decode_roundtrip_420() {
        let cfg = HybridUintConfig::new(4, 2, 0);
        for value in [0, 1, 15, 16, 17, 31, 32, 100, 255, 1000, 65535] {
            let enc = cfg.encode(value);

            // Write token + extra bits, then read back.
            let mut writer = BitWriter::new();
            writer
                .write(enc.nbits as usize, u64::from(enc.extra_bits))
                .unwrap();
            writer.write(32, 0).unwrap(); // padding
            let bytes = writer.finish();

            let mut br = BitReader::new(&bytes);
            let decoder = HybridUint::decode(
                5,
                &mut crate::bit_reader::BitReader::new(&{
                    let mut w = BitWriter::new();
                    cfg.write(&mut w, 5).unwrap();
                    w.finish()
                }),
            )
            .unwrap();
            let decoded = decoder.read(enc.token, &mut br);
            assert_eq!(
                decoded, value,
                "mismatch for value={}, token={}, nbits={}, extra={}",
                value, enc.token, enc.nbits, enc.extra_bits
            );
        }
    }

    #[test]
    fn test_hybrid_uint_encode_decode_roundtrip_000() {
        // Config (0,0,0): every value >= 1 needs extra bits.
        let cfg = HybridUintConfig::new(0, 0, 0);
        for value in [0, 1, 2, 3, 10, 127, 255, 511, 1023] {
            let enc = cfg.encode(value);

            let mut writer = BitWriter::new();
            writer
                .write(enc.nbits as usize, u64::from(enc.extra_bits))
                .unwrap();
            writer.write(32, 0).unwrap();
            let bytes = writer.finish();

            let mut br = BitReader::new(&bytes);
            let decoder = HybridUint::new(0, 0, 0);
            let decoded = decoder.read(enc.token, &mut br);
            assert_eq!(decoded, value, "mismatch for value={}", value);
        }
    }

    #[test]
    fn test_hybrid_uint_encode_roundtrip_420() {
        let cfg = HybridUintConfig::new(4, 2, 0);

        let mut writer = BitWriter::new();
        cfg.write(&mut writer, 5).unwrap();
        writer.write(2, 0).unwrap(); // extra bits for symbol=16 read below
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let decoded = HybridUint::decode(5, &mut br).unwrap();

        assert!(decoded.is_config_420());
        assert_eq!(decoded.read(16, &mut br), 16);
    }

    #[test]
    fn test_hybrid_uint_config_roundtrip_split_eq_log_alpha() {
        let cfg = HybridUintConfig::new(6, 0, 0);

        let mut writer = BitWriter::new();
        cfg.write(&mut writer, 6).unwrap();
        let bytes = writer.finish();

        let mut br = BitReader::new(&bytes);
        let decoded = HybridUint::decode(6, &mut br).unwrap();

        // With split_exponent == log_alpha_size, all symbols are direct.
        for s in [0, 1, 2, 7, 10, 63] {
            assert_eq!(decoded.read(s, &mut br), s);
        }
    }

    #[test]
    fn test_hybrid_uint_config_invalid() {
        let mut writer = BitWriter::new();
        let err = HybridUintConfig::new(7, 0, 0)
            .write(&mut writer, 6)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidUintConfig(7, 0, Some(0))));

        let err = HybridUintConfig::new(4, 3, 2)
            .write(&mut writer, 6)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidUintConfig(4, 3, Some(2))));
    }
}
