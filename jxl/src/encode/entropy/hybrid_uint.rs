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

impl HybridUintConfig {
    pub fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        Self {
            split_exponent,
            msb_in_token,
            lsb_in_token,
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
    fn test_hybrid_uint_config_roundtrip_420() {
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
