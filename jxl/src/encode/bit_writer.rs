// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::{BitReader, MAX_BITS_PER_CALL},
    error::{Error, Result},
};

/// Bitstream writer using the same LSB-first bit packing as `BitReader`.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    data: Vec<u8>,
    bit_buf: u64,
    bits_in_buf: usize,
    total_bits_written: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity_bytes: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity_bytes),
            bit_buf: 0,
            bits_in_buf: 0,
            total_bits_written: 0,
        }
    }

    /// Number of data bits explicitly written so far.
    pub fn total_bits_written(&self) -> usize {
        self.total_bits_written
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_buf == 0
    }

    /// Writes up to `MAX_BITS_PER_CALL` bits from `value`.
    ///
    /// Bits above `num_bits` are ignored.
    pub fn write(&mut self, num_bits: usize, value: u64) -> Result<()> {
        if num_bits > MAX_BITS_PER_CALL {
            return Err(Error::InvalidBitCount(num_bits));
        }

        if num_bits == 0 {
            return Ok(());
        }

        self.add_bits_checked(num_bits)?;

        let mask = (1u64 << num_bits) - 1;
        self.bit_buf |= (value & mask) << self.bits_in_buf;
        self.bits_in_buf += num_bits;
        self.flush_full_bytes();

        Ok(())
    }

    /// Pads with zero bits until the next byte boundary.
    ///
    /// Returns the number of zero bits appended.
    pub fn byte_align_zero_pad(&mut self) -> Result<usize> {
        if self.is_byte_aligned() {
            return Ok(0);
        }

        let pad_bits = 8 - self.bits_in_buf;
        self.write(pad_bits, 0)?;
        Ok(pad_bits)
    }

    /// Writes whole bytes. Requires byte-aligned state.
    pub fn write_aligned_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.is_byte_aligned() {
            return Err(Error::BitWriterNotByteAligned);
        }

        self.data.try_reserve(bytes.len())?;
        self.data.extend_from_slice(bytes);

        let bits = bytes.len().checked_mul(8).ok_or(Error::SizeOverflow)?;
        self.add_bits_checked(bits)
    }

    /// Finalizes and returns the encoded bytes.
    ///
    /// If there are partial bits, they are zero-padded to the next byte.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_buf > 0 {
            self.data.push(self.bit_buf as u8);
            self.bit_buf = 0;
            self.bits_in_buf = 0;
        }
        self.data
    }

    fn add_bits_checked(&mut self, bits: usize) -> Result<()> {
        self.total_bits_written = self
            .total_bits_written
            .checked_add(bits)
            .ok_or(Error::SizeOverflow)?;
        Ok(())
    }

    fn flush_full_bytes(&mut self) {
        while self.bits_in_buf >= 8 {
            self.data.push(self.bit_buf as u8);
            self.bit_buf >>= 8;
            self.bits_in_buf -= 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    fn bit_mask(bits: usize) -> u64 {
        if bits == 0 { 0 } else { (1u64 << bits) - 1 }
    }

    #[test]
    fn test_roundtrip_fixed() {
        let mut writer = BitWriter::new();

        let cases = [
            (1, 1),
            (2, 3),
            (7, 0x55),
            (8, 0xAB),
            (9, 0x1AB),
            (16, 0xBEEF),
            (31, 0x7FFFFFFF),
            (56, 0x00DEADBEEFCAFE),
        ];

        for &(n, v) in &cases {
            writer.write(n, v).unwrap();
        }

        let total_bits = writer.total_bits_written();
        let bytes = writer.finish();

        let mut reader = BitReader::new(&bytes);
        for &(n, v) in &cases {
            let got = reader.read(n).unwrap();
            assert_eq!(got, v & bit_mask(n));
        }

        let pad_bits = (8 - (total_bits % 8)) % 8;
        if pad_bits > 0 {
            assert_eq!(reader.read(pad_bits).unwrap(), 0);
        }
        assert!(reader.read(1).is_err());
    }

    #[test]
    fn test_byte_align_zero_pad() {
        let mut writer = BitWriter::new();
        writer.write(3, 0b101).unwrap();
        let pad = writer.byte_align_zero_pad().unwrap();
        assert_eq!(pad, 5);
        assert!(writer.is_byte_aligned());

        writer.write(8, 0xAB).unwrap();
        let out = writer.finish();
        assert_eq!(out, vec![0x05, 0xAB]);
    }

    #[test]
    fn test_write_aligned_bytes_requires_alignment() {
        let mut writer = BitWriter::new();
        writer.write(1, 1).unwrap();

        let err = writer.write_aligned_bytes(&[1, 2, 3]).unwrap_err();
        assert!(matches!(err, Error::BitWriterNotByteAligned));

        writer.byte_align_zero_pad().unwrap();
        writer.write_aligned_bytes(&[0xAA, 0xBB, 0xCC]).unwrap();
        let out = writer.finish();

        assert_eq!(out[1..], [0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_invalid_bit_count() {
        let mut writer = BitWriter::new();
        let err = writer.write(MAX_BITS_PER_CALL + 1, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidBitCount(_)));
    }

    #[test]
    fn test_random_roundtrip() {
        let mut rng = XorShiftRng::seed_from_u64(0xDEC0DE_u64);

        for _ in 0..64 {
            let mut writer = BitWriter::new();
            let mut sequence = Vec::new();

            for _ in 0..2000 {
                let bits = rng.random_range(0..=MAX_BITS_PER_CALL);
                let value = rng.random::<u64>();
                writer.write(bits, value).unwrap();
                sequence.push((bits, value & bit_mask(bits)));
            }

            let total_bits = writer.total_bits_written();
            let bytes = writer.finish();

            let mut reader = BitReader::new(&bytes);
            for (bits, expected) in sequence {
                let got = reader.read(bits).unwrap();
                assert_eq!(got, expected);
            }

            let pad_bits = (8 - (total_bits % 8)) % 8;
            if pad_bits > 0 {
                assert_eq!(reader.read(pad_bits).unwrap(), 0);
            }
            assert_eq!(reader.total_bits_read(), total_bits + pad_bits);
            assert!(reader.read(1).is_err());
        }
    }
}
