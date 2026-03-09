// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    headers::encodings::{U32, U32Coder},
};

use super::BitWriter;

fn can_encode_with_u32(coder: &U32, value: u32) -> bool {
    match *coder {
        U32::Bits(n) => {
            if n > 32 {
                return false;
            }
            if n == 32 {
                return true;
            }
            value < (1u32 << n)
        }
        U32::BitsOffset { n, off } => {
            if n > 32 {
                return false;
            }
            if value < off {
                return false;
            }
            let adjusted = value - off;
            if n == 32 {
                true
            } else {
                adjusted < (1u32 << n)
            }
        }
        U32::Val(v) => value == v,
    }
}

fn write_u32_direct(writer: &mut BitWriter, coder: &U32, value: u32) -> Result<()> {
    if !can_encode_with_u32(coder, value) {
        return Err(Error::U32EncodeOutOfRange { value });
    }

    match *coder {
        U32::Bits(n) => writer.write(n, value as u64),
        U32::BitsOffset { n, off } => writer.write(n, (value - off) as u64),
        U32::Val(_) => Ok(()),
    }
}

/// Encodes a `u32` value using a JPEG XL `U32Coder` descriptor.
pub fn write_u32(writer: &mut BitWriter, coder: &U32Coder, value: u32) -> Result<()> {
    match coder {
        U32Coder::Direct(c) => write_u32_direct(writer, c, value),
        U32Coder::Select(u0, u1, u2, u3) => {
            let coders = [u0, u1, u2, u3];
            for (selector, candidate) in coders.into_iter().enumerate() {
                if can_encode_with_u32(candidate, value) {
                    writer.write(2, selector as u64)?;
                    return write_u32_direct(writer, candidate, value);
                }
            }
            Err(Error::U32EncodeOutOfRange { value })
        }
    }
}

/// Packs a signed integer using JPEG XL signed packing format.
pub fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        let abs_minus_one = (-1i64 - i64::from(value)) as u32;
        (abs_minus_one << 1) | 1
    }
}

/// Encodes an `i32` value using a JPEG XL `U32Coder` and signed packing.
pub fn write_i32(writer: &mut BitWriter, coder: &U32Coder, value: i32) -> Result<()> {
    write_u32(writer, coder, pack_signed(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        entropy_coding::decode::unpack_signed,
        headers::encodings::{Empty, UnconditionalCoder},
    };

    fn decode_u32(coder: &U32Coder, bytes: &[u8]) -> u32 {
        let mut br = BitReader::new(bytes);
        let decoded = u32::read_unconditional(coder, &mut br, &Empty {}).unwrap();
        br.jump_to_byte_boundary().unwrap();
        decoded
    }

    #[test]
    fn test_write_u32_direct_roundtrip() {
        let coder = U32Coder::Direct(U32::BitsOffset { n: 5, off: 17 });

        for value in [17, 18, 31, 48] {
            let mut writer = BitWriter::new();
            write_u32(&mut writer, &coder, value).unwrap();
            let bytes = writer.finish();
            assert_eq!(decode_u32(&coder, &bytes), value);
        }
    }

    #[test]
    fn test_write_u32_select_picks_first_matching_variant() {
        let coder = U32Coder::Select(U32::Val(5), U32::Bits(8), U32::Bits(16), U32::Val(0));

        let mut writer = BitWriter::new();
        write_u32(&mut writer, &coder, 5).unwrap();
        let total_bits = writer.total_bits_written();
        let bytes = writer.finish();

        // selector only (00), because first variant is exact match U32::Val(5)
        assert_eq!(total_bits, 2);
        assert_eq!(decode_u32(&coder, &bytes), 5);
    }

    #[test]
    fn test_write_u32_out_of_range() {
        let coder = U32Coder::Direct(U32::Bits(3));
        let mut writer = BitWriter::new();
        let err = write_u32(&mut writer, &coder, 8).unwrap_err();
        assert!(matches!(err, Error::U32EncodeOutOfRange { value: 8 }));
    }

    #[test]
    fn test_pack_signed_roundtrip() {
        let samples = [i32::MIN, -1000, -1, 0, 1, 12345, i32::MAX];
        for value in samples {
            let packed = pack_signed(value);
            assert_eq!(unpack_signed(packed), value);
        }
    }

    #[test]
    fn test_write_i32_roundtrip() {
        let coder = U32Coder::Direct(U32::Bits(16));
        let mut writer = BitWriter::new();
        write_i32(&mut writer, &coder, -123).unwrap();
        let bytes = writer.finish();

        let decoded_unsigned = decode_u32(&coder, &bytes);
        assert_eq!(unpack_signed(decoded_unsigned), -123);
    }
}
