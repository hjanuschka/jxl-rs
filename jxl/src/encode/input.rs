// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::{Error, Result};

fn validate_packed_len(data_len: usize, expected: usize) -> Result<()> {
    if data_len != expected {
        return Err(Error::InvalidPixelBufferLength {
            expected,
            actual: data_len,
        });
    }
    Ok(())
}

fn pack_strided_bytes(
    data: &[u8],
    height: usize,
    stride: usize,
    row_bytes: usize,
) -> Result<Vec<u8>> {
    if stride < row_bytes {
        return Err(Error::InvalidPixelRowStride {
            minimum: row_bytes,
            actual: stride,
        });
    }

    let required = if height == 0 {
        0
    } else {
        (height - 1)
            .checked_mul(stride)
            .and_then(|x| x.checked_add(row_bytes))
            .ok_or(Error::ArithmeticOverflow)?
    };
    if data.len() < required {
        return Err(Error::InvalidPixelBufferLength {
            expected: required,
            actual: data.len(),
        });
    }

    let mut packed = Vec::with_capacity(
        row_bytes
            .checked_mul(height)
            .ok_or(Error::ArithmeticOverflow)?,
    );
    for y in 0..height {
        let row_start = y * stride;
        packed.extend_from_slice(&data[row_start..row_start + row_bytes]);
    }
    Ok(packed)
}

pub fn expand_gray8_to_rgb8(gray: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let px_count = width.checked_mul(height).ok_or(Error::ArithmeticOverflow)?;
    validate_packed_len(gray.len(), px_count)?;

    let mut rgb = Vec::with_capacity(px_count.checked_mul(3).ok_or(Error::ArithmeticOverflow)?);
    for &y in gray {
        rgb.extend_from_slice(&[y, y, y]);
    }
    Ok(rgb)
}

pub fn validate_rgb8_interleaved_len(rgb: &[u8], width: usize, height: usize) -> Result<()> {
    let px_count = width.checked_mul(height).ok_or(Error::ArithmeticOverflow)?;
    let expected = px_count.checked_mul(3).ok_or(Error::ArithmeticOverflow)?;
    validate_packed_len(rgb.len(), expected)
}

pub fn pack_rgb8_strided(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<u8>> {
    let row_bytes = width.checked_mul(3).ok_or(Error::ArithmeticOverflow)?;
    pack_strided_bytes(data, height, stride, row_bytes)
}

pub fn pack_gray8_strided(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<u8>> {
    let row_bytes = width;
    pack_strided_bytes(data, height, stride, row_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_rgb8_strided() {
        let width = 4usize;
        let height = 2usize;
        let row_bytes = width * 3;
        let stride = row_bytes + 2;

        let mut src = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..row_bytes {
                src[y * stride + x] = (x + y * 17) as u8;
            }
        }

        let packed = pack_rgb8_strided(&src, width, height, stride).unwrap();
        assert_eq!(packed.len(), row_bytes * height);
        for y in 0..height {
            assert_eq!(
                &packed[y * row_bytes..(y + 1) * row_bytes],
                &src[y * stride..y * stride + row_bytes]
            );
        }
    }

    #[test]
    fn test_pack_gray8_strided() {
        let width = 4usize;
        let height = 2usize;
        let stride = width + 1;

        let mut src = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                src[y * stride + x] = (x + y * 7) as u8;
            }
        }

        let packed = pack_gray8_strided(&src, width, height, stride).unwrap();
        assert_eq!(packed.len(), width * height);
        for y in 0..height {
            assert_eq!(
                &packed[y * width..(y + 1) * width],
                &src[y * stride..y * stride + width]
            );
        }
    }

    #[test]
    fn test_expand_gray8_to_rgb8() {
        let gray = vec![1u8, 2, 3, 4];
        let rgb = expand_gray8_to_rgb8(&gray, 2, 2).unwrap();
        assert_eq!(rgb, vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
    }
}
