// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Forward DCT-8 (DCT-II of size 8), the inverse of `idct_8` (DCT-III).
//!
//! The forward DCT is the transpose of the IDCT butterfly, with
//! identical constants but reversed data flow.

/// Scalar forward DCT-8 in-place on 8 values.
///
/// This is the exact inverse of `idct_8`: `dct8(idct8(x)) == x`.
///
/// The jxl IDCT basis is:
///   B[0][n] = 1                               (DC)
///   B[k][n] = sqrt(2) * cos(pi*(2n+1)*k/16)   (AC, k>0)
///
/// Since B * B^T = 8 * I, the forward DCT is c = (1/8) * B^T * x:
///   c[0] = (1/8) * sum_n x[n]
///   c[k] = sqrt(2)/8 * sum_n x[n] * cos(pi*(2n+1)*k/16)
#[inline]
pub(super) fn dct_8_scalar(v: &mut [f32; 8]) {
    let input = *v;
    // DC coefficient
    let mut dc_sum = 0.0f32;
    for n in 0..8 {
        dc_sum += input[n];
    }
    v[0] = dc_sum * 0.125; // 1/8

    // AC coefficients
    let ac_scale = std::f32::consts::SQRT_2 * 0.125; // sqrt(2)/8
    for k in 1..8 {
        let mut sum = 0.0f32;
        for n in 0..8 {
            let angle = std::f32::consts::PI * ((2 * n + 1) * k) as f32 / 16.0;
            sum += input[n] * angle.cos();
        }
        v[k] = sum * ac_scale;
    }
}

/// Forward DCT-8x8 (2D) on a row-major 64-element buffer.
///
/// Applies 1D forward DCT to each row, transposes, then applies to each column.
/// This is the exact inverse of `idct2d_8_8`: `dct2d_8(idct2d_8(x)) == x`.
pub fn dct2d_8_scalar(data: &mut [f32; 64]) {
    // Row DCTs
    for row in 0..8 {
        let start = row * 8;
        let mut tmp = [0.0f32; 8];
        tmp.copy_from_slice(&data[start..start + 8]);
        dct_8_scalar(&mut tmp);
        data[start..start + 8].copy_from_slice(&tmp);
    }

    // Transpose
    for i in 0..8 {
        for j in i + 1..8 {
            data.swap(i * 8 + j, j * 8 + i);
        }
    }

    // Column DCTs (now rows after transpose)
    for row in 0..8 {
        let start = row * 8;
        let mut tmp = [0.0f32; 8];
        tmp.copy_from_slice(&data[start..start + 8]);
        dct_8_scalar(&mut tmp);
        data[start..start + 8].copy_from_slice(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_simd::scalar::ScalarDescriptor;

    #[test]
    fn test_dct2d_8_roundtrip_via_jxl_idct() {
        // Start with known DCT coefficients.
        let mut original = [0.0f32; 64];
        for i in 0..64 {
            original[i] = ((i as f32 * 7.3 + 1.1).sin() * 100.0).round();
        }

        // IDCT using jxl's actual implementation -> spatial domain
        let mut spatial = original;
        crate::idct2d_8_8(ScalarDescriptor, &mut spatial);

        // Forward DCT should recover original coefficients
        let mut recovered = [0.0f32; 64];
        recovered.copy_from_slice(&spatial);
        dct2d_8_scalar(&mut recovered);

        // Print first few for debugging
        for i in 0..8 {
            eprintln!(
                "coeff[{i}]: original={:.4}, recovered={:.4}, diff={:.4}",
                original[i],
                recovered[i],
                recovered[i] - original[i]
            );
        }

        let max_err: f32 = original
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_err < 0.01,
            "max roundtrip error = {max_err} (expected < 0.01)"
        );
    }

    #[test]
    fn test_dct2d_8_roundtrip_reverse() {
        // Start with spatial data, forward DCT, then IDCT.
        let mut spatial = [0.0f32; 64];
        for i in 0..64 {
            spatial[i] = (i as f32 * 3.7).sin() * 50.0 + 128.0;
        }
        let original = spatial;

        // Forward DCT
        dct2d_8_scalar(&mut spatial);

        // IDCT using jxl's actual implementation
        crate::idct2d_8_8(ScalarDescriptor, &mut spatial);

        let max_err: f32 = original
            .iter()
            .zip(spatial.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_err < 0.01,
            "max roundtrip error = {max_err} (expected < 0.01)"
        );
    }
}
