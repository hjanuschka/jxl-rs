// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Forward DCT-8 (DCT-II of size 8), the inverse of `idct_8` (DCT-III).
//!
//! The forward DCT is the transpose of the IDCT butterfly, with
//! identical constants but reversed data flow.

use jxl_simd::{F32SimdVec, SimdDescriptor};

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

#[inline]
fn hsum_f32_vec<D: SimdDescriptor>(_d: D, v: D::F32Vec) -> f32 {
    let mut tmp = vec![0.0f32; D::F32Vec::LEN];
    v.store(&mut tmp);
    let mut s = 0.0f32;
    for t in tmp {
        s += t;
    }
    s
}

/// SIMD-assisted forward DCT-8 in-place on 8 values.
///
/// Uses lane-wise multiply-add over chunks while keeping exact scalar fallback
/// for tails and horizontal reduction.
pub fn dct_8_simd<D: SimdDescriptor>(d: D, v: &mut [f32; 8]) {
    let input = *v;

    let lanes = D::F32Vec::LEN;
    let mut dc_sum = 0.0f32;
    let mut n0 = 0usize;
    while n0 < 8 {
        let mut chunk = vec![0.0f32; lanes];
        let chunk_len = (8 - n0).min(lanes);
        chunk[..chunk_len].copy_from_slice(&input[n0..n0 + chunk_len]);
        let vv = D::F32Vec::load(d, &chunk);
        dc_sum += hsum_f32_vec(d, vv);
        n0 += lanes;
    }
    v[0] = dc_sum * 0.125;

    let ac_scale = std::f32::consts::SQRT_2 * 0.125;
    for k in 1..8 {
        let mut sum = 0.0f32;
        let mut n0 = 0usize;
        while n0 < 8 {
            let mut chunk_v = vec![0.0f32; lanes];
            let mut chunk_c = vec![0.0f32; lanes];
            let chunk_len = (8 - n0).min(lanes);
            for lane in 0..chunk_len {
                let n = n0 + lane;
                chunk_v[lane] = input[n];
                let angle = std::f32::consts::PI * ((2 * n + 1) * k) as f32 / 16.0;
                chunk_c[lane] = angle.cos();
            }
            let vv = D::F32Vec::load(d, &chunk_v);
            let cc = D::F32Vec::load(d, &chunk_c);
            sum += hsum_f32_vec(d, vv * cc);
            n0 += lanes;
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

/// SIMD-assisted forward DCT-8x8.
pub fn dct2d_8_simd<D: SimdDescriptor>(d: D, data: &mut [f32; 64]) {
    // Row DCTs
    for row in 0..8 {
        let start = row * 8;
        let mut tmp = [0.0f32; 8];
        tmp.copy_from_slice(&data[start..start + 8]);
        dct_8_simd(d, &mut tmp);
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
        dct_8_simd(d, &mut tmp);
        data[start..start + 8].copy_from_slice(&tmp);
    }
}

#[inline]
fn dct_n_simd<const N: usize, D: SimdDescriptor>(d: D, v: &mut [f32; N]) {
    let input = *v;

    let lanes = D::F32Vec::LEN;
    let mut dc_sum = 0.0f32;
    let mut n0 = 0usize;
    while n0 < N {
        let mut chunk = vec![0.0f32; lanes];
        let chunk_len = (N - n0).min(lanes);
        chunk[..chunk_len].copy_from_slice(&input[n0..n0 + chunk_len]);
        let vv = D::F32Vec::load(d, &chunk);
        dc_sum += hsum_f32_vec(d, vv);
        n0 += lanes;
    }
    v[0] = dc_sum * (1.0 / N as f32);

    let ac_scale = std::f32::consts::SQRT_2 * (1.0 / N as f32);
    for k in 1..N {
        let mut sum = 0.0f32;
        let mut n0 = 0usize;
        while n0 < N {
            let mut chunk_v = vec![0.0f32; lanes];
            let mut chunk_c = vec![0.0f32; lanes];
            let chunk_len = (N - n0).min(lanes);
            for lane in 0..chunk_len {
                let n = n0 + lane;
                chunk_v[lane] = input[n];
                let angle = std::f32::consts::PI * ((2 * n + 1) * k) as f32 / (2.0 * N as f32);
                chunk_c[lane] = angle.cos();
            }
            let vv = D::F32Vec::load(d, &chunk_v);
            let cc = D::F32Vec::load(d, &chunk_c);
            sum += hsum_f32_vec(d, vv * cc);
            n0 += lanes;
        }
        v[k] = sum * ac_scale;
    }
}

pub fn dct2d_16_scalar(data: &mut [f32; 256]) {
    // Row DCTs
    for row in 0..16 {
        let start = row * 16;
        let mut tmp = [0.0f32; 16];
        tmp.copy_from_slice(&data[start..start + 16]);

        let input = tmp;
        let mut out = [0.0f32; 16];
        let mut dc_sum = 0.0f32;
        for n in 0..16 {
            dc_sum += input[n];
        }
        out[0] = dc_sum * (1.0 / 16.0);
        let ac_scale = std::f32::consts::SQRT_2 * (1.0 / 16.0);
        for k in 1..16 {
            let mut sum = 0.0f32;
            for n in 0..16 {
                let angle = std::f32::consts::PI * ((2 * n + 1) * k) as f32 / 32.0;
                sum += input[n] * angle.cos();
            }
            out[k] = sum * ac_scale;
        }

        data[start..start + 16].copy_from_slice(&out);
    }

    // Transpose 16x16
    for i in 0..16 {
        for j in i + 1..16 {
            data.swap(i * 16 + j, j * 16 + i);
        }
    }

    // Column DCTs
    for row in 0..16 {
        let start = row * 16;
        let mut tmp = [0.0f32; 16];
        tmp.copy_from_slice(&data[start..start + 16]);

        let input = tmp;
        let mut out = [0.0f32; 16];
        let mut dc_sum = 0.0f32;
        for n in 0..16 {
            dc_sum += input[n];
        }
        out[0] = dc_sum * (1.0 / 16.0);
        let ac_scale = std::f32::consts::SQRT_2 * (1.0 / 16.0);
        for k in 1..16 {
            let mut sum = 0.0f32;
            for n in 0..16 {
                let angle = std::f32::consts::PI * ((2 * n + 1) * k) as f32 / 32.0;
                sum += input[n] * angle.cos();
            }
            out[k] = sum * ac_scale;
        }

        data[start..start + 16].copy_from_slice(&out);
    }
}

pub fn dct2d_16_simd<D: SimdDescriptor>(d: D, data: &mut [f32; 256]) {
    // Row DCTs
    for row in 0..16 {
        let start = row * 16;
        let mut tmp = [0.0f32; 16];
        tmp.copy_from_slice(&data[start..start + 16]);
        dct_n_simd::<16, D>(d, &mut tmp);
        data[start..start + 16].copy_from_slice(&tmp);
    }

    // Transpose 16x16
    for i in 0..16 {
        for j in i + 1..16 {
            data.swap(i * 16 + j, j * 16 + i);
        }
    }

    // Column DCTs
    for row in 0..16 {
        let start = row * 16;
        let mut tmp = [0.0f32; 16];
        tmp.copy_from_slice(&data[start..start + 16]);
        dct_n_simd::<16, D>(d, &mut tmp);
        data[start..start + 16].copy_from_slice(&tmp);
    }
}

pub fn dct2d_32_scalar(data: &mut [f32; 1024]) {
    // Row DCTs
    for row in 0..32 {
        let start = row * 32;
        let mut tmp = [0.0f32; 32];
        tmp.copy_from_slice(&data[start..start + 32]);

        let input = tmp;
        let mut out = [0.0f32; 32];
        let mut dc_sum = 0.0f32;
        for &v in &input {
            dc_sum += v;
        }
        out[0] = dc_sum * (1.0 / 32.0);
        let ac_scale = std::f32::consts::SQRT_2 * (1.0 / 32.0);
        for (k, out_k) in out.iter_mut().enumerate().skip(1) {
            let mut sum = 0.0f32;
            for (n, &v) in input.iter().enumerate() {
                let angle = std::f32::consts::PI * ((2 * n + 1) * k) as f32 / 64.0;
                sum += v * angle.cos();
            }
            *out_k = sum * ac_scale;
        }

        data[start..start + 32].copy_from_slice(&out);
    }

    // Transpose 32x32
    for i in 0..32 {
        for j in i + 1..32 {
            data.swap(i * 32 + j, j * 32 + i);
        }
    }

    // Column DCTs
    for row in 0..32 {
        let start = row * 32;
        let mut tmp = [0.0f32; 32];
        tmp.copy_from_slice(&data[start..start + 32]);

        let input = tmp;
        let mut out = [0.0f32; 32];
        let mut dc_sum = 0.0f32;
        for &v in &input {
            dc_sum += v;
        }
        out[0] = dc_sum * (1.0 / 32.0);
        let ac_scale = std::f32::consts::SQRT_2 * (1.0 / 32.0);
        for (k, out_k) in out.iter_mut().enumerate().skip(1) {
            let mut sum = 0.0f32;
            for (n, &v) in input.iter().enumerate() {
                let angle = std::f32::consts::PI * ((2 * n + 1) * k) as f32 / 64.0;
                sum += v * angle.cos();
            }
            *out_k = sum * ac_scale;
        }

        data[start..start + 32].copy_from_slice(&out);
    }
}

pub fn dct2d_32_simd<D: SimdDescriptor>(d: D, data: &mut [f32; 1024]) {
    // Row DCTs
    for row in 0..32 {
        let start = row * 32;
        let mut tmp = [0.0f32; 32];
        tmp.copy_from_slice(&data[start..start + 32]);
        dct_n_simd::<32, D>(d, &mut tmp);
        data[start..start + 32].copy_from_slice(&tmp);
    }

    // Transpose 32x32
    for i in 0..32 {
        for j in i + 1..32 {
            data.swap(i * 32 + j, j * 32 + i);
        }
    }

    // Column DCTs
    for row in 0..32 {
        let start = row * 32;
        let mut tmp = [0.0f32; 32];
        tmp.copy_from_slice(&data[start..start + 32]);
        dct_n_simd::<32, D>(d, &mut tmp);
        data[start..start + 32].copy_from_slice(&tmp);
    }
}

fn dct_1d_scalar(input: &[f32], out: &mut [f32]) {
    let n = input.len();
    debug_assert_eq!(n, out.len());

    let mut dc_sum = 0.0f32;
    for &v in input {
        dc_sum += v;
    }
    out[0] = dc_sum * (1.0 / n as f32);

    let ac_scale = std::f32::consts::SQRT_2 * (1.0 / n as f32);
    for (k, out_k) in out.iter_mut().enumerate().skip(1) {
        let mut sum = 0.0f32;
        for (n_idx, &v) in input.iter().enumerate() {
            let angle = std::f32::consts::PI * ((2 * n_idx + 1) * k) as f32 / (2.0 * n as f32);
            sum += v * angle.cos();
        }
        *out_k = sum * ac_scale;
    }
}

fn dct_1d_simd<D: SimdDescriptor>(d: D, input: &[f32], out: &mut [f32]) {
    let n = input.len();
    debug_assert_eq!(n, out.len());
    let lanes = D::F32Vec::LEN;

    let mut dc_sum = 0.0f32;
    let mut n0 = 0usize;
    while n0 < n {
        let mut chunk = vec![0.0f32; lanes];
        let chunk_len = (n - n0).min(lanes);
        chunk[..chunk_len].copy_from_slice(&input[n0..n0 + chunk_len]);
        dc_sum += hsum_f32_vec(d, D::F32Vec::load(d, &chunk));
        n0 += lanes;
    }
    out[0] = dc_sum * (1.0 / n as f32);

    let ac_scale = std::f32::consts::SQRT_2 * (1.0 / n as f32);
    for (k, out_k) in out.iter_mut().enumerate().skip(1) {
        let mut sum = 0.0f32;
        let mut n0 = 0usize;
        while n0 < n {
            let mut chunk_v = vec![0.0f32; lanes];
            let mut chunk_c = vec![0.0f32; lanes];
            let chunk_len = (n - n0).min(lanes);
            for lane in 0..chunk_len {
                let n_idx = n0 + lane;
                chunk_v[lane] = input[n_idx];
                let angle = std::f32::consts::PI * ((2 * n_idx + 1) * k) as f32 / (2.0 * n as f32);
                chunk_c[lane] = angle.cos();
            }
            let vv = D::F32Vec::load(d, &chunk_v);
            let cc = D::F32Vec::load(d, &chunk_c);
            sum += hsum_f32_vec(d, vv * cc);
            n0 += lanes;
        }
        *out_k = sum * ac_scale;
    }
}

fn dct2d_rect_scalar(data: &mut [f32], h: usize, w: usize) {
    debug_assert_eq!(data.len(), h * w);

    // Row transforms.
    let mut row_in = vec![0.0f32; w];
    let mut row_out = vec![0.0f32; w];
    for y in 0..h {
        let off = y * w;
        row_in.copy_from_slice(&data[off..off + w]);
        dct_1d_scalar(&row_in, &mut row_out);
        data[off..off + w].copy_from_slice(&row_out);
    }

    // Column transforms.
    let mut col_in = vec![0.0f32; h];
    let mut col_out = vec![0.0f32; h];
    for x in 0..w {
        for y in 0..h {
            col_in[y] = data[y * w + x];
        }
        dct_1d_scalar(&col_in, &mut col_out);
        for y in 0..h {
            data[y * w + x] = col_out[y];
        }
    }
}

fn dct2d_rect_simd<D: SimdDescriptor>(d: D, data: &mut [f32], h: usize, w: usize) {
    debug_assert_eq!(data.len(), h * w);

    // Row transforms.
    let mut row_in = vec![0.0f32; w];
    let mut row_out = vec![0.0f32; w];
    for y in 0..h {
        let off = y * w;
        row_in.copy_from_slice(&data[off..off + w]);
        dct_1d_simd(d, &row_in, &mut row_out);
        data[off..off + w].copy_from_slice(&row_out);
    }

    // Column transforms.
    let mut col_in = vec![0.0f32; h];
    let mut col_out = vec![0.0f32; h];
    for x in 0..w {
        for y in 0..h {
            col_in[y] = data[y * w + x];
        }
        dct_1d_simd(d, &col_in, &mut col_out);
        for y in 0..h {
            data[y * w + x] = col_out[y];
        }
    }
}

pub fn dct2d_16x8_scalar(data: &mut [f32; 128]) {
    dct2d_rect_scalar(data, 16, 8)
}

pub fn dct2d_8x16_scalar(data: &mut [f32; 128]) {
    dct2d_rect_scalar(data, 8, 16)
}

pub fn dct2d_16x8_simd<D: SimdDescriptor>(d: D, data: &mut [f32; 128]) {
    dct2d_rect_simd(d, data, 16, 8)
}

pub fn dct2d_8x16_simd<D: SimdDescriptor>(d: D, data: &mut [f32; 128]) {
    dct2d_rect_simd(d, data, 8, 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_simd::{test_all_instruction_sets, ScalarDescriptor, SimdDescriptor};

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

    fn dct2d_8_simd_equivalent<D: SimdDescriptor>(d: D) {
        let mut a = [0.0f32; 64];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32 * 5.31).sin() * 77.0) + 2.0;
        }
        let mut b = a;

        dct2d_8_scalar(&mut a);
        dct2d_8_simd(d, &mut b);

        for i in 0..64 {
            assert!(
                (a[i] - b[i]).abs() < 1e-4,
                "coeff {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    test_all_instruction_sets!(dct2d_8_simd_equivalent);

    fn dct2d_16_simd_equivalent<D: SimdDescriptor>(d: D) {
        let mut a = [0.0f32; 256];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32 * 2.17).sin() * 120.0) + 5.0;
        }
        let mut b = a;

        dct2d_16_scalar(&mut a);
        dct2d_16_simd(d, &mut b);

        for i in 0..256 {
            assert!(
                (a[i] - b[i]).abs() < 1e-3,
                "coeff {i}: {} vs {}",
                a[i],
                b[i]
            );
        }

        let mut inv = b;
        crate::idct2d_16_16(d, &mut inv);
        let mut max_err = 0.0f32;
        for i in 0..256 {
            max_err = max_err.max((inv[i] - (((i as f32 * 2.17).sin() * 120.0) + 5.0)).abs());
        }
        assert!(max_err < 0.2, "roundtrip max err too large: {max_err}");
    }

    test_all_instruction_sets!(dct2d_16_simd_equivalent);

    fn dct2d_16x8_simd_equivalent<D: SimdDescriptor>(d: D) {
        let mut a = [0.0f32; 128];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32 * 4.09).sin() * 95.0) + 1.0;
        }
        let mut b = a;

        dct2d_16x8_scalar(&mut a);
        dct2d_16x8_simd(d, &mut b);

        for i in 0..128 {
            assert!(
                (a[i] - b[i]).abs() < 1e-3,
                "coeff {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    fn dct2d_8x16_simd_equivalent<D: SimdDescriptor>(d: D) {
        let mut a = [0.0f32; 128];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32 * 3.37).sin() * 105.0) + 4.0;
        }
        let mut b = a;

        dct2d_8x16_scalar(&mut a);
        dct2d_8x16_simd(d, &mut b);

        for i in 0..128 {
            assert!(
                (a[i] - b[i]).abs() < 1e-3,
                "coeff {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    fn dct2d_32_simd_equivalent<D: SimdDescriptor>(d: D) {
        let mut a = [0.0f32; 1024];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i as f32 * 1.37).sin() * 90.0) + 3.0;
        }
        let mut b = a;

        dct2d_32_scalar(&mut a);
        dct2d_32_simd(d, &mut b);

        for i in 0..1024 {
            assert!(
                (a[i] - b[i]).abs() < 2e-3,
                "coeff {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    fn quantized_equivalence_transform_set<D: SimdDescriptor>(d: D) {
        // 8x8
        let mut s8 = [0.0f32; 64];
        for (i, v) in s8.iter_mut().enumerate() {
            *v = ((i as f32 * 0.91).sin() * 140.0) + 0.5;
        }
        let mut v8 = s8;
        dct2d_8_scalar(&mut s8);
        dct2d_8_simd(d, &mut v8);
        for i in 0..64 {
            assert_eq!(s8[i].round() as i32, v8[i].round() as i32, "8x8 coeff {i}");
        }

        // 16x16
        let mut s16 = [0.0f32; 256];
        for (i, v) in s16.iter_mut().enumerate() {
            *v = ((i as f32 * 1.11).sin() * 100.0) + 1.0;
        }
        let mut v16 = s16;
        dct2d_16_scalar(&mut s16);
        dct2d_16_simd(d, &mut v16);
        for i in 0..256 {
            assert_eq!(
                s16[i].round() as i32,
                v16[i].round() as i32,
                "16x16 coeff {i}"
            );
        }

        // 32x32
        let mut s32 = [0.0f32; 1024];
        for (i, v) in s32.iter_mut().enumerate() {
            *v = ((i as f32 * 1.51).sin() * 70.0) + 2.0;
        }
        let mut v32 = s32;
        dct2d_32_scalar(&mut s32);
        dct2d_32_simd(d, &mut v32);
        for i in 0..1024 {
            assert_eq!(
                s32[i].round() as i32,
                v32[i].round() as i32,
                "32x32 coeff {i}"
            );
        }
    }

    test_all_instruction_sets!(dct2d_16x8_simd_equivalent);
    test_all_instruction_sets!(dct2d_8x16_simd_equivalent);
    test_all_instruction_sets!(dct2d_32_simd_equivalent);
    test_all_instruction_sets!(quantized_equivalence_transform_set);
}
