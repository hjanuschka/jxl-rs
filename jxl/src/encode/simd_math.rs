// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Common SIMD math helpers for encoder-side kernels.
//!
//! These utilities are intentionally small and reusable so encode modules can
//! share one implementation for lane-wise arithmetic and clamped conversion.

use jxl_simd::{F32SimdVec, SimdDescriptor};

pub fn round_clamp_f32_to_u8<D: SimdDescriptor>(d: D, src: &[f32], dst: &mut [u8]) {
    assert!(dst.len() >= src.len());

    let lanes = D::F32Vec::LEN;
    let z = D::F32Vec::splat(d, 0.0);
    let maxv = D::F32Vec::splat(d, 255.0);

    let mut i = 0usize;
    while i + lanes <= src.len() {
        let v = D::F32Vec::load(d, &src[i..]);
        let v = v.max(z).min(maxv);
        v.round_store_u8(&mut dst[i..]);
        i += lanes;
    }

    while i < src.len() {
        dst[i] = src[i].round().clamp(0.0, 255.0) as u8;
        i += 1;
    }
}

pub fn mul_add_f32_inplace<D: SimdDescriptor>(d: D, data: &mut [f32], mul: f32, add: f32) {
    let lanes = D::F32Vec::LEN;
    let vmul = D::F32Vec::splat(d, mul);
    let vadd = D::F32Vec::splat(d, add);

    let mut i = 0usize;
    while i + lanes <= data.len() {
        let v = D::F32Vec::load(d, &data[i..]);
        let out = v.mul_add(vmul, vadd);
        out.store(&mut data[i..]);
        i += lanes;
    }

    while i < data.len() {
        data[i] = data[i] * mul + add;
        i += 1;
    }
}

pub fn deinterleave3_f32<D: SimdDescriptor>(
    d: D,
    src_interleaved: &[f32],
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
) {
    let n = out0.len();
    assert_eq!(out1.len(), n);
    assert_eq!(out2.len(), n);
    assert!(src_interleaved.len() >= n * 3);

    let lanes = D::F32Vec::LEN;
    let mut i = 0usize;

    while i + lanes <= n {
        let src_off = i * 3;
        let (a, b, c) = D::F32Vec::load_deinterleaved_3(d, &src_interleaved[src_off..]);
        a.store(&mut out0[i..]);
        b.store(&mut out1[i..]);
        c.store(&mut out2[i..]);
        i += lanes;
    }

    while i < n {
        let s = i * 3;
        out0[i] = src_interleaved[s];
        out1[i] = src_interleaved[s + 1];
        out2[i] = src_interleaved[s + 2];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_simd::ScalarDescriptor;

    #[test]
    fn test_round_clamp_f32_to_u8_scalar() {
        let src = [-10.4f32, -0.49, 0.49, 0.5, 12.2, 254.6, 255.4, 999.0];
        let mut out = [0u8; 8];
        round_clamp_f32_to_u8(ScalarDescriptor, &src, &mut out);
        assert_eq!(out, [0, 0, 0, 1, 12, 255, 255, 255]);
    }

    #[test]
    fn test_mul_add_f32_inplace_scalar() {
        let mut v = [1.0f32, -2.0, 3.5, 8.0, -7.0];
        mul_add_f32_inplace(ScalarDescriptor, &mut v, 2.0, -1.0);
        assert_eq!(v, [1.0, -5.0, 6.0, 15.0, -15.0]);
    }

    #[test]
    fn test_deinterleave3_f32_scalar() {
        let src = [
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        let mut c = [0.0f32; 3];
        deinterleave3_f32(ScalarDescriptor, &src, &mut a, &mut b, &mut c);
        assert_eq!(a, [1.0, 4.0, 7.0]);
        assert_eq!(b, [2.0, 5.0, 8.0]);
        assert_eq!(c, [3.0, 6.0, 9.0]);
    }
}
