// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Forward XYB color transform for the encoder.
//!
//! Converts sRGB u8 pixels to the XYB color space used by VarDCT encoding.
//! This is the exact inverse of the decoder's `XybStage`.

use crate::{
    color::tf,
    encode::simd::EncoderSimdBackend,
    error::{Error, Result},
    util::Matrix3x3,
};
use jxl_simd::{F32SimdVec, SimdDescriptor};

/// Default opsin biases (same for all 3 channels).
#[allow(clippy::excessive_precision)]
const DEFAULT_OPSIN_BIAS: f32 = -0.0037930732552754493;

/// Default intensity target (nits).
const DEFAULT_INTENSITY_TARGET: f32 = 255.0;

/// Default opsin inverse matrix (from `OpsinInverseMatrix` defaults in transform_data.rs).
#[cfg(test)]
#[allow(clippy::excessive_precision)]
const DEFAULT_INVERSE_MATRIX: [f32; 9] = [
    11.031566901960783,
    -9.866943921568629,
    -0.16462299647058826,
    -3.254147380392157,
    4.418770392156863,
    -0.16462299647058826,
    -3.6588512862745097,
    2.7129230470588235,
    1.9459282392156863,
];

/// Forward opsin absorbance matrix (inverse of DEFAULT_INVERSE_MATRIX).
const DEFAULT_FORWARD_MATRIX: Matrix3x3<f64> = [
    [0.2999999989467609, 0.6219999966883528, 0.0779999996754753],
    [0.2299999984251923, 0.6919999951851371, 0.0779999995041812],
    [0.2434226894556144, 0.2047674443555890, 0.5518098669709147],
];

/// Apply the sRGB transfer function in reverse: sRGB -> linear.
///
/// Uses the standard sRGB EOTF: if v <= 0.04045, linear = v/12.92,
/// else linear = ((v + 0.055) / 1.055)^2.4.
#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Pre-computed LUT for sRGB u8 [0..255] -> linear float.
/// Using a LUT ensures scalar and SIMD paths produce identical results,
/// avoiding rational polynomial approximation differences.
fn srgb_u8_to_linear_lut() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0.0f32; 256];
        for i in 0..256 {
            table[i] = srgb_to_linear(i as f32 / 255.0);
        }
        table
    })
}

/// Convert sRGB u8 RGB pixels to XYB float channels.
///
/// Input: `rgb` is interleaved R,G,B bytes (length = width*height*3).
/// Output: `out_x`, `out_y`, `out_b` are separate float channels
/// (each length = width*height), in XYB order.
///
/// The XYB values match what the decoder expects:
/// - Decoder's `XybStage` converts XYB -> linear sRGB
/// - This function converts sRGB u8 -> linear sRGB -> XYB
pub fn srgb_u8_to_xyb(
    rgb: &[u8],
    width: usize,
    height: usize,
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
) -> Result<()> {
    let npixels = width.checked_mul(height).ok_or(Error::ArithmeticOverflow)?;
    let expected_rgb = npixels.checked_mul(3).ok_or(Error::ArithmeticOverflow)?;
    if rgb.len() != expected_rgb {
        return Err(Error::InvalidPixelBufferLength {
            expected: expected_rgb,
            actual: rgb.len(),
        });
    }
    if out_x.len() != npixels || out_y.len() != npixels || out_b.len() != npixels {
        return Err(Error::InvalidPixelBufferLength {
            expected: npixels,
            actual: out_x.len().min(out_y.len()).min(out_b.len()),
        });
    }

    let forward_mat = DEFAULT_FORWARD_MATRIX;
    let bias = DEFAULT_OPSIN_BIAS;
    let bias_cbrt = bias.cbrt();
    let intensity_target = DEFAULT_INTENSITY_TARGET;
    // intensity_scale in decoder = 255.0 / intensity_target
    // For default intensity_target = 255.0, intensity_scale = 1.0
    let intensity_scale = 255.0_f32 / intensity_target;

    let lut = srgb_u8_to_linear_lut();

    for i in 0..npixels {
        // Step 1: sRGB u8 -> linear sRGB via LUT (byte-identical to SIMD path)
        let r_lin = lut[rgb[i * 3] as usize];
        let g_lin = lut[rgb[i * 3 + 1] as usize];
        let b_lin = lut[rgb[i * 3 + 2] as usize];

        // Step 2: Apply forward opsin matrix (linear sRGB -> linear LMS)
        let l_lin = forward_mat[0][0] as f32 * r_lin
            + forward_mat[0][1] as f32 * g_lin
            + forward_mat[0][2] as f32 * b_lin;
        let m_lin = forward_mat[1][0] as f32 * r_lin
            + forward_mat[1][1] as f32 * g_lin
            + forward_mat[1][2] as f32 * b_lin;
        let s_lin = forward_mat[2][0] as f32 * r_lin
            + forward_mat[2][1] as f32 * g_lin
            + forward_mat[2][2] as f32 * b_lin;

        // Step 3: Undo biased gamma
        // Decoder does: l_lin = l^3 * intensity_scale + bias * intensity_scale
        //             = (l^3 + bias) * intensity_scale
        // So: l^3 = l_lin / intensity_scale - bias
        //     l = cbrt(l_lin / intensity_scale - bias)
        let l = (l_lin / intensity_scale - bias).cbrt();
        let m = (m_lin / intensity_scale - bias).cbrt();
        let s = (s_lin / intensity_scale - bias).cbrt();

        // Step 4: Undo mixing
        // Decoder: l = y + x - cbrt(bias), m = y - x - cbrt(bias), s = b - cbrt(bias)
        // So:
        //   y + x = l + cbrt(bias)
        //   y - x = m + cbrt(bias)
        //   b_out = s + cbrt(bias)
        //
        //   y = (l + m) / 2 + cbrt(bias)
        //   x = (l - m) / 2
        let x = (l - m) * 0.5;
        let y = (l + m) * 0.5 + bias_cbrt;
        let b_out = s + bias_cbrt;

        out_x[i] = x;
        out_y[i] = y;
        out_b[i] = b_out;
    }

    Ok(())
}

/// Runtime-dispatched RGB->XYB path for the encoder.
///
/// SIMD is enabled by default through runtime dispatch. Use `JXL_ENC_SIMD=scalar`
/// to force scalar-only mode for deterministic reference output.
pub fn srgb_u8_to_xyb_auto(
    rgb: &[u8],
    width: usize,
    height: usize,
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
) -> Result<()> {
    if !crate::encode::simd::benchmark_force_scalar() {
        let backend = crate::encode::simd::detect_encoder_simd_backend();
        match backend {
            EncoderSimdBackend::Scalar => {}
            #[cfg(all(target_arch = "x86_64", any(feature = "sse42", feature = "all-simd")))]
            EncoderSimdBackend::Sse42 => {
                if let Some(d) = jxl_simd::Sse42Descriptor::new() {
                    return srgb_u8_to_xyb_simd_assisted(
                        d, rgb, width, height, out_x, out_y, out_b,
                    );
                }
            }
            #[cfg(all(target_arch = "x86_64", any(feature = "avx", feature = "all-simd")))]
            EncoderSimdBackend::Avx2 => {
                if let Some(d) = jxl_simd::AvxDescriptor::new() {
                    return srgb_u8_to_xyb_simd_assisted(
                        d, rgb, width, height, out_x, out_y, out_b,
                    );
                }
            }
            #[cfg(all(target_arch = "x86_64", any(feature = "avx512", feature = "all-simd")))]
            EncoderSimdBackend::Avx512 => {
                if let Some(d) = jxl_simd::Avx512Descriptor::new() {
                    return srgb_u8_to_xyb_simd_assisted(
                        d, rgb, width, height, out_x, out_y, out_b,
                    );
                }
            }
            #[cfg(all(target_arch = "aarch64", any(feature = "neon", feature = "all-simd")))]
            EncoderSimdBackend::Neon => {
                if let Some(d) = jxl_simd::NeonDescriptor::new() {
                    return srgb_u8_to_xyb_simd_assisted(
                        d, rgb, width, height, out_x, out_y, out_b,
                    );
                }
            }
        }
    }
    srgb_u8_to_xyb(rgb, width, height, out_x, out_y, out_b)
}

/// SIMD-assisted XYB conversion path using jxl_simd + jxl_transforms primitives.
///
/// This keeps final cbrt/mix scalar for parity with the existing reference path,
/// while accelerating transfer and matrix stages with SIMD lanes.
pub fn srgb_u8_to_xyb_simd_assisted<D: SimdDescriptor>(
    d: D,
    rgb: &[u8],
    width: usize,
    height: usize,
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
) -> Result<()> {
    let npixels = width.checked_mul(height).ok_or(Error::ArithmeticOverflow)?;
    let expected_rgb = npixels.checked_mul(3).ok_or(Error::ArithmeticOverflow)?;
    if rgb.len() != expected_rgb {
        return Err(Error::InvalidPixelBufferLength {
            expected: expected_rgb,
            actual: rgb.len(),
        });
    }
    if out_x.len() != npixels || out_y.len() != npixels || out_b.len() != npixels {
        return Err(Error::InvalidPixelBufferLength {
            expected: npixels,
            actual: out_x.len().min(out_y.len()).min(out_b.len()),
        });
    }

    // Use LUT for sRGB u8 -> linear (byte-identical to scalar path).
    let lut = srgb_u8_to_linear_lut();
    let mut r = vec![0f32; npixels];
    let mut g = vec![0f32; npixels];
    let mut b = vec![0f32; npixels];
    for i in 0..npixels {
        r[i] = lut[rgb[i * 3] as usize];
        g[i] = lut[rgb[i * 3 + 1] as usize];
        b[i] = lut[rgb[i * 3 + 2] as usize];
    }

    let lanes = D::F32Vec::LEN;
    let forward_mat = DEFAULT_FORWARD_MATRIX;
    let mut l_lin = vec![0f32; npixels];
    let mut m_lin = vec![0f32; npixels];
    let mut s_lin = vec![0f32; npixels];

    let r0 = D::F32Vec::splat(d, forward_mat[0][0] as f32);
    let r1 = D::F32Vec::splat(d, forward_mat[0][1] as f32);
    let r2 = D::F32Vec::splat(d, forward_mat[0][2] as f32);
    let g0 = D::F32Vec::splat(d, forward_mat[1][0] as f32);
    let g1 = D::F32Vec::splat(d, forward_mat[1][1] as f32);
    let g2 = D::F32Vec::splat(d, forward_mat[1][2] as f32);
    let b0 = D::F32Vec::splat(d, forward_mat[2][0] as f32);
    let b1 = D::F32Vec::splat(d, forward_mat[2][1] as f32);
    let b2 = D::F32Vec::splat(d, forward_mat[2][2] as f32);

    // S202: SIMD matrix stage.
    let mut i = 0usize;
    while i + lanes <= npixels {
        let rv = D::F32Vec::load(d, &r[i..]);
        let gv = D::F32Vec::load(d, &g[i..]);
        let bv = D::F32Vec::load(d, &b[i..]);

        // Use separate multiply+add (not mul_add/FMA) to match scalar FP order.
        (rv * r0 + gv * r1 + bv * r2).store(&mut l_lin[i..]);
        (rv * g0 + gv * g1 + bv * g2).store(&mut m_lin[i..]);
        (rv * b0 + gv * b1 + bv * b2).store(&mut s_lin[i..]);
        i += lanes;
    }
    while i < npixels {
        l_lin[i] = forward_mat[0][0] as f32 * r[i]
            + forward_mat[0][1] as f32 * g[i]
            + forward_mat[0][2] as f32 * b[i];
        m_lin[i] = forward_mat[1][0] as f32 * r[i]
            + forward_mat[1][1] as f32 * g[i]
            + forward_mat[1][2] as f32 * b[i];
        s_lin[i] = forward_mat[2][0] as f32 * r[i]
            + forward_mat[2][1] as f32 * g[i]
            + forward_mat[2][2] as f32 * b[i];
        i += 1;
    }

    let bias = DEFAULT_OPSIN_BIAS;
    let bias_cbrt = bias.cbrt();
    let intensity_scale = 255.0_f32 / DEFAULT_INTENSITY_TARGET;

    for i in 0..npixels {
        let l = (l_lin[i] / intensity_scale - bias).cbrt();
        let m = (m_lin[i] / intensity_scale - bias).cbrt();
        let s = (s_lin[i] / intensity_scale - bias).cbrt();

        out_x[i] = (l - m) * 0.5;
        out_y[i] = (l + m) * 0.5 + bias_cbrt;
        out_b[i] = s + bias_cbrt;
    }

    Ok(())
}

/// RGBA convenience path for SIMD-assisted XYB preprocessing.
///
/// Splits interleaved RGBA into RGB+alpha and runs SIMD-assisted RGB->XYB.
pub fn srgb_u8_rgba_to_xyb_with_alpha_simd_assisted<D: SimdDescriptor>(
    d: D,
    rgba: &[u8],
    width: usize,
    height: usize,
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_b: &mut [f32],
    out_alpha: &mut [u8],
) -> Result<()> {
    let npixels = width.checked_mul(height).ok_or(Error::ArithmeticOverflow)?;
    let expected_rgba = npixels.checked_mul(4).ok_or(Error::ArithmeticOverflow)?;
    if rgba.len() != expected_rgba {
        return Err(Error::InvalidPixelBufferLength {
            expected: expected_rgba,
            actual: rgba.len(),
        });
    }
    if out_alpha.len() != npixels {
        return Err(Error::InvalidPixelBufferLength {
            expected: npixels,
            actual: out_alpha.len(),
        });
    }

    let mut rgb = vec![0u8; npixels * 3];
    for i in 0..npixels {
        let src = i * 4;
        let dst = i * 3;
        rgb[dst] = rgba[src];
        rgb[dst + 1] = rgba[src + 1];
        rgb[dst + 2] = rgba[src + 2];
        out_alpha[i] = rgba[src + 3];
    }

    srgb_u8_to_xyb_simd_assisted(d, &rgb, width, height, out_x, out_y, out_b)
}

/// Convert XYB float channels back to sRGB u8 (for testing).
///
/// This implements the same transform as the decoder's XybStage + from_linear.
#[cfg(test)]
pub fn xyb_to_srgb_u8(
    x_chan: &[f32],
    y_chan: &[f32],
    b_chan: &[f32],
    width: usize,
    height: usize,
    out_rgb: &mut [u8],
) {
    let npixels = width * height;
    assert_eq!(x_chan.len(), npixels);
    assert_eq!(y_chan.len(), npixels);
    assert_eq!(b_chan.len(), npixels);
    assert_eq!(out_rgb.len(), npixels * 3);

    let bias = DEFAULT_OPSIN_BIAS;
    let bias_cbrt = bias.cbrt();
    let intensity_target = DEFAULT_INTENSITY_TARGET;
    let intensity_scale = 255.0_f32 / intensity_target;

    for i in 0..npixels {
        let x = x_chan[i];
        let y = y_chan[i];
        let b = b_chan[i];

        // Mix (decoder step 1)
        let l = y + x - bias_cbrt;
        let m = y - x - bias_cbrt;
        let s = b - bias_cbrt;

        // Cube + scale (decoder step 2)
        let l_lin = l * l * l * intensity_scale + bias * intensity_scale;
        let m_lin = m * m * m * intensity_scale + bias * intensity_scale;
        let s_lin = s * s * s * intensity_scale + bias * intensity_scale;

        // Inverse matrix (decoder step 3)
        let r_lin = DEFAULT_INVERSE_MATRIX[0] * l_lin
            + DEFAULT_INVERSE_MATRIX[1] * m_lin
            + DEFAULT_INVERSE_MATRIX[2] * s_lin;
        let g_lin = DEFAULT_INVERSE_MATRIX[3] * l_lin
            + DEFAULT_INVERSE_MATRIX[4] * m_lin
            + DEFAULT_INVERSE_MATRIX[5] * s_lin;
        let b_lin = DEFAULT_INVERSE_MATRIX[6] * l_lin
            + DEFAULT_INVERSE_MATRIX[7] * m_lin
            + DEFAULT_INVERSE_MATRIX[8] * s_lin;

        // Linear -> sRGB gamma
        let r_srgb = linear_to_srgb(r_lin);
        let g_srgb = linear_to_srgb(g_lin);
        let b_srgb = linear_to_srgb(b_lin);

        out_rgb[i * 3] = (r_srgb * 255.0).round().clamp(0.0, 255.0) as u8;
        out_rgb[i * 3 + 1] = (g_srgb * 255.0).round().clamp(0.0, 255.0) as u8;
        out_rgb[i * 3 + 2] = (b_srgb * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

/// Apply sRGB OETF: linear -> sRGB.
#[cfg(test)]
#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_simd::ScalarDescriptor;

    #[test]
    fn test_forward_matrix_invertible() {
        let fwd = DEFAULT_FORWARD_MATRIX;
        // Verify forward * inverse = identity
        let inv = [
            [
                DEFAULT_INVERSE_MATRIX[0] as f64,
                DEFAULT_INVERSE_MATRIX[1] as f64,
                DEFAULT_INVERSE_MATRIX[2] as f64,
            ],
            [
                DEFAULT_INVERSE_MATRIX[3] as f64,
                DEFAULT_INVERSE_MATRIX[4] as f64,
                DEFAULT_INVERSE_MATRIX[5] as f64,
            ],
            [
                DEFAULT_INVERSE_MATRIX[6] as f64,
                DEFAULT_INVERSE_MATRIX[7] as f64,
                DEFAULT_INVERSE_MATRIX[8] as f64,
            ],
        ];
        let product = crate::util::mul_3x3_matrix(&inv, &fwd);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-10,
                    "product[{i}][{j}] = {}, expected {expected}",
                    product[i][j]
                );
            }
        }
    }

    #[test]
    fn test_xyb_roundtrip_primary_colors() {
        // Test with pure red, green, blue, white, black
        let colors: Vec<[u8; 3]> = vec![
            [255, 0, 0],     // red
            [0, 255, 0],     // green
            [0, 0, 255],     // blue
            [255, 255, 255], // white
            [0, 0, 0],       // black
            [128, 128, 128], // mid gray
        ];

        for color in &colors {
            let rgb = [color[0], color[1], color[2]];
            let mut x = [0.0f32; 1];
            let mut y = [0.0f32; 1];
            let mut b = [0.0f32; 1];

            srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b).unwrap();

            // Convert back
            let mut out = [0u8; 3];
            xyb_to_srgb_u8(&x, &y, &b, 1, 1, &mut out);

            assert_eq!(
                out, rgb,
                "Roundtrip failed for color {:?}: got {:?}",
                color, out
            );
        }
    }

    #[test]
    fn test_xyb_roundtrip_all_gray() {
        // Test all 256 gray values
        for v in 0..=255u8 {
            let rgb = [v, v, v];
            let mut x = [0.0f32; 1];
            let mut y = [0.0f32; 1];
            let mut b = [0.0f32; 1];

            srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b).unwrap();

            let mut out = [0u8; 3];
            xyb_to_srgb_u8(&x, &y, &b, 1, 1, &mut out);

            assert_eq!(out, rgb, "Roundtrip failed for gray {v}: got {:?}", out);
        }
    }

    #[test]
    fn test_xyb_roundtrip_random_pixels() {
        // Larger test with pseudo-random pixels
        let npixels = 256;
        let mut rgb = vec![0u8; npixels * 3];
        for i in 0..npixels * 3 {
            rgb[i] = ((i as f32 * 17.3 + 7.1).sin() * 127.5 + 127.5) as u8;
        }

        let mut x = vec![0.0f32; npixels];
        let mut y = vec![0.0f32; npixels];
        let mut b = vec![0.0f32; npixels];

        srgb_u8_to_xyb(&rgb, npixels, 1, &mut x, &mut y, &mut b).unwrap();

        let mut out = vec![0u8; npixels * 3];
        xyb_to_srgb_u8(&x, &y, &b, npixels, 1, &mut out);

        // Allow +/- 1 due to rounding through float
        let mut max_diff = 0i32;
        for i in 0..npixels * 3 {
            let diff = (out[i] as i32 - rgb[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff <= 1,
            "max pixel difference = {max_diff} (expected <= 1)"
        );
    }

    #[test]
    fn test_xyb_black_is_near_zero() {
        // Black should produce near-zero XYB values
        let rgb = [0u8, 0, 0];
        let mut x = [0.0f32; 1];
        let mut y = [0.0f32; 1];
        let mut b = [0.0f32; 1];

        srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b).unwrap();

        // X should be very close to 0 (symmetric for achromatic colors)
        assert!(x[0].abs() < 0.001, "x for black = {}", x[0]);
        // Y and B should be very small (not exactly zero due to bias)
        assert!(y[0].abs() < 0.2, "y for black = {}", y[0]);
        assert!(b[0].abs() < 0.2, "b for black = {}", b[0]);
    }

    #[test]
    fn test_xyb_white_reasonable_range() {
        let rgb = [255u8, 255, 255];
        let mut x = [0.0f32; 1];
        let mut y = [0.0f32; 1];
        let mut b = [0.0f32; 1];

        srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b).unwrap();

        // X should be near 0 for achromatic
        assert!(x[0].abs() < 0.01, "x for white = {}", x[0]);
        // Y should be positive and substantial
        assert!(y[0] > 0.3, "y for white = {}", y[0]);
        // B should be positive
        assert!(b[0] > 0.3, "b for white = {}", b[0]);
    }

    #[test]
    fn test_xyb_matches_decoder_test_vectors() {
        // The decoder test in xyb.rs uses these known XYB -> linear RGB values:
        // XYB (0.028100073, 0.4881882, 0.471659) -> linear (1.0, 0.0, 0.0) [red]
        // XYB (-0.015386105, 0.71478134, 0.43707693) -> linear (0.0, 1.0, 0.0) [green]
        // XYB (0.0, 0.2781282, 0.66613984) -> linear (0.0, 0.0, 1.0) [blue]
        //
        // Our forward transform should produce these XYB values from the same
        // linear RGB values. But we start from sRGB u8, so we test with
        // the linearized sRGB primary values.

        // Pure red: linear (1,0,0) -> sRGB u8 (255,0,0)
        let rgb_red = [255u8, 0, 0];
        let mut x = [0.0f32; 1];
        let mut y = [0.0f32; 1];
        let mut b = [0.0f32; 1];
        srgb_u8_to_xyb(&rgb_red, 1, 1, &mut x, &mut y, &mut b).unwrap();

        // Check against decoder's expected XYB values
        assert!(
            (x[0] - 0.028100073).abs() < 0.001,
            "red x: expected ~0.028, got {}",
            x[0]
        );
        assert!(
            (y[0] - 0.4881882).abs() < 0.001,
            "red y: expected ~0.488, got {}",
            y[0]
        );
        assert!(
            (b[0] - 0.471659).abs() < 0.001,
            "red b: expected ~0.472, got {}",
            b[0]
        );
    }

    #[test]
    fn test_xyb_simd_assisted_close_to_scalar_reference() {
        let (w, h) = (37usize, 19usize);
        let npixels = w * h;
        let mut rgb = vec![0u8; npixels * 3];
        for i in 0..rgb.len() {
            rgb[i] = ((i as f32 * 11.17 + 5.3).sin() * 127.0 + 127.0) as u8;
        }

        let mut sx = vec![0.0f32; npixels];
        let mut sy = vec![0.0f32; npixels];
        let mut sb = vec![0.0f32; npixels];
        srgb_u8_to_xyb(&rgb, w, h, &mut sx, &mut sy, &mut sb).unwrap();

        let mut vx = vec![0.0f32; npixels];
        let mut vy = vec![0.0f32; npixels];
        let mut vb = vec![0.0f32; npixels];
        srgb_u8_to_xyb_simd_assisted(ScalarDescriptor, &rgb, w, h, &mut vx, &mut vy, &mut vb)
            .unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..npixels {
            max_diff = max_diff.max((sx[i] - vx[i]).abs());
            max_diff = max_diff.max((sy[i] - vy[i]).abs());
            max_diff = max_diff.max((sb[i] - vb[i]).abs());
        }
        assert!(
            max_diff < 1e-3,
            "max SIMD-assisted delta too large: {max_diff}"
        );
    }

    #[test]
    fn test_xyb_rgba_simd_assisted_preserves_alpha() {
        let (w, h) = (9usize, 7usize);
        let npixels = w * h;
        let mut rgba = vec![0u8; npixels * 4];
        for i in 0..npixels {
            let o = i * 4;
            rgba[o] = (i * 3 % 256) as u8;
            rgba[o + 1] = (i * 7 % 256) as u8;
            rgba[o + 2] = (i * 11 % 256) as u8;
            rgba[o + 3] = (i * 13 % 256) as u8;
        }

        let mut x = vec![0.0f32; npixels];
        let mut y = vec![0.0f32; npixels];
        let mut b = vec![0.0f32; npixels];
        let mut alpha = vec![0u8; npixels];
        srgb_u8_rgba_to_xyb_with_alpha_simd_assisted(
            ScalarDescriptor,
            &rgba,
            w,
            h,
            &mut x,
            &mut y,
            &mut b,
            &mut alpha,
        )
        .unwrap();

        for i in 0..npixels {
            assert_eq!(alpha[i], rgba[i * 4 + 3]);
        }
    }

    #[test]
    fn test_xyb_simd_edge_values() {
        let edges = [0u8, 1u8, 254u8, 255u8];
        for &r in &edges {
            for &g in &edges {
                for &bb in &edges {
                    let rgb = [r, g, bb];
                    let mut sx = [0.0f32; 1];
                    let mut sy = [0.0f32; 1];
                    let mut sb = [0.0f32; 1];
                    srgb_u8_to_xyb(&rgb, 1, 1, &mut sx, &mut sy, &mut sb).unwrap();

                    let mut vx = [0.0f32; 1];
                    let mut vy = [0.0f32; 1];
                    let mut vb = [0.0f32; 1];
                    srgb_u8_to_xyb_simd_assisted(
                        ScalarDescriptor,
                        &rgb,
                        1,
                        1,
                        &mut vx,
                        &mut vy,
                        &mut vb,
                    )
                    .unwrap();

                    assert!((sx[0] - vx[0]).abs() < 1e-5);
                    assert!((sy[0] - vy[0]).abs() < 1e-5);
                    assert!((sb[0] - vb[0]).abs() < 1e-5);
                }
            }
        }
    }
}
