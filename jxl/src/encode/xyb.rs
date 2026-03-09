// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Forward XYB color transform for the encoder.
//!
//! Converts sRGB u8 pixels to the XYB color space used by VarDCT encoding.
//! This is the exact inverse of the decoder's `XybStage`.

use crate::util::{Matrix3x3, inv_3x3_matrix};

/// Default opsin inverse matrix (from `OpsinInverseMatrix` defaults in transform_data.rs).
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

/// Default opsin biases (same for all 3 channels).
#[allow(clippy::excessive_precision)]
const DEFAULT_OPSIN_BIAS: f32 = -0.0037930732552754493;

/// Default intensity target (nits).
const DEFAULT_INTENSITY_TARGET: f32 = 255.0;

/// Compute the forward opsin absorbance matrix (inverse of the inverse matrix).
fn compute_forward_matrix() -> Matrix3x3<f64> {
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
    // The forward matrix is the inverse of the inverse matrix.
    inv_3x3_matrix(&inv).expect("default opsin inverse matrix is invertible")
}

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
) {
    assert_eq!(rgb.len(), width * height * 3);
    let npixels = width * height;
    assert_eq!(out_x.len(), npixels);
    assert_eq!(out_y.len(), npixels);
    assert_eq!(out_b.len(), npixels);

    let forward_mat = compute_forward_matrix();
    let bias = DEFAULT_OPSIN_BIAS;
    let bias_cbrt = bias.cbrt();
    let intensity_target = DEFAULT_INTENSITY_TARGET;
    // intensity_scale in decoder = 255.0 / intensity_target
    // For default intensity_target = 255.0, intensity_scale = 1.0
    let intensity_scale = 255.0_f32 / intensity_target;

    for i in 0..npixels {
        // Step 1: sRGB u8 -> sRGB float [0,1] -> linear sRGB [0,1]
        let r_lin = srgb_to_linear(rgb[i * 3] as f32 / 255.0);
        let g_lin = srgb_to_linear(rgb[i * 3 + 1] as f32 / 255.0);
        let b_lin = srgb_to_linear(rgb[i * 3 + 2] as f32 / 255.0);

        // Step 2: Apply forward opsin matrix (linear sRGB -> linear LMS)
        let l_lin =
            forward_mat[0][0] as f32 * r_lin + forward_mat[0][1] as f32 * g_lin + forward_mat[0][2] as f32 * b_lin;
        let m_lin =
            forward_mat[1][0] as f32 * r_lin + forward_mat[1][1] as f32 * g_lin + forward_mat[1][2] as f32 * b_lin;
        let s_lin =
            forward_mat[2][0] as f32 * r_lin + forward_mat[2][1] as f32 * g_lin + forward_mat[2][2] as f32 * b_lin;

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

    #[test]
    fn test_forward_matrix_invertible() {
        let fwd = compute_forward_matrix();
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

            srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b);

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

            srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b);

            let mut out = [0u8; 3];
            xyb_to_srgb_u8(&x, &y, &b, 1, 1, &mut out);

            assert_eq!(
                out, rgb,
                "Roundtrip failed for gray {v}: got {:?}",
                out
            );
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

        srgb_u8_to_xyb(&rgb, npixels, 1, &mut x, &mut y, &mut b);

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

        srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b);

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

        srgb_u8_to_xyb(&rgb, 1, 1, &mut x, &mut y, &mut b);

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
        srgb_u8_to_xyb(&rgb_red, 1, 1, &mut x, &mut y, &mut b);

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
}
