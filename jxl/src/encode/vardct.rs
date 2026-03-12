// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! VarDCT lossy encoding pipeline.
//!
//! Converts sRGB u8 input to a VarDCT-encoded JXL codestream.
//! Currently supports DCT8x8-only, single-pass encoding.

use crate::encode::bit_writer::BitWriter;
use crate::encode::container::wrap_codestream;
use crate::encode::encodings::write_u32;
use crate::encode::entropy::context_map::write_context_map;
use crate::encode::entropy::huffman_encode::build_huffman_code;
use crate::encode::headers::write_file_header;
use crate::encode::toc::write_toc;
use crate::encode::xyb::srgb_u8_to_xyb;
use crate::error::Result;
use crate::frame::block_context_map::{self, NON_ZERO_BUCKETS, ZERO_DENSITY_CONTEXT_COUNT};
use crate::headers::encodings::{U32, U32Coder};
use jxl_transforms::{
    dct2d_8_scalar,
    transform::transform_to_pixels,
    transform_map::{HfTransformType, block_shape_id, covered_blocks_x, covered_blocks_y},
};

// If true, evaluate both ANS and Huffman AC entropy paths and choose the smaller payload.
// If false, restrict to Huffman only.
const USE_ANS_AC_ENTROPY: bool = true;
const TRANSFORM_FIRST_BLOCK_FLAG: u8 = 0x80;
const DCT8_TRANSFORM_ID: u8 = 0;
const IDENTITY_TRANSFORM_ID: u8 = 1;
const DCT2X2_TRANSFORM_ID: u8 = 2;
const DCT4X4_TRANSFORM_ID: u8 = 3;
const DCT16_TRANSFORM_ID: u8 = 4;
const DCT32_TRANSFORM_ID: u8 = 5;
const DCT16X8_TRANSFORM_ID: u8 = 6;
const DCT8X16_TRANSFORM_ID: u8 = 7;
const DCT32X8_TRANSFORM_ID: u8 = 8;
const DCT8X32_TRANSFORM_ID: u8 = 9;
const DCT32X16_TRANSFORM_ID: u8 = 10;
const DCT16X32_TRANSFORM_ID: u8 = 11;
const DCT4X8_TRANSFORM_ID: u8 = 12;
const DCT8X4_TRANSFORM_ID: u8 = 13;
const AFV0_TRANSFORM_ID: u8 = 14;
const AFV1_TRANSFORM_ID: u8 = 15;
const AFV2_TRANSFORM_ID: u8 = 16;
const AFV3_TRANSFORM_ID: u8 = 17;
const DCT64_TRANSFORM_ID: u8 = 18;
const DCT64X32_TRANSFORM_ID: u8 = 19;
const DCT32X64_TRANSFORM_ID: u8 = 20;
const DCT128_TRANSFORM_ID: u8 = 21;
const DCT128X64_TRANSFORM_ID: u8 = 22;
const DCT64X128_TRANSFORM_ID: u8 = 23;
const DCT256_TRANSFORM_ID: u8 = 24;
const DCT256X128_TRANSFORM_ID: u8 = 25;
const DCT128X256_TRANSFORM_ID: u8 = 26;

fn is_supported_nonzero_transform_id(transform_id: u8) -> bool {
    matches!(
        transform_id,
        IDENTITY_TRANSFORM_ID
            | DCT2X2_TRANSFORM_ID
            | DCT4X4_TRANSFORM_ID
            | DCT16_TRANSFORM_ID
            | DCT32_TRANSFORM_ID
            | DCT16X8_TRANSFORM_ID
            | DCT8X16_TRANSFORM_ID
            | DCT32X8_TRANSFORM_ID
            | DCT8X32_TRANSFORM_ID
            | DCT32X16_TRANSFORM_ID
            | DCT16X32_TRANSFORM_ID
            | DCT4X8_TRANSFORM_ID
            | DCT8X4_TRANSFORM_ID
            | AFV0_TRANSFORM_ID
            | AFV1_TRANSFORM_ID
            | AFV2_TRANSFORM_ID
            | AFV3_TRANSFORM_ID
            | DCT64_TRANSFORM_ID
            | DCT64X32_TRANSFORM_ID
            | DCT32X64_TRANSFORM_ID
            | DCT128_TRANSFORM_ID
            | DCT128X64_TRANSFORM_ID
            | DCT64X128_TRANSFORM_ID
            | DCT256_TRANSFORM_ID
            | DCT256X128_TRANSFORM_ID
            | DCT128X256_TRANSFORM_ID
    )
}

fn canonical_transform_for_shape_id(shape_id: usize) -> Option<HfTransformType> {
    Some(match shape_id {
        0 => HfTransformType::DCT,
        1 => HfTransformType::AFV0,
        2 => HfTransformType::DCT16X16,
        3 => HfTransformType::DCT32X32,
        4 => HfTransformType::DCT8X16,
        5 => HfTransformType::DCT8X32,
        6 => HfTransformType::DCT16X32,
        7 => HfTransformType::DCT64X64,
        8 => HfTransformType::DCT32X64,
        9 => HfTransformType::DCT128X128,
        10 => HfTransformType::DCT64X128,
        11 => HfTransformType::DCT256X256,
        12 => HfTransformType::DCT128X256,
        _ => return None,
    })
}

static TOKEN_SHAPE_ORDERS: [std::sync::OnceLock<Vec<usize>>; 13] = [
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
];

fn token_shape_order(shape_id: usize) -> Option<&'static [usize]> {
    let canonical = canonical_transform_for_shape_id(shape_id)?;
    Some(
        TOKEN_SHAPE_ORDERS[shape_id]
            .get_or_init(|| natural_coeff_order_for_transform(canonical))
            .as_slice(),
    )
}

/// VarDCT encoder configuration.
pub struct VarDctConfig {
    /// Quality distance parameter. Lower = better quality. 1.0 = visually lossless.
    pub distance: f32,
    /// Effort tier in [1..=9], where higher values spend more CPU for better R-D.
    pub effort: u8,
    /// Emit multi-pass progressive VarDCT AC sections.
    pub progressive: bool,
}

impl Default for VarDctConfig {
    fn default() -> Self {
        Self {
            distance: 1.0,
            effort: 7,
            progressive: false,
        }
    }
}

#[derive(Clone, Copy)]
struct EffortParams {
    max_total_encodes: usize,
    enable_entropy_merge: bool,
    enable_custom_coeff_orders: bool,
}

/// Maps encoder effort `1..=9` to a libjxl-like speed tier index.
///
/// libjxl maps effort to `SpeedTier` via `speed_tier = 10 - effort`.
/// We use the same numeric mapping to gate heuristics consistently.
fn effort_to_speed_tier_index(effort: u8) -> u8 {
    let e = effort.clamp(1, 9);
    10 - e
}

fn effort_params(effort: u8) -> EffortParams {
    let speed_tier = effort_to_speed_tier_index(effort);

    // Candidate budget: larger at slower (higher quality) effort tiers.
    let max_total_encodes = match speed_tier {
        9 => 4,  // Lightning (effort 1)
        8 => 6,  // Thunder   (effort 2)
        7 => 8,  // Falcon    (effort 3)
        6 => 12, // Cheetah   (effort 4)
        5 => 18, // Hare      (effort 5)
        4 => 26, // Wombat    (effort 6)
        3 => 36, // Squirrel  (effort 7)
        2 => 48, // Kitten    (effort 8)
        _ => 64, // Tortoise-ish (effort 9)
    };

    EffortParams {
        // Enable entropy-merge heuristics from Hare-and-slower style tiers.
        enable_entropy_merge: speed_tier <= 5,
        // Enable custom coefficient orders from Squirrel-and-slower tiers.
        enable_custom_coeff_orders: speed_tier <= 3,
        max_total_encodes,
    }
}

fn choose_progressive_pass_plan(
    progressive: bool,
    has_alpha: bool,
    effort: u8,
    width: usize,
    height: usize,
) -> (usize, Vec<u32>) {
    if !progressive || has_alpha {
        return (1, vec![]);
    }

    let pixels = width.saturating_mul(height);
    // Fuller progressive scheduling: use 3 passes at highest effort for larger
    // images, otherwise keep robust 2-pass mode.
    if effort >= 9 && pixels >= 768 * 512 {
        (3, vec![2, 1])
    } else {
        (2, vec![1])
    }
}

/// Map distance parameter to (global_scale, quant_lf).
/// Compute (global_scale, quant_lf) from butteraugli distance.
#[cfg(test)]
fn distance_to_quant_params(distance: f32) -> (u32, u32) {
    let (gs, ql, _) = distance_to_full_quant_params(distance);
    (gs, ql)
}

/// Compute (global_scale, quant_lf, base_raw_quant) from butteraugli distance.
///
/// Mirrors libjxl's Quantizer::ComputeGlobalScaleAndQuant + InitialQuantDC:
///   quant_ac = 0.79 / distance             (AC quant field value, fast mode)
///   quant_dc = min(kDcQuant / dc_target, 50)
///   global_scale = kGlobalScaleDenom * quant_ac / kQuantFieldTarget
///   quant_lf = round(quant_dc * inv_global_scale + 0.5)
///   base_raw_quant = ClampVal(quant_ac * inv_global_scale + 0.5)
/// Compute global_scale, quant_lf, and base_raw_quant from distance.
/// Uses quant_ac = 0.79/d for the base raw_quant calculation.
fn distance_to_full_quant_params(distance: f32) -> (u32, u32, u8) {
    let quant_ac = 0.79f32 / distance;
    let (global_scale, quant_lf, _) = compute_global_scale_and_quant(distance, quant_ac);
    let inv_global_scale = 65536.0 / global_scale as f32;
    let base_raw_quant = ((quant_ac * inv_global_scale + 0.5) as u8).clamp(1, 255);
    (global_scale, quant_lf, base_raw_quant)
}

/// Core global_scale + quant_lf computation, matching libjxl's
/// Quantizer::ComputeGlobalScaleAndQuant exactly.
/// `quant_median` and `quant_median_absd` are the median and median absolute
/// deviation of the quant field.
fn compute_global_scale_and_quant(distance: f32, quant_median: f32) -> (u32, u32, u8) {
    compute_global_scale_and_quant_full(distance, quant_median, 0.0)
}

fn compute_global_scale_and_quant_full(
    distance: f32,
    quant_median: f32,
    quant_median_absd: f32,
) -> (u32, u32, u8) {
    const K_DC_QUANT: f32 = 1.095_924;
    const K_DC_QUANT_POW: f32 = 0.83;
    const K_DC_MUL: f32 = 0.3;
    const K_GLOBAL_SCALE_DENOM: f32 = 65536.0;
    const K_QUANT_FIELD_TARGET: f32 = 5.0;
    const K_GLOBAL_SCALE_NUMERATOR: f32 = 4096.0;

    // DC quant with non-linearity for low distances.
    let dc_target_raw = distance;
    let dc_target = (0.5 * dc_target_raw).max(
        (K_DC_MUL * ((1.0 / K_DC_MUL) * dc_target_raw).powf(K_DC_QUANT_POW)).min(dc_target_raw),
    );
    let quant_dc = (K_DC_QUANT / dc_target).min(50.0);

    // libjxl: scale = kGlobalScaleDenom * (quant_median - quant_median_absd) / kQuantFieldTarget
    // Subtracting absd gives higher resolution on highly varying quant fields.
    let mut global_scale =
        (K_GLOBAL_SCALE_DENOM * (quant_median - quant_median_absd) / K_QUANT_FIELD_TARGET) as i32;
    // Clamp so quant_dc won't be too small.
    let scaled_quant_dc = (quant_dc * K_GLOBAL_SCALE_NUMERATOR * 1.6) as i32;
    if global_scale > scaled_quant_dc {
        global_scale = scaled_quant_dc;
    }
    let global_scale = (global_scale.max(1) as u32).min(1 << 15);

    let inv_global_scale = K_GLOBAL_SCALE_DENOM / global_scale as f32;
    let quant_lf = ((quant_dc * inv_global_scale + 0.5).min(65536.0) as u32).max(1);
    let base_raw_quant = ((quant_median * inv_global_scale + 0.5) as u8).clamp(1, 255);

    (global_scale, quant_lf, base_raw_quant)
}

#[allow(dead_code)]
/// Apply inverse Gaborish (5x5 symmetric sharpening filter) to XYB pixel data.
///
/// Mirrors libjxl's `GaborishInverse` from enc_gaborish.cc.
/// The decoder applies forward Gaborish (3x3 smoothing) when gab=true.
/// The encoder compensates by applying this approximate inverse BEFORE the DCT.
fn apply_inverse_gaborish(channels: &mut [&mut [f32]; 3], width: usize, height: usize) {
    apply_inverse_gaborish_weighted(channels, width, height, [1.0, 1.0, 1.0]);
}

fn apply_inverse_gaborish_weighted(
    channels: &mut [&mut [f32]; 3],
    width: usize,
    height: usize,
    weights: [f32; 3],
) {
    // libjxl inverse Gaborish kernel (symmetric 5x5).
    const K: [f32; 5] = [
        -0.094_958_16,
        -0.041_031_725,
        0.013_710_005,
        0.006_510_206,
        -0.001_478_906_4,
    ];

    for (ch_idx, chan) in channels.iter_mut().enumerate() {
        let mul = weights[ch_idx];
        let sum = 1.0 + mul * 4.0 * (K[0] + K[1] + K[2] + K[4] + 2.0 * K[3]);
        let normalize = 1.0 / sum;
        let nm = mul * normalize;
        let w_center = normalize;
        let w_adj = nm * K[0];
        let w_diag = nm * K[1];
        let w_far_card = nm * K[2];
        let w_knight = nm * K[3];
        let w_far_diag = nm * K[4];

        let input = chan.to_vec();
        // Mirror boundary conditions, matching libjxl's Symmetric5 convolution.
        let mirror = |v: isize, max: isize| -> usize {
            if v < 0 {
                (-v) as usize
            } else if v >= max {
                (2 * max - 2 - v) as usize
            } else {
                v as usize
            }
        };
        let w_i = width as isize;
        let h_i = height as isize;
        let get = |x: isize, y: isize| -> f32 { input[mirror(y, h_i) * width + mirror(x, w_i)] };

        for y in 0..height {
            let iy = y as isize;
            for x in 0..width {
                let ix = x as isize;
                let mut v = get(ix, iy) * w_center;
                // Adjacent (r): 4 positions at distance 1 cardinal
                v +=
                    (get(ix - 1, iy) + get(ix + 1, iy) + get(ix, iy - 1) + get(ix, iy + 1)) * w_adj;
                // Diagonal (d): 4 positions at (1,1)
                v += (get(ix - 1, iy - 1)
                    + get(ix + 1, iy - 1)
                    + get(ix - 1, iy + 1)
                    + get(ix + 1, iy + 1))
                    * w_diag;
                // Far cardinal (R): 4 positions at distance 2 cardinal
                v += (get(ix - 2, iy) + get(ix + 2, iy) + get(ix, iy - 2) + get(ix, iy + 2))
                    * w_far_card;
                // Knight-move (L): 8 positions at (1,2)/(2,1)
                v += (get(ix - 1, iy - 2)
                    + get(ix + 1, iy - 2)
                    + get(ix - 1, iy + 2)
                    + get(ix + 1, iy + 2)
                    + get(ix - 2, iy - 1)
                    + get(ix + 2, iy - 1)
                    + get(ix - 2, iy + 1)
                    + get(ix + 2, iy + 1))
                    * w_knight;
                // Far diagonal/corner (D): 4 positions at (2,2)
                v += (get(ix - 2, iy - 2)
                    + get(ix + 2, iy - 2)
                    + get(ix - 2, iy + 2)
                    + get(ix + 2, iy + 2))
                    * w_far_diag;
                chan[y * width + x] = v;
            }
        }
    }
}

// Port of libjxl enc_frame.cc:SimplifyInvisible.
// For lossy non-associated alpha, replace fully invisible (alpha==0) color
// pixels with a local weighted blend of nearby pixels. This improves
// compression and reduces color halos near transparency edges.
fn simplify_invisible_channel(chan: &mut [f32], alpha: &[u8], width: usize, height: usize) {
    for y in 0..height {
        let row_off = y * width;
        let prow_off = y.checked_sub(1).map(|yy| yy * width);
        let nrow_off = (y + 1 < height).then_some((y + 1) * width);

        for x in 0..width {
            let idx = row_off + x;
            if alpha[idx] != 0 {
                continue;
            }

            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;

            if x > 0 {
                let left = chan[idx - 1];
                sum += left;
                wsum += 1.0;
                if alpha[idx - 1] > 0 {
                    sum += left;
                    wsum += 1.0;
                }
            }

            if x + 1 < width {
                if let Some(po) = prow_off {
                    sum += chan[po + x + 1];
                    wsum += 1.0;
                }
                if alpha[idx + 1] > 0 {
                    sum += 2.0 * chan[idx + 1];
                    wsum += 2.0;
                }
                if let Some(po) = prow_off {
                    if alpha[po + x + 1] > 0 {
                        sum += 2.0 * chan[po + x + 1];
                        wsum += 2.0;
                    }
                }
                if let Some(no) = nrow_off {
                    if alpha[no + x + 1] > 0 {
                        sum += 2.0 * chan[no + x + 1];
                        wsum += 2.0;
                    }
                }
            }

            if let Some(po) = prow_off {
                if alpha[po + x] > 0 {
                    sum += 2.0 * chan[po + x];
                    wsum += 2.0;
                }
            }
            if let Some(no) = nrow_off {
                if alpha[no + x] > 0 {
                    sum += 2.0 * chan[no + x];
                    wsum += 2.0;
                }
            }

            if wsum > 1.0 {
                chan[idx] = sum / wsum;
            }
        }
    }
}

fn simplify_invisible_xyb(
    x: &mut [f32],
    y: &mut [f32],
    b: &mut [f32],
    alpha: &[u8],
    width: usize,
    height: usize,
) {
    simplify_invisible_channel(x, alpha, width, height);
    simplify_invisible_channel(y, alpha, width, height);
    simplify_invisible_channel(b, alpha, width, height);
}

fn quantize_alpha_for_lossy_step(alpha: &[u8], step: u16) -> Vec<u8> {
    if step <= 1 {
        return alpha.to_vec();
    }

    alpha
        .iter()
        .map(|&a| {
            if a == 0 || a == 255 {
                return a;
            }
            let aa = a as u16;
            let q = ((aa + step / 2) / step) * step;
            q.clamp(1, 254) as u8
        })
        .collect()
}

fn alpha_psnr_db(orig: &[u8], cand: &[u8]) -> f32 {
    if orig.is_empty() {
        return 99.0;
    }
    let mse = orig
        .iter()
        .zip(cand.iter())
        .map(|(&o, &c)| {
            let d = o as f32 - c as f32;
            d * d
        })
        .sum::<f32>()
        / orig.len() as f32;
    if mse <= 1e-9 {
        99.0
    } else {
        10.0 * ((255.0f32 * 255.0) / mse).log10()
    }
}

fn estimate_alpha_modular_bytes(alpha: &[u8], width: usize, height: usize) -> Result<usize> {
    let mut w = BitWriter::new();
    let alpha_i32: Vec<i32> = alpha.iter().map(|&a| a as i32).collect();
    crate::encode::modular_encode::encode_modular_signed_stream(&mut w, width, height, 1, &alpha_i32)?;
    w.byte_align_zero_pad()?;
    Ok(w.finish().len())
}

fn choose_lossy_alpha_candidate(
    alpha: &[u8],
    width: usize,
    height: usize,
    distance: f32,
    effort: u8,
) -> Result<Vec<u8>> {
    if distance <= 0.0 || effort < 5 {
        return Ok(alpha.to_vec());
    }

    let psnr_floor = if distance <= 1.0 {
        49.0
    } else if distance <= 2.0 {
        44.0
    } else {
        40.0
    };

    let mut best = alpha.to_vec();
    let mut best_bytes = estimate_alpha_modular_bytes(&best, width, height)?;

    for &step in &[2u16, 3, 4, 6, 8, 12, 16, 24, 32] {
        let cand = quantize_alpha_for_lossy_step(alpha, step);
        let psnr = alpha_psnr_db(alpha, &cand);
        if psnr < psnr_floor {
            continue;
        }
        let bytes = estimate_alpha_modular_bytes(&cand, width, height)?;
        if bytes < best_bytes {
            best_bytes = bytes;
            best = cand;
        }
    }

    Ok(best)
}

// Heuristic: spend more bits on blocks that mix transparent and opaque pixels.
// This targets RGBA edge halos ("glow") by reducing color quantization error
// exactly where alpha compositing is most sensitive.
fn boost_quant_on_alpha_edges(
    raw_quant_map: &mut [u8],
    alpha: &[u8],
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
) {
    debug_assert_eq!(raw_quant_map.len(), bw * bh);
    debug_assert_eq!(alpha.len(), width * height);

    for by in 0..bh {
        for bx in 0..bw {
            let mut has_nonzero = false;
            let mut has_nonopaque = false;
            let y0 = by * 8;
            let x0 = bx * 8;

            for yy in y0..(y0 + 8).min(height) {
                let row = yy * width;
                for xx in x0..(x0 + 8).min(width) {
                    let a = alpha[row + xx];
                    if a > 0 {
                        has_nonzero = true;
                    }
                    if a < 255 {
                        has_nonopaque = true;
                    }
                }
            }

            // Mixed block: contains both visible and non-opaque pixels.
            if has_nonzero && has_nonopaque {
                let idx = by * bw + bx;
                let boosted = (raw_quant_map[idx] as f32 * 3.00).round() as u32;
                raw_quant_map[idx] = boosted.min(255) as u8;
            }
        }
    }
}

// Flat-region optimization for line-art / logo-like inputs.
// Adds an additional quant-map candidate that spends fewer bits on large,
// very flat interiors while keeping edge blocks less affected.
fn detect_flat_graphic(y_chan: &[f32], width: usize, height: usize, bw: usize, bh: usize) -> bool {
    let num_blocks = bw * bh;
    if num_blocks == 0 {
        return false;
    }

    let mut very_flat = 0usize;
    let mut high_detail = 0usize;

    for by in 0..bh {
        for bx in 0..bw {
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for yy in (by * 8)..((by * 8 + 8).min(height)) {
                let row = yy * width;
                for xx in (bx * 8)..((bx * 8 + 8).min(width)) {
                    let v = y_chan[row + xx];
                    min_v = min_v.min(v);
                    max_v = max_v.max(v);
                }
            }
            let r = max_v - min_v;
            if r < 0.004 {
                very_flat += 1;
            }
            if r > 0.03 {
                high_detail += 1;
            }
        }
    }

    very_flat * 100 >= num_blocks * 45 && high_detail * 100 <= num_blocks * 20
}

fn apply_flat_region_quant_boost(
    raw_quant_map: &[u8],
    y_chan: &[f32],
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
    strength: f32,
) -> Option<Vec<u8>> {
    let num_blocks = bw * bh;
    if num_blocks == 0 {
        return None;
    }

    let mut ranges = vec![0.0f32; num_blocks];
    let mut very_flat = 0usize;

    for by in 0..bh {
        for bx in 0..bw {
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for yy in (by * 8)..((by * 8 + 8).min(height)) {
                let row = yy * width;
                for xx in (bx * 8)..((bx * 8 + 8).min(width)) {
                    let v = y_chan[row + xx];
                    min_v = min_v.min(v);
                    max_v = max_v.max(v);
                }
            }
            let r = max_v - min_v;
            let idx = by * bw + bx;
            ranges[idx] = r;
            if r < 0.004 {
                very_flat += 1;
            }
        }
    }

    // Activate only when the image is dominantly flat.
    if very_flat * 100 < num_blocks * 45 {
        return None;
    }

    let mut out = raw_quant_map.to_vec();
    let mut boosted_any = false;

    for by in 0..bh {
        for bx in 0..bw {
            let idx = by * bw + bx;
            let r = ranges[idx];

            if r >= 0.010 {
                continue;
            }

            // Keep edge-adjacent blocks milder to avoid haloing.
            let mut has_nonflat_neighbor = false;
            for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let nx = bx as isize + dx;
                let ny = by as isize + dy;
                if nx < 0 || ny < 0 || nx >= bw as isize || ny >= bh as isize {
                    continue;
                }
                let nidx = ny as usize * bw + nx as usize;
                if ranges[nidx] >= 0.012 {
                    has_nonflat_neighbor = true;
                    break;
                }
            }

            let base_factor = if r < 0.003 && !has_nonflat_neighbor {
                2.4
            } else if r < 0.006 {
                1.6
            } else {
                1.25
            };
            let factor = 1.0 + (base_factor - 1.0) * strength;

            let boosted = (out[idx] as f32 * factor).round() as u32;
            let new_v = boosted.min(255) as u8;
            if new_v != out[idx] {
                boosted_any = true;
                out[idx] = new_v;
            }
        }
    }

    if boosted_any { Some(out) } else { None }
}

/// Encode an sRGB u8 RGB image as a VarDCT JXL file (container-wrapped).
pub fn encode_vardct_u8_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: &VarDctConfig,
) -> Result<Vec<u8>> {
    let codestream = encode_vardct_u8_rgb_codestream(rgb, width, height, config)?;
    wrap_codestream(&codestream)
}

/// Encode an sRGB u8 RGBA image as a VarDCT JXL with alpha channel.
/// `rgba` is interleaved RGBA (4 bytes per pixel).
pub fn encode_vardct_u8_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    config: &VarDctConfig,
) -> Result<Vec<u8>> {
    assert_eq!(rgba.len(), width * height * 4);
    let npixels = width * height;
    // Split into RGB + alpha
    let mut rgb = vec![0u8; npixels * 3];
    let mut alpha = vec![0u8; npixels];
    for i in 0..npixels {
        rgb[i * 3] = rgba[i * 4];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
        alpha[i] = rgba[i * 4 + 3];
    }
    let codestream = encode_single_rgba_frame(&rgb, width, height, config, None, Some(&alpha))?;
    wrap_codestream(&codestream)
}

/// Encode multiple sRGB u8 RGB frames as an animated JXL.
/// `frames`: slice of (rgb_data, duration_ms) pairs.
/// All frames must have the same dimensions.
/// Returns a JXL container wrapping the animation codestream.
pub fn encode_vardct_animation_u8_rgb(
    frames: &[(&[u8], u32)], // (rgb_data, duration_ms)
    width: usize,
    height: usize,
    config: &VarDctConfig,
) -> Result<Vec<u8>> {
    assert!(!frames.is_empty(), "need at least one frame");
    assert!(width > 0 && height > 0);

    // Compute ticks per second from frame durations.
    // Use 1000 TPS (millisecond precision) for simplicity.
    let tps_num: u32 = 1000;
    let tps_den: u32 = 1;

    let mut codestream = Vec::new();

    // Write animation file header
    let mut header_writer = BitWriter::new();
    crate::encode::headers::write_file_header_animated(
        &mut header_writer,
        width as u32,
        height as u32,
        tps_num,
        tps_den,
        0, // num_loops = 0 (infinite)
    )?;
    codestream.extend_from_slice(&header_writer.finish());

    let npixels = width * height;
    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);

    for (frame_idx, &(rgb, duration_ms)) in frames.iter().enumerate() {
        assert_eq!(rgb.len(), npixels * 3, "frame {frame_idx} wrong size");
        let is_last = frame_idx == frames.len() - 1;

        let anim = AnimFrameParams {
            duration: duration_ms,
            is_last,
        };
        let frame_bytes = encode_single_rgb_frame(rgb, width, height, config, Some(&anim))?;
        codestream.extend_from_slice(&frame_bytes);
    }

    wrap_codestream(&codestream)
}

/// Encode multiple sRGB u8 RGBA frames as an animated JXL with alpha.
/// `frames`: slice of (rgba_data, duration_ms) pairs.
pub fn encode_vardct_animation_u8_rgba(
    frames: &[(&[u8], u32)], // (rgba_data, duration_ms)
    width: usize,
    height: usize,
    config: &VarDctConfig,
) -> Result<Vec<u8>> {
    assert!(!frames.is_empty(), "need at least one frame");
    assert!(width > 0 && height > 0);
    let npixels = width * height;

    let tps_num: u32 = 1000;
    let tps_den: u32 = 1;

    let mut codestream = Vec::new();

    // Write animation file header with alpha
    let mut header_writer = BitWriter::new();
    crate::encode::headers::write_file_header_animated_with_alpha(
        &mut header_writer,
        width as u32,
        height as u32,
        tps_num,
        tps_den,
        0,
    )?;
    codestream.extend_from_slice(&header_writer.finish());

    for (frame_idx, &(rgba, duration_ms)) in frames.iter().enumerate() {
        assert_eq!(rgba.len(), npixels * 4, "frame {frame_idx} wrong size");
        let is_last = frame_idx == frames.len() - 1;

        // Split RGBA -> RGB + alpha
        let mut rgb = vec![0u8; npixels * 3];
        let mut alpha = vec![0u8; npixels];
        for i in 0..npixels {
            rgb[i * 3] = rgba[i * 4];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
            alpha[i] = rgba[i * 4 + 3];
        }

        let anim = AnimFrameParams {
            duration: duration_ms,
            is_last,
        };
        let frame_bytes =
            encode_single_rgba_frame(&rgb, width, height, config, Some(&anim), Some(&alpha))?;
        codestream.extend_from_slice(&frame_bytes);
    }

    wrap_codestream(&codestream)
}

/// Encode an sRGB u8 RGB image as a raw VarDCT JXL codestream.
pub fn encode_vardct_u8_rgb_codestream(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: &VarDctConfig,
) -> Result<Vec<u8>> {
    encode_single_rgb_frame(rgb, width, height, config, None)
}

/// Encode a single RGB frame. If `anim_params` is None, includes the file header
/// (for standalone images). If Some, writes only the frame (for animation).
fn encode_single_rgb_frame(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: &VarDctConfig,
    anim_params: Option<&AnimFrameParams>,
) -> Result<Vec<u8>> {
    encode_single_rgba_frame(rgb, width, height, config, anim_params, None)
}

/// Encode a single frame from RGB + optional alpha.
/// If `anim_params` is None, includes the file header (standalone image).
/// If Some, writes only the frame (for animation).
fn encode_single_rgba_frame(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: &VarDctConfig,
    anim_params: Option<&AnimFrameParams>,
    alpha: Option<&[u8]>,
) -> Result<Vec<u8>> {
    assert_eq!(rgb.len(), width * height * 3);
    if let Some(a) = alpha {
        assert_eq!(a.len(), width * height);
    }
    assert!(width > 0 && height > 0);

    let mut alpha_owned = None;
    let alpha = if let Some(a) = alpha {
        if config.distance > 0.0 {
            alpha_owned = Some(choose_lossy_alpha_candidate(
                a,
                width,
                height,
                config.distance,
                config.effort,
            )?);
            alpha_owned.as_deref()
        } else {
            Some(a)
        }
    } else {
        None
    };

    let npixels = width * height;
    let mut x_chan = vec![0.0f32; npixels];
    let mut y_chan = vec![0.0f32; npixels];
    let mut b_chan = vec![0.0f32; npixels];
    srgb_u8_to_xyb(rgb, width, height, &mut x_chan, &mut y_chan, &mut b_chan)?;

    // Match libjxl enc_frame.cc:SimplifyInvisible for lossy non-associated alpha.
    // This only touches fully invisible pixels (alpha == 0).
    if let Some(a) = alpha {
        simplify_invisible_xyb(&mut x_chan, &mut y_chan, &mut b_chan, a, width, height);
    }

    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);
    let num_blocks = bw * bh;
    let is_flat_graphic_pre = detect_flat_graphic(&y_chan, width, height, bw, bh);

    // --- libjxl encoder flow (enc_heuristics.cc, Squirrel default speed) ---
    // 1. Compute AQ map on ORIGINAL opsin (before inverse gaborish).
    //    kAcQuant = 0.765 for AdaptiveQuantizationMap scale.
    //    global_scale from ComputeGlobalScaleAndQuant(quant_dc, 0.39/d, 0).
    // Scale quant_ac by 1.15 to reduce quantization aggressiveness.
    // Our AQ map distributes bits less optimally than libjxl's. Scaling up
    // compensates uniformly, closing PSNR gaps at the cost of ~5% larger files.
    let quant_ac = (0.765f32 * 1.15) / config.distance;
    let (adaptive_map, global_scale, quant_lf, aq_float_map) = build_adaptive_raw_quant_map_full(
        &x_chan,
        &y_chan,
        &b_chan,
        width,
        height,
        bw,
        bh,
        config.distance,
        quant_ac,
    );
    let (_, _, base_raw_quant) = distance_to_full_quant_params(config.distance);
    let mut raw_quant_map_candidates = vec![
        vec![base_raw_quant; bw * bh], // uniform candidate
        adaptive_map,                  // libjxl-style adaptive candidate
    ];

    // For RGBA input, add boosted quant candidates for translucent edges,
    // but keep non-boosted maps available for size wins.
    if let Some(a) = alpha {
        let mut boosted = raw_quant_map_candidates.clone();
        for map in &mut boosted {
            boost_quant_on_alpha_edges(map, a, width, height, bw, bh);
        }
        for map in boosted {
            if !raw_quant_map_candidates.contains(&map) {
                raw_quant_map_candidates.push(map);
            }
        }
    }

    // Flat-region optimization candidates (line art / logos).
    if let Some(flat_map) = apply_flat_region_quant_boost(
        &raw_quant_map_candidates[1],
        &y_chan,
        width,
        height,
        bw,
        bh,
        1.0,
    ) {
        raw_quant_map_candidates.push(flat_map);
    }
    if config.distance > 1.2
        && let Some(flat_map_aggr) = apply_flat_region_quant_boost(
            &raw_quant_map_candidates[1],
            &y_chan,
            width,
            height,
            bw,
            bh,
            1.8,
        )
    {
        raw_quant_map_candidates.push(flat_map_aggr);
    }
    if config.distance > 1.6
        && is_flat_graphic_pre
        && let Some(flat_map_ultra) = apply_flat_region_quant_boost(
            &raw_quant_map_candidates[1],
            &y_chan,
            width,
            height,
            bw,
            bh,
            2.6,
        )
    {
        raw_quant_map_candidates.push(flat_map_ultra);
    }

    // Compute per-pixel masking field for AC strategy loss estimation.
    // Must be computed on ORIGINAL opsin (before inverse gaborish),
    // matching libjxl's AdaptiveQuantizationImpl::ComputeTile.
    let masking1x1 = compute_masking_1x1(&y_chan, width, height);

    // Save original opsin (pre-inverse-gaborish) for MSE comparison in
    // per-block transform selection. The decoder applies gaborish smoothing,
    // so the viewer sees approximately the original opsin, not the sharpened version.
    let orig_y_chan = y_chan.clone();

    // 2. Apply inverse gaborish to opsin (libjxl: GaborishInverse after AQ).
    //    For very flat graphics/logos, skip gab to avoid edge halo overhead.
    let use_gab = config.distance >= 0.3 && !is_flat_graphic_pre;
    if use_gab {
        apply_inverse_gaborish(&mut [&mut x_chan, &mut y_chan, &mut b_chan], width, height);
    }

    // 3. CfL input for B channel in HF coding: in_b = b - y
    //    (computed on gaborished opsin, matching libjxl)
    let b_minus_y_chan: Vec<f32> = b_chan.iter().zip(&y_chan).map(|(&b, &y)| b - y).collect();

    // 4. Forward DCT on gaborished opsin
    let mut dct_x = vec![0.0f32; num_blocks * 64];
    let mut dct_y = vec![0.0f32; num_blocks * 64];
    let mut dct_b = vec![0.0f32; num_blocks * 64];
    forward_dct_channel(&x_chan, width, height, bw, bh, &mut dct_x);
    forward_dct_channel(&y_chan, width, height, bw, bh, &mut dct_y);
    forward_dct_channel(&b_chan, width, height, bw, bh, &mut dct_b);

    // INV_LF_QUANT from the spec/decoder: [4096.0, 512.0, 256.0] for channels [X, Y, B]
    let inv_lf_quant = [4096.0f32, 512.0, 256.0];

    // Get default dequant weights for DCT8x8 (3*64 floats: X=0..64, Y=64..128, B=128..192)
    let dequant_weights = default_dct8x8_dequant_weights();

    // dm_multiplier for x and b channels (from x_qm_scale=3, b_qm_scale=2 defaults)
    let x_dm_multiplier = (1.0f32 / 1.25).powf(3.0 - 2.0); // = 0.8
    let b_dm_multiplier = (1.0f32 / 1.25).powf(2.0 - 2.0); // = 1.0

    // Cache forward transformed non-8x8 blocks across candidate evaluation.
    let mut forward_transform_cache = ForwardTransformCoeffCache::new();

    // Compute per-tile CfL maps (ytox and ytob).
    // Skip CfL optimization at near-lossless distances where the factor quantization
    // (1/84 granularity) would dominate error.
    let (ytox_map, ytob_map) = if config.distance >= 0.5 && !is_flat_graphic_pre {
        compute_cfl_maps(&dct_x, &dct_y, &dct_b, bw, bh)
    } else {
        let cr_size = bw.div_ceil(8) * bh.div_ceil(8);
        (vec![0i32; cr_size], vec![0i32; cr_size])
    };

    // Evaluate candidates by exact encoded frame size.
    // Budget and heuristics depend on effort tier.
    let effort_cfg = effort_params(config.effort);
    let is_flat_graphic = is_flat_graphic_pre;
    let max_total_encodes = effort_cfg.max_total_encodes;
    let mut total_encodes = 0usize;
    let mut candidate_frames = Vec::with_capacity(raw_quant_map_candidates.len());
    for raw_quant_map in &raw_quant_map_candidates {
        let quantized = quantize_vardct_blocks(
            &dct_x,
            &dct_y,
            &dct_b,
            global_scale,
            quant_lf,
            raw_quant_map,
            &inv_lf_quant,
            &dequant_weights,
            x_dm_multiplier,
            b_dm_multiplier,
            bw,
            &ytox_map,
            &ytob_map,
        );
        let mut transform_map_candidates =
            build_transform_map_candidates_from_quantized_ac_with_flags(
                &quantized.ac_x,
                &quantized.ac_y,
                &quantized.ac_b,
                bw,
                bh,
                config.distance,
                is_flat_graphic,
            );
        // Entropy-based DCT16/32 merge using full libjxl EstimateEntropy model
        // (entropy + information loss with perceptual masking).
        if effort_cfg.enable_entropy_merge && bw >= 2 && bh >= 2 {
            let entropy_map = build_entropy_merge_transform_map(
                &x_chan,
                &y_chan,
                &b_minus_y_chan,
                width,
                height,
                bw,
                bh,
                &quantized.ac_y,
                &quantized.ac_x,
                &quantized.ac_b,
                global_scale,
                raw_quant_map,
                &dequant_weights,
                x_dm_multiplier,
                b_dm_multiplier,
                config.distance,
                &masking1x1,
                &orig_y_chan,
                &aq_float_map,
            );
            let default_map = build_default_transform_map(bw, bh);
            if entropy_map != default_map && !transform_map_candidates.contains(&entropy_map) {
                transform_map_candidates.push(entropy_map);
            }
        }

        let mut best_frame = None;
        let mut best_size = usize::MAX;
        for transform_map in transform_map_candidates {
            if total_encodes >= max_total_encodes {
                break;
            }
            // AdjustQuantField: for merged blocks, set quant to MAX of constituents.
            // This matches libjxl's behavior and ensures merged blocks use the
            // finest quantization step among their constituent 8x8 blocks.
            let adj_quant =
                adjust_quant_field(raw_quant_map, &transform_map, bw, bh, config.distance);
            let quant_for_encode = &adj_quant;

            // Candidate A: legacy bootstrap behavior for non-8x8 transforms (all-zero AC).
            let mut ac_x_zero = quantized.ac_x.clone();
            let mut ac_y_zero = quantized.ac_y.clone();
            let mut ac_b_zero = quantized.ac_b.clone();
            zero_non_dct8_ac_coeffs(&mut ac_x_zero, &transform_map, bw, bh)?;
            zero_non_dct8_ac_coeffs(&mut ac_y_zero, &transform_map, bw, bh)?;
            zero_non_dct8_ac_coeffs(&mut ac_b_zero, &transform_map, bw, bh)?;
            let frame_zero = encode_vardct_frame_inner(
                width,
                height,
                bw,
                bh,
                global_scale,
                quant_lf,
                &quantized.dc_y,
                &quantized.dc_x,
                &quantized.dc_b,
                &ac_x_zero,
                &ac_y_zero,
                &ac_b_zero,
                quant_for_encode,
                &transform_map,
                &ytox_map,
                &ytob_map,
                use_gab,
                anim_params,
                anim_params.is_none(), // include file header only for standalone images
                alpha,
                config.effort,
                config.progressive,
            )?;

            let has_supported_nonzero_transform = transform_map.iter().any(|&t| {
                (t & TRANSFORM_FIRST_BLOCK_FLAG) != 0
                    && is_supported_nonzero_transform_id(t & !TRANSFORM_FIRST_BLOCK_FLAG)
            });

            total_encodes += 1;

            let frame = if has_supported_nonzero_transform && total_encodes < max_total_encodes {
                // Candidate B: non-zero coefficient path for supported larger transforms.
                let (ac_x_for_map, ac_y_for_map, ac_b_for_map) =
                    prepare_ac_for_transform_map_with_cache(
                        &quantized.ac_x,
                        &quantized.ac_y,
                        &quantized.ac_b,
                        &x_chan,
                        &y_chan,
                        &b_minus_y_chan,
                        width,
                        height,
                        bw,
                        bh,
                        global_scale,
                        quant_for_encode,
                        &transform_map,
                        Some(&mut forward_transform_cache),
                        x_dm_multiplier,
                        b_dm_multiplier,
                    )?;
                let frame_nonzero = encode_vardct_frame_inner(
                    width,
                    height,
                    bw,
                    bh,
                    global_scale,
                    quant_lf,
                    &quantized.dc_y,
                    &quantized.dc_x,
                    &quantized.dc_b,
                    &ac_x_for_map,
                    &ac_y_for_map,
                    &ac_b_for_map,
                    quant_for_encode,
                    &transform_map,
                    &ytox_map,
                    &ytob_map,
                    use_gab,
                    anim_params,
                    anim_params.is_none(),
                    alpha,
                    config.effort,
                    config.progressive,
                )?;

                total_encodes += 1;
                if frame_nonzero.len() < frame_zero.len() {
                    frame_nonzero
                } else {
                    frame_zero
                }
            } else {
                frame_zero
            };
            if frame.len() < best_size {
                best_size = frame.len();
                best_frame = Some(frame);
            }
        }
        if let Some(frame) = best_frame {
            candidate_frames.push(frame);
        }
        if total_encodes >= max_total_encodes {
            break;
        }
    }

    if candidate_frames.is_empty() {
        return Err(crate::error::Error::InvalidVarDCTTransformMap);
    }
    // Candidate 0 is always uniform raw_quant=base.
    let uniform_size = candidate_frames[0].len();
    let mut best_idx = 0usize;
    let mut best_size = uniform_size;
    for (idx, frame) in candidate_frames.iter().enumerate().skip(1) {
        if frame.len() < best_size {
            best_size = frame.len();
            best_idx = idx;
        }
    }

    // Regression-safe threshold: keep adaptive map only with a clear byte gain.
    const ADAPTIVE_RAW_QUANT_MIN_GAIN_BYTES: usize = 0;
    if best_idx > 0 && best_size + ADAPTIVE_RAW_QUANT_MIN_GAIN_BYTES < uniform_size {
        Ok(candidate_frames.swap_remove(best_idx))
    } else {
        Ok(candidate_frames.swap_remove(0))
    }
}

struct QuantizedVardct {
    dc_x: Vec<i32>,
    dc_y: Vec<i32>,
    dc_b: Vec<i32>,
    ac_x: Vec<i32>,
    ac_y: Vec<i32>,
    ac_b: Vec<i32>,
}

#[allow(clippy::too_many_arguments)]
/// Compute per-tile CfL maps (ytox_map and ytob_map) via least-squares regression.
/// Returns (ytox_map, ytob_map), each cr_w * cr_h values.
/// Actual factors: base_correlation_x + x_factor/84 for X, base_correlation_b + b_factor/84 for B.
/// libjxl's towards_zero shrinkage for CfL multipliers.
/// Reduces oscillations by pulling small values to zero.
const TOWARDS_ZERO: f64 = 2.6;

/// Approximate quantization weights for CfL regression.
/// libjxl multiplies DCT coefficients by q * inv_dequant_matrix[k].
/// The inv_dequant_matrix is ~1/dequant_weight, which is high for DC and
/// low-frequency coefficients, dropping off for HF. We approximate this
/// with 1/(1 + distance_from_dc). This weights low-frequency AC
/// coefficients more heavily, matching libjxl's behavior.
static CFL_QUANT_WEIGHTS: [f32; 64] = {
    let mut w = [0.0f32; 64];
    let mut ky = 0;
    while ky < 8 {
        let mut kx = 0;
        while kx < 8 {
            let k = ky * 8 + kx;
            // Manhattan distance from DC, scaled to approximate quant weight
            let dist = (ky + kx) as f32;
            w[k] = 1.0 / (1.0 + dist * dist * 0.25);
            kx += 1;
        }
        ky += 1;
    }
    w
};

fn towards_zero_shrink(x: f64, threshold: f64) -> f64 {
    if x >= threshold {
        x - threshold
    } else if x <= -threshold {
        x + threshold
    } else {
        0.0
    }
}

fn compute_cfl_maps(
    dct_x: &[f32],
    dct_y: &[f32],
    dct_b: &[f32],
    bw: usize,
    bh: usize,
) -> (Vec<i32>, Vec<i32>) {
    // Get DCT8 dequant weights for weighting CfL regression.
    // Table 0 (Dct) has 3*64 = 192 entries: [0..64]=Y, [64..128]=X, [128..192]=B.
    // libjxl uses InvMatrix(strategy, channel) which returns 1/dequant_weight.
    // We use the raw dequant weights as multipliers: 1/dw = quant weight.
    let dw_table = crate::frame::quant_weights::DequantMatrices::get_library_table(0);
    // Invert to get quant weights (high for positions that get quantized hard)
    let mut qw_x = [0.0f64; 64];
    let mut qw_b = [0.0f64; 64];
    for k in 0..64 {
        // dw_table contains dequant weights. InvMatrix = 1/dw.
        // CfL multiplies by InvMatrix (= 1/dw = quant weight).
        qw_x[k] = 1.0 / (dw_table[64 + k] as f64).max(1e-10);
        qw_b[k] = 1.0 / (dw_table[128 + k] as f64).max(1e-10);
    }
    const K_COLOR_FACTOR: f32 = 84.0;
    let cr_w = bw.div_ceil(8);
    let cr_h = bh.div_ceil(8);
    let mut ytox_map = vec![0i32; cr_w * cr_h];
    let mut ytob_map = vec![0i32; cr_w * cr_h];

    for ty in 0..cr_h {
        for tx in 0..cr_w {
            let mut sum_yy_x = 0.0f64; // Y*Y weighted by X-channel quant
            let mut sum_yx = 0.0f64; // Y*X weighted by X-channel quant
            let mut sum_yy_b = 0.0f64; // Y*Y weighted by B-channel quant
            let mut sum_yb = 0.0f64; // Y*B weighted by B-channel quant

            // libjxl weights DCT coefficients by q * inv_dequant_matrix[k]
            // (the quantization strength). This ensures that coefficients
            // which will be quantized to zero don't influence the CfL
            // regression. Critical for images with large uniform areas.
            for by in (ty * 8)..((ty + 1) * 8).min(bh) {
                for bx in (tx * 8)..((tx + 1) * 8).min(bw) {
                    let blk = by * bw + bx;
                    for k in 1..64 {
                        let y_for_x = dct_y[blk * 64 + k] as f64 * qw_x[k];
                        let x_val = dct_x[blk * 64 + k] as f64 * qw_x[k];
                        let y_for_b = dct_y[blk * 64 + k] as f64 * qw_b[k];
                        let b_val = dct_b[blk * 64 + k] as f64 * qw_b[k];
                        sum_yy_x += y_for_x * y_for_x;
                        sum_yx += y_for_x * x_val;
                        sum_yy_b += y_for_b * y_for_b;
                        sum_yb += y_for_b * b_val;
                    }
                }
            }

            // libjxl's FindBestMultiplier fast path:
            // x = -sum(a*b) / (sum(a*a) + num * distance_mul * 0.5)
            // where a = y_coeff / COLOR_FACTOR, b = base * y_coeff - target_coeff
            // Our formulation: optimal = sum_yx / sum_yy
            // Then convert to integer factor and apply towards_zero shrinkage.

            // libjxl FindBestMultiplier fast path:
            // x = -sum(a*b) / (sum(a*a) + num * distance_mul * 0.5)
            // where a = y * qw / COLOR_FACTOR
            //       b = base * y * qw - target * qw
            // We reformulate: optimal = sum_yx / sum_yy is the raw ratio.
            // Adding distance_mul regularization:
            // factor = sum_cross / (sum_yy + regularizer)
            // where regularizer = num * distance_mul * 0.5 * K^2
            let n_blocks = ((ty * 8 + 8).min(bh) - ty * 8) * ((tx * 8 + 8).min(bw) - tx * 8);
            let num_coeffs = (n_blocks * 63) as f64;
            // kDistanceMultiplierAC = 1e-3 (libjxl uses 1e-9 but with much
            // larger q*qm weights; we scale up to compensate)
            let dist_mul = 1e-3;

            // X channel: base_correlation_x = 0
            let reg_x = num_coeffs * dist_mul * 0.5;
            if sum_yy_x + reg_x > 1e-10 {
                let x_raw = (sum_yx / (sum_yy_x + reg_x)) * K_COLOR_FACTOR as f64;
                let x_shrunk = towards_zero_shrink(x_raw, TOWARDS_ZERO);
                ytox_map[ty * cr_w + tx] = (x_shrunk.round() as i32).clamp(-127, 127);
            }

            // B channel: base_correlation_b = 1.0
            let reg_b = num_coeffs * dist_mul * 0.5;
            if sum_yy_b + reg_b > 1e-10 {
                let b_raw = ((sum_yb / (sum_yy_b + reg_b)) - 1.0) * K_COLOR_FACTOR as f64;
                let b_shrunk = towards_zero_shrink(b_raw, TOWARDS_ZERO);
                ytob_map[ty * cr_w + tx] = (b_shrunk.round() as i32).clamp(-127, 127);
            }
        }
    }

    (ytox_map, ytob_map)
}

fn quantize_vardct_blocks(
    dct_x: &[f32],
    dct_y: &[f32],
    dct_b: &[f32],
    global_scale: u32,
    quant_lf: u32,
    raw_quant_map: &[u8],
    inv_lf_quant: &[f32; 3],
    dequant_weights: &[f32],
    x_dm_multiplier: f32,
    b_dm_multiplier: f32,
    bw: usize,
    ytox_map: &[i32],
    ytob_map: &[i32],
) -> QuantizedVardct {
    const K_COLOR_FACTOR: f32 = 84.0;
    let num_blocks = raw_quant_map.len();
    let cr_w = bw.div_ceil(8);
    let mut dc_x = vec![0i32; num_blocks];
    let mut dc_y = vec![0i32; num_blocks];
    let mut dc_b = vec![0i32; num_blocks];
    let mut qx = vec![0i32; num_blocks * 64];
    let mut qy = vec![0i32; num_blocks * 64];
    let mut qb = vec![0i32; num_blocks * 64];

    for blk in 0..num_blocks {
        let raw_quant = raw_quant_map[blk] as u32;
        let bx = blk % bw;
        let by = blk / bw;
        let tx = bx / 8;
        let ty = by / 8;
        let x_factor = ytox_map[ty * cr_w + tx];
        let b_factor = ytob_map[ty * cr_w + tx];
        let ytox_ratio = x_factor as f32 / K_COLOR_FACTOR;
        let ytob_ratio = 1.0 + b_factor as f32 / K_COLOR_FACTOR;

        // DC: apply CfL decorrelation and proper DC quantization.
        let raw_dc_x = dct_x[blk * 64];
        let raw_dc_y = dct_y[blk * 64];
        let raw_dc_b = dct_b[blk * 64];
        let cfl_dc_x = raw_dc_x; // in_x = dc_x (y_to_x_lf=0)
        let cfl_dc_y = raw_dc_y; // in_y = dc_y
        let cfl_dc_b = raw_dc_b - raw_dc_y; // in_b = dc_b - dc_y (y_to_b_lf=1.0)

        dc_x[blk] = quantize_dc(cfl_dc_x, global_scale, quant_lf, inv_lf_quant[0]);
        dc_y[blk] = quantize_dc(cfl_dc_y, global_scale, quant_lf, inv_lf_quant[1]);
        dc_b[blk] = quantize_dc(cfl_dc_b, global_scale, quant_lf, inv_lf_quant[2]);

        // AC: apply CfL decorrelation with per-tile ytox and ytob factors.
        for k in 1..64 {
            let dw_x = dequant_weights[k] * x_dm_multiplier;
            let dw_y = dequant_weights[64 + k];
            let dw_b = dequant_weights[128 + k] * b_dm_multiplier;

            let ac_x = dct_x[blk * 64 + k] - ytox_ratio * dct_y[blk * 64 + k];
            let ac_y = dct_y[blk * 64 + k];
            let ac_b = dct_b[blk * 64 + k] - ytob_ratio * dct_y[blk * 64 + k];

            qx[blk * 64 + k] = quantize_ac(ac_x, global_scale, raw_quant, dw_x);
            qy[blk * 64 + k] = quantize_ac(ac_y, global_scale, raw_quant, dw_y);
            qb[blk * 64 + k] = quantize_ac(ac_b, global_scale, raw_quant, dw_b);
        }
    }

    QuantizedVardct {
        dc_x,
        dc_y,
        dc_b,
        ac_x: qx,
        ac_y: qy,
        ac_b: qb,
    }
}

// ==================== libjxl adaptive quantization port ====================
// Direct port of enc_adaptive_quantization.cc: AdaptiveQuantizationMap +
// PerBlockModulations pipeline. All constants and math from libjxl.

/// SimpleGamma constants from libjxl.
const K_SG_MUL: f32 = 226.77216153508914;
const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_INV_LOG2E: f32 = 1.0 / std::f32::consts::LOG2_E;
const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
const K_SG_V_OFFSET: f32 = 7.7825991679894591;

/// Ratio of derivatives of cubic root to SimpleGamma.
/// Maps opsin space to psychovisual space for masking computations.
fn ratio_of_derivatives(v: f32, invert: bool) -> f32 {
    let v = v.max(0.0);
    let k_epsilon = 1e-2f32;
    let k_num_mul = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    let k_v_offset = K_SG_V_OFFSET * K_INV_LOG2E + k_epsilon;
    let k_den_mul = K_INV_LOG2E * K_SG_MUL;

    let v2 = v * v;
    let num = k_num_mul * v2 + k_epsilon;
    let den = k_den_mul * v * v2 + k_v_offset;
    if invert { den / num } else { num / den }
}

/// Compute per-pixel masking field for information loss estimation.
/// Direct port of libjxl's mask1x1 computation from
/// AdaptiveQuantizationImpl::ComputeTile in enc_adaptive_quantization.cc.
///
/// For each pixel: compute Laplacian of Y channel intensity, apply gamma
/// correction, convert to masking weight: high activity = lower masking
/// (errors more visible).
fn compute_masking_1x1(y_chan: &[f32], width: usize, height: usize) -> Vec<f32> {
    let match_gamma_offset = 0.019f32;
    let mut mask = vec![0.0f32; width * height];

    for y in 0..height {
        let y1 = if y > 0 { y - 1 } else { y };
        let y2 = if y + 1 < height { y + 1 } else { y };
        for x in 0..width {
            let x1 = if x > 0 { x - 1 } else { x };
            let x2 = if x + 1 < width { x + 1 } else { x };

            let center = y_chan[y * width + x];
            let base = 0.25
                * (y_chan[y2 * width + x]
                    + y_chan[y1 * width + x]
                    + y_chan[y * width + x1]
                    + y_chan[y * width + x2]);

            let gammac = ratio_of_derivatives(center + match_gamma_offset, false);
            let diff = (gammac * (center - base)).abs();
            // kScaler = 1.0, so no scaling
            let diff = diff.ln_1p();
            let k_mul = 1.0f32;
            let k_offset = 0.01f32;
            mask[y * width + x] = k_mul / (diff + k_offset);
        }
    }
    mask
}

/// MaskingSqrt from libjxl.
fn masking_sqrt(v: f32) -> f32 {
    const K_LOG_OFFSET: f32 = 27.505837037000106;
    const K_MUL: f32 = 211.66567973503678;
    let mul_v = K_MUL * 1e8;
    0.25 * (v * mul_v.sqrt() + K_LOG_OFFSET).sqrt()
}

/// ComputeMask from libjxl: maps aq_map values to masking multipliers.
fn compute_mask(out_val: f32) -> f32 {
    const K_BASE: f32 = -0.7647;
    const K_MUL4: f32 = 9.4708735624378946;
    const K_MUL2: f32 = 17.35036561631863;
    const K_OFFSET2: f32 = 302.59587815579727;
    const K_MUL3: f32 = 6.7943250517376494;
    const K_OFFSET3: f32 = 3.7179635626140772;
    const K_OFFSET4: f32 = 0.25 * K_OFFSET3;
    const K_MUL0: f32 = 0.80061762862741759;

    let v1 = (out_val * K_MUL0).max(1e-3);
    let v2 = 1.0 / (v1 + K_OFFSET2);
    let v3 = 1.0 / (v1 * v1 + K_OFFSET3);
    let v4 = 1.0 / (v1 * v1 + K_OFFSET4);
    K_BASE + K_MUL4 * v4 + K_MUL2 * v2 + K_MUL3 * v3
}

/// Fast log2 approximation matching libjxl's FastLog2f.
fn fast_log2f(v: f32) -> f32 {
    v.max(1e-30).log2()
}

/// GammaModulation from libjxl: adjusts mask based on opsin gamma.
fn gamma_modulation(
    bx: usize,
    by: usize,
    xyb_x: &[f32],
    xyb_y: &[f32],
    img_w: usize,
    img_h: usize,
) -> f32 {
    const K_BIAS: f32 = 0.16;
    let mut overall_ratio = 0.0f32;
    for dy in 0..8usize {
        let py = (by * 8 + dy).min(img_h - 1);
        for dx in 0..8usize {
            let px = (bx * 8 + dx).min(img_w - 1);
            let iny = xyb_y[py * img_w + px] + K_BIAS;
            let inx = xyb_x[py * img_w + px];
            let r = iny - inx;
            overall_ratio += ratio_of_derivatives(r, true);
            let g = iny + inx;
            overall_ratio += ratio_of_derivatives(g, true);
        }
    }
    overall_ratio *= 0.5 / 64.0;
    const K_GAMMA: f32 = 0.1005613337192697;
    K_GAMMA * fast_log2f(overall_ratio)
}

/// HfModulation from libjxl: per-block HF activity from pixel gradients.
fn hf_modulation(bx: usize, by: usize, xyb_y: &[f32], img_w: usize, img_h: usize) -> f32 {
    const VAL_CLAMP: f32 = 0.0206;
    let mut sum_y = 0.0f32;
    for dy in 0..8usize {
        let py = (by * 8 + dy).min(img_h - 1);
        let py_next = if dy < 7 {
            (by * 8 + dy + 1).min(img_h - 1)
        } else {
            py
        };
        for dx in 0..8usize {
            let px = (bx * 8 + dx).min(img_w - 1);
            let v = xyb_y[py * img_w + px];
            // Right neighbor (skip last col)
            if dx < 7 {
                let px2 = (bx * 8 + dx + 1).min(img_w - 1);
                sum_y += (v - xyb_y[py * img_w + px2]).abs().min(VAL_CLAMP);
            }
            // Bottom neighbor
            sum_y += (v - xyb_y[py_next * img_w + px]).abs().min(VAL_CLAMP);
        }
    }
    const K_MUL_Y: f32 = -0.38;
    const K_OFFSET: f32 = 0.42;
    sum_y * K_MUL_Y + K_OFFSET
}

/// BlueModulation from libjxl: boosts quality for blue-dominant blocks.
fn blue_modulation(
    bx: usize,
    by: usize,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    img_w: usize,
    img_h: usize,
) -> f32 {
    const K_LIMIT: f32 = 0.010474084867598155;
    const K_OFFSET: f32 = 0.0031994768654636393;
    let mut sum = 0.0f32;
    for dy in 0..8usize {
        let py = (by * 8 + dy).min(img_h - 1);
        for dx in 0..8usize {
            let px = (bx * 8 + dx).min(img_w - 1);
            let idx = py * img_w + px;
            let p_x = xyb_x[idx];
            let p_b = xyb_b[idx];
            let p_y_raw = xyb_y[idx] + K_OFFSET;
            let p_y_effective = p_y_raw + p_x.abs();
            if p_b > p_y_effective {
                sum += (p_b - p_y_effective).min(K_LIMIT);
            }
        }
    }
    // If all blue, don't boost
    if sum >= 32.0 * K_LIMIT {
        sum = 64.0 * K_LIMIT - sum;
    }
    const K_MAX_LIMIT: f32 = 15.463398341612438;
    if sum >= K_MAX_LIMIT * K_LIMIT {
        sum = K_MAX_LIMIT * K_LIMIT;
    }
    const K_MUL: f32 = 0.90590804735610064;
    sum * K_MUL
}

/// Compute per-pixel Laplacian-based diff in Y channel, squared, clamped,
/// then MaskingSqrt'd, downsampled 4x, and FuzzyErosion'd to get per-block
/// aq_map values. Direct port of libjxl ComputeTile.
fn compute_aq_map(
    xyb_y: &[f32],
    img_w: usize,
    img_h: usize,
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<f32> {
    let pw = bw * 8; // padded width (block-aligned)
    let ph = bh * 8;
    const MATCH_GAMMA_OFFSET: f32 = 0.019;
    const LIMIT: f32 = 0.2;

    // Step 1: Per-pixel Laplacian diff -> squared -> clamped -> MaskingSqrt
    // libjxl accumulates 4 rows into row_out, then averages groups of 4 columns.
    // This gives a 4x downsampled image (ds_w x ds_h) = (bw*2 x bh*2).
    let ds_w = pw / 4;
    let ds_h = ph / 4;
    let mut pre_erosion = vec![0.0f32; ds_w * ds_h];
    let mut row_buf = vec![0.0f32; pw]; // accumulator for 4 rows

    let get_y = |x: usize, y: usize| -> f32 {
        let cx = x.min(img_w - 1);
        let cy = y.min(img_h - 1);
        xyb_y[cy * img_w + cx]
    };

    for y in 0..ph {
        for x in 0..pw {
            let x1 = if x > 0 { x - 1 } else { x };
            let x2 = if x + 1 < pw { x + 1 } else { x };
            let y1 = if y > 0 { y - 1 } else { y };
            let y2 = if y + 1 < ph { y + 1 } else { y };

            let center = get_y(x, y);
            let base = 0.25 * (get_y(x2, y) + get_y(x1, y) + get_y(x, y1) + get_y(x, y2));
            let gammac = ratio_of_derivatives(center + MATCH_GAMMA_OFFSET, false);
            let mut diff = gammac * (center - base);
            diff *= diff;
            diff = diff.min(LIMIT);
            diff = masking_sqrt(diff);

            if y % 4 != 0 {
                row_buf[x] += diff;
            } else {
                row_buf[x] = diff;
            }
        }
        // At end of each 4-row group, downsample columns by averaging groups of 4
        if y % 4 == 3 {
            let dy = y / 4;
            if dy < ds_h {
                for qx in 0..ds_w {
                    let avg = (row_buf[qx * 4]
                        + row_buf[qx * 4 + 1]
                        + row_buf[qx * 4 + 2]
                        + row_buf[qx * 4 + 3])
                        * 0.25;
                    pre_erosion[dy * ds_w + qx] = avg;
                }
            }
        }
    }

    // Step 2: FuzzyErosion - 3x3 min-4-of-9 weighted, downsampled 2x -> per-block
    // pre_erosion is (ds_w x ds_h) = (bw*2 x bh*2), output is (bw x bh)
    let pe_w = ds_w;
    let pe_h = ds_h;

    // FuzzyErosion weights from libjxl
    let mut k_mul_base = [0.125f32, 0.1, 0.09, 0.06];
    let k_mul_add = [0.0f32, -0.1, -0.09, -0.06];
    let mul = if distance < 2.0 {
        (2.0 - distance) * 0.5
    } else {
        0.0
    };
    let mut norm_sum = 0.0f32;
    for i in 0..4 {
        k_mul_base[i] += mul * k_mul_add[i];
        norm_sum += k_mul_base[i];
    }
    const K_TOTAL: f32 = 0.29959705784054957;
    let k_mul: [f32; 4] = [
        k_mul_base[0] * K_TOTAL / norm_sum,
        k_mul_base[1] * K_TOTAL / norm_sum,
        k_mul_base[2] * K_TOTAL / norm_sum,
        k_mul_base[3] * K_TOTAL / norm_sum,
    ];

    let mut aq_map = vec![0.0f32; bw * bh];
    let pe_get =
        |x: usize, y: usize| -> f32 { pre_erosion[y.min(pe_h - 1) * pe_w + x.min(pe_w - 1)] };

    for fy in 0..pe_h.min(bh * 2) {
        for fx in 0..pe_w.min(bw * 2) {
            let x = fx;
            let y = fy;
            let xm1 = if x > 0 { x - 1 } else { x };
            let xp1 = if x + 1 < pe_w { x + 1 } else { x };
            let ym1 = if y > 0 { y - 1 } else { y };
            let yp1 = if y + 1 < pe_h { y + 1 } else { y };

            // Collect all 9 neighbors
            let mut vals = [
                pe_get(x, y),
                pe_get(xm1, y),
                pe_get(xp1, y),
                pe_get(xm1, ym1),
                pe_get(x, ym1),
                pe_get(xp1, ym1),
                pe_get(xm1, yp1),
                pe_get(x, yp1),
                pe_get(xp1, yp1),
            ];
            // Sort to find 4 smallest
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let v =
                k_mul[0] * vals[0] + k_mul[1] * vals[1] + k_mul[2] * vals[2] + k_mul[3] * vals[3];

            let bx = fx / 2;
            let by = fy / 2;
            if bx < bw && by < bh {
                if fx % 2 == 0 && fy % 2 == 0 {
                    aq_map[by * bw + bx] = v;
                } else {
                    aq_map[by * bw + bx] += v;
                }
            }
        }
    }

    aq_map
}

/// Apply PerBlockModulations from libjxl: ComputeMask + GammaModulation +
/// HfModulation + BlueModulation, then exp2 scaling.
fn apply_per_block_modulations(
    aq_map: &mut [f32],
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    img_w: usize,
    img_h: usize,
    bw: usize,
    bh: usize,
    distance: f32,
    scale: f32,
) {
    let base_level = 0.48 * scale;
    let dampen = if distance >= 2.0 {
        let d = 1.0 - (distance - 2.0) / 12.0;
        d.max(0.0)
    } else {
        1.0
    };
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;

    for by in 0..bh {
        for bx in 0..bw {
            let idx = by * bw + bx;
            let mut out_val = compute_mask(aq_map[idx]);
            out_val = gamma_modulation(bx, by, xyb_x, xyb_y, img_w, img_h) + out_val;
            let hf_val = hf_modulation(bx, by, xyb_y, img_w, img_h) + out_val;
            let blue_val = blue_modulation(bx, by, xyb_x, xyb_y, xyb_b, img_w, img_h) + out_val;
            out_val = hf_val.min(blue_val);
            // exp2(out_val * log2(e)^-1 * log2(e)) = exp2(out_val * 1.442695...)
            aq_map[idx] = (out_val * std::f32::consts::LOG2_E).exp2() * mul + add;
        }
    }
}

/// Build per-block adaptive quant map using libjxl's full pipeline.
/// Returns (raw_quant_map, global_scale, quant_lf).
fn build_adaptive_raw_quant_map_full(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    img_w: usize,
    img_h: usize,
    bw: usize,
    bh: usize,
    distance: f32,
    quant_ac: f32,
) -> (Vec<u8>, u32, u32, Vec<f32>) {
    let num_blocks = bw * bh;
    if distance < 1.0 || num_blocks < 64 {
        let (gs, qlf, base) = distance_to_full_quant_params(distance);
        let q = 0.79 / distance;
        return (vec![base; num_blocks], gs, qlf, vec![q; num_blocks]);
    }

    // Step 1: Compute initial aq_map from Y-channel Laplacian masking
    let mut aq_map = compute_aq_map(xyb_y, img_w, img_h, bw, bh, distance);

    // Step 2: Apply per-block modulations (ComputeMask + Gamma + HF + Blue).
    // libjxl PerBlockModulations uses quant_ac as the `scale` parameter.
    let scale = quant_ac;
    apply_per_block_modulations(
        &mut aq_map,
        xyb_x,
        xyb_y,
        xyb_b,
        img_w,
        img_h,
        bw,
        bh,
        distance,
        scale,
    );

    // Step 3: Convert float aq_map to raw_quant (integer).
    // libjxl Squirrel path: ComputeGlobalScaleAndQuant(quant_dc, 0.39/d, 0)
    // sets global_scale from 0.39/d with absd=0, NOT from the AQ map median.
    // Then SetQuantFieldRect converts each aq_map[i] to
    //   raw_quant = clamp(aq_map[i] * inv_global_scale + 0.5, 1, 255).
    let q_for_global_scale = 0.39 / distance;
    let (global_scale, quant_lf, _) = compute_global_scale_and_quant(distance, q_for_global_scale);
    let inv_global_scale = 65536.0 / global_scale as f32;

    let mut raw_quant_map = Vec::with_capacity(num_blocks);
    for &v in &aq_map {
        let rq = (v * inv_global_scale + 0.5).clamp(1.0, 255.0) as u8;
        raw_quant_map.push(rq);
    }

    // Return aq_map as well -- libjxl's EstimateEntropy uses the float
    // quant field values (not integer raw_quant) for quant_norm16.
    (raw_quant_map, global_scale, quant_lf, aq_map)
}

/// Port of libjxl's `AdjustQuantField`: for merged (non-8x8) blocks, replace
/// all constituent 8x8 blocks' raw_quant with the MAX of the group.
/// At d <= 1.54, uses pure max. At higher d, interpolates towards mean.
/// This ensures merged blocks use the finest quantization step among their
/// constituents, preventing quality loss from coarse-quant blocks in the group.
fn adjust_quant_field(
    raw_quant_map: &[u8],
    transform_map: &[u8],
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<u8> {
    let mut adjusted = raw_quant_map.to_vec();

    // libjxl constants for max/mean interpolation
    const K_LIMIT: f32 = 1.54138;
    const K_MUL: f32 = 0.56391;
    const K_MIN: f32 = 0.0;
    let mean_max_mixer = if distance > K_LIMIT {
        (1.0 - (distance - K_LIMIT) * K_MUL).max(K_MIN)
    } else {
        1.0
    };

    for by in 0..bh {
        for bx in 0..bw {
            let idx = by * bw + bx;
            let t = transform_map[idx];
            if (t & TRANSFORM_FIRST_BLOCK_FLAG) == 0 {
                continue;
            }
            let tid_raw = (t & !TRANSFORM_FIRST_BLOCK_FLAG) as usize;
            let Some(tid) = HfTransformType::from_usize(tid_raw) else {
                continue;
            };
            let cbx = covered_blocks_x(tid) as usize;
            let cby = covered_blocks_y(tid) as usize;
            if cbx == 1 && cby == 1 {
                continue; // 8x8 block, nothing to adjust
            }

            // Compute max and mean of constituent blocks
            let mut max_val = 0u8;
            let mut sum = 0u32;
            let count = (cbx * cby) as u32;
            for iy in 0..cby {
                for ix in 0..cbx {
                    let qi = (by + iy) * bw + (bx + ix);
                    if qi < raw_quant_map.len() {
                        let v = raw_quant_map[qi];
                        max_val = max_val.max(v);
                        sum += v as u32;
                    }
                }
            }
            let mean = sum as f32 / count as f32;

            // Interpolate between max and mean
            let result = if count >= 4 {
                let max_f = max_val as f32;
                max_f * mean_max_mixer + (1.0 - mean_max_mixer) * mean
            } else {
                max_val as f32
            };
            let rq = (result + 0.5).clamp(1.0, 255.0) as u8;

            // Set all constituent blocks to the adjusted value
            for iy in 0..cby {
                for ix in 0..cbx {
                    let qi = (by + iy) * bw + (bx + ix);
                    if qi < adjusted.len() {
                        adjusted[qi] = rq;
                    }
                }
            }
        }
    }

    adjusted
}

fn build_default_transform_map(bw: usize, bh: usize) -> Vec<u8> {
    vec![DCT8_TRANSFORM_ID | TRANSFORM_FIRST_BLOCK_FLAG; bw * bh]
}

fn quantized_block_ac_all_zero(ac: &[i32], block_idx: usize) -> bool {
    ac[block_idx * 64 + 1..block_idx * 64 + 64]
        .iter()
        .all(|&v| v == 0)
}

fn quantized_transform_region_ac_all_zero(
    ac: &[i32],
    bw: usize,
    bx: usize,
    by: usize,
    cx: usize,
    cy: usize,
) -> bool {
    for iy in 0..cy {
        for ix in 0..cx {
            let block_idx = (by + iy) * bw + (bx + ix);
            if !quantized_block_ac_all_zero(ac, block_idx) {
                return false;
            }
        }
    }
    true
}

fn build_zero_merge_transform_map(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    transform_priority: &[u8],
) -> Vec<u8> {
    let mut transform_map = build_default_transform_map(bw, bh);
    let mut covered = vec![false; bw * bh];

    let fits_hf_group = |bx: usize, by: usize, cx: usize, cy: usize| {
        bx / 32 == (bx + cx - 1) / 32 && by / 32 == (by + cy - 1) / 32
    };

    for &transform_id in transform_priority {
        let Some(transform_type) = HfTransformType::from_usize(transform_id as usize) else {
            continue;
        };
        let cx = covered_blocks_x(transform_type) as usize;
        let cy = covered_blocks_y(transform_type) as usize;
        if cx == 1 && cy == 1 {
            continue;
        }

        for by in 0..bh {
            for bx in 0..bw {
                if bx + cx > bw || by + cy > bh || !fits_hf_group(bx, by, cx, cy) {
                    continue;
                }

                let mut free = true;
                'free_check: for iy in 0..cy {
                    for ix in 0..cx {
                        if covered[(by + iy) * bw + (bx + ix)] {
                            free = false;
                            break 'free_check;
                        }
                    }
                }
                if !free {
                    continue;
                }

                let all_zero = quantized_transform_region_ac_all_zero(ac_x, bw, bx, by, cx, cy)
                    && quantized_transform_region_ac_all_zero(ac_y, bw, bx, by, cx, cy)
                    && quantized_transform_region_ac_all_zero(ac_b, bw, bx, by, cx, cy);
                if !all_zero {
                    continue;
                }

                for iy in 0..cy {
                    for ix in 0..cx {
                        let idx = (by + iy) * bw + (bx + ix);
                        transform_map[idx] = if ix == 0 && iy == 0 {
                            TRANSFORM_FIRST_BLOCK_FLAG | transform_id
                        } else {
                            transform_id
                        };
                        covered[idx] = true;
                    }
                }
            }
        }
    }

    transform_map
}

fn quantized_transform_region_abs_sum(
    ac: &[i32],
    bw: usize,
    bx: usize,
    by: usize,
    cx: usize,
    cy: usize,
) -> u64 {
    let mut sum = 0u64;
    for iy in 0..cy {
        for ix in 0..cx {
            let block_idx = (by + iy) * bw + (bx + ix);
            let base = block_idx * 64;
            for &v in &ac[base + 1..base + 64] {
                sum += v.unsigned_abs() as u64;
            }
        }
    }
    sum
}

fn build_low_energy_merge_transform_map(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    transform_priority: &[u8],
    max_abs_sum_per_block: u64,
) -> Vec<u8> {
    let mut transform_map = build_default_transform_map(bw, bh);
    let mut covered = vec![false; bw * bh];

    let fits_hf_group = |bx: usize, by: usize, cx: usize, cy: usize| {
        bx / 32 == (bx + cx - 1) / 32 && by / 32 == (by + cy - 1) / 32
    };

    for &transform_id in transform_priority {
        let Some(transform_type) = HfTransformType::from_usize(transform_id as usize) else {
            continue;
        };
        let cx = covered_blocks_x(transform_type) as usize;
        let cy = covered_blocks_y(transform_type) as usize;
        if cx == 1 && cy == 1 {
            continue;
        }

        let region_budget = max_abs_sum_per_block * (cx * cy) as u64;
        for by in 0..bh {
            for bx in 0..bw {
                if bx + cx > bw || by + cy > bh || !fits_hf_group(bx, by, cx, cy) {
                    continue;
                }

                let mut free = true;
                'free_check: for iy in 0..cy {
                    for ix in 0..cx {
                        if covered[(by + iy) * bw + (bx + ix)] {
                            free = false;
                            break 'free_check;
                        }
                    }
                }
                if !free {
                    continue;
                }

                let energy = quantized_transform_region_abs_sum(ac_x, bw, bx, by, cx, cy)
                    + quantized_transform_region_abs_sum(ac_y, bw, bx, by, cx, cy)
                    + quantized_transform_region_abs_sum(ac_b, bw, bx, by, cx, cy);
                if energy > region_budget {
                    continue;
                }

                for iy in 0..cy {
                    for ix in 0..cx {
                        let idx = (by + iy) * bw + (bx + ix);
                        transform_map[idx] = if ix == 0 && iy == 0 {
                            TRANSFORM_FIRST_BLOCK_FLAG | transform_id
                        } else {
                            transform_id
                        };
                        covered[idx] = true;
                    }
                }
            }
        }
    }

    transform_map
}

// NOTE: this is intentionally scalar for now.
// SIMD forward-transform acceleration is deferred until algorithmic parity work
// (transform selection + quant modeling + entropy contexting) stabilizes.
fn forward_dct_1d_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let n = input.len();
    let inv_n = 1.0f32 / n as f32;
    for (k, out) in output.iter_mut().enumerate() {
        let scale = if k == 0 {
            inv_n
        } else {
            std::f32::consts::SQRT_2 * inv_n
        };
        let mut sum = 0.0f32;
        for (i, &v) in input.iter().enumerate() {
            let angle = std::f32::consts::PI * ((2 * i + 1) * k) as f32 / (2.0 * n as f32);
            sum += v * angle.cos();
        }
        *out = sum * scale;
    }
}

fn forward_dct2d_scalar(block: &mut [f32], width: usize, height: usize) {
    assert_eq!(block.len(), width * height);

    if width == height {
        // Match the layout convention used by jxl_transforms IDCT implementations:
        // row transform -> transpose -> row transform (no final transpose).
        let n = width;
        let mut src = vec![0.0f32; n];
        let mut dst = vec![0.0f32; n];

        for row in 0..n {
            let start = row * n;
            src.copy_from_slice(&block[start..start + n]);
            forward_dct_1d_scalar(&src, &mut dst);
            block[start..start + n].copy_from_slice(&dst);
        }

        for i in 0..n {
            for j in i + 1..n {
                block.swap(i * n + j, j * n + i);
            }
        }

        for row in 0..n {
            let start = row * n;
            src.copy_from_slice(&block[start..start + n]);
            forward_dct_1d_scalar(&src, &mut dst);
            block[start..start + n].copy_from_slice(&dst);
        }
        return;
    }

    // Rectangular fallback: row pass + column pass.
    let mut row_src = vec![0.0f32; width];
    let mut row_dst = vec![0.0f32; width];
    let mut tmp = vec![0.0f32; width * height];
    for y in 0..height {
        let start = y * width;
        row_src.copy_from_slice(&block[start..start + width]);
        forward_dct_1d_scalar(&row_src, &mut row_dst);
        tmp[start..start + width].copy_from_slice(&row_dst);
    }

    let mut col_src = vec![0.0f32; height];
    let mut col_dst = vec![0.0f32; height];
    for x in 0..width {
        for y in 0..height {
            col_src[y] = tmp[y * width + x];
        }
        forward_dct_1d_scalar(&col_src, &mut col_dst);
        for y in 0..height {
            block[y * width + x] = col_dst[y];
        }
    }
}

fn gather_clamped_block(
    chan: &[f32],
    width: usize,
    height: usize,
    px0: usize,
    py0: usize,
    block_w: usize,
    block_h: usize,
) -> Vec<f32> {
    let mut block = vec![0.0f32; block_w * block_h];
    for y in 0..block_h {
        for x in 0..block_w {
            let sx = (px0 + x).min(width - 1);
            let sy = (py0 + y).min(height - 1);
            block[y * block_w + x] = chan[sy * width + sx];
        }
    }
    block
}

fn transform_coeff_index_to_block_storage(
    full_bw: usize,
    bx: usize,
    by: usize,
    cx: usize,
    coeff_index: usize,
) -> usize {
    let xsize = cx * 8;
    let x = coeff_index % xsize;
    let y = coeff_index / xsize;

    let block_x = x / 8;
    let block_y = y / 8;
    let inner_x = x % 8;
    let inner_y = y % 8;

    let global_block_idx = (by + block_y) * full_bw + (bx + block_x);
    global_block_idx * 64 + inner_y * 8 + inner_x
}

fn zero_transform_region_ac(
    ac: &mut [i32],
    full_bw: usize,
    bx: usize,
    by: usize,
    cx: usize,
    cy: usize,
) {
    for iy in 0..cy {
        for ix in 0..cx {
            let bidx = (by + iy) * full_bw + (bx + ix);
            let base = bidx * 64;
            for coeff in &mut ac[base + 1..base + 64] {
                *coeff = 0;
            }
        }
    }
}

fn zero_non_dct8_ac_coeffs(
    ac: &mut [i32],
    transform_map: &[u8],
    bw: usize,
    bh: usize,
) -> Result<()> {
    for by in 0..bh {
        for bx in 0..bw {
            let idx = by * bw + bx;
            let raw_transform = transform_map[idx];
            if raw_transform & TRANSFORM_FIRST_BLOCK_FLAG == 0 {
                continue;
            }

            let transform_id = raw_transform & !TRANSFORM_FIRST_BLOCK_FLAG;
            if transform_id == DCT8_TRANSFORM_ID {
                continue;
            }

            let transform_type = HfTransformType::from_usize(transform_id as usize).ok_or(
                crate::error::Error::InvalidVarDCTTransform(transform_id as usize),
            )?;
            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            if bx + cx > bw || by + cy > bh {
                return Err(crate::error::Error::HFBlockOutOfBounds);
            }

            zero_transform_region_ac(ac, bw, bx, by, cx, cy);
        }
    }
    Ok(())
}

type ForwardTransformCoeffCache = std::collections::HashMap<(u8, usize, usize), [Vec<f32>; 3]>;

static SPECIAL_8X8_FORWARD_INVERSE_MATRICES: [std::sync::OnceLock<Vec<f32>>; 9] = [
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
];

fn special_8x8_transform_index(transform_id: u8) -> Option<(usize, HfTransformType)> {
    Some(match transform_id {
        IDENTITY_TRANSFORM_ID => (0usize, HfTransformType::IDENTITY),
        DCT2X2_TRANSFORM_ID => (1usize, HfTransformType::DCT2X2),
        DCT4X4_TRANSFORM_ID => (2usize, HfTransformType::DCT4X4),
        DCT4X8_TRANSFORM_ID => (3usize, HfTransformType::DCT4X8),
        DCT8X4_TRANSFORM_ID => (4usize, HfTransformType::DCT8X4),
        AFV0_TRANSFORM_ID => (5usize, HfTransformType::AFV0),
        AFV1_TRANSFORM_ID => (6usize, HfTransformType::AFV1),
        AFV2_TRANSFORM_ID => (7usize, HfTransformType::AFV2),
        AFV3_TRANSFORM_ID => (8usize, HfTransformType::AFV3),
        _ => return None,
    })
}

fn is_special_8x8_transform_id(transform_id: u8) -> bool {
    special_8x8_transform_index(transform_id).is_some()
}

#[cfg(test)]
fn forward_dct4x8_from_8x8(block: &[f32]) -> Vec<f32> {
    forward_special_8x8_from_8x8(block, DCT4X8_TRANSFORM_ID)
}

#[cfg(test)]
fn forward_dct8x4_from_8x8(block: &[f32]) -> Vec<f32> {
    forward_special_8x8_from_8x8(block, DCT8X4_TRANSFORM_ID)
}

#[cfg(test)]
fn forward_dct2x2_from_8x8(block: &[f32]) -> Vec<f32> {
    forward_special_8x8_from_8x8(block, DCT2X2_TRANSFORM_ID)
}

#[cfg(test)]
fn forward_dct4x4_from_8x8(block: &[f32]) -> Vec<f32> {
    forward_special_8x8_from_8x8(block, DCT4X4_TRANSFORM_ID)
}

#[cfg(test)]
fn forward_identity_from_8x8(block: &[f32]) -> Vec<f32> {
    forward_special_8x8_from_8x8(block, IDENTITY_TRANSFORM_ID)
}

fn invert_square_matrix_f64(a: &[f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n);
    let mut aug = vec![0.0f64; n * 2 * n];
    for r in 0..n {
        for c in 0..n {
            aug[r * 2 * n + c] = a[r * n + c];
        }
        aug[r * 2 * n + n + r] = 1.0;
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = aug[col * 2 * n + col].abs();
        for r in col + 1..n {
            let v = aug[r * 2 * n + col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot_row = r;
            }
        }
        if pivot_abs < 1e-12 {
            return None;
        }

        if pivot_row != col {
            for c in 0..2 * n {
                aug.swap(col * 2 * n + c, pivot_row * 2 * n + c);
            }
        }

        let pivot = aug[col * 2 * n + col];
        for c in 0..2 * n {
            aug[col * 2 * n + c] /= pivot;
        }

        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r * 2 * n + col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..2 * n {
                aug[r * 2 * n + c] -= factor * aug[col * 2 * n + c];
            }
        }
    }

    let mut inv = vec![0.0f64; n * n];
    for r in 0..n {
        for c in 0..n {
            inv[r * n + c] = aug[r * 2 * n + n + c];
        }
    }
    Some(inv)
}

fn build_forward_inverse_matrix(transform: HfTransformType) -> Vec<f32> {
    let mut matrix = vec![0.0f64; 64 * 64];

    for j in 0..64 {
        let mut lf = vec![0.0f32; 1];
        let mut coeffs = vec![0.0f32; 64];
        if j == 0 {
            lf[0] = 1.0;
        } else {
            coeffs[j] = 1.0;
        }
        transform_to_pixels(transform, &mut lf, &mut coeffs);
        for i in 0..64 {
            matrix[i * 64 + j] = coeffs[i] as f64;
        }
    }

    let inv = invert_square_matrix_f64(&matrix, 64)
        .expect("forward matrix should be invertible for supported 8x8 block transforms");
    inv.into_iter().map(|v| v as f32).collect()
}

fn special_8x8_forward_inverse_matrix(transform_id: u8) -> &'static [f32] {
    let (idx, transform) = special_8x8_transform_index(transform_id)
        .expect("special_8x8_forward_inverse_matrix called for non-special transform id");
    SPECIAL_8X8_FORWARD_INVERSE_MATRICES[idx]
        .get_or_init(|| build_forward_inverse_matrix(transform))
        .as_slice()
}

fn forward_special_8x8_from_8x8(block: &[f32], transform_id: u8) -> Vec<f32> {
    debug_assert_eq!(block.len(), 64);
    let inv = special_8x8_forward_inverse_matrix(transform_id);
    let mut coeffs = vec![0.0f32; 64];
    for r in 0..64 {
        let mut sum = 0.0f32;
        for c in 0..64 {
            sum += inv[r * 64 + c] * block[c];
        }
        coeffs[r] = sum;
    }
    coeffs
}

#[cfg(test)]
fn forward_afv_from_8x8(block: &[f32], transform_id: u8) -> Vec<f32> {
    forward_special_8x8_from_8x8(block, transform_id)
}

struct TransformLinearForwardSolver {
    coeff_count: usize,
    lf_count: usize,
    hf_coeff_indices: Vec<usize>,
    inverse: Vec<f32>,
}

const RECTANGULAR_SOLVER_MAX_BLOCKS: usize = 1024;
const SQUARE_SOLVER_MAX_BLOCKS: usize = 1024;

static SQUARE_FORWARD_SOLVERS: [std::sync::OnceLock<TransformLinearForwardSolver>; 2] =
    [std::sync::OnceLock::new(), std::sync::OnceLock::new()];

static RECTANGULAR_FORWARD_SOLVERS: [std::sync::OnceLock<TransformLinearForwardSolver>; 6] = [
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
    std::sync::OnceLock::new(),
];

fn square_transform_index(transform_id: u8) -> Option<(usize, HfTransformType)> {
    Some(match transform_id {
        DCT16_TRANSFORM_ID => (0usize, HfTransformType::DCT16X16),
        DCT32_TRANSFORM_ID => (1usize, HfTransformType::DCT32X32),
        _ => return None,
    })
}

fn rectangular_transform_index(transform_id: u8) -> Option<(usize, HfTransformType)> {
    Some(match transform_id {
        DCT16X8_TRANSFORM_ID => (0usize, HfTransformType::DCT16X8),
        DCT8X16_TRANSFORM_ID => (1usize, HfTransformType::DCT8X16),
        DCT32X8_TRANSFORM_ID => (2usize, HfTransformType::DCT32X8),
        DCT8X32_TRANSFORM_ID => (3usize, HfTransformType::DCT8X32),
        DCT32X16_TRANSFORM_ID => (4usize, HfTransformType::DCT32X16),
        DCT16X32_TRANSFORM_ID => (5usize, HfTransformType::DCT16X32),
        _ => return None,
    })
}

fn build_linear_forward_solver(transform: HfTransformType) -> TransformLinearForwardSolver {
    let cx = covered_blocks_x(transform) as usize;
    let cy = covered_blocks_y(transform) as usize;
    let coeff_count = cx * cy * 64;
    let lf_count = cx * cy;

    // Identify the LF-overwritten coefficient indices by canonical shape order.
    let shape_id = block_shape_id(transform) as usize;
    let canonical = canonical_transform_for_shape_id(shape_id)
        .expect("missing canonical transform mapping for shape id");
    let mut lowfreq_indices = natural_coeff_order_for_transform(canonical)[..lf_count].to_vec();
    lowfreq_indices.sort_unstable();

    let mut is_lowfreq = vec![false; coeff_count];
    for idx in lowfreq_indices {
        is_lowfreq[idx] = true;
    }
    let hf_coeff_indices: Vec<usize> = (0..coeff_count).filter(|&i| !is_lowfreq[i]).collect();

    let mut matrix = vec![0.0f64; coeff_count * coeff_count];
    for j in 0..coeff_count {
        let mut lf = vec![0.0f32; lf_count];
        let mut coeffs = vec![0.0f32; coeff_count];
        if j < lf_count {
            lf[j] = 1.0;
        } else {
            coeffs[hf_coeff_indices[j - lf_count]] = 1.0;
        }
        transform_to_pixels(transform, &mut lf, &mut coeffs);
        for i in 0..coeff_count {
            matrix[i * coeff_count + j] = coeffs[i] as f64;
        }
    }

    let inv = invert_square_matrix_f64(&matrix, coeff_count)
        .expect("forward matrix should be invertible for linear transform solver")
        .into_iter()
        .map(|v| v as f32)
        .collect();

    TransformLinearForwardSolver {
        coeff_count,
        lf_count,
        hf_coeff_indices,
        inverse: inv,
    }
}

fn square_forward_solver(transform_id: u8) -> Option<&'static TransformLinearForwardSolver> {
    let (idx, transform) = square_transform_index(transform_id)?;
    Some(SQUARE_FORWARD_SOLVERS[idx].get_or_init(|| build_linear_forward_solver(transform)))
}

fn rectangular_forward_solver(transform_id: u8) -> Option<&'static TransformLinearForwardSolver> {
    let (idx, transform) = rectangular_transform_index(transform_id)?;
    Some(RECTANGULAR_FORWARD_SOLVERS[idx].get_or_init(|| build_linear_forward_solver(transform)))
}

/// Forward 1D DCT of size N, matching jxl's basis convention:
///   B[0][n] = 1               (DC)
///   B[k][n] = sqrt(2) * cos(pi*(2n+1)*k/(2N))  (AC, k>0)
///
/// Forward: c[0] = (1/N) * sum_n x[n]
///          c[k] = (sqrt(2)/N) * sum_n x[n] * cos(pi*(2n+1)*k/(2N))
#[allow(dead_code)]
fn dct_1d_n(input: &[f32], output: &mut [f32], n: usize) {
    let inv_n = 1.0 / n as f32;
    let ac_scale = std::f32::consts::SQRT_2 * inv_n;
    // DC
    output[0] = input.iter().sum::<f32>() * inv_n;
    // AC
    for k in 1..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += input[i]
                * (std::f32::consts::PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32).cos();
        }
        output[k] = sum * ac_scale;
    }
}

/// Forward DCT NxN using separable 1D DCTs with jxl basis normalization.
/// Matches libjxl's ComputeScaledDCT<N,N> + the decoder's inverse.
#[allow(dead_code)]
fn forward_dct_nxn(pixels: &[f32], coeffs: &mut [f32], n: usize) {
    let mut temp = vec![0.0f32; n * n];
    let mut row_in = vec![0.0f32; n];
    let mut row_out = vec![0.0f32; n];

    // Row-wise DCT
    for y in 0..n {
        row_in.copy_from_slice(&pixels[y * n..(y + 1) * n]);
        dct_1d_n(&row_in, &mut row_out, n);
        temp[y * n..(y + 1) * n].copy_from_slice(&row_out);
    }

    // Transpose
    let mut transposed = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            transposed[x * n + y] = temp[y * n + x];
        }
    }

    // Column-wise DCT (on transposed = now rows)
    for y in 0..n {
        row_in.copy_from_slice(&transposed[y * n..(y + 1) * n]);
        dct_1d_n(&row_in, &mut row_out, n);
        // Store back transposed
        for x in 0..n {
            coeffs[x * n + y] = row_out[x];
        }
    }
}

fn forward_with_linear_solver(block: &[f32], solver: &TransformLinearForwardSolver) -> Vec<f32> {
    debug_assert_eq!(block.len(), solver.coeff_count);
    let n = solver.coeff_count;

    let mut solved = vec![0.0f32; n];
    for r in 0..n {
        let mut sum = 0.0f32;
        for c in 0..n {
            sum += solver.inverse[r * n + c] * block[c];
        }
        solved[r] = sum;
    }

    let mut coeffs = vec![0.0f32; n];
    for (i, &coeff_index) in solver.hf_coeff_indices.iter().enumerate() {
        coeffs[coeff_index] = solved[solver.lf_count + i];
    }
    coeffs
}

fn compute_forward_transform_coeffs(
    transform_id: u8,
    x_chan: &[f32],
    y_chan: &[f32],
    b_minus_y_chan: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block_w: usize,
    block_h: usize,
) -> [Vec<f32>; 3] {
    if is_special_8x8_transform_id(transform_id) {
        let block_x = gather_clamped_block(x_chan, width, height, bx * 8, by * 8, 8, 8);
        let block_y = gather_clamped_block(y_chan, width, height, bx * 8, by * 8, 8, 8);
        let block_b = gather_clamped_block(b_minus_y_chan, width, height, bx * 8, by * 8, 8, 8);

        let synth = |block: &[f32]| forward_special_8x8_from_8x8(block, transform_id);

        return [synth(&block_x), synth(&block_y), synth(&block_b)];
    }

    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);

    let allow_expensive_square_solver = bw * bh <= SQUARE_SOLVER_MAX_BLOCKS;
    let use_square_solver = if transform_id == DCT32_TRANSFORM_ID {
        allow_expensive_square_solver
    } else {
        transform_id == DCT16_TRANSFORM_ID
    };
    if use_square_solver {
        if let Some(solver) = square_forward_solver(transform_id) {
            let block_x =
                gather_clamped_block(x_chan, width, height, bx * 8, by * 8, block_w, block_h);
            let block_y =
                gather_clamped_block(y_chan, width, height, bx * 8, by * 8, block_w, block_h);
            let block_b = gather_clamped_block(
                b_minus_y_chan,
                width,
                height,
                bx * 8,
                by * 8,
                block_w,
                block_h,
            );
            debug_assert_eq!(block_w * block_h, solver.coeff_count);
            let synth = |block: &[f32]| forward_with_linear_solver(block, solver);
            return [synth(&block_x), synth(&block_y), synth(&block_b)];
        }
    }

    let allow_expensive_rect_solver = bw * bh <= RECTANGULAR_SOLVER_MAX_BLOCKS;
    let use_rect_solver = if matches!(transform_id, DCT32X16_TRANSFORM_ID | DCT16X32_TRANSFORM_ID) {
        allow_expensive_rect_solver
    } else {
        true
    };

    if use_rect_solver {
        if let Some(solver) = rectangular_forward_solver(transform_id) {
            let block_x =
                gather_clamped_block(x_chan, width, height, bx * 8, by * 8, block_w, block_h);
            let block_y =
                gather_clamped_block(y_chan, width, height, bx * 8, by * 8, block_w, block_h);
            let block_b = gather_clamped_block(
                b_minus_y_chan,
                width,
                height,
                bx * 8,
                by * 8,
                block_w,
                block_h,
            );
            debug_assert_eq!(block_w * block_h, solver.coeff_count);
            let synth = |block: &[f32]| forward_with_linear_solver(block, solver);
            return [synth(&block_x), synth(&block_y), synth(&block_b)];
        }
    }

    let mut block_x = gather_clamped_block(x_chan, width, height, bx * 8, by * 8, block_w, block_h);
    let mut block_y = gather_clamped_block(y_chan, width, height, bx * 8, by * 8, block_w, block_h);
    let mut block_b = gather_clamped_block(
        b_minus_y_chan,
        width,
        height,
        bx * 8,
        by * 8,
        block_w,
        block_h,
    );

    forward_dct2d_scalar(&mut block_x, block_w, block_h);
    forward_dct2d_scalar(&mut block_y, block_w, block_h);
    forward_dct2d_scalar(&mut block_b, block_w, block_h);

    [block_x, block_y, block_b]
}

#[allow(dead_code, clippy::too_many_arguments)]
fn prepare_ac_for_transform_map(
    ac_x_base: &[i32],
    ac_y_base: &[i32],
    ac_b_base: &[i32],
    x_chan: &[f32],
    y_chan: &[f32],
    b_minus_y_chan: &[f32],
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
    global_scale: u32,
    raw_quant_map: &[u8],
    transform_map: &[u8],
    x_dm_multiplier: f32,
    b_dm_multiplier: f32,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    prepare_ac_for_transform_map_with_cache(
        ac_x_base,
        ac_y_base,
        ac_b_base,
        x_chan,
        y_chan,
        b_minus_y_chan,
        width,
        height,
        bw,
        bh,
        global_scale,
        raw_quant_map,
        transform_map,
        None,
        x_dm_multiplier,
        b_dm_multiplier,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_ac_for_transform_map_with_cache(
    ac_x_base: &[i32],
    ac_y_base: &[i32],
    ac_b_base: &[i32],
    x_chan: &[f32],
    y_chan: &[f32],
    b_minus_y_chan: &[f32],
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
    global_scale: u32,
    raw_quant_map: &[u8],
    transform_map: &[u8],
    forward_cache: Option<&mut ForwardTransformCoeffCache>,
    x_dm_multiplier: f32,
    b_dm_multiplier: f32,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    assert_eq!(ac_x_base.len(), bw * bh * 64);
    assert_eq!(ac_y_base.len(), bw * bh * 64);
    assert_eq!(ac_b_base.len(), bw * bh * 64);
    assert_eq!(raw_quant_map.len(), bw * bh);
    assert_eq!(transform_map.len(), bw * bh);

    let mut ac_x = ac_x_base.to_vec();
    let mut ac_y = ac_y_base.to_vec();
    let mut ac_b = ac_b_base.to_vec();
    let mut forward_cache = forward_cache;

    let identity_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(1);
    let dct2_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(2);
    let dct4_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(3);
    let dct16_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(4);
    let dct32_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(5);
    let dct8x16_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(6);
    let dct8x32_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(7);
    let dct16x32_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(8);
    let dct4x8_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(9);
    let afv_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(10);
    let dct64_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(11);
    let dct32x64_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(12);
    let dct128_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(13);
    let dct64x128_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(14);
    let dct256_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(15);
    let dct128x256_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(16);

    for by in 0..bh {
        for bx in 0..bw {
            let idx = by * bw + bx;
            let raw_transform = transform_map[idx];
            if raw_transform & TRANSFORM_FIRST_BLOCK_FLAG == 0 {
                continue;
            }

            let transform_id = raw_transform & !TRANSFORM_FIRST_BLOCK_FLAG;
            if transform_id == DCT8_TRANSFORM_ID {
                continue;
            }

            let transform_type = HfTransformType::from_usize(transform_id as usize).ok_or(
                crate::error::Error::InvalidVarDCTTransform(transform_id as usize),
            )?;
            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            if bx + cx > bw || by + cy > bh {
                return Err(crate::error::Error::HFBlockOutOfBounds);
            }

            let (block_w, block_h, coeff_count, weights) = match transform_id {
                IDENTITY_TRANSFORM_ID => (8usize, 8usize, 64usize, identity_weights),
                DCT2X2_TRANSFORM_ID => (8usize, 8usize, 64usize, dct2_weights),
                DCT4X4_TRANSFORM_ID => (8usize, 8usize, 64usize, dct4_weights),
                DCT16_TRANSFORM_ID => (16usize, 16usize, 256usize, dct16_weights),
                DCT32_TRANSFORM_ID => (32usize, 32usize, 1024usize, dct32_weights),
                DCT16X8_TRANSFORM_ID => (8usize, 16usize, 128usize, dct8x16_weights),
                DCT8X16_TRANSFORM_ID => (16usize, 8usize, 128usize, dct8x16_weights),
                DCT32X8_TRANSFORM_ID => (8usize, 32usize, 256usize, dct8x32_weights),
                DCT8X32_TRANSFORM_ID => (32usize, 8usize, 256usize, dct8x32_weights),
                DCT32X16_TRANSFORM_ID => (16usize, 32usize, 512usize, dct16x32_weights),
                DCT16X32_TRANSFORM_ID => (32usize, 16usize, 512usize, dct16x32_weights),
                DCT4X8_TRANSFORM_ID => (8usize, 8usize, 64usize, dct4x8_weights),
                DCT8X4_TRANSFORM_ID => (8usize, 8usize, 64usize, dct4x8_weights),
                AFV0_TRANSFORM_ID => (8usize, 8usize, 64usize, afv_weights),
                AFV1_TRANSFORM_ID => (8usize, 8usize, 64usize, afv_weights),
                AFV2_TRANSFORM_ID => (8usize, 8usize, 64usize, afv_weights),
                AFV3_TRANSFORM_ID => (8usize, 8usize, 64usize, afv_weights),
                DCT64_TRANSFORM_ID => (64usize, 64usize, 4096usize, dct64_weights),
                DCT64X32_TRANSFORM_ID => (32usize, 64usize, 2048usize, dct32x64_weights),
                DCT32X64_TRANSFORM_ID => (64usize, 32usize, 2048usize, dct32x64_weights),
                DCT128_TRANSFORM_ID => (128usize, 128usize, 16384usize, dct128_weights),
                DCT128X64_TRANSFORM_ID => (64usize, 128usize, 8192usize, dct64x128_weights),
                DCT64X128_TRANSFORM_ID => (128usize, 64usize, 8192usize, dct64x128_weights),
                DCT256_TRANSFORM_ID => (256usize, 256usize, 65536usize, dct256_weights),
                DCT256X128_TRANSFORM_ID => (128usize, 256usize, 32768usize, dct128x256_weights),
                DCT128X256_TRANSFORM_ID => (256usize, 128usize, 32768usize, dct128x256_weights),
                _ => {
                    return Err(crate::error::Error::InvalidVarDCTTransform(
                        transform_id as usize,
                    ));
                }
            };

            let wx = &weights[..coeff_count];
            let wy = &weights[coeff_count..2 * coeff_count];
            let wb = &weights[2 * coeff_count..3 * coeff_count];

            let raw_quant = raw_quant_map[idx].max(1) as u32;

            let coeffs_owned;
            let coeffs = if let Some(cache) = forward_cache.as_mut() {
                let key = (transform_id, bx, by);
                if !cache.contains_key(&key) {
                    cache.insert(
                        key,
                        compute_forward_transform_coeffs(
                            transform_id,
                            x_chan,
                            y_chan,
                            b_minus_y_chan,
                            width,
                            height,
                            bx,
                            by,
                            block_w,
                            block_h,
                        ),
                    );
                }
                cache.get(&key).unwrap()
            } else {
                coeffs_owned = compute_forward_transform_coeffs(
                    transform_id,
                    x_chan,
                    y_chan,
                    b_minus_y_chan,
                    width,
                    height,
                    bx,
                    by,
                    block_w,
                    block_h,
                );
                &coeffs_owned
            };

            for coeff_index in 0..coeff_count {
                let storage_index =
                    transform_coeff_index_to_block_storage(bw, bx, by, cx, coeff_index);
                let dw_x = wx[coeff_index] * x_dm_multiplier;
                let dw_y = wy[coeff_index];
                let dw_b = wb[coeff_index] * b_dm_multiplier;
                ac_x[storage_index] =
                    quantize_ac(coeffs[0][coeff_index], global_scale, raw_quant, dw_x);
                ac_y[storage_index] =
                    quantize_ac(coeffs[1][coeff_index], global_scale, raw_quant, dw_y);
                ac_b[storage_index] =
                    quantize_ac(coeffs[2][coeff_index], global_scale, raw_quant, dw_b);
            }
        }
    }

    Ok((ac_x, ac_y, ac_b))
}

fn build_transform_map_from_quantized_ac(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<u8> {
    if distance >= 1.5 {
        return build_zero_merge_transform_map(ac_x, ac_y, ac_b, bw, bh, &[DCT16_TRANSFORM_ID]);
    }
    build_default_transform_map(bw, bh)
}

/// Full port of libjxl's EstimateEntropy from enc_ac_strategy.cc.
///
/// Combines two components:
///   1. Entropy estimate: sum(sqrt(|quantized|)) + zero-run cost
///   2. Information loss: inverse-transforms quantization error to pixels,
///      weights by perceptual masking, computes L8 norm.
///
/// This prevents merging blocks where quantization would cause visible ringing
/// (the loss term detects it even when entropy looks favorable).
#[allow(dead_code)]
fn estimate_transform_entropy_full(
    coeffs: &[Vec<f32>; 3],
    weights: &[f32],
    coeff_count: usize,
    global_scale: u32,
    raw_quant: u32,
    x_dm_multiplier: f32,
    b_dm_multiplier: f32,
    // Masking field (per-pixel, Y-channel based). If None, loss term is skipped.
    masking1x1: Option<(&[f32], usize)>, // (data, stride=image_width)
    pixel_x_origin: usize,
    pixel_y_origin: usize,
    block_w_pixels: usize,
    block_h_pixels: usize,
    transform_id: u8,
    num_blocks: usize,
    // libjxl config constants (distance-dependent)
    cost_delta: f32,
    zeros_mul: f32,
    info_loss_multiplier: f32,
) -> f32 {
    // libjxl's EstimateEntropy uses: val = coeff * inv_matrix * quant_norm16
    // where quant_norm16 = raw_quant (integer quant field value).
    // Our actual encoding uses: val = coeff * (global_scale * raw_quant / 65536) / dw
    //   = coeff * inv_dw * (gs * rq / 65536)
    // libjxl uses: val = coeff * inv_dw * rq
    // For the loss term to work correctly, we need to match libjxl's scale.
    // But for the entropy term (which decides encoding), we need our actual scale.
    // Solution: use our actual scale for entropy, libjxl's scale for loss.
    let encode_scale = (global_scale as f32 * raw_quant as f32) / 65536.0;
    let quant_norm16 = raw_quant as f32; // libjxl's scale for loss term

    let mut total_entropy = 0.0f32;

    for c in 0..3usize {
        let dw_base = &weights[c * coeff_count..(c + 1) * coeff_count];
        let dm_mul = match c {
            0 => x_dm_multiplier,
            1 => 1.0,
            _ => b_dm_multiplier,
        };
        // CfL: libjxl subtracts Y * cmap_factor from X and B before quantizing.
        // We've already done CfL subtraction for the B channel (pixel_b = b - y).
        // For X channel, cmap_factor is typically 0 (base_correlation_x = 0).
        // So we skip explicit CfL here -- the coefficients already reflect it.

        let mut entropy_v = 0.0f32;
        let mut num_nzeros = 0usize;

        for k in 1..coeff_count {
            let dw = dw_base[k] * dm_mul;
            if dw.abs() < 1e-10 {
                continue;
            }
            let inv_dw = 1.0 / dw;
            // Use our actual encoding scale for quantization (entropy term)
            let val = coeffs[c][k] * encode_scale * inv_dw;
            let rval = val.round();

            let q_abs = rval.abs();
            entropy_v += q_abs.sqrt();
            if q_abs > 0.5 {
                num_nzeros += 1;
            }
        }

        // libjxl: cost for encoding the number of non-zeros
        let nbits = if num_nzeros > 0 {
            (num_nzeros + 1).next_power_of_two().trailing_zeros() + 1
        } else {
            1u32
        };
        total_entropy += cost_delta * entropy_v;
        total_entropy +=
            zeros_mul * ((nbits + 17).next_power_of_two().trailing_zeros() as f32 + nbits as f32);

        // libjxl: X channel penalty for large blocks (ringing in red-green)
        if c == 0 && num_blocks >= 2 {
            let w = 1.0 + (num_blocks as f32 / 8.0).min(3.0);
            total_entropy *= w;
        }
    }

    // Information loss term: compute actual pixel-domain quantization error
    // by quantizing + dequantizing coefficients and inverse-transforming.
    // Weight by perceptual masking field.
    if let Some((mask_data, mask_stride)) = masking1x1 {
        let mask_offsets: [f32; 3] = [12.0, 0.0, 4.0];
        let channel_muls: [f64; 3] = [8.2f64.powi(8), 1.0f64.powi(8), 1.03f64.powi(8)];

        let mut total_loss = 0.0f64;

        for c in 0..3usize {
            // Compute quantization error in DCT domain:
            // error[k] = dequant_weight * (round(val) - val)
            // where val = coeff * encode_scale / dw
            // This is the actual error that the decoder will see.
            let mut error_coeffs = vec![0.0f32; coeff_count];
            let dw_base = &weights[c * coeff_count..(c + 1) * coeff_count];
            let dm_mul = match c {
                0 => x_dm_multiplier,
                1 => 1.0,
                _ => b_dm_multiplier,
            };
            for k in 1..coeff_count {
                let dw = dw_base[k] * dm_mul;
                if dw.abs() < 1e-10 {
                    continue;
                }
                let val = coeffs[c][k] * encode_scale / dw;
                let rval = val.round();
                // Dequantized value = rval * dw / encode_scale
                // Error in pixel-equivalent = (rval * dw / encode_scale - coeffs[c][k])
                // In DCT domain for inverse transform: rval * dw / encode_scale - coeffs[c][k]
                // But we need it in the transform's coefficient space.
                // Actually: error_coeff[k] = (rval - val) * dw / encode_scale
                //   because original = coeffs[c][k], reconstructed = rval * dw / encode_scale
                //   error = reconstructed - original = (rval - val) * dw / encode_scale
                // No wait: original_coeff = coeffs[c][k]
                //   val = coeffs[c][k] * encode_scale / dw
                //   rval = round(val)
                //   reconstructed_coeff = rval * dw / encode_scale
                //   error_coeff = reconstructed - original = (rval - val) * dw / encode_scale
                error_coeffs[k] = (rval - val) * dw / encode_scale;
            }

            // Inverse-transform to get pixel-domain error
            let error_pixels = inverse_transform_error(
                &error_coeffs,
                transform_id,
                block_w_pixels,
                block_h_pixels,
            );

            let masku_off = mask_offsets[c];
            let mut loss_c = 0.0f64;

            for py_local in 0..block_h_pixels {
                for px_local in 0..block_w_pixels {
                    let local_idx = py_local * block_w_pixels + px_local;
                    if local_idx >= error_pixels.len() {
                        continue;
                    }
                    let err = error_pixels[local_idx] as f64;

                    let px = pixel_x_origin + px_local;
                    let py = pixel_y_origin + py_local;
                    let mask_val = if py < mask_stride && px < mask_stride {
                        let midx = py * mask_stride + px;
                        if midx < mask_data.len() {
                            mask_data[midx] as f64
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };

                    let masked = (mask_val + masku_off as f64) * err;
                    let m2 = masked * masked;
                    let m4 = m2 * m2;
                    let m8 = m4 * m4;
                    loss_c += m8;
                }
            }

            loss_c *= channel_muls[c];
            total_loss += loss_c;
        }

        let num_pixels = (num_blocks * 64) as f64;
        let loss_scalar =
            (total_loss / num_pixels).powf(1.0 / 8.0) * num_pixels as f64 / quant_norm16 as f64;
        total_entropy += info_loss_multiplier * loss_scalar as f32;
    }

    total_entropy
}

/// Inverse-transform quantization error from DCT domain to pixel domain.
/// Inverse transform for all 3 channels of an 8x8 block (any special transform type).
/// Returns [x_pixels, y_pixels, b_pixels], each 64 floats.
#[allow(dead_code)]
fn inverse_transform_8x8_all_channels(transform_id: u8, coeffs: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
    use jxl_transforms::transform_map::HfTransformType;

    let transform =
        HfTransformType::from_usize(transform_id as usize).unwrap_or(HfTransformType::DCT);

    let mut result = [vec![0.0f32; 64], vec![0.0f32; 64], vec![0.0f32; 64]];
    for c in 0..3 {
        // For 8x8 transforms: lf is 1 element (DC), hf is 64 elements (full block)
        // transform_to_pixels places lf[0] into hf[0] then runs inverse transform
        let mut lf = [coeffs[c][0]];
        let mut hf = vec![0.0f32; 64];
        hf.copy_from_slice(&coeffs[c][..64]);

        transform_to_pixels(transform, &mut lf, &mut hf);
        result[c] = hf;
    }
    result
}

/// Inverse transform a single channel of 8x8 coefficients to pixels.
#[allow(dead_code)]
fn inverse_transform_8x8_single_channel(transform_id: u8, coeffs: &[f32; 64]) -> [f32; 64] {
    use jxl_transforms::transform_map::HfTransformType;
    let transform =
        HfTransformType::from_usize(transform_id as usize).unwrap_or(HfTransformType::DCT);
    let mut lf = [coeffs[0]];
    let mut hf = [0.0f32; 64];
    hf.copy_from_slice(coeffs);
    transform_to_pixels(transform, &mut lf, &mut hf);
    hf
}

/// Uses the decoder's inverse transform to convert error coefficients to pixels.
#[allow(dead_code)]
fn inverse_transform_error(
    error_dct: &[f32],
    transform_id: u8,
    block_w: usize,
    block_h: usize,
) -> Vec<f32> {
    let n = block_w * block_h;
    if n == 0 {
        return vec![];
    }

    if transform_id == DCT8_TRANSFORM_ID || is_special_8x8_transform_id(transform_id) {
        // For 8x8: use idct2d_8
        let mut data = [0.0f32; 64];
        data[..error_dct.len().min(64)].copy_from_slice(&error_dct[..error_dct.len().min(64)]);
        jxl_transforms::idct2d_8_8(jxl_simd::scalar::ScalarDescriptor, &mut data);
        return data.to_vec();
    }

    // For larger transforms (DCT16, DCT32, etc.): use the decoder's inverse.
    // The error_dct is in natural coefficient order. We need to split into
    // LF and HF parts for the decoder's transform_to_pixels.
    use jxl_transforms::transform_map::HfTransformType;

    let (transform, lf_w, lf_h) = match transform_id {
        DCT16_TRANSFORM_ID => (HfTransformType::DCT16X16, 2, 2),
        DCT32_TRANSFORM_ID => (HfTransformType::DCT32X32, 4, 4),
        _ => {
            // Fallback: no inverse, return zeros (loss term won't penalize)
            return vec![0.0f32; n];
        }
    };

    let lf_count = lf_w * lf_h;
    // Split error into LF (DC-like) and HF arrays matching decoder expectations.
    let mut lf = vec![0.0f32; lf_count];
    let mut hf = vec![0.0f32; n];

    // The LF positions for DCT16 are the 2x2 top-left corner of the 16x16 grid.
    // Positions (0,0), (0,1), (1,0), (1,1) in row-major = indices 0, 1, 16, 17.
    let block_dim = block_w; // square transforms
    let mut lf_idx = 0;
    for ly in 0..lf_h {
        for lx in 0..lf_w {
            let coeff_idx = ly * block_dim + lx;
            lf[lf_idx] = error_dct[coeff_idx];
            lf_idx += 1;
        }
    }
    // All other positions are HF
    hf[..n].copy_from_slice(&error_dct[..n]);
    for ly in 0..lf_h {
        for lx in 0..lf_w {
            hf[ly * block_dim + lx] = 0.0;
        }
    }

    transform_to_pixels(transform, &mut lf, &mut hf);
    hf
}

/// Full port of libjxl's AC strategy selection (EstimateEntropy-based DCT16/32
/// merging) from enc_ac_strategy.cc. Uses entropy + information loss model to
/// decide when merging 8x8 blocks into larger transforms is beneficial.
#[allow(clippy::too_many_arguments)]
fn build_entropy_merge_transform_map(
    pixel_x: &[f32],
    pixel_y: &[f32],
    pixel_b: &[f32], // b_minus_y for CfL-adjusted comparison
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
    _dct8_ac_y: &[i32],
    _dct8_ac_x: &[i32],
    _dct8_ac_b: &[i32],
    global_scale: u32,
    raw_quant_map: &[u8],
    _dequant_weights_8x8: &[f32],
    x_dm_multiplier: f32,
    b_dm_multiplier: f32,
    _distance: f32,
    _masking1x1: &[f32],   // per-pixel masking field (for future loss term)
    _orig_y: &[f32],       // original Y channel (pre-inverse-gaborish) for MSE
    _aq_float_map: &[f32], // float quant field (libjxl's quant_norm16)
) -> Vec<u8> {
    let mut map = build_default_transform_map(bw, bh);

    // Entropy multipliers: without libjxl's perceptual loss term working,
    // we use high multipliers to avoid quality-destroying merges.
    // PSNR parity with libjxl is the priority over file size.
    let entropy_mul_16 = 2.5f32;
    let entropy_mul_32 = 3.5f32;

    // Phase 0: Per-block 8x8 transform selection.
    // DISABLED: At d <= ~4.0, libjxl's EstimateEntropy produces all-zero
    // quantized values, making all transforms score identically. DCT8 wins
    // by being first. libjxl also keeps all-DCT8 at Squirrel/e3 for d=1.0.
    // The PSNR gap vs libjxl comes from other factors (CfL, quant field
    // calibration, EPF tuning), not from 8x8 transform selection.

    // Simple entropy estimate using sqrt(|quantized|) from DCT8 coefficients.
    let estimate_8x8_entropy = |blk: usize, _bx: usize, _by: usize| -> f32 {
        let base = blk * 64;
        let mut e = 0.0f32;
        for k in 1..64 {
            e += (_dct8_ac_y[base + k].abs() as f32).sqrt();
            e += (_dct8_ac_x[base + k].abs() as f32).sqrt() * 0.3;
            e += (_dct8_ac_b[base + k].abs() as f32).sqrt() * 0.3;
        }
        e
    };

    let dct16_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(4);
    let allow_square_solver = bw * bh <= SQUARE_SOLVER_MAX_BLOCKS;

    // Phase 0.5: DCT16x8 / DCT8x16 rectangular merges (pairs of 8x8 blocks).
    // These are common in libjxl and help in areas smooth in one direction.
    let dct8x16_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(6);
    let entropy_mul_rect = 2.0f32; // Conservative multiplier for rect merges

    // DCT16X8 (id=6): 16 rows x 8 cols = 1 block wide, 2 blocks tall
    // Merge 2 vertically adjacent blocks (must not cross group boundaries)
    let group_dim = 32usize; // 256 pixels / 8 = 32 blocks per group dimension
    for bx in 0..bw {
        let mut by = 0;
        while by + 1 < bh {
            // Don't cross group boundary vertically
            if by / group_dim != (by + 1) / group_dim {
                by += 1;
                continue;
            }
            if map[by * bw + bx] != (TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID)
                || map[(by + 1) * bw + bx] != (TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID)
            {
                by += 1;
                continue;
            }

            let rq0 = raw_quant_map[by * bw + bx] as f32;
            let rq1 = raw_quant_map[(by + 1) * bw + bx] as f32;
            if rq0.max(rq1) > rq0.min(rq1) * 1.5 + 1.0 {
                by += 1;
                continue;
            }

            let e8_sum = estimate_8x8_entropy(by * bw + bx, bx, by)
                + estimate_8x8_entropy((by + 1) * bw + bx, bx, by + 1);

            let rq = rq0.max(rq1) as u32;
            // DCT16X8: 8 cols (1 block), 16 rows (2 blocks)
            let coeffs = compute_forward_transform_coeffs(
                DCT16X8_TRANSFORM_ID,
                pixel_x,
                pixel_y,
                pixel_b,
                width,
                height,
                bx,
                by,
                8,
                16,
            );
            let scale_rect = (global_scale as f32 * rq as f32) / 65536.0;
            let mut e_rect = 0.0f32;
            for c in 0..3usize {
                let dw_base = &dct8x16_weights[c * 128..(c + 1) * 128];
                let dm_mul = match c {
                    0 => x_dm_multiplier,
                    1 => 1.0,
                    _ => b_dm_multiplier,
                };
                let chan_w = match c {
                    0 => 0.3f32,
                    1 => 1.0,
                    _ => 0.3,
                };
                for k in 1..128 {
                    let dw = dw_base[k] * dm_mul;
                    if dw.abs() < 1e-10 {
                        continue;
                    }
                    let q = (coeffs[c][k] * scale_rect / dw).round();
                    e_rect += q.abs().sqrt() * chan_w;
                }
            }
            e_rect *= entropy_mul_rect;

            if e_rect < e8_sum {
                map[by * bw + bx] = TRANSFORM_FIRST_BLOCK_FLAG | DCT16X8_TRANSFORM_ID;
                map[(by + 1) * bw + bx] = DCT16X8_TRANSFORM_ID;
                by += 2;
            } else {
                by += 1;
            }
        }
    }

    // DCT8X16 (id=7): 8 rows x 16 cols = 2 blocks wide, 1 block tall
    // Merge 2 horizontally adjacent blocks (must not cross group boundaries)
    for by in 0..bh {
        let mut bx = 0;
        while bx + 1 < bw {
            // Don't cross group boundary horizontally
            if bx / group_dim != (bx + 1) / group_dim {
                bx += 1;
                continue;
            }
            if map[by * bw + bx] != (TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID)
                || map[by * bw + bx + 1] != (TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID)
            {
                bx += 1;
                continue;
            }

            let rq0 = raw_quant_map[by * bw + bx] as f32;
            let rq1 = raw_quant_map[by * bw + bx + 1] as f32;
            if rq0.max(rq1) > rq0.min(rq1) * 1.5 + 1.0 {
                bx += 1;
                continue;
            }

            let e8_sum = estimate_8x8_entropy(by * bw + bx, bx, by)
                + estimate_8x8_entropy(by * bw + bx + 1, bx + 1, by);

            let rq = rq0.max(rq1) as u32;
            // DCT8X16: 16 cols (2 blocks), 8 rows (1 block)
            let coeffs = compute_forward_transform_coeffs(
                DCT8X16_TRANSFORM_ID,
                pixel_x,
                pixel_y,
                pixel_b,
                width,
                height,
                bx,
                by,
                16,
                8,
            );
            let scale_rect = (global_scale as f32 * rq as f32) / 65536.0;
            let mut e_rect = 0.0f32;
            for c in 0..3usize {
                let dw_base = &dct8x16_weights[c * 128..(c + 1) * 128];
                let dm_mul = match c {
                    0 => x_dm_multiplier,
                    1 => 1.0,
                    _ => b_dm_multiplier,
                };
                let chan_w = match c {
                    0 => 0.3f32,
                    1 => 1.0,
                    _ => 0.3,
                };
                for k in 1..128 {
                    let dw = dw_base[k] * dm_mul;
                    if dw.abs() < 1e-10 {
                        continue;
                    }
                    let q = (coeffs[c][k] * scale_rect / dw).round();
                    e_rect += q.abs().sqrt() * chan_w;
                }
            }
            e_rect *= entropy_mul_rect;

            if e_rect < e8_sum {
                map[by * bw + bx] = TRANSFORM_FIRST_BLOCK_FLAG | DCT8X16_TRANSFORM_ID;
                map[by * bw + bx + 1] = DCT8X16_TRANSFORM_ID;
                bx += 2;
            } else {
                bx += 1;
            }
        }
    }

    // Phase 1: DCT16x16 merge (2x2 groups of 8x8 blocks).
    let h2 = bh / 2;
    let w2 = bw / 2;
    for by2 in 0..h2 {
        for bx2 in 0..w2 {
            let bx = bx2 * 2;
            let by = by2 * 2;

            // All 4 blocks must still be DCT8 (not already rect-merged)
            let all_dct8 = [(by, bx), (by, bx + 1), (by + 1, bx), (by + 1, bx + 1)]
                .iter()
                .all(|&(r, c)| map[r * bw + c] == (TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID));
            if !all_dct8 {
                continue;
            }

            // Guard: skip merge if raw_quant values vary too much across the
            // 2x2 group. This prevents merging a smooth block with an edge
            // block, which would over-quantize the edge (libjxl handles this
            // via the loss term in EstimateEntropy, but our loss term is not
            // effective due to forward transform normalization differences).
            let rq_vals: [u8; 4] = [
                raw_quant_map[by * bw + bx],
                raw_quant_map[by * bw + bx + 1],
                raw_quant_map[(by + 1) * bw + bx],
                raw_quant_map[(by + 1) * bw + bx + 1],
            ];
            let rq_min = *rq_vals.iter().min().unwrap() as f32;
            let rq_max = *rq_vals.iter().max().unwrap() as f32;
            if rq_max > rq_min * 1.5 + 1.0 {
                continue; // Too much variance in quantization -- skip merge
            }

            // Sum of 4x DCT8 entropies using full EstimateEntropy model.
            let e8_sum = estimate_8x8_entropy(by * bw + bx, bx, by)
                + estimate_8x8_entropy(by * bw + bx + 1, bx + 1, by)
                + estimate_8x8_entropy((by + 1) * bw + bx, bx, by + 1)
                + estimate_8x8_entropy((by + 1) * bw + bx + 1, bx + 1, by + 1);

            // libjxl: quant_norm16 for >= 4 blocks uses L16 norm
            let rq: u32 = {
                let mut sum16 = 0.0f64;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let qval = raw_quant_map[(by + dy) * bw + (bx + dx)] as f64;
                        let q2 = qval * qval;
                        let q4 = q2 * q2;
                        let q8 = q4 * q4;
                        sum16 += q8 * q8;
                    }
                }
                (sum16 / 4.0).powf(1.0 / 16.0).round().max(1.0) as u32
            };

            let coeffs = compute_forward_transform_coeffs(
                DCT16_TRANSFORM_ID,
                pixel_x,
                pixel_y,
                pixel_b,
                width,
                height,
                bx,
                by,
                16,
                16,
            );
            let scale16 = (global_scale as f32 * rq as f32) / 65536.0;
            let mut e16 = 0.0f32;
            for c in 0..3usize {
                let dw_base = &dct16_weights[c * 256..(c + 1) * 256];
                let dm_mul = match c {
                    0 => x_dm_multiplier,
                    1 => 1.0,
                    _ => b_dm_multiplier,
                };
                let chan_w = match c {
                    0 => 0.3f32,
                    1 => 1.0,
                    _ => 0.3,
                };
                for k in 1..256 {
                    let dw = dw_base[k] * dm_mul;
                    if dw.abs() < 1e-10 {
                        continue;
                    }
                    let q = (coeffs[c][k] * scale16 / dw).round();
                    e16 += q.abs().sqrt() * chan_w;
                }
            }
            e16 *= entropy_mul_16;

            if e16 < e8_sum {
                map[by * bw + bx] = TRANSFORM_FIRST_BLOCK_FLAG | DCT16_TRANSFORM_ID;
                map[by * bw + bx + 1] = DCT16_TRANSFORM_ID;
                map[(by + 1) * bw + bx] = DCT16_TRANSFORM_ID;
                map[(by + 1) * bw + bx + 1] = DCT16_TRANSFORM_ID;
            }
        }
    }

    // Phase 2: DCT32x32 merge (2x2 groups of DCT16 blocks = 4x4 of 8x8).
    if allow_square_solver && bw >= 4 && bh >= 4 {
        let dct32_weights = crate::frame::quant_weights::DequantMatrices::get_library_table(5);
        let h4 = bh / 4;
        let w4 = bw / 4;
        for by4 in 0..h4 {
            for bx4 in 0..w4 {
                let bx = bx4 * 4;
                let by = by4 * 4;

                // Check: all 4 constituent DCT16 blocks must already be DCT16.
                let all_dct16 = (0..2).all(|dy| {
                    (0..2).all(|dx| {
                        let idx = (by + dy * 2) * bw + (bx + dx * 2);
                        map[idx] & 0x3F == DCT16_TRANSFORM_ID
                            && map[idx] & TRANSFORM_FIRST_BLOCK_FLAG != 0
                    })
                });
                if !all_dct16 {
                    continue;
                }

                // Sum of 4x DCT16 entropies (full model).
                let mut e16_sum = 0.0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx16 = bx + dx * 2;
                        let by16 = by + dy * 2;
                        let coeffs = compute_forward_transform_coeffs(
                            DCT16_TRANSFORM_ID,
                            pixel_x,
                            pixel_y,
                            pixel_b,
                            width,
                            height,
                            bx16,
                            by16,
                            16,
                            16,
                        );
                        let rq16: u32 = (0..2)
                            .flat_map(|ddy| {
                                (0..2).map(move |ddx| {
                                    raw_quant_map[(by16 + ddy) * bw + (bx16 + ddx)] as u32
                                })
                            })
                            .max()
                            .unwrap();
                        let scale16 = (global_scale as f32 * rq16 as f32) / 65536.0;
                        for c in 0..3usize {
                            let dw_base = &dct16_weights[c * 256..(c + 1) * 256];
                            let dm_mul = match c {
                                0 => x_dm_multiplier,
                                1 => 1.0,
                                _ => b_dm_multiplier,
                            };
                            let chan_w = match c {
                                0 => 0.3f32,
                                1 => 1.0,
                                _ => 0.3,
                            };
                            for k in 1..256 {
                                let dw = dw_base[k] * dm_mul;
                                if dw.abs() < 1e-10 {
                                    continue;
                                }
                                let q = (coeffs[c][k] * scale16 / dw).round();
                                e16_sum += q.abs().sqrt() * chan_w;
                            }
                        }
                    }
                }

                let rq32: u32 = {
                    let mut sum16 = 0.0f64;
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let qval = raw_quant_map[(by + dy) * bw + (bx + dx)] as f64;
                            let q2 = qval * qval;
                            let q4 = q2 * q2;
                            let q8 = q4 * q4;
                            sum16 += q8 * q8;
                        }
                    }
                    (sum16 / 16.0).powf(1.0 / 16.0).round().max(1.0) as u32
                };

                let coeffs = compute_forward_transform_coeffs(
                    DCT32_TRANSFORM_ID,
                    pixel_x,
                    pixel_y,
                    pixel_b,
                    width,
                    height,
                    bx,
                    by,
                    32,
                    32,
                );
                let scale32 = (global_scale as f32 * rq32 as f32) / 65536.0;
                let mut e32 = 0.0f32;
                for c in 0..3usize {
                    let dw_base = &dct32_weights[c * 1024..(c + 1) * 1024];
                    let dm_mul = match c {
                        0 => x_dm_multiplier,
                        1 => 1.0,
                        _ => b_dm_multiplier,
                    };
                    let chan_w = match c {
                        0 => 0.3f32,
                        1 => 1.0,
                        _ => 0.3,
                    };
                    for k in 1..1024 {
                        let dw = dw_base[k] * dm_mul;
                        if dw.abs() < 1e-10 {
                            continue;
                        }
                        let q = (coeffs[c][k] * scale32 / dw).round();
                        e32 += q.abs().sqrt() * chan_w;
                    }
                }
                e32 *= entropy_mul_32;

                if e32 < e16_sum {
                    // Set all 16 blocks to DCT32.
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let idx = (by + dy) * bw + (bx + dx);
                            if dy == 0 && dx == 0 {
                                map[idx] = TRANSFORM_FIRST_BLOCK_FLAG | DCT32_TRANSFORM_ID;
                            } else {
                                map[idx] = DCT32_TRANSFORM_ID;
                            }
                        }
                    }
                }
            }
        }
    }

    map
}

fn build_afv_transform_map_from_quantized_ac(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<u8> {
    let mut map = build_default_transform_map(bw, bh);
    if distance < 2.5 {
        return map;
    }

    let num_blocks = bw * bh;
    let max_afv_blocks = (num_blocks / 64).clamp(1, 64);
    let mut scored = Vec::<(i64, usize, u8)>::new();

    for blk in 0..num_blocks {
        let base = blk * 64;
        let mut total = 0i64;
        let mut low = 0i64;
        let mut high = 0i64;

        for k in 1..64 {
            let u = k & 7;
            let v = k >> 3;
            let y = ac_y[base + k].abs() as i64;
            let xb = (ac_x[base + k].abs() + ac_b[base + k].abs()) as i64;
            let mag = y * 2 + xb / 2;
            total += mag;
            if u < 3 && v < 3 {
                low += mag;
            }
            if u >= 3 && v >= 3 {
                high += mag;
            }
        }

        // Keep AFV sparse and only for blocks with pronounced high-frequency content.
        if total < 40 || high * 2 < low {
            continue;
        }

        let horiz = (ac_y[base + 1].abs() + ac_y[base + 2].abs() + ac_y[base + 3].abs()) as i64;
        let vert = (ac_y[base + 8].abs() + ac_y[base + 16].abs() + ac_y[base + 24].abs()) as i64;
        let anis = (horiz - vert).abs();
        if anis < 4 {
            continue;
        }

        let sx = ac_y[base + 1] + ac_x[base + 1] - ac_b[base + 1];
        let sy = ac_y[base + 8] + ac_x[base + 8] - ac_b[base + 8];
        let afv = match (sx < 0, sy < 0) {
            (false, false) => AFV0_TRANSFORM_ID,
            (true, false) => AFV1_TRANSFORM_ID,
            (false, true) => AFV2_TRANSFORM_ID,
            (true, true) => AFV3_TRANSFORM_ID,
        };

        let score = high + anis * 3 - low;
        if score > 0 {
            scored.push((score, blk, afv));
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    for &(_score, blk, afv) in scored.iter().take(max_afv_blocks) {
        map[blk] = TRANSFORM_FIRST_BLOCK_FLAG | afv;
    }

    map
}

fn build_directional_special_transform_map_from_quantized_ac(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<u8> {
    let mut map = build_default_transform_map(bw, bh);
    if distance < 3.0 {
        return map;
    }

    let num_blocks = bw * bh;
    let max_directional_blocks = (num_blocks / 96).clamp(1, 48);
    let mut scored = Vec::<(i64, usize, u8)>::new();

    for blk in 0..num_blocks {
        let base = blk * 64;
        let mut total = 0i64;
        let mut high = 0i64;

        for k in 1..64 {
            let u = k & 7;
            let v = k >> 3;
            let y = ac_y[base + k].abs() as i64;
            let xb = (ac_x[base + k].abs() + ac_b[base + k].abs()) as i64;
            let mag = y * 2 + xb / 2;
            total += mag;
            if u >= 2 || v >= 2 {
                high += mag;
            }
        }

        if total < 48 || high * 3 < total {
            continue;
        }

        let horiz = (ac_y[base + 1].abs()
            + ac_y[base + 2].abs()
            + ac_y[base + 3].abs()
            + ac_x[base + 1].abs()
            + ac_b[base + 1].abs()) as i64;
        let vert = (ac_y[base + 8].abs()
            + ac_y[base + 16].abs()
            + ac_y[base + 24].abs()
            + ac_x[base + 8].abs()
            + ac_b[base + 8].abs()) as i64;
        let anis = (horiz - vert).abs();
        if anis < 10 {
            continue;
        }

        let transform = if horiz >= vert {
            DCT4X8_TRANSFORM_ID
        } else {
            DCT8X4_TRANSFORM_ID
        };
        let score = anis * 4 + high - total / 2;
        if score > 0 {
            scored.push((score, blk, transform));
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    for &(_score, blk, transform) in scored.iter().take(max_directional_blocks) {
        map[blk] = TRANSFORM_FIRST_BLOCK_FLAG | transform;
    }

    map
}

fn build_compact_special_transform_map_from_quantized_ac(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<u8> {
    let mut map = build_default_transform_map(bw, bh);
    if distance < 3.0 {
        return map;
    }

    let num_blocks = bw * bh;
    let max_compact_blocks = (num_blocks / 128).clamp(1, 32);
    let mut scored = Vec::<(i64, usize, u8)>::new();

    for blk in 0..num_blocks {
        let base = blk * 64;
        let mut total = 0i64;
        let mut low = 0i64;
        let mut high = 0i64;
        let mut peak = 0i64;

        for k in 1..64 {
            let u = k & 7;
            let v = k >> 3;
            let y = ac_y[base + k].abs() as i64;
            let xb = (ac_x[base + k].abs() + ac_b[base + k].abs()) as i64;
            let mag = y * 2 + xb / 2;
            total += mag;
            peak = peak.max(mag);
            if u < 2 && v < 2 {
                low += mag;
            }
            if u >= 3 || v >= 3 {
                high += mag;
            }
        }

        if total < 4 {
            continue;
        }

        let (transform, score) = if total <= 28 && high <= 6 {
            (
                DCT2X2_TRANSFORM_ID,
                40i64 - total + (low - high).clamp(0, i64::MAX),
            )
        } else if total <= 64 && high * 3 <= low * 2 {
            (
                DCT4X4_TRANSFORM_ID,
                80i64 - total + (low - high / 2).clamp(0, i64::MAX),
            )
        } else if peak >= 36 && peak * 3 > total * 2 && total <= 160 {
            (IDENTITY_TRANSFORM_ID, peak * 2 - total)
        } else {
            continue;
        };

        if score > 0 {
            scored.push((score, blk, transform));
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    for &(_score, blk, transform) in scored.iter().take(max_compact_blocks) {
        map[blk] = TRANSFORM_FIRST_BLOCK_FLAG | transform;
    }

    map
}

fn build_mixed_special_transform_map_from_quantized_ac(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<u8> {
    let mut map = build_default_transform_map(bw, bh);
    if distance < 3.0 {
        return map;
    }

    let num_blocks = bw * bh;
    let max_special_blocks = (num_blocks / 16).clamp(1, 64);
    let mut scored = Vec::<(i64, usize, u8)>::new();

    for blk in 0..num_blocks {
        let base = blk * 64;
        let mut total = 0i64;
        let mut low2 = 0i64;
        let mut low3 = 0i64;
        let mut high3 = 0i64;
        let mut high2 = 0i64;
        let mut peak = 0i64;

        for k in 1..64 {
            let u = k & 7;
            let v = k >> 3;
            let y = ac_y[base + k].abs() as i64;
            let xb = (ac_x[base + k].abs() + ac_b[base + k].abs()) as i64;
            let mag = y * 2 + xb / 2;
            total += mag;
            peak = peak.max(mag);
            if u < 2 && v < 2 {
                low2 += mag;
            }
            if u < 3 && v < 3 {
                low3 += mag;
            }
            if u >= 3 && v >= 3 {
                high3 += mag;
            }
            if u >= 2 || v >= 2 {
                high2 += mag;
            }
        }

        if total < 4 {
            continue;
        }

        let mut best_score = 0i64;
        let mut best_transform = DCT8_TRANSFORM_ID;

        // Compact/smooth/sparse special transform preference.
        if total <= 28 && high3 <= 6 {
            let score = 40i64 - total + (low2 - high3).clamp(0, i64::MAX);
            if score > best_score {
                best_score = score;
                best_transform = DCT2X2_TRANSFORM_ID;
            }
        } else if total <= 64 && high3 * 3 <= low2 * 2 {
            let score = 80i64 - total + (low2 - high3 / 2).clamp(0, i64::MAX);
            if score > best_score {
                best_score = score;
                best_transform = DCT4X4_TRANSFORM_ID;
            }
        } else if peak >= 36 && peak * 3 > total * 2 && total <= 160 {
            let score = peak * 2 - total;
            if score > best_score {
                best_score = score;
                best_transform = IDENTITY_TRANSFORM_ID;
            }
        }

        // Directional small transforms.
        let horiz_dir = (ac_y[base + 1].abs()
            + ac_y[base + 2].abs()
            + ac_y[base + 3].abs()
            + ac_x[base + 1].abs()
            + ac_b[base + 1].abs()) as i64;
        let vert_dir = (ac_y[base + 8].abs()
            + ac_y[base + 16].abs()
            + ac_y[base + 24].abs()
            + ac_x[base + 8].abs()
            + ac_b[base + 8].abs()) as i64;
        let anis_dir = (horiz_dir - vert_dir).abs();
        if total >= 48 && high2 * 3 >= total && anis_dir >= 10 {
            let transform = if horiz_dir >= vert_dir {
                DCT4X8_TRANSFORM_ID
            } else {
                DCT8X4_TRANSFORM_ID
            };
            let score = anis_dir * 4 + high2 - total / 2;
            if score > best_score {
                best_score = score;
                best_transform = transform;
            }
        }

        // AFV: pronounced high frequency plus direction/sign variants.
        let horiz_afv = (ac_y[base + 1].abs() + ac_y[base + 2].abs() + ac_y[base + 3].abs()) as i64;
        let vert_afv =
            (ac_y[base + 8].abs() + ac_y[base + 16].abs() + ac_y[base + 24].abs()) as i64;
        let anis_afv = (horiz_afv - vert_afv).abs();
        if total >= 40 && high3 * 2 >= low3 && anis_afv >= 4 {
            let sx = ac_y[base + 1] + ac_x[base + 1] - ac_b[base + 1];
            let sy = ac_y[base + 8] + ac_x[base + 8] - ac_b[base + 8];
            let transform = match (sx < 0, sy < 0) {
                (false, false) => AFV0_TRANSFORM_ID,
                (true, false) => AFV1_TRANSFORM_ID,
                (false, true) => AFV2_TRANSFORM_ID,
                (true, true) => AFV3_TRANSFORM_ID,
            };
            let score = high3 + anis_afv * 3 - low3;
            if score > best_score {
                best_score = score;
                best_transform = transform;
            }
        }

        if best_score > 0 && best_transform != DCT8_TRANSFORM_ID {
            scored.push((best_score, blk, best_transform));
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    for &(_score, blk, transform) in scored.iter().take(max_special_blocks) {
        map[blk] = TRANSFORM_FIRST_BLOCK_FLAG | transform;
    }

    map
}

fn build_transform_map_candidates_from_quantized_ac(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    distance: f32,
) -> Vec<Vec<u8>> {
    build_transform_map_candidates_from_quantized_ac_with_flags(
        ac_x, ac_y, ac_b, bw, bh, distance, false,
    )
}

fn build_transform_map_candidates_from_quantized_ac_with_flags(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    bw: usize,
    bh: usize,
    distance: f32,
    prefer_large_transforms_for_flat: bool,
) -> Vec<Vec<u8>> {
    let distance = if prefer_large_transforms_for_flat {
        distance.max(2.5)
    } else {
        distance
    };

    let default_map = build_default_transform_map(bw, bh);
    if distance < 1.5 {
        // At low distances, only use default (DCT8). The entropy-based DCT16
        // merge in build_entropy_merge_transform_map handles DCT16 selection.
        return vec![default_map];
    }

    if bw * bh > 8192 {
        let mut candidates = vec![default_map.clone()];
        candidates.push(build_transform_map_from_quantized_ac(
            ac_x, ac_y, ac_b, bw, bh, distance,
        ));
        candidates.push(build_zero_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[DCT32_TRANSFORM_ID, DCT16_TRANSFORM_ID],
        ));
        candidates.push(build_zero_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[
                DCT32_TRANSFORM_ID,
                DCT16X32_TRANSFORM_ID,
                DCT32X16_TRANSFORM_ID,
                DCT8X32_TRANSFORM_ID,
                DCT32X8_TRANSFORM_ID,
                DCT8X16_TRANSFORM_ID,
                DCT16X8_TRANSFORM_ID,
                DCT16_TRANSFORM_ID,
            ],
        ));
        if distance >= 2.0 {
            candidates.push(build_low_energy_merge_transform_map(
                ac_x,
                ac_y,
                ac_b,
                bw,
                bh,
                &[
                    DCT16X32_TRANSFORM_ID,
                    DCT32X16_TRANSFORM_ID,
                    DCT8X32_TRANSFORM_ID,
                    DCT32X8_TRANSFORM_ID,
                    DCT8X16_TRANSFORM_ID,
                    DCT16X8_TRANSFORM_ID,
                    DCT16_TRANSFORM_ID,
                ],
                3,
            ));
        }
        if distance >= 2.5 {
            candidates.push(build_low_energy_merge_transform_map(
                ac_x,
                ac_y,
                ac_b,
                bw,
                bh,
                &[
                    DCT32_TRANSFORM_ID,
                    DCT16X32_TRANSFORM_ID,
                    DCT32X16_TRANSFORM_ID,
                    DCT8X32_TRANSFORM_ID,
                    DCT32X8_TRANSFORM_ID,
                    DCT8X16_TRANSFORM_ID,
                    DCT16X8_TRANSFORM_ID,
                    DCT16_TRANSFORM_ID,
                ],
                if distance >= 3.0 { 8 } else { 5 },
            ));
        }

        let mut unique = Vec::new();
        for candidate in candidates {
            if !unique.iter().any(|u: &Vec<u8>| *u == candidate) {
                unique.push(candidate);
            }
        }
        return unique;
    }

    let mut candidates = vec![default_map];
    candidates.push(build_transform_map_from_quantized_ac(
        ac_x, ac_y, ac_b, bw, bh, distance,
    ));
    candidates.push(build_zero_merge_transform_map(
        ac_x,
        ac_y,
        ac_b,
        bw,
        bh,
        &[DCT32_TRANSFORM_ID, DCT16_TRANSFORM_ID],
    ));
    candidates.push(build_zero_merge_transform_map(
        ac_x,
        ac_y,
        ac_b,
        bw,
        bh,
        &[
            DCT32_TRANSFORM_ID,
            DCT16X32_TRANSFORM_ID,
            DCT32X16_TRANSFORM_ID,
            DCT8X32_TRANSFORM_ID,
            DCT32X8_TRANSFORM_ID,
            DCT8X16_TRANSFORM_ID,
            DCT16X8_TRANSFORM_ID,
            DCT16_TRANSFORM_ID,
        ],
    ));

    if distance >= 1.8 {
        candidates.push(build_low_energy_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[DCT16_TRANSFORM_ID],
            3,
        ));
    }
    if distance >= 2.0 {
        candidates.push(build_low_energy_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[DCT16_TRANSFORM_ID],
            2,
        ));
        candidates.push(build_low_energy_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[
                DCT16X32_TRANSFORM_ID,
                DCT32X16_TRANSFORM_ID,
                DCT8X32_TRANSFORM_ID,
                DCT32X8_TRANSFORM_ID,
                DCT16_TRANSFORM_ID,
            ],
            3,
        ));
        candidates.push(build_low_energy_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[
                DCT16X32_TRANSFORM_ID,
                DCT32X16_TRANSFORM_ID,
                DCT8X32_TRANSFORM_ID,
                DCT32X8_TRANSFORM_ID,
                DCT8X16_TRANSFORM_ID,
                DCT16X8_TRANSFORM_ID,
                DCT16_TRANSFORM_ID,
            ],
            3,
        ));
    }
    if distance >= 2.5 {
        if bw * bh <= 4096 {
            candidates.push(build_afv_transform_map_from_quantized_ac(
                ac_x, ac_y, ac_b, bw, bh, distance,
            ));
        }
        candidates.push(build_low_energy_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[DCT32_TRANSFORM_ID, DCT16_TRANSFORM_ID],
            4,
        ));
        candidates.push(build_low_energy_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[
                DCT32_TRANSFORM_ID,
                DCT16X32_TRANSFORM_ID,
                DCT32X16_TRANSFORM_ID,
                DCT8X32_TRANSFORM_ID,
                DCT32X8_TRANSFORM_ID,
                DCT8X16_TRANSFORM_ID,
                DCT16X8_TRANSFORM_ID,
                DCT16_TRANSFORM_ID,
            ],
            5,
        ));
        if bw * bh <= 4096 {
            candidates.push(build_low_energy_merge_transform_map(
                ac_x,
                ac_y,
                ac_b,
                bw,
                bh,
                &[
                    DCT64_TRANSFORM_ID,
                    DCT32X64_TRANSFORM_ID,
                    DCT64X32_TRANSFORM_ID,
                    DCT32_TRANSFORM_ID,
                    DCT16X32_TRANSFORM_ID,
                    DCT32X16_TRANSFORM_ID,
                    DCT16_TRANSFORM_ID,
                ],
                6,
            ));
        }
    }
    if distance >= 3.0 {
        if bw * bh <= 4096 {
            candidates.push(build_directional_special_transform_map_from_quantized_ac(
                ac_x, ac_y, ac_b, bw, bh, distance,
            ));
            candidates.push(build_compact_special_transform_map_from_quantized_ac(
                ac_x, ac_y, ac_b, bw, bh, distance,
            ));
            candidates.push(build_mixed_special_transform_map_from_quantized_ac(
                ac_x, ac_y, ac_b, bw, bh, distance,
            ));
        }
        candidates.push(build_low_energy_merge_transform_map(
            ac_x,
            ac_y,
            ac_b,
            bw,
            bh,
            &[
                DCT32_TRANSFORM_ID,
                DCT16X32_TRANSFORM_ID,
                DCT32X16_TRANSFORM_ID,
                DCT8X32_TRANSFORM_ID,
                DCT32X8_TRANSFORM_ID,
                DCT8X16_TRANSFORM_ID,
                DCT16X8_TRANSFORM_ID,
                DCT16_TRANSFORM_ID,
            ],
            8,
        ));
        if bw * bh <= 4096 {
            candidates.push(build_low_energy_merge_transform_map(
                ac_x,
                ac_y,
                ac_b,
                bw,
                bh,
                &[
                    DCT64_TRANSFORM_ID,
                    DCT32X64_TRANSFORM_ID,
                    DCT64X32_TRANSFORM_ID,
                    DCT32_TRANSFORM_ID,
                    DCT16X32_TRANSFORM_ID,
                    DCT32X16_TRANSFORM_ID,
                    DCT8X32_TRANSFORM_ID,
                    DCT32X8_TRANSFORM_ID,
                    DCT16_TRANSFORM_ID,
                ],
                10,
            ));
            if bw >= 16 && bh >= 16 {
                candidates.push(build_low_energy_merge_transform_map(
                    ac_x,
                    ac_y,
                    ac_b,
                    bw,
                    bh,
                    &[
                        DCT128_TRANSFORM_ID,
                        DCT64X128_TRANSFORM_ID,
                        DCT128X64_TRANSFORM_ID,
                        DCT64_TRANSFORM_ID,
                        DCT32X64_TRANSFORM_ID,
                        DCT64X32_TRANSFORM_ID,
                        DCT32_TRANSFORM_ID,
                        DCT16X32_TRANSFORM_ID,
                        DCT32X16_TRANSFORM_ID,
                        DCT16_TRANSFORM_ID,
                    ],
                    14,
                ));
            }
            if bw * bh <= 2048 && bw >= 32 && bh >= 32 {
                candidates.push(build_low_energy_merge_transform_map(
                    ac_x,
                    ac_y,
                    ac_b,
                    bw,
                    bh,
                    &[
                        DCT256_TRANSFORM_ID,
                        DCT128X256_TRANSFORM_ID,
                        DCT256X128_TRANSFORM_ID,
                        DCT128_TRANSFORM_ID,
                        DCT64X128_TRANSFORM_ID,
                        DCT128X64_TRANSFORM_ID,
                        DCT64_TRANSFORM_ID,
                        DCT32X64_TRANSFORM_ID,
                        DCT64X32_TRANSFORM_ID,
                        DCT32_TRANSFORM_ID,
                        DCT16_TRANSFORM_ID,
                    ],
                    18,
                ));
            }
        }
    }

    let mut unique = Vec::new();
    for candidate in candidates {
        if !unique.iter().any(|u: &Vec<u8>| *u == candidate) {
            unique.push(candidate);
        }
    }
    unique
}

fn collect_transform_entries_for_rect(
    transform_map: &[u8],
    raw_quant_map: &[u8],
    bw: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
) -> Vec<(u8, u8)> {
    let mut entries = Vec::new();
    for y in 0..rh {
        for x in 0..rw {
            let idx = (y0 + y) * bw + (x0 + x);
            let transform = transform_map[idx];
            if transform & TRANSFORM_FIRST_BLOCK_FLAG == 0 {
                continue;
            }
            let transform_id = transform & !TRANSFORM_FIRST_BLOCK_FLAG;
            entries.push((transform_id, raw_quant_map[idx]));
        }
    }
    entries
}

/// Apply forward DCT8x8 to a channel with edge-clamp padding.
fn forward_dct_channel(
    chan: &[f32],
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
    out: &mut [f32],
) {
    for by in 0..bh {
        for bx in 0..bw {
            let blk_idx = by * bw + bx;
            let mut block = [0.0f32; 64];
            for dy in 0..8 {
                for dx in 0..8 {
                    let sy = (by * 8 + dy).min(height - 1);
                    let sx = (bx * 8 + dx).min(width - 1);
                    block[dy * 8 + dx] = chan[sy * width + sx];
                }
            }
            dct2d_8_scalar(&mut block);
            out[blk_idx * 64..blk_idx * 64 + 64].copy_from_slice(&block);
        }
    }
}

/// Quantize DC coefficients for a single channel.
///
/// The decoder dequantizes DC as:
///   dc_float = quantized * LF_QUANT[c] * 2^16 / (global_scale * quant_lf)
///
/// So forward quantization is:
///   quantized = round(dc_float * global_scale * quant_lf / (2^16 * LF_QUANT[c]))
///             = round(dc_float * global_scale * quant_lf * INV_LF_QUANT[c] / 2^16)
fn quantize_dc(dc_float: f32, global_scale: u32, quant_lf: u32, inv_lf_quant: f32) -> i32 {
    let scale = (global_scale as f32) * (quant_lf as f32) * inv_lf_quant / (1u32 << 16) as f32;
    (dc_float * scale).round() as i32
}

/// Get the default DCT8x8 dequant matrix weights.
///
/// Returns 3*64 floats: 64 weights per channel (X, Y, B).
/// These are the same weights computed by the decoder from the library
/// encoding with `all_default=true`.
fn default_dct8x8_dequant_weights() -> &'static [f32] {
    use crate::frame::quant_weights::DequantMatrices;
    DequantMatrices::get_library_table(0)
}

/// Quantize a single AC coefficient using the dequant matrix.
///
/// The decoder dequantizes as:
///   ac_float = adjust_quant_bias(quantized) * dequant_weight[k] * inv_global_scale / raw_quant
///
/// For forward quantization (ignoring quant bias):
///   quantized = round(ac_float * raw_quant / (dequant_weight[k] * inv_global_scale))
///             = round(ac_float * raw_quant * global_scale / (dequant_weight[k] * 2^16))
fn quantize_ac(ac_float: f32, global_scale: u32, raw_quant: u32, dequant_weight: f32) -> i32 {
    if dequant_weight.abs() < 1e-10 {
        return 0;
    }
    let scale = (global_scale as f32 * raw_quant as f32) / ((1u32 << 16) as f32 * dequant_weight);
    // Standard round-to-nearest, matching libjxl's kZeroBiasDefault = 0.5.
    (ac_float * scale).round() as i32
}

/// Dead-zone for AC quantization, varying by dequant weight.
/// Low dequant weight = visually important = small dead-zone (preserve detail).
/// High dequant weight = less important = large dead-zone (save bytes).
/// Returns dead-zone given the dequant_weight for the coefficient.

/// Animation parameters for a single frame.
struct AnimFrameParams {
    duration: u32,
    is_last: bool,
}

/// Build the complete VarDCT frame bitstream.
#[allow(clippy::too_many_arguments)]
fn encode_vardct_frame(
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
    global_scale: u32,
    quant_lf: u32,
    dc_y: &[i32],
    dc_x: &[i32],
    dc_b: &[i32],
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    raw_quant_map: &[u8],
    transform_map: &[u8],
    ytox_map: &[i32],
    ytob_map: &[i32],
    use_gab: bool,
) -> Result<Vec<u8>> {
    encode_vardct_frame_inner(
        width,
        height,
        bw,
        bh,
        global_scale,
        quant_lf,
        dc_y,
        dc_x,
        dc_b,
        ac_x,
        ac_y,
        ac_b,
        raw_quant_map,
        transform_map,
        ytox_map,
        ytob_map,
        use_gab,
        None,
        true, // no animation, include file header
        None, // no alpha
        7,
        false,
    )
}

/// Inner function: encode a VarDCT frame.
/// If `include_file_header` is true, writes the codestream file header first.
/// If `anim_params` is Some, writes animation-aware frame header with duration/is_last.
fn encode_vardct_frame_inner(
    width: usize,
    height: usize,
    bw: usize,
    bh: usize,
    global_scale: u32,
    quant_lf: u32,
    dc_y: &[i32],
    dc_x: &[i32],
    dc_b: &[i32],
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    raw_quant_map: &[u8],
    transform_map: &[u8],
    ytox_map: &[i32],
    ytob_map: &[i32],
    use_gab: bool,
    anim_params: Option<&AnimFrameParams>,
    include_file_header: bool,
    alpha: Option<&[u8]>,
    effort: u8,
    progressive: bool,
) -> Result<Vec<u8>> {
    let has_alpha = alpha.is_some();
    let num_extra_channels = if has_alpha { 1u32 } else { 0 };
    let mut writer = BitWriter::new();

    if include_file_header {
        // Codestream header
        if has_alpha {
            crate::encode::headers::write_file_header_with_alpha(
                &mut writer,
                width as u32,
                height as u32,
            )?;
        } else {
            write_file_header(&mut writer, width as u32, height as u32, true, false)?;
        }
    }

    // The decoder byte-aligns before frame-header parsing.
    writer.byte_align_zero_pad()?;

    let (progressive_num_passes, progressive_pass_shifts) =
        choose_progressive_pass_plan(progressive, alpha.is_some(), effort, width, height);

    // Frame header (VarDCT)
    write_vardct_frame_header_full(
        &mut writer,
        &FrameHeaderConfig {
            use_gab,
            num_extra_channels,
            have_animation: anim_params.is_some(),
            duration: anim_params.map_or(0, |ap| ap.duration),
            is_last: anim_params.map_or(true, |ap| ap.is_last),
            num_passes: progressive_num_passes as u32,
            pass_shifts: progressive_pass_shifts.clone(),
        },
    )?;

    // Group layout
    let group_dim_blocks = 32usize; // 256 pixels / 8
    let num_groups_x = bw.div_ceil(group_dim_blocks);
    let num_groups_y = bh.div_ceil(group_dim_blocks);
    let num_groups = num_groups_x * num_groups_y;

    assert_eq!(raw_quant_map.len(), bw * bh);
    assert_eq!(transform_map.len(), bw * bh);

    if num_groups == 1 && !(progressive && alpha.is_none()) {
        // Single-group image: 1 TOC entry, everything in one section.
        let section = encode_single_group_section(
            bw,
            bh,
            width,
            height,
            global_scale,
            quant_lf,
            dc_y,
            dc_x,
            dc_b,
            ac_x,
            ac_y,
            ac_b,
            raw_quant_map,
            transform_map,
            ytox_map,
            ytob_map,
            alpha,
            effort,
        )?;

        write_toc(&mut writer, &[section.len() as u32])?;
        writer.byte_align_zero_pad()?;

        let mut result = writer.finish();
        result.extend_from_slice(&section);
        Ok(result)
    } else {
        // Multi-group: LfGlobal + LfGroups + HfGlobal + HfGroups
        //
        // LF groups operate on DC blocks; each LF group covers group_dim blocks
        // (= group_dim * 8 pixels = 2048px with default group_dim=256).
        // HF groups operate on AC data; each covers group_dim pixels (256px).
        //
        // Section order: [LfGlobal, LfGroup0..NLF-1, HfGlobal, HfGroup0..NHF-1]
        // TOC entries:  2 + num_lf_groups + num_groups (from FrameHeader::num_toc_entries)
        let lf_group_dim_blocks = group_dim_blocks * 8; // 256 blocks = 2048px
        let num_lf_groups_x = bw.div_ceil(lf_group_dim_blocks);
        let num_lf_groups_y = bh.div_ceil(lf_group_dim_blocks);
        let num_lf_groups = num_lf_groups_x * num_lf_groups_y;

        let num_passes = progressive_num_passes;
        let total_sections = 2 + num_lf_groups + num_groups * num_passes;
        let mut sections: Vec<Vec<u8>> = Vec::with_capacity(total_sections);

        // Block context map: currently disabled as the overhead exceeds savings
        // without per-block transform selection (all DCT8, uniform shape).
        // TODO: enable when FindBest8x8Transform gives shape variety.
        let block_ctx: Option<CustomBlockCtx> = None;
        let num_contexts = 15; // default

        // Pre-compute alpha tiles for multi-group encoding
        let alpha_tiles: Vec<Option<(Vec<i32>, usize, usize)>> = if let Some(alpha_data) = alpha {
            let group_dim = 256usize;
            (0..num_groups)
                .map(|g| {
                    let gx = g % num_groups_x;
                    let gy = g / num_groups_x;
                    let px0 = gx * group_dim;
                    let py0 = gy * group_dim;
                    let pw = (px0 + group_dim).min(width) - px0;
                    let ph = (py0 + group_dim).min(height) - py0;
                    let mut tile = vec![0i32; pw * ph];
                    for y in 0..ph {
                        for x in 0..pw {
                            tile[y * pw + x] = alpha_data[(py0 + y) * width + (px0 + x)] as i32;
                        }
                    }
                    Some((tile, pw, ph))
                })
                .collect()
        } else {
            vec![None; num_groups]
        };

        // Optional custom coefficient order for DCT8 (global for all HF groups).
        let effort_cfg = effort_params(effort);
        let custom_orders_8x8 = if effort_cfg.enable_custom_coeff_orders && num_passes == 1 {
            let orders =
                compute_optimal_coeff_orders_8x8(ac_y, ac_x, ac_b, transform_map, bw, 0, 0, bw, bh);
            if orders != [natural_coeff_order_8x8(); 3] {
                Some(orders)
            } else {
                None
            }
        } else {
            None
        };

        // --- Phase 1: Tokenize ALL HF groups' AC data to build global histogram ---
        let num_ac_contexts = num_contexts * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT);

        // Per-pass, per-HF-group token lists.
        let mut group_tokens_passes: Vec<Vec<Vec<AcToken>>> =
            vec![Vec::with_capacity(num_groups); num_passes];

        // Progressive split: early passes carry coarse coefficients, final pass
        // carries residual refinement.
        let mut pass_ac_x: Vec<Vec<i32>> = vec![vec![0i32; ac_x.len()]; num_passes];
        let mut pass_ac_y: Vec<Vec<i32>> = vec![vec![0i32; ac_y.len()]; num_passes];
        let mut pass_ac_b: Vec<Vec<i32>> = vec![vec![0i32; ac_b.len()]; num_passes];
        if num_passes == 1 {
            pass_ac_x[0].copy_from_slice(ac_x);
            pass_ac_y[0].copy_from_slice(ac_y);
            pass_ac_b[0].copy_from_slice(ac_b);
        } else {
            let pass_shifts: Vec<i32> = progressive_pass_shifts
                .iter()
                .copied()
                .map(|s| s as i32)
                .collect();

            let prog_residual_keep_coeffs: usize = if effort <= 3 {
                40
            } else if effort <= 5 {
                48
            } else if effort <= 7 {
                52
            } else {
                64
            };
            let prog_mid_keep_coeffs: usize = if num_passes >= 3 {
                if effort <= 7 { 56 } else { 48 }
            } else {
                64
            };

            for i in 0..ac_x.len() {
                let mut rx = ac_x[i];
                let mut ry = ac_y[i];
                let mut rb = ac_b[i];
                let k = i % 64;

                for p in 0..(num_passes - 1) {
                    let s = pass_shifts.get(p).copied().unwrap_or(0).clamp(0, 3);
                    let bx = rx >> s;
                    let by = ry >> s;
                    let bb = rb >> s;

                    let keep_this_pass = if num_passes >= 3 && p == num_passes - 2 {
                        k < prog_mid_keep_coeffs
                    } else {
                        true
                    };

                    if keep_this_pass {
                        pass_ac_x[p][i] = bx;
                        pass_ac_y[p][i] = by;
                        pass_ac_b[p][i] = bb;
                        rx -= bx << s;
                        ry -= by << s;
                        rb -= bb << s;
                    }
                }

                if k < prog_residual_keep_coeffs {
                    pass_ac_x[num_passes - 1][i] = rx;
                    pass_ac_y[num_passes - 1][i] = ry;
                    pass_ac_b[num_passes - 1][i] = rb;
                }
            }
        }

        for g in 0..num_groups {
            let gx = g % num_groups_x;
            let gy = g / num_groups_x;
            let x0 = gx * group_dim_blocks;
            let y0 = gy * group_dim_blocks;
            let gw = (x0 + group_dim_blocks).min(bw) - x0;
            let gh = (y0 + group_dim_blocks).min(bh) - y0;

            for pass in 0..num_passes {
                let tokens_p = tokenize_hf_region(
                    &pass_ac_x[pass],
                    &pass_ac_y[pass],
                    &pass_ac_b[pass],
                    transform_map,
                    bw,
                    x0,
                    y0,
                    gw,
                    gh,
                    num_contexts,
                    0,
                    custom_orders_8x8.as_ref(),
                    Some(raw_quant_map),
                    block_ctx.as_ref(),
                )?;
                group_tokens_passes[pass].push(tokens_p);
            }
        }

        // Build token encodings for each pass/group.
        let per_pass_uint_configs: Vec<crate::encode::entropy::HybridUintConfig> =
            if num_passes > 1 {
                vec![crate::encode::entropy::HybridUintConfig::new(4, 2, 0); num_passes]
            } else {
                vec![crate::encode::entropy::HybridUintConfig::new(4, 1, 2)]
            };
        let all_encoded_passes: Vec<Vec<Vec<crate::encode::entropy::HybridUintEncoded>>> =
            group_tokens_passes
                .iter()
                .enumerate()
                .map(|(pass, group_tokens)| {
                    let uint_config = per_pass_uint_configs
                        .get(pass)
                        .copied()
                        .unwrap_or(per_pass_uint_configs[0]);
                    group_tokens
                        .iter()
                        .map(|tokens| {
                            tokens
                                .iter()
                                .map(|t| uint_config.encode(t.value))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
        let max_token = all_encoded_passes
            .iter()
            .flat_map(|per_pass| per_pass.iter())
            .flat_map(|enc| enc.iter().map(|e| e.token))
            .max()
            .unwrap_or(0);
        let alphabet_size = (max_token as usize + 1).max(1);

        let mut merged_group_tokens: Vec<Vec<AcToken>> = vec![Vec::new(); num_groups];
        let mut merged_all_encoded: Vec<Vec<crate::encode::entropy::HybridUintEncoded>> =
            vec![Vec::new(); num_groups];
        for pass in 0..num_passes {
            for g in 0..num_groups {
                merged_group_tokens[g].extend(group_tokens_passes[pass][g].iter().copied());
                merged_all_encoded[g].extend(all_encoded_passes[pass][g].iter().copied());
            }
        }

        // --- Phase 2: Write sections ---

        // LfGlobal
        sections.push(encode_lf_global_section(
            global_scale,
            quant_lf,
            block_ctx.as_ref(),
            has_alpha,
        )?);

        // LfGroups (use LF group dimensions)
        for g in 0..num_lf_groups {
            let gx = g % num_lf_groups_x;
            let gy = g / num_lf_groups_x;
            sections.push(encode_lf_group_section(
                gx,
                gy,
                bw,
                bh,
                lf_group_dim_blocks,
                dc_y,
                dc_x,
                dc_b,
                raw_quant_map,
                transform_map,
                ytox_map,
                ytob_map,
            )?);
        }

        let mut context_counts = vec![0u64; num_ac_contexts];
        for tokens in &merged_group_tokens {
            for token in tokens {
                context_counts[token.context] += 1;
            }
        }

        // Build context map candidates.
        let mut context_map_candidates = if num_passes > 1 {
            // Conservative progressive mode: single AC histogram cluster for robustness.
            vec![vec![0u8; num_ac_contexts]]
        } else {
            build_ac_context_map_candidates(num_ac_contexts, &context_counts)
        };
        if num_passes == 1 {
            // Build combined tokens+encoded for greedy clustering.
            let all_tokens_flat: Vec<AcToken> = merged_group_tokens
                .iter()
                .flat_map(|t| t.iter().copied())
                .collect();
            let all_encoded_flat: Vec<crate::encode::entropy::HybridUintEncoded> =
                merged_all_encoded
                    .iter()
                    .flat_map(|e| e.iter().copied())
                    .collect();
            for max_c in [2, 4, 8, 16, 32] {
                if max_c <= num_ac_contexts {
                    let greedy_map = build_greedy_clustered_context_map(
                        num_ac_contexts,
                        alphabet_size,
                        &all_tokens_flat,
                        &all_encoded_flat,
                        max_c,
                    );
                    if !context_map_candidates.iter().any(|m| *m == greedy_map) {
                        context_map_candidates.push(greedy_map);
                    }
                }
            }
        }

        // num_histograms is serialized with ceil_log2(num_groups) bits in HFGlobal,
        // so cluster count must not exceed num_groups.
        context_map_candidates.retain(|m| num_clusters_in_context_map(m) <= num_groups);
        if context_map_candidates.is_empty() {
            context_map_candidates.push(vec![0u8; num_ac_contexts]);
        }

        // Evaluate entropy candidates by exact encoded section size and keep the best.
        let mut best_bits = usize::MAX;
        let mut best_hf_global = None;
        let mut best_hf_groups: Vec<Vec<u8>> = Vec::new();

        for context_map in context_map_candidates {
            let cluster_frequencies = build_cluster_frequencies_for_groups(
                &context_map,
                alphabet_size,
                &merged_group_tokens,
                &merged_all_encoded,
            )?;

            if USE_ANS_AC_ENTROPY {
                let distributions = build_ans_distributions(&cluster_frequencies);
                let hf_global = encode_hf_global_section_with_ans(
                    num_groups,
                    &context_map,
                    &per_pass_uint_configs[0],
                    &distributions,
                    custom_orders_8x8.as_ref(),
                    num_passes,
                )?;
                let mut hf_groups = Vec::with_capacity(num_groups * num_passes);
                for pass in 0..num_passes {
                    for g in 0..num_groups {
                        let at = if pass == 0 {
                            alpha_tiles[g]
                                .as_ref()
                                .map(|(d, w, h)| (d.as_slice(), *w, *h))
                        } else {
                            None
                        };
                        hf_groups.push(encode_hf_group_tokens_ans(
                            1,
                            &group_tokens_passes[pass][g],
                            &all_encoded_passes[pass][g],
                            &context_map,
                            &distributions,
                            at,
                        )?);
                    }
                }
                let bits =
                    hf_global.len() * 8 + hf_groups.iter().map(|s| s.len() * 8).sum::<usize>();
                if bits < best_bits {
                    best_bits = bits;
                    best_hf_global = Some(hf_global);
                    best_hf_groups = hf_groups;
                }
            }

            let mut codes_per_pass: Vec<Vec<crate::encode::entropy::huffman_encode::HuffmanCode>> =
                Vec::with_capacity(num_passes);
            for pass in 0..num_passes {
                let freqs_pass = build_cluster_frequencies_for_groups(
                    &context_map,
                    alphabet_size,
                    &group_tokens_passes[pass],
                    &all_encoded_passes[pass],
                )?;
                codes_per_pass.push(build_huffman_codes_from_frequencies(&freqs_pass)?);
            }

            let hf_global = encode_hf_global_section_with_code(
                num_groups,
                &context_map,
                &per_pass_uint_configs,
                &codes_per_pass,
                custom_orders_8x8.as_ref(),
                num_passes,
            )?;
            let mut hf_groups = Vec::with_capacity(num_groups * num_passes);
            for pass in 0..num_passes {
                for g in 0..num_groups {
                    let at = if pass == 0 {
                        alpha_tiles[g]
                            .as_ref()
                            .map(|(d, w, h)| (d.as_slice(), *w, *h))
                    } else {
                        None
                    };
                    hf_groups.push(encode_hf_group_tokens(
                        1,
                        &group_tokens_passes[pass][g],
                        &all_encoded_passes[pass][g],
                        &context_map,
                        &codes_per_pass[pass],
                        at,
                    )?);
                }
            }
            let bits = hf_global.len() * 8 + hf_groups.iter().map(|s| s.len() * 8).sum::<usize>();
            if bits < best_bits {
                best_bits = bits;
                best_hf_global = Some(hf_global);
                best_hf_groups = hf_groups;
            }
        }

        sections.push(best_hf_global.ok_or(crate::error::Error::InvalidHuffman)?);
        sections.extend(best_hf_groups);

        let section_sizes: Vec<u32> = sections.iter().map(|s| s.len() as u32).collect();
        write_toc(&mut writer, &section_sizes)?;
        writer.byte_align_zero_pad()?;

        let mut result = writer.finish();
        for section in &sections {
            result.extend_from_slice(section);
        }
        Ok(result)
    }
}

/// Encode HF metadata as a modular stream with 4 channels of different sizes.
///
/// Channels:
///   0: ytox_map (cr_w x cr_h)
///   1: ytob_map (cr_w x cr_h)
///   2: transform_image (count x 2)
///   3: epf_map (bw x bh)
///
/// Values are currently simple (zero chroma-correlation/EPF and DCT8x8 transform type),
/// with per-block `raw_quant - 1` carried in transform_image row 1.
fn encode_hf_metadata_modular(
    w: &mut BitWriter,
    cr_w: usize,
    cr_h: usize,
    count: usize,
    bw: usize,
    bh: usize,
    data: &[i32],
) -> Result<()> {
    // Channels:
    //   ch0: ytox_map         (cr_w * cr_h)
    //   ch1: ytob_map         (cr_w * cr_h)
    //   ch2: transform_image  (count * 2)
    //   ch3: epf_map          (bw * bh)
    let total = cr_w * cr_h + cr_w * cr_h + count * 2 + bw * bh;
    assert_eq!(data.len(), total);

    // Encode as a regular modular section with a single zero-predictor tree.
    // The residual stream is channel-concatenated in decode order.
    // For HF metadata, data contains mixed-size channels (ytox, ytob, transform, epf),
    // so per-channel prediction isn't straightforward. Use zero predictor.
    let uint_config = crate::encode::entropy::HybridUintConfig::new(4, 1, 2);
    crate::encode::modular_encode::write_modular_section_huffman(
        w,
        0, // offset
        0, // predictor = Zero
        data,
        uint_config,
        false, // use_global_tree
    )
}

// ==================== AC coefficient tokenization ====================

/// A token in the AC coefficient stream, with its context.
#[derive(Clone, Copy)]
struct AcToken {
    /// Context index for this token (used for multi-histogram routing).
    #[allow(dead_code)]
    context: usize,
    /// The unsigned value to encode (via HybridUint).
    value: u32,
}

#[allow(clippy::too_many_arguments)]
fn tokenize_hf_region(
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    transform_map: &[u8],
    full_bw: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
    num_contexts: usize,
    context_offset: usize,
    custom_orders_8x8: Option<&[[usize; 64]; 3]>,
    // For custom block context maps:
    raw_quant_map: Option<&[u8]>,
    block_ctx_info: Option<&CustomBlockCtx>,
) -> Result<Vec<AcToken>> {
    let mut tokens = Vec::new();
    let mut num_nzeros = [
        vec![0u32; rw * rh],
        vec![0u32; rw * rh],
        vec![0u32; rw * rh],
    ];

    // Decoder order is Y, X, B (channel indices 1,0,2).
    let ac_channels = [ac_y, ac_x, ac_b];
    let channel_indices = [1usize, 0, 2];

    for by in 0..rh {
        for bx in 0..rw {
            let global_idx = (y0 + by) * full_bw + (x0 + bx);
            let raw_transform = transform_map[global_idx];
            if raw_transform & TRANSFORM_FIRST_BLOCK_FLAG == 0 {
                continue;
            }

            let transform_id = raw_transform & !TRANSFORM_FIRST_BLOCK_FLAG;
            let transform_type = HfTransformType::from_usize(transform_id as usize).ok_or(
                crate::error::Error::InvalidVarDCTTransform(transform_id as usize),
            )?;
            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            if bx + cx > rw || by + cy > rh {
                return Err(crate::error::Error::HFBlockOutOfBounds);
            }
            let shape_id = block_shape_id(transform_type) as usize;

            for (ci, &c) in channel_indices.iter().enumerate() {
                let ac = ac_channels[ci];
                let block_ctx = if let (Some(rqm), Some(bci)) = (raw_quant_map, block_ctx_info) {
                    custom_block_context(
                        c,
                        shape_id,
                        rqm[global_idx] as u32,
                        &bci.qf_thresholds,
                        &bci.context_map,
                    )
                } else {
                    default_block_context(c, shape_id, 0)
                };
                let predicted = predict_num_nonzeros(&num_nzeros[c], rw, bx, by);

                if transform_id == DCT8_TRANSFORM_ID {
                    let blk_coeffs = &ac[global_idx * 64..(global_idx + 1) * 64];
                    // Map channel index to order index:
                    // ci=0->Y (order[0]), ci=1->X (order[1]), ci=2->B (order[2])
                    let order_for_chan = custom_orders_8x8.map(|orders| &orders[ci]);
                    let nz = tokenize_block_8x8(
                        blk_coeffs,
                        c,
                        block_ctx,
                        num_contexts,
                        context_offset,
                        predicted,
                        &mut tokens,
                        order_for_chan,
                    );
                    num_nzeros[c][by * rw + bx] = nz as u32;
                    continue;
                }

                if is_supported_nonzero_transform_id(transform_id) {
                    let num_blocks = cx * cy;
                    let num_coeffs = num_blocks * 64;
                    let log_num_blocks = num_blocks.ilog2() as usize;
                    let order = token_shape_order(shape_id).ok_or(
                        crate::error::Error::InvalidVarDCTTransform(transform_id as usize),
                    )?;

                    let coeff_at = |k: usize| {
                        let coeff_index = order[k] as usize;
                        let storage_index = transform_coeff_index_to_block_storage(
                            full_bw,
                            x0 + bx,
                            y0 + by,
                            cx,
                            coeff_index,
                        );
                        ac[storage_index]
                    };

                    let mut nonzeros = 0usize;
                    for k in num_blocks..num_coeffs {
                        if coeff_at(k) != 0 {
                            nonzeros += 1;
                        }
                    }

                    let nz_context = nonzero_context(predicted as usize, block_ctx, num_contexts)
                        + context_offset;
                    tokens.push(AcToken {
                        context: nz_context,
                        value: nonzeros as u32,
                    });

                    let histo_offset =
                        zero_density_context_offset(block_ctx, num_contexts) + context_offset;
                    let mut nz_left = nonzeros;
                    let mut prev: usize = if nonzeros > num_coeffs / 16 { 0 } else { 1 };
                    for k in num_blocks..num_coeffs {
                        if nz_left == 0 {
                            break;
                        }
                        let ctx = histo_offset
                            + block_context_map::zero_density_context(
                                nz_left,
                                k,
                                log_num_blocks,
                                prev,
                            );
                        let coeff = coeff_at(k);
                        tokens.push(AcToken {
                            context: ctx,
                            value: pack_signed(coeff),
                        });
                        prev = if coeff != 0 { 1 } else { 0 };
                        if coeff != 0 {
                            nz_left -= 1;
                        }
                    }

                    let per_block_nz = nonzeros.div_ceil(num_blocks) as u32;
                    for iy in 0..cy {
                        for ix in 0..cx {
                            num_nzeros[c][(by + iy) * rw + (bx + ix)] = per_block_nz;
                        }
                    }
                    continue;
                }

                return Err(crate::error::Error::InvalidVarDCTTransform(
                    transform_id as usize,
                ));
            }
        }
    }

    Ok(tokens)
}

/// Pack a signed integer into an unsigned value for HybridUint encoding.
/// This is the inverse of the decoder's `unpack_signed`.
fn pack_signed(x: i32) -> u32 {
    if x >= 0 {
        (x as u32) << 1
    } else {
        ((-x as u32) << 1) - 1
    }
}

fn natural_coeff_order_from_dims(cx: usize, cy: usize) -> Vec<usize> {
    if cx < cy {
        // Build order for the transposed shape and transpose indices back.
        let transposed = natural_coeff_order_from_dims(cy, cx);
        let xsize = cx * 8;
        let transposed_xsize = cy * 8;
        return transposed
            .into_iter()
            .map(|idx| {
                let tx = idx % transposed_xsize;
                let ty = idx / transposed_xsize;
                tx * xsize + ty
            })
            .collect();
    }

    let xsize = cx * 8;
    let xs = cx / cy;
    let xsm = xs - 1;
    let xss = xs.ilog2() as usize;

    let mut out = vec![0usize; cx * cy * 64];
    let mut cur = cx * cy;

    for i in 0..xsize {
        for j in 0..=i {
            let mut x = j;
            let mut y = i - j;
            if i % 2 != 0 {
                std::mem::swap(&mut x, &mut y);
            }
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;
            let val = if x < cx && y < cy {
                y * cx + x
            } else {
                let v = cur;
                cur += 1;
                v
            };
            out[val] = y * xsize + x;
        }
    }

    for ir in 1..xsize {
        let ip = xsize - ir;
        let i = ip - 1;
        for j in 0..=i {
            let mut x = xsize - 1 - (i - j);
            let mut y = xsize - 1 - j;
            if i % 2 != 0 {
                std::mem::swap(&mut x, &mut y);
            }
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;
            out[cur] = y * xsize + x;
            cur += 1;
        }
    }

    out
}

fn natural_coeff_order_for_transform(transform: HfTransformType) -> Vec<usize> {
    let cx = covered_blocks_x(transform) as usize;
    let cy = covered_blocks_y(transform) as usize;
    natural_coeff_order_from_dims(cx, cy)
}

/// Natural (zigzag) coefficient order for DCT8x8.
/// Maps scan position k (0..64) to coefficient index in the 8x8 block.
fn natural_coeff_order_8x8() -> [usize; 64] {
    // Standard JPEG/JXL zigzag order
    let order: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];
    order
}

/// Compute optimal coefficient order for 8x8 DCT by counting non-zero frequencies.
///
/// Returns 3 custom orders (Y, X, B channels) based on zero density analysis.
/// Positions with more non-zero coefficients are scanned first, improving entropy coding.
fn compute_optimal_coeff_orders_8x8(
    ac_y: &[i32],
    ac_x: &[i32],
    ac_b: &[i32],
    transform_map: &[u8],
    full_bw: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
) -> [[usize; 64]; 3] {
    let natural = natural_coeff_order_8x8();
    let ac_channels = [ac_y, ac_x, ac_b];
    let mut result = [natural; 3];

    for (ci, ac) in ac_channels.iter().enumerate() {
        // Count non-zeros at each of the 64 coefficient positions
        let mut nonzero_count = [0u64; 64];
        let mut total_blocks = 0u64;

        for by in 0..rh {
            for bx in 0..rw {
                let global_idx = (y0 + by) * full_bw + (x0 + bx);
                let raw_transform = transform_map[global_idx];
                if raw_transform & TRANSFORM_FIRST_BLOCK_FLAG == 0 {
                    continue;
                }
                let transform_id = raw_transform & !TRANSFORM_FIRST_BLOCK_FLAG;
                if transform_id != DCT8_TRANSFORM_ID {
                    continue;
                }
                total_blocks += 1;
                for k in 1..64 {
                    if ac[global_idx * 64 + natural[k]] != 0 {
                        nonzero_count[k] += 1;
                    }
                }
            }
        }

        if total_blocks == 0 {
            continue;
        }

        // Sort scan positions 1..64 by non-zero count DESCENDING (most non-zeros first).
        // Position 0 (DC) stays fixed.
        let mut positions: Vec<(usize, u64, usize)> = (1..64)
            .map(|k| (k, nonzero_count[k], k)) // (scan_pos, count, original_idx for stability)
            .collect();
        // Sort descending by count, then by original position for stability
        positions.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

        // Build the optimized order: scan position k maps to coefficient natural[positions[k-1].0]
        result[ci][0] = natural[0]; // DC stays at position 0
        for (new_k, &(old_k, _, _)) in positions.iter().enumerate() {
            result[ci][new_k + 1] = natural[old_k];
        }
    }

    result
}

/// Compute Lehmer code for a permutation relative to natural order.
///
/// Given a custom coefficient order `custom` and the `natural` order,
/// computes the Lehmer code representation that the decoder needs to reconstruct
/// the custom order from the natural order.
///
/// `skip` is the number of leading elements to skip (LF coefficients, typically 1 for 8x8).
///
/// Returns (lehmer_codes, end) where end is the length of meaningful codes.
fn compute_lehmer_code(custom: &[usize], natural: &[usize], skip: usize) -> (Vec<u32>, usize) {
    let n = custom.len();
    assert_eq!(n, natural.len());

    // The permutation the decoder applies: natural_order[permutation[i]] = custom_order[i]
    // So we need permutation[i] = natural_inverse[custom[i]]
    let mut natural_inverse = vec![0usize; 64]; // Assumes max size 64
    for (i, &v) in natural.iter().enumerate() {
        natural_inverse[v] = i;
    }
    let mut perm: Vec<usize> = custom.iter().map(|&c| natural_inverse[c]).collect();

    // Only consider elements from skip onwards
    let perm_slice = &mut perm[skip..];
    let m = perm_slice.len();

    // Compute Lehmer codes using O(n^2) (fine for n=63)
    let mut lehmer = Vec::with_capacity(m);
    let mut available: Vec<usize> = (0..m).collect();
    // Remap perm_slice to be relative to available set
    // perm_slice values are indices into the full perm; we need to offset by skip
    let adjusted: Vec<usize> = perm_slice.iter().map(|&v| v - skip).collect();

    for i in 0..m {
        let val = adjusted[i];
        // Find rank of val in available
        let rank = available.iter().position(|&a| a == val).unwrap();
        lehmer.push(rank as u32);
        available.remove(rank);
        // Adjust remaining values (not needed since we track available set)
    }

    // Find effective end (trim trailing zeros)
    let end = lehmer
        .iter()
        .rposition(|&v| v != 0)
        .map(|p| p + 1)
        .unwrap_or(0);

    (lehmer, end)
}

/// Encode coefficient order permutations into the bitstream.
///
/// Uses Huffman-coded Lehmer code, matching the decoder's `decode_coeff_orders`.
fn encode_coeff_orders(
    w: &mut BitWriter,
    orders: &[[usize; 64]; 3], // Y, X, B channel orders
) -> Result<()> {
    let natural = natural_coeff_order_8x8();
    let natural_slice = natural.as_slice();
    let skip = 1usize; // 1 LF coefficient for 8x8

    // Collect all Lehmer codes for all 3 channels
    let mut all_codes: Vec<(Vec<u32>, usize)> = Vec::new();
    for order in orders {
        let (codes, end) = compute_lehmer_code(order, natural_slice, skip);
        all_codes.push((codes, end));
    }

    // Encode using the permutation entropy coding format:
    // First: a histogram with NUM_PERMUTATION_CONTEXTS=8 contexts
    // Then: for each channel, encode (end, lehmer_codes[0..end])

    // Collect all tokens: (context, value) pairs
    let size = 64u32;
    let mut tokens: Vec<(usize, u32)> = Vec::new();
    for (codes, end) in &all_codes {
        // First token: end value, context = get_context(size)
        let ctx = permutation_context(size);
        tokens.push((ctx, *end as u32));

        // Then: lehmer codes with context = get_context(prev_val)
        let mut prev_val = 0u32;
        for &code in codes.iter().take(*end) {
            let ctx = permutation_context(prev_val);
            tokens.push((ctx, code));
            prev_val = code;
        }
    }

    // Build Huffman code for the permutation tokens (8 contexts, map all to cluster 0)
    let context_map = [0u8; 8];
    let uint_config = crate::encode::entropy::HybridUintConfig::new(4, 1, 2);

    let encoded: Vec<_> = tokens.iter().map(|(_, v)| uint_config.encode(*v)).collect();
    let max_symbol = encoded.iter().map(|e| e.token).max().unwrap_or(0) as usize + 1;

    let mut freqs = vec![0u64; max_symbol.max(1)];
    for e in &encoded {
        freqs[e.token as usize] += 1;
    }
    let code = build_huffman_code(&freqs).ok_or(crate::error::Error::InvalidHuffman)?;

    use crate::encode::entropy::huffman_encode::{write_huffman_histograms, write_huffman_symbol};
    write_huffman_histograms(w, &context_map, &[uint_config], &[code.clone()])?;

    for ((ctx, _), enc) in tokens.iter().zip(encoded.iter()) {
        let cluster = context_map[*ctx] as usize;
        write_huffman_symbol(w, &code, enc.token as usize)?;
        if enc.nbits > 0 {
            w.write(enc.nbits as usize, enc.extra_bits as u64)?;
        }
        let _ = cluster; // All mapped to cluster 0
    }

    Ok(())
}

/// Context function for permutation encoding, matching decoder's get_context.
fn permutation_context(x: u32) -> usize {
    let log2 = if x == 0 { 0 } else { 32 - (x).leading_zeros() };
    (log2 as usize).min(7)
}

/// Tokenize AC coefficients for a single block (DCT8x8).
///
/// Produces tokens matching the decoder's reading order in `decode_vardct_group`.
fn tokenize_block_8x8(
    coeffs: &[i32], // 64 coefficients, DC position is 0 (not used)
    _channel: usize,
    block_context: usize,
    num_contexts: usize,
    context_offset: usize,
    num_nzeros_left: u32, // predicted nonzeros from neighbors
    tokens: &mut Vec<AcToken>,
    custom_order: Option<&[usize; 64]>,
) -> usize {
    let default_order = natural_coeff_order_8x8();
    let order = custom_order.unwrap_or(&default_order);

    // Count actual nonzeros (positions 1..64 in scan order)
    let mut nonzeros = 0usize;
    for k in 1..64 {
        if coeffs[order[k]] != 0 {
            nonzeros += 1;
        }
    }

    // Emit nonzeros count token
    let predicted = num_nzeros_left;
    let nz_context =
        nonzero_context(predicted as usize, block_context, num_contexts) + context_offset;
    tokens.push(AcToken {
        context: nz_context,
        value: nonzeros as u32,
    });

    // Emit coefficient tokens
    let histo_offset = zero_density_context_offset(block_context, num_contexts) + context_offset;
    let mut nz_left = nonzeros;
    let mut prev: usize = if nonzeros > 64 / 16 { 0 } else { 1 };

    for k in 1..64 {
        if nz_left == 0 {
            break;
        }
        let ctx = histo_offset + block_context_map::zero_density_context(nz_left, k, 0, prev);
        let coeff = coeffs[order[k]];
        let unsigned = pack_signed(coeff);
        tokens.push(AcToken {
            context: ctx,
            value: unsigned,
        });
        prev = if coeff != 0 { 1 } else { 0 };
        if coeff != 0 {
            nz_left -= 1;
        }
    }

    nonzeros
}

/// Compute block context using a custom BlockContextMap.
fn custom_block_context(
    channel: usize,
    shape_id: usize,
    qf: u32,
    qf_thresholds: &[u32],
    context_map: &[u8],
) -> usize {
    let ch_remap = if channel < 2 { channel ^ 1 } else { 2 };
    let num_orders = 13;
    let shape_id = shape_id.min(num_orders - 1);
    let mut qf_idx = 0usize;
    for t in qf_thresholds {
        if qf > *t {
            qf_idx += 1;
        }
    }
    // No lf_thresholds -> num_lf_contexts = 1, lf_idx = 0
    let idx = ch_remap * num_orders + shape_id;
    let idx = idx * (qf_thresholds.len() + 1) + qf_idx;
    // idx * num_lf_contexts + lf_idx = idx * 1 + 0
    if idx < context_map.len() {
        context_map[idx] as usize
    } else {
        0
    }
}

/// Compute block context using the default BlockContextMap.
/// Simplified version of BlockContextMap::block_context for default map.
fn default_block_context(channel: usize, shape_id: usize, _quant_lf_idx: usize) -> usize {
    // Default context map has:
    //   no lf thresholds (num_lf_contexts=1), no qf thresholds
    //   context_map indices for (channel, shape, qf, lf) -> block_context
    //
    // idx = channel_remap * NUM_ORDERS + shape
    // idx = idx * (qf_thresholds.len()+1) + qf_idx
    // idx = idx * num_lf_contexts + lf_idx
    // channel_remap: 0->1(Y), 1->0(X), 2->2(B)
    let ch_remap = if channel < 2 { channel ^ 1 } else { 2 };
    let num_orders = 13; // NUM_ORDERS
    let shape_id = shape_id.min(num_orders - 1);
    let idx = ch_remap * num_orders + shape_id;
    // Default: no qf thresholds and no lf thresholds -> idx * 1 + 0 = idx

    // Default context_map lookup (from decoder):
    // [0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6,
    //  7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
    //  7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14]
    const DEFAULT_CTX_MAP: [u8; 39] = [
        0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 7,
        8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
    ];
    DEFAULT_CTX_MAP[idx] as usize
}

/// Compute nonzero context. Matches BlockContextMap::nonzero_context.
fn nonzero_context(nonzeros: usize, block_context: usize, num_contexts: usize) -> usize {
    let bucket = if nonzeros < 8 {
        nonzeros
    } else if nonzeros < 64 {
        4 + nonzeros / 2
    } else {
        36
    };
    bucket * num_contexts + block_context
}

/// Compute zero_density_context_offset. Matches BlockContextMap::zero_density_context_offset.
fn zero_density_context_offset(block_context: usize, num_contexts: usize) -> usize {
    num_contexts * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_context
}

/// Predict number of nonzeros for a block based on already encoded neighbors.
/// Matches decoder-side `frame::group::predict_num_nonzeros` for DCT8x8 blocks.
fn predict_num_nonzeros(num_nzeros_map: &[u32], stride: usize, bx: usize, by: usize) -> u32 {
    if bx == 0 {
        if by == 0 {
            32
        } else {
            num_nzeros_map[(by - 1) * stride]
        }
    } else if by == 0 {
        num_nzeros_map[by * stride + (bx - 1)]
    } else {
        (num_nzeros_map[(by - 1) * stride + bx] + num_nzeros_map[by * stride + (bx - 1)])
            .div_ceil(2)
    }
}

fn global_scale_coder() -> U32Coder {
    U32Coder::Select(
        U32::BitsOffset { n: 11, off: 1 },
        U32::BitsOffset { n: 11, off: 2049 },
        U32::BitsOffset { n: 12, off: 4097 },
        U32::BitsOffset { n: 16, off: 8193 },
    )
}

fn quant_lf_coder() -> U32Coder {
    U32Coder::Select(
        U32::Val(16),
        U32::BitsOffset { n: 5, off: 1 },
        U32::BitsOffset { n: 8, off: 1 },
        U32::BitsOffset { n: 16, off: 1 },
    )
}

/// Configuration for VarDCT frame header writing.
struct FrameHeaderConfig {
    use_gab: bool,
    num_extra_channels: u32,
    have_animation: bool,
    duration: u32,
    is_last: bool,
    num_passes: u32,
    pass_shifts: Vec<u32>,
}

/// Unified VarDCT frame header writer.
fn write_vardct_frame_header_full(writer: &mut BitWriter, cfg: &FrameHeaderConfig) -> Result<()> {
    // 1. all_default = false (we need VarDCT settings)
    writer.write(1, 0)?;
    // 2. frame_type = RegularFrame (0)
    writer.write(2, 0)?;
    // 3. encoding = VarDCT (0)
    writer.write(1, 0)?;
    // 4. flags = 0
    writer.write(2, 0)?;
    // 7. upsampling = 1
    writer.write(2, 0)?;
    // 8. ec_upsampling: one entry per extra channel, each = 1 (u2S(1,2,4,8), selector 00)
    for _ in 0..cfg.num_extra_channels {
        writer.write(2, 0)?; // upsampling = 1
    }
    // 10. x_qm_scale = 3
    writer.write(3, 3)?;
    // 11. b_qm_scale = 2
    writer.write(3, 2)?;
    // 12. passes
    let num_passes = cfg.num_passes.max(1);
    match num_passes {
        1 => writer.write(2, 0)?,
        2 => writer.write(2, 1)?,
        3 => writer.write(2, 2)?,
        n => {
            writer.write(2, 3)?;
            writer.write(3, (n - 4) as u64)?;
        }
    }
    if num_passes != 1 {
        // num_ds = 0
        writer.write(2, 0)?;
        // shift for each pass except last (num_passes - 1 entries).
        for i in 0..(num_passes - 1) {
            let s = cfg.pass_shifts.get(i as usize).copied().unwrap_or(0).min(3);
            writer.write(2, s as u64)?;
        }
        // no downsample / last_pass entries because num_ds == 0
    }
    // 14. have_crop = false
    writer.write(1, 0)?;
    // 16. blending_info: mode = Replace (0)
    writer.write(2, 0)?;
    // For Replace mode: alpha_channel (cond Blend/AlphaWeightedAdd) NOT WRITTEN
    //                    clamp (cond Blend/AlphaWeightedAdd/Mul) NOT WRITTEN
    //                    source (cond !(full_frame && Replace)) NOT WRITTEN for full_frame
    // 17. ec_blending_info: one per extra channel
    for _ in 0..cfg.num_extra_channels {
        // mode = Replace (0)
        writer.write(2, 0)?;
        // alpha_channel: cond num_extra_channels>0 && mode not Replace/None...
        // Actually for Replace mode: alpha_channel, clamp, source conditions:
        //   alpha_channel: cond num_extra_channels > 0 && (mode == kBlend || mode == kAlphaWeightedAdd)
        //   For Replace: condition is false, NOT WRITTEN
        //   clamp: same condition, NOT WRITTEN
        //   source: cond !(full_frame && Replace) = false => NOT WRITTEN
    }
    // 18. duration: cond have_animation
    if cfg.have_animation {
        write_u32(writer, &duration_coder(), cfg.duration)?;
    }
    // 20. is_last
    writer.write(1, if cfg.is_last { 1 } else { 0 })?;
    // 21. save_as_reference: cond !is_last
    if !cfg.is_last {
        writer.write(2, 0)?;
    }
    // 22. save_before_ct: #[condition(false)] - never serialized
    // 23. name: size = 0
    writer.write(2, 0)?;
    // 24. restoration_filter
    writer.write(1, 0)?; // all_default = false
    writer.write(1, if cfg.use_gab { 1 } else { 0 })?;
    if cfg.use_gab {
        writer.write(1, 0)?; // gab_custom = false
    }
    writer.write(2, 2)?; // epf_iters = 2
    writer.write(1, 0)?; // epf_sharp_custom = false
    writer.write(1, 0)?; // epf_weight_custom = false
    writer.write(1, 0)?; // epf_sigma_custom = false
    writer.write(2, 0)?; // LoopFilter extensions = 0
    // 25. FrameHeader extensions = 0
    writer.write(2, 0)?;
    Ok(())
}

/// Legacy wrapper: no extra channels, no animation, is_last=true.
fn write_vardct_frame_header(
    writer: &mut BitWriter,
    _width: u32,
    _height: u32,
    use_gab: bool,
) -> Result<()> {
    write_vardct_frame_header_full(
        writer,
        &FrameHeaderConfig {
            use_gab,
            num_extra_channels: 0,
            have_animation: false,
            duration: 0,
            is_last: true,
            num_passes: 1,
            pass_shifts: vec![],
        },
    )
}

/// Legacy wrapper for animation: no extra channels.
fn write_vardct_frame_header_animated(
    writer: &mut BitWriter,
    _width: u32,
    _height: u32,
    use_gab: bool,
    duration: u32,
    is_last: bool,
) -> Result<()> {
    write_vardct_frame_header_full(
        writer,
        &FrameHeaderConfig {
            use_gab,
            num_extra_channels: 0,
            have_animation: true,
            duration,
            is_last,
            num_passes: 1,
            pass_shifts: vec![],
        },
    )
}

fn duration_coder() -> crate::headers::encodings::U32Coder {
    use crate::headers::encodings::{U32, U32Coder};
    // u2S(0, 1, Bits(8), Bits(32))
    U32Coder::Select(U32::Val(0), U32::Val(1), U32::Bits(8), U32::Bits(32))
}

/// Encode all frame data as a single section (for single-group images).
#[allow(clippy::too_many_arguments)]
fn encode_single_group_section(
    bw: usize,
    bh: usize,
    width: usize,
    height: usize,
    global_scale: u32,
    quant_lf: u32,
    dc_y: &[i32],
    dc_x: &[i32],
    dc_b: &[i32],
    ac_x: &[i32],
    ac_y: &[i32],
    ac_b: &[i32],
    raw_quant_map: &[u8],
    transform_map: &[u8],
    ytox_map: &[i32],
    ytob_map: &[i32],
    alpha: Option<&[u8]>, // u8 alpha channel, width*height pixels
    effort: u8,
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    let num_blocks = bw * bh;
    assert_eq!(raw_quant_map.len(), num_blocks);
    assert_eq!(transform_map.len(), num_blocks);

    // === LfGlobal ===
    // LfQuantFactors: all_default = true
    w.write(1, 1)?;
    // QuantizerParams
    write_u32(&mut w, &global_scale_coder(), global_scale)?;
    write_u32(&mut w, &quant_lf_coder(), quant_lf)?;
    // BlockContextMap: all_default = true
    w.write(1, 1)?;
    // ColorCorrelationParams: all_default = true
    w.write(1, 1)?;
    // Global tree: not present
    w.write(1, 0)?;
    // Modular global: for VarDCT, only extra channels go here.
    // With 0 extra channels: empty. With alpha: encode it as modular.
    if let Some(alpha_data) = alpha {
        // Alpha channel goes in the modular global subbitstream.
        // For single-group images, all modular channels that are "small enough"
        // (size <= group_dim) go into the global section.
        // Our single-group images are always <= 256x256 pixels, so alpha fits.
        let alpha_i32: Vec<i32> = alpha_data.iter().map(|&a| a as i32).collect();
        crate::encode::modular_encode::encode_modular_signed_stream(
            &mut w, width, height, 1, &alpha_i32,
        )?;
    }

    // === LfGroup0: VarDCT DC ===
    // extra_precision = 0 (2 bits)
    w.write(2, 0)?;
    // DC coefficients as modular (3 channels: Y, X, B order as per decode_vardct_lf)
    // The decoder creates channels in order: [shrink_rect(1), shrink_rect(0), shrink_rect(2)]
    // which for non-subsampled is [Y_chan, X_chan, B_chan]
    let mut dc_data = vec![0i32; num_blocks * 3];
    for i in 0..num_blocks {
        dc_data[i] = dc_y[i]; // Channel 0: Y
        dc_data[num_blocks + i] = dc_x[i]; // Channel 1: X
        dc_data[2 * num_blocks + i] = dc_b[i]; // Channel 2: B
    }
    crate::encode::modular_encode::encode_modular_signed_stream(&mut w, bw, bh, 3, &dc_data)?;

    // === LfGroup0: ModularLF (empty for 0 extra channels) ===

    // === LfGroup0: HF metadata ===
    // The HF metadata encodes 4 modular channels:
    //   ch0: ytox_map (cr_w x cr_h, where cr = blocks/8 ceiled)
    //   ch1: ytob_map (same size)
    //   ch2: transform_image (count x 2)
    //   ch3: epf_map (bw x bh)
    //
    // First: count is read from ceil_log2(bw*bh) bits, value = count-1.
    let upper_bound = bw * bh;
    let count_num_bits = if upper_bound <= 1 {
        0
    } else {
        32 - (upper_bound as u32 - 1).leading_zeros()
    };

    let transform_entries =
        collect_transform_entries_for_rect(transform_map, raw_quant_map, bw, 0, 0, bw, bh);
    let count = transform_entries.len();
    assert!(count > 0 && count <= upper_bound);

    // Write count-1 in count_num_bits bits.
    if count_num_bits > 0 {
        w.write(count_num_bits as usize, (count - 1) as u64)?;
    }

    // Build modular channels for HF metadata
    let cr_w = bw.div_ceil(8); // chroma correlation map size
    let cr_h = bh.div_ceil(8);
    let ch0_size = cr_w * cr_h; // ytox
    let ch1_size = cr_w * cr_h; // ytob
    let ch2_size = count * 2; // transform (count x 2)
    let ch3_size = bw * bh; // epf
    let total = ch0_size + ch1_size + ch2_size + ch3_size;

    let mut hf_meta = vec![0i32; total];
    // ch0 (ytox): copy from ytox_map
    for i in 0..ch0_size {
        hf_meta[i] = ytox_map[i];
    }
    // ch1 (ytob): copy from ytob_map
    for i in 0..ch1_size {
        hf_meta[ch0_size + i] = ytob_map[i];
    }
    // ch2 (transform_image):
    //   row 0: transform type ids
    //   row 1: raw_quant - 1
    let ch2_off = ch0_size + ch1_size;
    for (i, (transform_id, raw_quant)) in transform_entries.iter().copied().enumerate() {
        hf_meta[ch2_off + i] = transform_id as i32;
        hf_meta[ch2_off + count + i] = raw_quant.saturating_sub(1) as i32;
    }
    // ch3 (epf): per-block sharpness (epf_iters=2)
    let ch3_off = ch2_off + 2 * count;
    for i in 0..count {
        hf_meta[ch3_off + i] = 4; // EPF sharpness default
    }

    // The 4 channels have different sizes and are encoded as a modular subbitstream.
    encode_hf_metadata_modular(&mut w, cr_w, cr_h, count, bw, bh, &hf_meta)?;

    // === HfGlobal + HfGroup0: AC coefficients ===
    // Compute optimal coefficient order for 8x8 DCT
    let custom_orders =
        compute_optimal_coeff_orders_8x8(ac_y, ac_x, ac_b, transform_map, bw, 0, 0, bw, bh);
    let use_custom_order = effort >= 6 && custom_orders != [natural_coeff_order_8x8(); 3];

    // Tokenize all blocks' AC coefficients
    let num_contexts = 15; // default BlockContextMap has 15 block contexts
    let num_ac_contexts = num_contexts * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT);
    let context_offset = 0; // single histogram set

    let tokens = tokenize_hf_region(
        ac_x,
        ac_y,
        ac_b,
        transform_map,
        bw,
        0,
        0,
        bw,
        bh,
        num_contexts,
        context_offset,
        if use_custom_order {
            Some(&custom_orders)
        } else {
            None
        },
        None, // no custom block context for single-group
        None,
    )?;

    // === HfGlobal ===
    // DequantMatrices: all_default = true
    w.write(1, 1)?;
    // num_histograms: ceil_log2(1) = 0 bits (1 group -> 0 bits)
    if use_custom_order {
        // used_orders = 1 (bit 0 = DCT8x8 order customized)
        // kOrderEnc = U32Enc(Val(0x5F), Val(0x13), Val(0), Bits(13))
        // value 1 needs Bits(13) = selector 3
        w.write(2, 3)?; // selector 3 = Bits(13)
        w.write(13, 1)?; // value = 1 (only DCT8x8)
        // Encode the permutation for 3 channels.
        // Decoder reads in order c=0,1,2 = X,Y,B, but our compute returns [Y,X,B].
        // Reorder to match decoder expectations: [X, Y, B].
        let decoder_order = [custom_orders[1], custom_orders[0], custom_orders[2]];
        encode_coeff_orders(&mut w, &decoder_order)?;
    } else {
        // used_orders: selector 2 = no custom orders (value 0)
        w.write(2, 2)?;
    }

    // Build and write AC entropy histograms
    write_ac_histograms_and_tokens(&mut w, num_ac_contexts, &tokens, USE_ANS_AC_ENTROPY)?;

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Encode the LfGlobal section.
/// Custom block context map parameters.
struct CustomBlockCtx {
    qf_thresholds: Vec<u32>, // quant field thresholds
    context_map: Vec<u8>,    // context map entries
    num_contexts: usize,     // max(context_map) + 1
}

/// Compute an optimized block context map (port of libjxl's FindBestBlockEntropyModel).
/// Returns None if the image is too small to benefit.
fn compute_block_context_map(
    raw_quant_map: &[u8],
    _transform_map: &[u8],
    bw: usize,
    bh: usize,
) -> Option<CustomBlockCtx> {
    let tot = bw * bh;
    // libjxl: size_for_ctx_model = 1024 * distance.
    // Only create custom map for sufficiently large images where the overhead
    // of encoding the context map is amortized.
    if tot < 8192 {
        return None;
    }

    // Count qf value occurrences
    let mut qf_counts = [0usize; 256];
    for &rq in raw_quant_map {
        qf_counts[rq as usize] += 1;
    }

    // Find median qf to use as threshold
    let mut cumsum = 0usize;
    let half = tot / 2;
    let mut median_qf = 0u32;
    for j in 0..256 {
        cumsum += qf_counts[j];
        if cumsum > half {
            median_qf = j as u32;
            break;
        }
    }

    // Only create custom map if there's actual variance in qf
    let qf_min = raw_quant_map.iter().copied().min().unwrap_or(0);
    let qf_max = raw_quant_map.iter().copied().max().unwrap_or(0);
    if qf_min == qf_max {
        return None; // Uniform quant field, no benefit
    }

    let qf_thresholds = vec![median_qf];

    // Build context map: 3 channels * 13 orders * 2 qf segments = 78 entries.
    // Use the default 15-context structure but split by qf for the most
    // common orders (DCT8 = order 0). This gives better entropy separation
    // for blocks with different quantization levels.
    //
    // Default context map for qf=0 (low): same as default 15-context map.
    // For qf=1 (high): offset the Y channel contexts by num_default_y_contexts.
    let num_orders = 13usize;
    let num_qf_segments = 2usize;
    let map_size = 3 * num_orders * num_qf_segments;

    // Start with the default map structure (39 entries) replicated for 2 qf segments
    const DEFAULT_CTX_MAP: [u8; 39] = [
        0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 7,
        8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
    ];
    // Y uses contexts 0-6 (7 unique), X uses 7-14 (8 unique), B same as X
    // For qf split: Y-low=0-6, Y-high=7-13 (shifted by 7), X/B use shared 14-15
    // Total: 7 (Y-low) + 7 (Y-high) + 2 (X/B shared) = 16 -- too many (max 16)
    // Simpler: just split Y's DCT8 context (order 0) by qf.
    // Y-order0-lowqf = 0, Y-order0-highqf = new cluster.
    // This just adds 1 extra cluster for the most common case.

    let mut context_map = vec![0u8; map_size];
    let mut max_ctx = 0u8;

    for ch_remap in 0..3usize {
        for order in 0..num_orders {
            for qf_seg in 0..num_qf_segments {
                let flat_idx = ch_remap * num_orders + order;
                let base_ctx = DEFAULT_CTX_MAP[flat_idx.min(38)];
                let ctx = if ch_remap == 1 && order == 0 && qf_seg == 1 {
                    // Y channel, DCT8, high quant -> new context = 15
                    15
                } else {
                    base_ctx
                };
                let out_idx =
                    ch_remap * num_orders * num_qf_segments + order * num_qf_segments + qf_seg;
                if out_idx < map_size {
                    context_map[out_idx] = ctx;
                    if ctx > max_ctx {
                        max_ctx = ctx;
                    }
                }
            }
        }
    }

    let num_contexts = max_ctx as usize + 1;
    if num_contexts > 16 {
        return None; // Can't exceed 16 contexts
    }

    Some(CustomBlockCtx {
        qf_thresholds,
        context_map,
        num_contexts,
    })
}

fn encode_lf_global_section(
    global_scale: u32,
    quant_lf: u32,
    block_ctx: Option<&CustomBlockCtx>,
    has_alpha: bool,
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    // LfQuantFactors: all_default = true
    w.write(1, 1)?;
    // QuantizerParams
    write_u32(&mut w, &global_scale_coder(), global_scale)?;
    write_u32(&mut w, &quant_lf_coder(), quant_lf)?;

    if let Some(ctx) = block_ctx {
        // BlockContextMap: all_default = false
        w.write(1, 0)?;

        // lf_thresholds: 3 channels, each with 0 thresholds
        for _ in 0..3 {
            w.write(4, 0)?; // num_lf_thresholds = 0
        }

        // qf_thresholds
        let num_qf = ctx.qf_thresholds.len();
        w.write(4, num_qf as u64)?;
        for &t in &ctx.qf_thresholds {
            // Encode threshold value: t is 1-based (threshold = t)
            // Val = t - 1 (since we read +1 in decoder)
            let val = t.saturating_sub(1) as u64;
            if val < 4 {
                w.write(2, 0)?; // tag 0
                w.write(2, val)?;
            } else if val < 12 {
                w.write(2, 1)?; // tag 1
                w.write(3, val - 4)?;
            } else if val < 44 {
                w.write(2, 2)?; // tag 2
                w.write(5, val - 12)?;
            } else {
                w.write(2, 3)?; // tag 3
                w.write(8, val - 44)?;
            }
        }

        // Context map
        write_context_map(&mut w, &ctx.context_map)?;
    } else {
        // BlockContextMap: all_default = true
        w.write(1, 1)?;
    }

    // ColorCorrelationParams: all_default = true
    w.write(1, 1)?;
    // Global tree: not present
    w.write(1, 0)?;
    // Modular global:
    // For VarDCT with 0 extra channels, nothing is read (channels list empty).
    // For VarDCT with alpha (multi-group), the decoder reads a GroupHeader
    // because channels is non-empty, but the global section has no channels to
    // decode (all alpha data is in HF groups). We still must write the GroupHeader.
    if has_alpha {
        // Write empty GroupHeader: use_global_tree=false, no transforms
        // GroupHeader:
        //   use_global_tree: Bool(default false)
        //   wp_params: WPHeader (default, only if !use_global_tree)
        //   num_transforms: u2S(0, 1, Bits(4)+2, Bits(8)+18) = 0 (selector 00)
        w.write(1, 0)?; // use_global_tree = false
        // WPHeader: all_default = true
        w.write(1, 1)?;
        // num_transforms = 0
        w.write(2, 0)?;
    }
    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Encode an LfGroup section (DC coefficients + HF metadata).
fn encode_lf_group_section(
    gx: usize,
    gy: usize,
    bw: usize,
    bh: usize,
    group_dim_blocks: usize,
    dc_y: &[i32],
    dc_x: &[i32],
    dc_b: &[i32],
    raw_quant_map: &[u8],
    transform_map: &[u8],
    ytox_map: &[i32],
    ytob_map: &[i32],
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();

    let x0 = gx * group_dim_blocks;
    let y0 = gy * group_dim_blocks;
    let gw = (x0 + group_dim_blocks).min(bw) - x0;
    let gh = (y0 + group_dim_blocks).min(bh) - y0;
    assert_eq!(raw_quant_map.len(), bw * bh);
    assert_eq!(transform_map.len(), bw * bh);

    // === VarDCT LF: DC coefficients ===
    // extra_precision = 0 (2 bits)
    w.write(2, 0)?;

    // DC as modular (3 channels: Y, X, B order matching decoder's decode_vardct_lf)
    let npixels = gw * gh;
    let mut dc_data = vec![0i32; npixels * 3];
    for y in 0..gh {
        for x in 0..gw {
            let src = (y0 + y) * bw + (x0 + x);
            let dst = y * gw + x;
            dc_data[dst] = dc_y[src]; // Channel 0: Y
            dc_data[npixels + dst] = dc_x[src]; // Channel 1: X
            dc_data[2 * npixels + dst] = dc_b[src]; // Channel 2: B
        }
    }
    crate::encode::modular_encode::encode_modular_signed_stream(&mut w, gw, gh, 3, &dc_data)?;

    // === ModularLF: empty (0 extra channels) ===
    // Nothing to write.

    // === HF metadata: 4 channels ===
    // Same format as single-group: count field + 4-channel modular stream.
    // Channels: ytox_map (cr_w x cr_h), ytob_map, transform_image (count x 2), epf_map (gw x gh)
    let upper_bound = gw * gh;
    let count_num_bits = if upper_bound <= 1 {
        0
    } else {
        32 - (upper_bound as u32 - 1).leading_zeros()
    };

    let transform_entries =
        collect_transform_entries_for_rect(transform_map, raw_quant_map, bw, x0, y0, gw, gh);
    let count = transform_entries.len();
    assert!(count > 0 && count <= upper_bound);

    if count_num_bits > 0 {
        w.write(count_num_bits as usize, (count - 1) as u64)?;
    }

    let cr_w = gw.div_ceil(8);
    let cr_h = gh.div_ceil(8);
    let global_cr_w = bw.div_ceil(8);
    let ch0_size = cr_w * cr_h;
    let ch1_size = cr_w * cr_h;
    let total = ch0_size + ch1_size + count * 2 + gw * gh;
    let mut hf_meta_data = vec![0i32; total];
    // ch0 (ytox) and ch1 (ytob): copy from global maps for this group's tile range
    let tile_x0 = x0 / 8;
    let tile_y0 = y0 / 8;
    for ty in 0..cr_h {
        for tx in 0..cr_w {
            let global_idx = (tile_y0 + ty) * global_cr_w + (tile_x0 + tx);
            hf_meta_data[ty * cr_w + tx] = ytox_map[global_idx];
            hf_meta_data[ch0_size + ty * cr_w + tx] = ytob_map[global_idx];
        }
    }
    let ch2_off = ch0_size + ch1_size;
    for (i, (transform_id, raw_quant)) in transform_entries.iter().copied().enumerate() {
        hf_meta_data[ch2_off + i] = transform_id as i32;
        hf_meta_data[ch2_off + count + i] = raw_quant.saturating_sub(1) as i32;
    }
    // EPF sharpness = 4 (default) for all blocks
    let ch3_off = ch2_off + 2 * count;
    for i in 0..count {
        hf_meta_data[ch3_off + i] = 4;
    }
    encode_hf_metadata_modular(&mut w, cr_w, cr_h, count, gw, gh, &hf_meta_data)?;

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Entropy cost of a histogram (sum of -count * log2(count/total) for each symbol).
fn histogram_entropy_cost(freqs: &[u64]) -> f64 {
    let total: u64 = freqs.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let inv = 1.0 / total as f64;
    let mut cost = 0.0f64;
    for &f in freqs {
        if f > 0 {
            let p = f as f64 * inv;
            cost -= f as f64 * p.log2();
        }
    }
    cost
}

/// Build a context map by seed-based clustering of per-context histograms.
///
/// Algorithm (mirrors libjxl's FastClusterHistograms):
/// 1. Build per-context frequency histograms.
/// 2. Pick seed clusters starting from the most populated context.
/// 3. Add new seeds while max-min-distance exceeds threshold.
/// 4. Assign each context to its nearest seed.
/// Runtime: O(num_used_contexts * max_clusters * alphabet_size).
fn build_greedy_clustered_context_map(
    num_contexts: usize,
    alphabet_size: usize,
    tokens: &[AcToken],
    encoded: &[crate::encode::entropy::HybridUintEncoded],
    max_clusters: usize,
) -> Vec<u8> {
    if num_contexts == 0 || max_clusters == 0 {
        return vec![0u8; num_contexts];
    }

    // Build per-context histograms and totals.
    let mut per_ctx = vec![vec![0u64; alphabet_size]; num_contexts];
    let mut per_ctx_total = vec![0u64; num_contexts];
    for (token, enc) in tokens.iter().zip(encoded.iter()) {
        per_ctx[token.context][enc.token as usize] += 1;
        per_ctx_total[token.context] += 1;
    }

    let per_ctx_entropy: Vec<f64> = per_ctx.iter().map(|h| histogram_entropy_cost(h)).collect();

    // Collect contexts that have data, sorted by total count descending.
    let mut used: Vec<usize> = (0..num_contexts)
        .filter(|&c| per_ctx_total[c] > 0)
        .collect();
    used.sort_by_key(|&c| std::cmp::Reverse(per_ctx_total[c]));

    if used.is_empty() {
        return vec![0u8; num_contexts];
    }

    let target = max_clusters.min(used.len()).max(1);

    // Merged-histogram distance: entropy(a+b) - entropy(a) - entropy(b).
    let merged_entropy = |a: &[u64], b: &[u64]| -> f64 {
        let mut cost = 0.0f64;
        let mut total = 0u64;
        for i in 0..alphabet_size {
            total += a[i] + b[i];
        }
        if total == 0 {
            return 0.0;
        }
        let inv = 1.0 / total as f64;
        for i in 0..alphabet_size {
            let f = a[i] + b[i];
            if f > 0 {
                cost -= f as f64 * (f as f64 * inv).log2();
            }
        }
        cost
    };

    // Seed selection.
    let mut seeds: Vec<usize> = vec![used[0]];
    let mut min_dist = vec![f64::MAX; num_contexts];
    const MIN_DISTANCE_FOR_DISTINCT: f64 = 48.0;

    while seeds.len() < target {
        let latest = *seeds.last().unwrap();
        for &ctx in &used {
            let d = merged_entropy(&per_ctx[ctx], &per_ctx[latest])
                - per_ctx_entropy[ctx]
                - per_ctx_entropy[latest];
            min_dist[ctx] = min_dist[ctx].min(d);
        }
        min_dist[latest] = 0.0;

        let best = used
            .iter()
            .copied()
            .filter(|&c| min_dist[c] > 0.0)
            .max_by(|&a, &b| {
                min_dist[a]
                    .partial_cmp(&min_dist[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        match best {
            Some(ctx) if min_dist[ctx] >= MIN_DISTANCE_FOR_DISTINCT => {
                seeds.push(ctx);
            }
            _ => break,
        }
    }

    // Build seed cluster histograms.
    let seed_entropy: Vec<f64> = seeds.iter().map(|&s| per_ctx_entropy[s]).collect();
    let seed_hists: Vec<&[u64]> = seeds.iter().map(|&s| per_ctx[s].as_slice()).collect();

    // Assign each context to nearest seed.
    let mut context_map = vec![0u8; num_contexts];
    for &ctx in &used {
        let mut best_cluster = 0u8;
        let mut best_dist = f64::MAX;
        for (ci, &seed_hist) in seed_hists.iter().enumerate() {
            let d =
                merged_entropy(&per_ctx[ctx], seed_hist) - per_ctx_entropy[ctx] - seed_entropy[ci];
            if d < best_dist {
                best_dist = d;
                best_cluster = ci as u8;
            }
        }
        context_map[ctx] = best_cluster;
    }

    context_map
}

fn build_clustered_ac_context_map(num_ac_contexts: usize) -> Vec<u8> {
    // Legacy clustered map: keep all nonzero-count contexts in cluster 0,
    // split coefficient contexts by block-context family.
    let contexts_per_block = NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT;
    debug_assert_eq!(num_ac_contexts % contexts_per_block, 0);

    let num_block_contexts = (num_ac_contexts / contexts_per_block).max(1);
    let nonzero_contexts = num_block_contexts * NON_ZERO_BUCKETS;
    let coeff_clusters = num_block_contexts.min(4).max(1);

    let mut context_map = vec![0u8; num_ac_contexts];
    for (ctx, slot) in context_map.iter_mut().enumerate().skip(nonzero_contexts) {
        let block_context = ((ctx - nonzero_contexts) / ZERO_DENSITY_CONTEXT_COUNT)
            .min(num_block_contexts.saturating_sub(1));
        let cluster = 1 + (block_context * coeff_clusters / num_block_contexts);
        *slot = cluster as u8;
    }
    context_map
}

fn build_split_ac_context_map(
    num_ac_contexts: usize,
    nonzero_clusters: usize,
    coeff_clusters: usize,
) -> Vec<u8> {
    let contexts_per_block = NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT;
    debug_assert_eq!(num_ac_contexts % contexts_per_block, 0);

    let num_block_contexts = (num_ac_contexts / contexts_per_block).max(1);
    let nonzero_clusters = nonzero_clusters.max(1).min(num_block_contexts);
    let coeff_clusters = coeff_clusters.max(1).min(num_block_contexts);
    let nonzero_contexts = num_block_contexts * NON_ZERO_BUCKETS;

    let mut context_map = vec![0u8; num_ac_contexts];

    // Layout for nonzero contexts: bucket-major, then block_context.
    for (ctx, slot) in context_map.iter_mut().enumerate().take(nonzero_contexts) {
        let block_context = ctx % num_block_contexts;
        let cluster = block_context * nonzero_clusters / num_block_contexts;
        *slot = cluster as u8;
    }

    // Layout for coefficient contexts: contiguous block_context slabs.
    for (ctx, slot) in context_map.iter_mut().enumerate().skip(nonzero_contexts) {
        let block_context = ((ctx - nonzero_contexts) / ZERO_DENSITY_CONTEXT_COUNT)
            .min(num_block_contexts.saturating_sub(1));
        let cluster = nonzero_clusters + (block_context * coeff_clusters / num_block_contexts);
        *slot = cluster as u8;
    }

    context_map
}

fn build_popularity_split_ac_context_map(
    num_ac_contexts: usize,
    context_counts: &[u64],
    nonzero_clusters: usize,
    coeff_clusters: usize,
) -> Vec<u8> {
    let contexts_per_block = NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT;
    debug_assert_eq!(num_ac_contexts % contexts_per_block, 0);

    let num_block_contexts = (num_ac_contexts / contexts_per_block).max(1);
    let nonzero_clusters = nonzero_clusters.max(1).min(num_block_contexts);
    let coeff_clusters = coeff_clusters.max(1).min(num_block_contexts);
    let nonzero_contexts = num_block_contexts * NON_ZERO_BUCKETS;

    let mut nonzero_usage = vec![0u64; num_block_contexts];
    for (ctx, &count) in context_counts.iter().enumerate().take(nonzero_contexts) {
        nonzero_usage[ctx % num_block_contexts] += count;
    }

    let mut coeff_usage = vec![0u64; num_block_contexts];
    for (ctx, &count) in context_counts.iter().enumerate().skip(nonzero_contexts) {
        let block_context = ((ctx - nonzero_contexts) / ZERO_DENSITY_CONTEXT_COUNT)
            .min(num_block_contexts.saturating_sub(1));
        coeff_usage[block_context] += count;
    }

    let mut nonzero_order: Vec<usize> = (0..num_block_contexts).collect();
    nonzero_order.sort_by_key(|&bc| std::cmp::Reverse(nonzero_usage[bc]));
    let mut coeff_order: Vec<usize> = (0..num_block_contexts).collect();
    coeff_order.sort_by_key(|&bc| std::cmp::Reverse(coeff_usage[bc]));

    let mut nonzero_cluster_for_bc = vec![0u8; num_block_contexts];
    for (rank, &bc) in nonzero_order.iter().enumerate() {
        let cluster = if rank + 1 < nonzero_clusters {
            rank
        } else {
            nonzero_clusters - 1
        };
        nonzero_cluster_for_bc[bc] = cluster as u8;
    }

    let mut coeff_cluster_for_bc = vec![0u8; num_block_contexts];
    for (rank, &bc) in coeff_order.iter().enumerate() {
        let cluster = if rank + 1 < coeff_clusters {
            rank
        } else {
            coeff_clusters - 1
        };
        coeff_cluster_for_bc[bc] = cluster as u8;
    }

    let mut context_map = vec![0u8; num_ac_contexts];
    for (ctx, slot) in context_map.iter_mut().enumerate().take(nonzero_contexts) {
        let block_context = ctx % num_block_contexts;
        *slot = nonzero_cluster_for_bc[block_context];
    }
    for (ctx, slot) in context_map.iter_mut().enumerate().skip(nonzero_contexts) {
        let block_context = ((ctx - nonzero_contexts) / ZERO_DENSITY_CONTEXT_COUNT)
            .min(num_block_contexts.saturating_sub(1));
        *slot = (nonzero_clusters + coeff_cluster_for_bc[block_context] as usize) as u8;
    }

    context_map
}

fn build_ac_context_map_candidates(num_ac_contexts: usize, context_counts: &[u64]) -> Vec<Vec<u8>> {
    let contexts_per_block = NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT;
    debug_assert_eq!(num_ac_contexts % contexts_per_block, 0);
    let num_block_contexts = (num_ac_contexts / contexts_per_block).max(1);

    let mut candidates = vec![
        vec![0u8; num_ac_contexts],
        build_clustered_ac_context_map(num_ac_contexts),
    ];

    // Candidate grid over (nonzero_clusters, coeff_clusters).
    // These are clamped to the available block-context cardinality.
    let presets = [
        (1usize, 1usize),
        (1, 2),
        (1, 4),
        (1, 8),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (3, 3),
        (3, 6),
        (4, 4),
        (4, 8),
    ];
    for (nz_clusters, coeff_clusters) in presets {
        candidates.push(build_split_ac_context_map(
            num_ac_contexts,
            nz_clusters.min(num_block_contexts),
            coeff_clusters.min(num_block_contexts),
        ));
    }

    // Fully split candidate (one nonzero and one coeff family per block-context).
    candidates.push(build_split_ac_context_map(
        num_ac_contexts,
        num_block_contexts,
        num_block_contexts,
    ));

    // Popularity-guided candidates: keep the most used block-context families
    // in dedicated clusters and merge the tail.
    let popularity_presets = [(2usize, 4usize), (3, 6), (4, 8), (6, 10)];
    for (nz_clusters, coeff_clusters) in popularity_presets {
        candidates.push(build_popularity_split_ac_context_map(
            num_ac_contexts,
            context_counts,
            nz_clusters.min(num_block_contexts),
            coeff_clusters.min(num_block_contexts),
        ));
    }

    // Deduplicate identical maps (small candidate set, O(n^2) is fine).
    let mut unique = Vec::new();
    for candidate in candidates.drain(..) {
        if !unique.iter().any(|u: &Vec<u8>| *u == candidate) {
            unique.push(candidate);
        }
    }
    unique
}

fn num_clusters_in_context_map(context_map: &[u8]) -> usize {
    context_map
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(1)
}

fn build_cluster_frequencies_for_tokens(
    context_map: &[u8],
    alphabet_size: usize,
    tokens: &[AcToken],
    encoded: &[crate::encode::entropy::HybridUintEncoded],
) -> Result<Vec<Vec<u64>>> {
    let num_clusters = num_clusters_in_context_map(context_map);
    let mut frequencies = vec![vec![0u64; alphabet_size]; num_clusters];
    for (token, enc) in tokens.iter().zip(encoded.iter()) {
        let cluster = *context_map
            .get(token.context)
            .ok_or(crate::error::Error::InvalidAnsHistogram)? as usize;
        frequencies[cluster][enc.token as usize] += 1;
    }
    Ok(frequencies)
}

fn build_cluster_frequencies_for_groups(
    context_map: &[u8],
    alphabet_size: usize,
    group_tokens: &[Vec<AcToken>],
    all_encoded: &[Vec<crate::encode::entropy::HybridUintEncoded>],
) -> Result<Vec<Vec<u64>>> {
    let num_clusters = num_clusters_in_context_map(context_map);
    let mut frequencies = vec![vec![0u64; alphabet_size]; num_clusters];
    for (tokens, encoded_group) in group_tokens.iter().zip(all_encoded.iter()) {
        for (token, enc) in tokens.iter().zip(encoded_group.iter()) {
            let cluster = *context_map
                .get(token.context)
                .ok_or(crate::error::Error::InvalidAnsHistogram)?
                as usize;
            frequencies[cluster][enc.token as usize] += 1;
        }
    }
    Ok(frequencies)
}

fn build_ans_distributions(
    cluster_frequencies: &[Vec<u64>],
) -> Vec<crate::encode::entropy::ans::AnsDistribution> {
    cluster_frequencies
        .iter()
        .map(|freqs| {
            crate::encode::entropy::ans::AnsDistribution::from_frequencies(freqs)
                .unwrap_or_else(|| crate::encode::entropy::ans::AnsDistribution::single_symbol(0))
        })
        .collect()
}

fn estimate_ans_payload_bits(
    context_map: &[u8],
    uint_config: crate::encode::entropy::HybridUintConfig,
    distributions: &[crate::encode::entropy::ans::AnsDistribution],
    ans_tokens: &[crate::encode::entropy::ans::AnsToken],
) -> Result<usize> {
    let mut w = BitWriter::new();
    let uint_configs = vec![uint_config; distributions.len()];
    crate::encode::entropy::ans::write_ans_histograms(
        &mut w,
        context_map,
        &uint_configs,
        distributions,
    )?;
    crate::encode::entropy::ans::write_ans_stream(&mut w, distributions, ans_tokens)?;
    Ok(w.total_bits_written())
}

fn build_huffman_codes_from_frequencies(
    cluster_frequencies: &[Vec<u64>],
) -> Result<Vec<crate::encode::entropy::huffman_encode::HuffmanCode>> {
    let mut codes = Vec::with_capacity(cluster_frequencies.len());
    for freqs in cluster_frequencies {
        let code = if freqs.iter().all(|&f| f == 0) {
            build_huffman_code(&[1]).ok_or(crate::error::Error::InvalidHuffman)?
        } else {
            build_huffman_code(freqs).ok_or(crate::error::Error::InvalidHuffman)?
        };
        codes.push(code);
    }
    Ok(codes)
}

fn estimate_huffman_payload_bits(
    context_map: &[u8],
    uint_config: crate::encode::entropy::HybridUintConfig,
    codes: &[crate::encode::entropy::huffman_encode::HuffmanCode],
    tokens: &[AcToken],
    encoded: &[crate::encode::entropy::HybridUintEncoded],
) -> Result<usize> {
    use crate::encode::entropy::huffman_encode::write_huffman_symbol;

    let mut w = BitWriter::new();
    let uint_configs = vec![uint_config; codes.len()];
    crate::encode::entropy::huffman_encode::write_huffman_histograms(
        &mut w,
        context_map,
        &uint_configs,
        codes,
    )?;

    for (token, enc) in tokens.iter().zip(encoded.iter()) {
        let cluster = context_map[token.context] as usize;
        write_huffman_symbol(&mut w, &codes[cluster], enc.token as usize)?;
        if enc.nbits > 0 {
            w.write(enc.nbits as usize, enc.extra_bits as u64)?;
        }
    }

    Ok(w.total_bits_written())
}

/// Encode the HfGlobal section with pre-computed global Huffman codes.
///
/// Writes: DequantMatrices, num_histograms, used_orders, Huffman histogram header.
/// The histogram header defines the tables; token data goes in HfGroup sections.
fn encode_hf_global_section_with_code(
    num_groups: usize,
    context_map: &[u8],
    uint_configs_per_pass: &[crate::encode::entropy::HybridUintConfig],
    codes_per_pass: &[Vec<crate::encode::entropy::huffman_encode::HuffmanCode>],
    custom_orders_8x8: Option<&[[usize; 64]; 3]>, // [Y, X, B]
    num_passes: usize,
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();

    // DequantMatrices: all_default = true
    w.write(1, 1)?;

    // num_histograms: ceil_log2(num_groups) bits, value = 0 (meaning 1 histogram)
    let num_histo_bits = if num_groups <= 1 {
        0
    } else {
        32 - (num_groups as u32 - 1).leading_zeros()
    };
    if num_histo_bits > 0 {
        w.write(num_histo_bits as usize, 0)?;
    }

    // Per-pass data.
    for pass in 0..num_passes {
        if let Some(orders) = custom_orders_8x8 {
            // used_orders = 1 (only DCT8x8), encoded with selector=3 (Bits(13)).
            w.write(2, 3)?;
            w.write(13, 1)?;
            // Decoder expects coeff-order channels in [X, Y, B]; `orders` is [Y, X, B].
            let decoder_order = [orders[1], orders[0], orders[2]];
            encode_coeff_orders(&mut w, &decoder_order)?;
        } else {
            // used_orders selector 2 = value 0 (natural order)
            w.write(2, 2)?;
        }

        // Write Histograms header (tables only, no token data)
        let codes = &codes_per_pass[pass.min(codes_per_pass.len().saturating_sub(1))];
        let uint_config = uint_configs_per_pass
            .get(pass)
            .copied()
            .unwrap_or_else(|| uint_configs_per_pass[0]);
        let uint_configs = vec![uint_config; codes.len()];
        crate::encode::entropy::huffman_encode::write_huffman_histograms(
            &mut w,
            context_map,
            &uint_configs,
            codes,
        )?;
    }

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Encode the HfGlobal section with pre-computed global ANS distribution.
fn encode_hf_global_section_with_ans(
    num_groups: usize,
    context_map: &[u8],
    uint_config: &crate::encode::entropy::HybridUintConfig,
    distributions: &[crate::encode::entropy::ans::AnsDistribution],
    custom_orders_8x8: Option<&[[usize; 64]; 3]>, // [Y, X, B]
    num_passes: usize,
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();

    // DequantMatrices: all_default = true
    w.write(1, 1)?;

    // num_histograms: ceil_log2(num_groups) bits, value = 0 (meaning 1 histogram)
    let num_histo_bits = if num_groups <= 1 {
        0
    } else {
        32 - (num_groups as u32 - 1).leading_zeros()
    };
    if num_histo_bits > 0 {
        w.write(num_histo_bits as usize, 0)?;
    }

    // Per-pass data.
    for _pass in 0..num_passes {
        if let Some(orders) = custom_orders_8x8 {
            w.write(2, 3)?;
            w.write(13, 1)?;
            let decoder_order = [orders[1], orders[0], orders[2]];
            encode_coeff_orders(&mut w, &decoder_order)?;
        } else {
            w.write(2, 2)?;
        }

        // Write ANS histograms header (tables only, no token data)
        let uint_configs = vec![*uint_config; distributions.len()];
        crate::encode::entropy::ans::write_ans_histograms(
            &mut w,
            context_map,
            &uint_configs,
            distributions,
        )?;
    }

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Write AC entropy stream header and token data.
///
/// Supports both Huffman (legacy path) and ANS (new path).
fn write_ac_histograms_and_tokens(
    w: &mut BitWriter,
    num_ac_contexts: usize,
    tokens: &[AcToken],
    use_ans: bool,
) -> Result<()> {
    use crate::encode::entropy::huffman_encode::write_huffman_symbol;

    enum EntropyChoice {
        Ans {
            context_map: Vec<u8>,
            distributions: Vec<crate::encode::entropy::ans::AnsDistribution>,
            uint_config: crate::encode::entropy::HybridUintConfig,
        },
        Huffman {
            context_map: Vec<u8>,
            codes: Vec<crate::encode::entropy::huffman_encode::HuffmanCode>,
            uint_config: crate::encode::entropy::HybridUintConfig,
        },
    }

    // HybridUint config candidates for AC. libjxl kFast tries these.
    let uint_configs_to_try = [
        crate::encode::entropy::HybridUintConfig::new(4, 2, 0), // default
        crate::encode::entropy::HybridUintConfig::new(4, 1, 2), // libjxl e3
        crate::encode::entropy::HybridUintConfig::new(0, 0, 0), // smallest histograms
        crate::encode::entropy::HybridUintConfig::new(2, 0, 1), // good for ctx map
    ];

    // Caller byte-aligns right after AC payload in single-group sections.
    let start_mod8 = w.total_bits_written() % 8;
    let with_final_alignment =
        |payload_bits: usize| payload_bits + ((8 - ((start_mod8 + payload_bits) % 8)) % 8);

    let mut best_choice: Option<EntropyChoice> = None;
    let mut best_effective_bits = usize::MAX;

    let mut context_counts = vec![0u64; num_ac_contexts];
    for token in tokens {
        context_counts[token.context] += 1;
    }

    // Build context map candidates once using the base uint_config.
    let base_uint = crate::encode::entropy::HybridUintConfig::new(4, 1, 2);
    let base_encoded: Vec<_> = tokens.iter().map(|t| base_uint.encode(t.value)).collect();
    let base_alphabet =
        (base_encoded.iter().map(|e| e.token).max().unwrap_or(0) as usize + 1).max(1);

    let mut context_map_candidates =
        build_ac_context_map_candidates(num_ac_contexts, &context_counts);
    for max_c in [2, 4, 8, 16, 32] {
        if max_c <= num_ac_contexts {
            let greedy_map = build_greedy_clustered_context_map(
                num_ac_contexts,
                base_alphabet,
                tokens,
                &base_encoded,
                max_c,
            );
            if !context_map_candidates.iter().any(|m| *m == greedy_map) {
                context_map_candidates.push(greedy_map);
            }
        }
    }

    for &uint_config in &uint_configs_to_try {
        let encoded: Vec<_> = tokens.iter().map(|t| uint_config.encode(t.value)).collect();
        let max_token = encoded.iter().map(|e| e.token).max().unwrap_or(0);
        let alphabet_size = (max_token as usize + 1).max(1);

        for context_map in &context_map_candidates {
            let cluster_frequencies =
                build_cluster_frequencies_for_tokens(context_map, alphabet_size, tokens, &encoded)?;

            if use_ans {
                let distributions = build_ans_distributions(&cluster_frequencies);
                let ans_tokens: Vec<crate::encode::entropy::ans::AnsToken> = tokens
                    .iter()
                    .zip(encoded.iter())
                    .map(|(token, enc)| crate::encode::entropy::ans::AnsToken {
                        symbol: enc.token,
                        cluster: context_map[token.context] as usize,
                        extra_bits: enc.extra_bits,
                        extra_nbits: enc.nbits as usize,
                    })
                    .collect();
                let ans_bits = with_final_alignment(estimate_ans_payload_bits(
                    &context_map,
                    uint_config,
                    &distributions,
                    &ans_tokens,
                )?);
                if ans_bits < best_effective_bits {
                    best_effective_bits = ans_bits;
                    best_choice = Some(EntropyChoice::Ans {
                        context_map: context_map.clone(),
                        distributions,
                        uint_config,
                    });
                }
            }

            let codes = build_huffman_codes_from_frequencies(&cluster_frequencies)?;
            let huffman_bits = with_final_alignment(estimate_huffman_payload_bits(
                context_map,
                uint_config,
                &codes,
                tokens,
                &encoded,
            )?);
            if huffman_bits < best_effective_bits || best_choice.is_none() {
                best_effective_bits = huffman_bits;
                best_choice = Some(EntropyChoice::Huffman {
                    context_map: context_map.clone(),
                    codes,
                    uint_config,
                });
            }
        }
    } // end for &uint_config

    match best_choice.ok_or(crate::error::Error::InvalidHuffman)? {
        EntropyChoice::Ans {
            context_map,
            distributions,
            uint_config,
        } => {
            let uint_configs = vec![uint_config; distributions.len()];
            crate::encode::entropy::ans::write_ans_histograms(
                w,
                &context_map,
                &uint_configs,
                &distributions,
            )?;

            let encoded: Vec<_> = tokens.iter().map(|t| uint_config.encode(t.value)).collect();
            let ans_tokens: Vec<crate::encode::entropy::ans::AnsToken> = tokens
                .iter()
                .zip(encoded.iter())
                .map(|(token, enc)| crate::encode::entropy::ans::AnsToken {
                    symbol: enc.token,
                    cluster: context_map[token.context] as usize,
                    extra_bits: enc.extra_bits,
                    extra_nbits: enc.nbits as usize,
                })
                .collect();
            crate::encode::entropy::ans::write_ans_stream(w, &distributions, &ans_tokens)?;
        }
        EntropyChoice::Huffman {
            context_map,
            codes,
            uint_config,
        } => {
            let uint_configs = vec![uint_config; codes.len()];
            crate::encode::entropy::huffman_encode::write_huffman_histograms(
                w,
                &context_map,
                &uint_configs,
                &codes,
            )?;

            let encoded: Vec<_> = tokens.iter().map(|t| uint_config.encode(t.value)).collect();
            for (token, enc) in tokens.iter().zip(encoded.iter()) {
                let cluster = context_map[token.context] as usize;
                write_huffman_symbol(w, &codes[cluster], enc.token as usize)?;
                if enc.nbits > 0 {
                    w.write(enc.nbits as usize, enc.extra_bits as u64)?;
                }
            }
        }
    }

    Ok(())
}

/// Encode an HfGroup section: histogram_index + AC token data.
fn encode_hf_group_tokens(
    num_histograms: usize,
    tokens: &[AcToken],
    encoded: &[crate::encode::entropy::HybridUintEncoded],
    context_map: &[u8],
    codes: &[crate::encode::entropy::huffman_encode::HuffmanCode],
    alpha_tile: Option<(&[i32], usize, usize)>, // (data, tile_w, tile_h)
) -> Result<Vec<u8>> {
    use crate::encode::entropy::huffman_encode::write_huffman_symbol;

    let mut w = BitWriter::new();

    // histogram_index: 0
    let num_histo_bits = if num_histograms <= 1 {
        0
    } else {
        (32 - (num_histograms as u32 - 1).leading_zeros()) as usize
    };
    if num_histo_bits > 0 {
        w.write(num_histo_bits, 0)?;
    }

    for (token, enc) in tokens.iter().zip(encoded.iter()) {
        let cluster = context_map[token.context] as usize;
        write_huffman_symbol(&mut w, &codes[cluster], enc.token as usize)?;
        if enc.nbits > 0 {
            w.write(enc.nbits as usize, enc.extra_bits as u64)?;
        }
    }

    // Write modular alpha data after AC tokens, before byte-alignment
    if let Some((tile_data, tw, th)) = alpha_tile {
        crate::encode::modular_encode::encode_modular_signed_stream(&mut w, tw, th, 1, tile_data)?;
    }

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Encode an HfGroup section using ANS: histogram_index + ANS token stream.
fn encode_hf_group_tokens_ans(
    num_histograms: usize,
    tokens: &[AcToken],
    encoded: &[crate::encode::entropy::HybridUintEncoded],
    context_map: &[u8],
    distributions: &[crate::encode::entropy::ans::AnsDistribution],
    alpha_tile: Option<(&[i32], usize, usize)>,
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();

    // histogram_index: 0
    let num_histo_bits = if num_histograms <= 1 {
        0
    } else {
        (32 - (num_histograms as u32 - 1).leading_zeros()) as usize
    };
    if num_histo_bits > 0 {
        w.write(num_histo_bits, 0)?;
    }

    let ans_tokens: Vec<crate::encode::entropy::ans::AnsToken> = tokens
        .iter()
        .zip(encoded.iter())
        .map(|(token, enc)| crate::encode::entropy::ans::AnsToken {
            symbol: enc.token,
            cluster: context_map[token.context] as usize,
            extra_bits: enc.extra_bits,
            extra_nbits: enc.nbits as usize,
        })
        .collect();

    crate::encode::entropy::ans::write_ans_stream(&mut w, distributions, &ans_tokens)?;

    // Write modular alpha data after AC tokens, before byte-alignment
    if let Some((tile_data, tw, th)) = alpha_tile {
        crate::encode::modular_encode::encode_modular_signed_stream(&mut w, tw, th, 1, tile_data)?;
    }

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effort_to_speed_tier_mapping_monotonic() {
        // libjxl-style mapping: speed_tier = 10 - effort
        assert_eq!(effort_to_speed_tier_index(1), 9);
        assert_eq!(effort_to_speed_tier_index(7), 3);
        assert_eq!(effort_to_speed_tier_index(9), 1);

        for e in 1..9 {
            let a = effort_params(e);
            let b = effort_params(e + 1);
            assert!(b.max_total_encodes >= a.max_total_encodes);
            assert!(!a.enable_entropy_merge || b.enable_entropy_merge);
            assert!(!a.enable_custom_coeff_orders || b.enable_custom_coeff_orders);
        }
    }

    #[test]
    fn test_choose_progressive_pass_plan() {
        assert_eq!(
            choose_progressive_pass_plan(false, false, 9, 1024, 768),
            (1, vec![])
        );
        assert_eq!(
            choose_progressive_pass_plan(true, true, 9, 1024, 768),
            (1, vec![])
        );
        assert_eq!(
            choose_progressive_pass_plan(true, false, 7, 1024, 768),
            (2, vec![1])
        );
        assert_eq!(
            choose_progressive_pass_plan(true, false, 9, 768, 512),
            (3, vec![2, 1])
        );
    }

    #[test]
    fn test_distance_to_quant_params() {
        // At d=1.0, libjxl-aligned: global_scale ~ 65536 * 0.765 / 5.0 ~ 10026
        let (gs, ql) = distance_to_quant_params(1.0);
        assert!(gs > 5000 && gs < 15000, "global_scale at d=1: {gs}");
        assert!(ql > 0, "quant_lf must be positive: {ql}");

        // Higher distance -> lower global_scale
        let (gs2, _) = distance_to_quant_params(2.0);
        assert!(gs2 < gs, "d=2 global_scale ({gs2}) should be < d=1 ({gs})");

        // Lower distance -> higher global_scale
        let (gs05, _) = distance_to_quant_params(0.5);
        assert!(
            gs05 > gs,
            "d=0.5 global_scale ({gs05}) should be > d=1 ({gs})"
        );
    }

    #[test]
    fn test_predict_num_nonzeros_matches_decoder_behavior() {
        let stride = 4;
        let mut map = vec![0u32; stride * stride];

        // Top-left bootstrap value.
        assert_eq!(predict_num_nonzeros(&map, stride, 0, 0), 32);

        // Top row uses left neighbor.
        map[0] = 5;
        assert_eq!(predict_num_nonzeros(&map, stride, 1, 0), 5);

        // Left column uses top neighbor.
        assert_eq!(predict_num_nonzeros(&map, stride, 0, 1), 5);

        // Interior uses ceil((top + left) / 2).
        map[1] = 7; // top at (1, 0)
        map[stride] = 9; // left at (0, 1)
        assert_eq!(predict_num_nonzeros(&map, stride, 1, 1), 8);
    }

    #[test]
    fn test_ac_context_map_candidates_basic() {
        let num_contexts = 15;
        let num_ac_contexts = num_contexts * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT);
        let context_counts = vec![0u64; num_ac_contexts];
        let candidates = build_ac_context_map_candidates(num_ac_contexts, &context_counts);

        assert!(
            candidates.len() >= 4,
            "expected multiple context-map candidates"
        );
        assert!(
            candidates.iter().any(|m| m.iter().all(|&v| v == 0)),
            "expected all-zero context map candidate"
        );

        for map in &candidates {
            assert_eq!(map.len(), num_ac_contexts);
        }
    }

    #[test]
    fn test_adaptive_raw_quant_map_gating_and_range() {
        let bw = 8;
        let bh = 8;
        let num_blocks = bw * bh;
        let mut dct_x = vec![0.0f32; num_blocks * 64];
        let mut dct_y = vec![0.0f32; num_blocks * 64];
        let mut dct_b = vec![0.0f32; num_blocks * 64];

        // First half smooth, second half textured.
        for blk in 0..num_blocks {
            for k in 1..64 {
                let v = if blk < num_blocks / 2 { 0.01 } else { 8.0 };
                dct_x[blk * 64 + k] = v * 0.7;
                dct_y[blk * 64 + k] = v;
                dct_b[blk * 64 + k] = v * 0.5;
            }
        }

        // Create a dummy Y pixel channel for the test
        let (img_w, img_h) = (bw * 8, bh * 8);
        // Create non-uniform XYB channels with per-pixel variation
        let mut xyb_x = vec![0.0f32; img_w * img_h];
        let mut xyb_y = vec![0.0f32; img_w * img_h];
        let mut xyb_b = vec![0.0f32; img_w * img_h];
        for y in 0..img_h {
            for x in 0..img_w {
                let idx = y * img_w + x;
                // Checkerboard pattern creates high-frequency content in some blocks
                let t = ((x + y) % 2) as f32;
                if x < img_w / 2 {
                    xyb_y[idx] = 0.5; // smooth region
                } else {
                    xyb_y[idx] = 0.3 + 0.4 * t; // textured region
                }
                xyb_x[idx] = 0.1 + 0.05 * t;
                xyb_b[idx] = 0.3 + 0.1 * t;
            }
        }

        let quant_ac_low = 0.79 / 0.8;
        let (low_dist_map, _, _, _) = build_adaptive_raw_quant_map_full(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            img_w,
            img_h,
            bw,
            bh,
            0.8,
            quant_ac_low,
        );
        assert!(
            low_dist_map.iter().all(|&q| q == low_dist_map[0]),
            "distance gating should keep raw_quant uniform at d<1"
        );

        let quant_ac_high = 0.79 / 2.0;
        let (high_dist_map, _, _, _) = build_adaptive_raw_quant_map_full(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            img_w,
            img_h,
            bw,
            bh,
            2.0,
            quant_ac_high,
        );
        assert_eq!(high_dist_map.len(), num_blocks);
        assert!(high_dist_map.iter().all(|&q| q >= 1));
        assert!(
            high_dist_map.iter().any(|&q| q != high_dist_map[0]),
            "expected non-uniform quant map at higher distance"
        );
    }

    #[test]
    fn test_collect_transform_entries_for_rect_scan_order() {
        let bw = 4;
        let bh = 2;
        let mut transform_map = vec![0u8; bw * bh];
        let raw_quant_map = vec![1u8, 2, 3, 4, 5, 6, 7, 8];

        // First blocks at global indices 0, 3, 5.
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID;
        transform_map[3] = TRANSFORM_FIRST_BLOCK_FLAG | 4; // pretend DCT16x16 id
        transform_map[5] = TRANSFORM_FIRST_BLOCK_FLAG | 7; // pretend DCT8x16 id

        let entries =
            collect_transform_entries_for_rect(&transform_map, &raw_quant_map, bw, 0, 0, bw, bh);

        assert_eq!(entries, vec![(0, 1), (4, 4), (7, 6)]);
    }

    #[test]
    fn test_supported_nonzero_transform_ids_cover_all_non_dct8_strategies() {
        for id in 1u8..=26u8 {
            assert!(
                is_supported_nonzero_transform_id(id),
                "transform id {} should be supported in non-zero path",
                id
            );
        }
        assert!(!is_supported_nonzero_transform_id(DCT8_TRANSFORM_ID));
    }

    #[test]
    fn test_canonical_transform_for_shape_id_mapping() {
        assert_eq!(
            canonical_transform_for_shape_id(0),
            Some(HfTransformType::DCT)
        );
        assert_eq!(
            canonical_transform_for_shape_id(1),
            Some(HfTransformType::AFV0)
        );
        assert_eq!(
            canonical_transform_for_shape_id(4),
            Some(HfTransformType::DCT8X16)
        );
        assert_eq!(
            canonical_transform_for_shape_id(5),
            Some(HfTransformType::DCT8X32)
        );
        assert_eq!(
            canonical_transform_for_shape_id(6),
            Some(HfTransformType::DCT16X32)
        );
        assert_eq!(
            canonical_transform_for_shape_id(12),
            Some(HfTransformType::DCT128X256)
        );
        assert_eq!(canonical_transform_for_shape_id(13), None);
    }

    #[test]
    fn test_build_transform_map_from_quantized_ac_places_dct16_for_zero_regions() {
        let bw = 4;
        let bh = 4;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        let map = build_transform_map_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 2.0);
        assert_eq!(map.len(), num_blocks);

        // Top-left 2x2 should be one DCT16x16 placement.
        assert_eq!(map[0], TRANSFORM_FIRST_BLOCK_FLAG | DCT16_TRANSFORM_ID);
        assert_eq!(map[1], DCT16_TRANSFORM_ID);
        assert_eq!(map[bw], DCT16_TRANSFORM_ID);
        assert_eq!(map[bw + 1], DCT16_TRANSFORM_ID);

        // At low distance we keep all DCT8.
        let low_map = build_transform_map_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 1.0);
        assert!(
            low_map
                .iter()
                .all(|&t| t == (TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID))
        );
    }

    #[test]
    fn test_build_afv_transform_map_from_quantized_ac_selects_sparse_afv() {
        let bw = 8;
        let bh = 8;
        let num_blocks = bw * bh;
        let mut ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let mut ac_b = vec![0i32; num_blocks * 64];

        // Seed a few strong directional/high-frequency blocks.
        for blk in 0..4 {
            let base = blk * 64;
            ac_y[base + 1] = if blk % 2 == 0 { 20 } else { -20 };
            ac_y[base + 8] = if blk < 2 { 2 } else { -2 };
            ac_y[base + 27] = 45; // u=3,v=3 => "high" bucket in heuristic
            ac_x[base + 27] = 8;
            ac_b[base + 27] = -6;
        }

        let map = build_afv_transform_map_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        let afv_blocks = map
            .iter()
            .filter(|&&t| {
                let id = t & !TRANSFORM_FIRST_BLOCK_FLAG;
                matches!(
                    id,
                    AFV0_TRANSFORM_ID | AFV1_TRANSFORM_ID | AFV2_TRANSFORM_ID | AFV3_TRANSFORM_ID
                )
            })
            .count();

        assert!(afv_blocks >= 1, "expected at least one AFV-marked block");
        assert!(
            afv_blocks <= 1,
            "heuristic should remain sparse on 8x8 blocks"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_can_include_afv_map() {
        let bw = 8;
        let bh = 8;
        let num_blocks = bw * bh;
        let mut ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let mut ac_b = vec![0i32; num_blocks * 64];

        for blk in 0..4 {
            let base = blk * 64;
            ac_y[base + 1] = if blk % 2 == 0 { 20 } else { -20 };
            ac_y[base + 8] = if blk < 2 { 2 } else { -2 };
            ac_y[base + 27] = 45;
            ac_x[base + 27] = 8;
            ac_b[base + 27] = -6;
        }

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().any(|map| map.iter().any(|&t| {
                matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    AFV0_TRANSFORM_ID | AFV1_TRANSFORM_ID | AFV2_TRANSFORM_ID | AFV3_TRANSFORM_ID
                )
            })),
            "expected at least one AFV candidate map"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_skip_afv_when_grid_is_large() {
        let bw = 65;
        let bh = 65;
        let num_blocks = bw * bh;
        let mut ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let mut ac_b = vec![0i32; num_blocks * 64];

        // Seed many blocks with AFV-like energy, but grid size should disable AFV candidate insertion.
        for blk in 0..64 {
            let base = blk * 64;
            ac_y[base + 1] = 20;
            ac_y[base + 8] = 2;
            ac_y[base + 27] = 45;
            ac_x[base + 27] = 8;
            ac_b[base + 27] = -6;
        }

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().all(|map| map.iter().all(|&t| {
                !matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    AFV0_TRANSFORM_ID | AFV1_TRANSFORM_ID | AFV2_TRANSFORM_ID | AFV3_TRANSFORM_ID
                )
            })),
            "did not expect AFV candidates for large grids"
        );
    }

    #[test]
    fn test_build_directional_special_transform_map_selects_expected_orientation() {
        let bw = 4;
        let bh = 4;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        // Strongly horizontal directional energy in block 0.
        ac_y[1] = 50;
        ac_y[2] = 40;
        ac_y[3] = 30;
        ac_y[8] = 2;
        ac_y[16] = 1;
        ac_y[24] = 1;

        let map = build_directional_special_transform_map_from_quantized_ac(
            &ac_x, &ac_y, &ac_b, bw, bh, 3.0,
        );
        assert_eq!(map[0], TRANSFORM_FIRST_BLOCK_FLAG | DCT4X8_TRANSFORM_ID);

        let directional_blocks = map
            .iter()
            .filter(|&&t| {
                matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    DCT4X8_TRANSFORM_ID | DCT8X4_TRANSFORM_ID
                )
            })
            .count();
        assert_eq!(
            directional_blocks, 1,
            "directional map should remain sparse"
        );
    }

    #[test]
    fn test_build_compact_special_transform_map_ignores_zero_blocks() {
        let bw = 4;
        let bh = 4;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        let map =
            build_compact_special_transform_map_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            map.iter()
                .all(|&t| t == (TRANSFORM_FIRST_BLOCK_FLAG | DCT8_TRANSFORM_ID)),
            "all-zero AC should keep default DCT8 map"
        );
    }

    #[test]
    fn test_build_compact_special_transform_map_selects_dct2x2_for_smooth_block() {
        let bw = 4;
        let bh = 4;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        ac_y[1] = 6;
        ac_y[8] = 4;

        let map =
            build_compact_special_transform_map_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert_eq!(map[0], TRANSFORM_FIRST_BLOCK_FLAG | DCT2X2_TRANSFORM_ID);
    }

    #[test]
    fn test_build_compact_special_transform_map_selects_identity_for_sparse_peak() {
        let bw = 4;
        let bh = 4;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        ac_y[1] = 6;
        ac_y[27] = 40;

        let map =
            build_compact_special_transform_map_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert_eq!(map[0], TRANSFORM_FIRST_BLOCK_FLAG | IDENTITY_TRANSFORM_ID);
    }

    #[test]
    fn test_build_mixed_special_transform_map_selects_expected_blocks() {
        let bw = 8;
        let bh = 8;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let mut ac_b = vec![0i32; num_blocks * 64];

        // Block 0: smooth/low-energy compact transform candidate.
        ac_y[1] = 6;
        ac_y[8] = 4;

        // Block 1: strong horizontal directional content.
        let b1 = 64;
        ac_y[b1 + 1] = 35;
        ac_y[b1 + 2] = 28;
        ac_y[b1 + 3] = 16;
        ac_y[b1 + 8] = 2;
        ac_y[b1 + 16] = 1;
        ac_y[b1 + 24] = 1;

        // Block 2: AFV-like high-frequency content.
        let b2 = 128;
        ac_y[b2 + 1] = 12;
        ac_y[b2 + 8] = 2;
        ac_y[b2 + 27] = 45;
        ac_b[b2 + 27] = -6;

        // Block 3: sparse strong peak for IDENTITY.
        let b3 = 192;
        ac_y[b3 + 1] = 2;
        ac_y[b3 + 27] = 60;

        let map =
            build_mixed_special_transform_map_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);

        assert!(matches!(
            map[0] & !TRANSFORM_FIRST_BLOCK_FLAG,
            DCT2X2_TRANSFORM_ID | DCT4X4_TRANSFORM_ID
        ));
        assert_eq!(map[1], TRANSFORM_FIRST_BLOCK_FLAG | DCT4X8_TRANSFORM_ID);
        assert_eq!(map[2], TRANSFORM_FIRST_BLOCK_FLAG | AFV0_TRANSFORM_ID);
        assert_eq!(map[3], TRANSFORM_FIRST_BLOCK_FLAG | IDENTITY_TRANSFORM_ID);
    }

    #[test]
    fn test_build_transform_map_candidates_can_include_mixed_special_map() {
        let bw = 8;
        let bh = 8;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let mut ac_b = vec![0i32; num_blocks * 64];

        ac_y[1] = 6;
        ac_y[8] = 4;

        let b1 = 64;
        ac_y[b1 + 1] = 35;
        ac_y[b1 + 2] = 28;
        ac_y[b1 + 3] = 16;
        ac_y[b1 + 8] = 2;
        ac_y[b1 + 16] = 1;
        ac_y[b1 + 24] = 1;

        let b2 = 128;
        ac_y[b2 + 1] = 12;
        ac_y[b2 + 8] = 2;
        ac_y[b2 + 27] = 45;
        ac_b[b2 + 27] = -6;

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().any(|map| {
                matches!(
                    map[1] & !TRANSFORM_FIRST_BLOCK_FLAG,
                    DCT4X8_TRANSFORM_ID | DCT8X4_TRANSFORM_ID
                ) && matches!(
                    map[2] & !TRANSFORM_FIRST_BLOCK_FLAG,
                    AFV0_TRANSFORM_ID | AFV1_TRANSFORM_ID | AFV2_TRANSFORM_ID | AFV3_TRANSFORM_ID
                )
            }),
            "expected at least one mixed special-transform candidate map"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_can_include_directional_special_map() {
        let bw = 8;
        let bh = 8;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        for blk in 0..4 {
            let base = blk * 64;
            ac_y[base + 1] = 35;
            ac_y[base + 2] = 28;
            ac_y[base + 3] = 16;
            ac_y[base + 8] = 2;
            ac_y[base + 16] = 1;
            ac_y[base + 24] = 1;
        }

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().any(|map| map.iter().any(|&t| {
                matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    DCT4X8_TRANSFORM_ID | DCT8X4_TRANSFORM_ID
                )
            })),
            "expected at least one directional small-transform candidate map"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_can_include_compact_special_map() {
        let bw = 8;
        let bh = 8;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        for blk in 0..8 {
            let base = blk * 64;
            ac_y[base + 1] = 6;
            ac_y[base + 8] = 4;
        }

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().any(|map| map.iter().any(|&t| {
                matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    IDENTITY_TRANSFORM_ID | DCT2X2_TRANSFORM_ID | DCT4X4_TRANSFORM_ID
                )
            })),
            "expected at least one compact special-transform candidate map"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_skip_directional_special_when_grid_is_large() {
        let bw = 65;
        let bh = 65;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        for blk in 0..64 {
            let base = blk * 64;
            ac_y[base + 1] = 40;
            ac_y[base + 2] = 30;
            ac_y[base + 3] = 20;
            ac_y[base + 8] = 2;
            ac_y[base + 16] = 1;
            ac_y[base + 24] = 1;
        }

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().all(|map| map.iter().all(|&t| {
                !matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    DCT4X8_TRANSFORM_ID | DCT8X4_TRANSFORM_ID
                )
            })),
            "did not expect directional small-transform candidates for large grids"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_skip_compact_special_when_grid_is_large() {
        let bw = 65;
        let bh = 65;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        for blk in 0..64 {
            let base = blk * 64;
            ac_y[base + 1] = 6;
            ac_y[base + 8] = 4;
        }

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().all(|map| map.iter().all(|&t| {
                !matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    IDENTITY_TRANSFORM_ID | DCT2X2_TRANSFORM_ID | DCT4X4_TRANSFORM_ID
                )
            })),
            "did not expect compact special-transform candidates for large grids"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_can_include_256_family_map() {
        let bw = 32;
        let bh = 32;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().any(|map| map.iter().any(|&t| {
                matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    DCT256_TRANSFORM_ID | DCT256X128_TRANSFORM_ID | DCT128X256_TRANSFORM_ID
                )
            })),
            "expected at least one 256-family candidate map"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_skip_256_family_when_grid_is_too_large() {
        let bw = 64;
        let bh = 64;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let ac_y = vec![0i32; num_blocks * 64];
        let ac_b = vec![0i32; num_blocks * 64];

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().all(|map| map.iter().all(|&t| {
                !matches!(
                    t & !TRANSFORM_FIRST_BLOCK_FLAG,
                    DCT256_TRANSFORM_ID | DCT256X128_TRANSFORM_ID | DCT128X256_TRANSFORM_ID
                )
            })),
            "did not expect 256-family candidates for large grids"
        );
    }

    #[test]
    fn test_build_transform_map_candidates_skip_mixed_special_when_grid_is_large() {
        let bw = 65;
        let bh = 65;
        let num_blocks = bw * bh;
        let ac_x = vec![0i32; num_blocks * 64];
        let mut ac_y = vec![0i32; num_blocks * 64];
        let mut ac_b = vec![0i32; num_blocks * 64];

        ac_y[1] = 6;
        ac_y[8] = 4;

        let b1 = 64;
        ac_y[b1 + 1] = 35;
        ac_y[b1 + 2] = 28;
        ac_y[b1 + 3] = 16;
        ac_y[b1 + 8] = 2;
        ac_y[b1 + 16] = 1;
        ac_y[b1 + 24] = 1;

        let b2 = 128;
        ac_y[b2 + 1] = 12;
        ac_y[b2 + 8] = 2;
        ac_y[b2 + 27] = 45;
        ac_b[b2 + 27] = -6;

        let candidates =
            build_transform_map_candidates_from_quantized_ac(&ac_x, &ac_y, &ac_b, bw, bh, 3.0);
        assert!(
            candidates.iter().all(|map| {
                !matches!(
                    map[1] & !TRANSFORM_FIRST_BLOCK_FLAG,
                    DCT2X2_TRANSFORM_ID
                        | DCT4X4_TRANSFORM_ID
                        | DCT4X8_TRANSFORM_ID
                        | DCT8X4_TRANSFORM_ID
                ) && !matches!(
                    map[2] & !TRANSFORM_FIRST_BLOCK_FLAG,
                    AFV0_TRANSFORM_ID | AFV1_TRANSFORM_ID | AFV2_TRANSFORM_ID | AFV3_TRANSFORM_ID
                )
            }),
            "did not expect mixed special-transform candidates for large grids"
        );
    }

    #[test]
    fn test_zero_non_dct8_ac_coeffs_clears_region() {
        let bw = 2;
        let bh = 2;
        let mut ac = vec![0i32; bw * bh * 64];
        for blk in 0..(bw * bh) {
            ac[blk * 64 + 5] = 7;
            ac[blk * 64 + 17] = -3;
        }

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT16_TRANSFORM_ID;
        transform_map[1] = DCT16_TRANSFORM_ID;
        transform_map[bw] = DCT16_TRANSFORM_ID;
        transform_map[bw + 1] = DCT16_TRANSFORM_ID;

        zero_non_dct8_ac_coeffs(&mut ac, &transform_map, bw, bh).unwrap();
        for blk in 0..(bw * bh) {
            assert!(ac[blk * 64 + 1..blk * 64 + 64].iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn test_forward_dct2d_scalar_roundtrip_via_idct8() {
        let mut coeffs = [0.0f32; 64];
        for (i, v) in coeffs.iter_mut().enumerate() {
            *v = (i as f32 * 0.37).sin() * 20.0 + (i % 7) as f32;
        }

        let mut spatial = coeffs;
        jxl_transforms::idct2d_8_8(jxl_simd::scalar::ScalarDescriptor, &mut spatial);

        let mut recovered = spatial.to_vec();
        forward_dct2d_scalar(&mut recovered, 8, 8);

        let max_err = coeffs
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.02,
            "generic 8x8 DCT roundtrip mismatch: max_err={max_err}"
        );
    }

    #[test]
    fn test_forward_special_8x8_roundtrip_via_decoder_transform() {
        let mut block = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                block[y * 8 + x] = (x as f32 * 0.21 + y as f32 * 0.17).sin() * 6.0;
            }
        }

        for &(transform_type, forward, err_limit) in &[
            (
                HfTransformType::IDENTITY,
                forward_identity_from_8x8 as fn(&[f32]) -> Vec<f32>,
                0.02f32,
            ),
            (
                HfTransformType::DCT2X2,
                forward_dct2x2_from_8x8 as fn(&[f32]) -> Vec<f32>,
                0.02f32,
            ),
            (
                HfTransformType::DCT4X4,
                forward_dct4x4_from_8x8 as fn(&[f32]) -> Vec<f32>,
                0.02f32,
            ),
            (
                HfTransformType::DCT4X8,
                forward_dct4x8_from_8x8 as fn(&[f32]) -> Vec<f32>,
                0.03f32,
            ),
            (
                HfTransformType::DCT8X4,
                forward_dct8x4_from_8x8 as fn(&[f32]) -> Vec<f32>,
                0.03f32,
            ),
        ] {
            let coeffs = forward(&block);
            let mut lf = vec![coeffs[0]];
            let mut pixels = coeffs.clone();
            transform_to_pixels(transform_type, &mut lf, &mut pixels);

            let max_err = block
                .iter()
                .zip(pixels.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < err_limit,
                "special-transform roundtrip mismatch for {:?}: max_err={}",
                transform_type,
                max_err
            );
        }
    }

    #[test]
    fn test_compute_forward_transform_coeffs_special_8x8_roundtrip_with_clamp() {
        let width = 11;
        let height = 9;
        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.13 + y as f32 * 0.29).sin() * 4.0;
                x_chan[idx] = v * 0.8 + 0.3;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * -0.2 + (x as f32 * 0.05);
            }
        }

        let bx = 1;
        let by = 1;
        for &transform_id in &[
            IDENTITY_TRANSFORM_ID,
            DCT2X2_TRANSFORM_ID,
            DCT4X4_TRANSFORM_ID,
            DCT4X8_TRANSFORM_ID,
            DCT8X4_TRANSFORM_ID,
            AFV0_TRANSFORM_ID,
            AFV1_TRANSFORM_ID,
            AFV2_TRANSFORM_ID,
            AFV3_TRANSFORM_ID,
        ] {
            let coeffs = compute_forward_transform_coeffs(
                transform_id,
                &x_chan,
                &y_chan,
                &b_minus_y_chan,
                width,
                height,
                bx,
                by,
                8,
                8,
            );
            let transform_type = HfTransformType::from_usize(transform_id as usize).unwrap();

            for (chan_idx, source) in [&x_chan[..], &y_chan[..], &b_minus_y_chan[..]]
                .into_iter()
                .enumerate()
            {
                let expected = gather_clamped_block(source, width, height, bx * 8, by * 8, 8, 8);
                let mut lf = vec![coeffs[chan_idx][0]];
                let mut pixels = coeffs[chan_idx].clone();
                transform_to_pixels(transform_type, &mut lf, &mut pixels);

                let max_err = expected
                    .iter()
                    .zip(pixels.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 0.04,
                    "special compute_forward mismatch for id {} channel {}: max_err={}",
                    transform_id,
                    chan_idx,
                    max_err
                );
            }
        }
    }

    #[test]
    #[ignore] // ~48s: heavy solver build, run explicitly with --ignored
    fn test_compute_forward_transform_coeffs_non_special_square_roundtrip_with_clamp() {
        let width = 140;
        let height = 120;
        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v =
                    (x as f32 * 0.07 + y as f32 * 0.11).sin() * 5.0 + (x as f32 * 0.03).cos() * 1.5;
                x_chan[idx] = v * 0.9 + 0.2;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * -0.35 + (y as f32 * 0.02);
            }
        }

        let bx = 10;
        let by = 7;
        for &(transform_id, block_w, block_h, err_limit) in &[
            (DCT16_TRANSFORM_ID, 16usize, 16usize, 0.15f32),
            (DCT32_TRANSFORM_ID, 32usize, 32usize, 0.15f32),
            (DCT64_TRANSFORM_ID, 64usize, 64usize, 0.30f32),
        ] {
            let coeffs = compute_forward_transform_coeffs(
                transform_id,
                &x_chan,
                &y_chan,
                &b_minus_y_chan,
                width,
                height,
                bx,
                by,
                block_w,
                block_h,
            );
            let transform_type = HfTransformType::from_usize(transform_id as usize).unwrap();

            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            for (chan_idx, source) in [&x_chan[..], &y_chan[..], &b_minus_y_chan[..]]
                .into_iter()
                .enumerate()
            {
                let expected =
                    gather_clamped_block(source, width, height, bx * 8, by * 8, block_w, block_h);

                let mut lf = vec![0.0f32; cx * cy];
                for ly in 0..cy {
                    for lx in 0..cx {
                        let sub = gather_clamped_block(
                            source,
                            width,
                            height,
                            bx * 8 + lx * 8,
                            by * 8 + ly * 8,
                            8,
                            8,
                        );
                        let mut dct = [0.0f32; 64];
                        dct.copy_from_slice(&sub);
                        dct2d_8_scalar(&mut dct);
                        lf[ly * cx + lx] = dct[0];
                    }
                }

                let mut pixels = coeffs[chan_idx].clone();
                transform_to_pixels(transform_type, &mut lf, &mut pixels);

                let max_err = expected
                    .iter()
                    .zip(pixels.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < err_limit,
                    "non-special compute_forward mismatch for id {} channel {}: max_err={}",
                    transform_id,
                    chan_idx,
                    max_err
                );
            }
        }
    }

    #[test]
    #[ignore] // ~14s: heavy solver build, run explicitly with --ignored
    fn test_compute_forward_transform_coeffs_rectangular_roundtrip_with_clamp() {
        let width = 140;
        let height = 120;
        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v =
                    (x as f32 * 0.07 + y as f32 * 0.11).sin() * 5.0 + (x as f32 * 0.03).cos() * 1.5;
                x_chan[idx] = v * 0.9 + 0.2;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * -0.35 + (y as f32 * 0.02);
            }
        }

        let bx = 10;
        let by = 7;
        for &(transform_id, block_w, block_h, err_limit) in &[
            (DCT16X8_TRANSFORM_ID, 8usize, 16usize, 0.20f32),
            (DCT8X16_TRANSFORM_ID, 16usize, 8usize, 0.20f32),
            (DCT32X8_TRANSFORM_ID, 8usize, 32usize, 0.24f32),
            (DCT8X32_TRANSFORM_ID, 32usize, 8usize, 0.24f32),
            (DCT32X16_TRANSFORM_ID, 16usize, 32usize, 0.30f32),
            (DCT16X32_TRANSFORM_ID, 32usize, 16usize, 0.30f32),
        ] {
            let coeffs = compute_forward_transform_coeffs(
                transform_id,
                &x_chan,
                &y_chan,
                &b_minus_y_chan,
                width,
                height,
                bx,
                by,
                block_w,
                block_h,
            );
            let transform_type = HfTransformType::from_usize(transform_id as usize).unwrap();

            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            for (chan_idx, source) in [&x_chan[..], &y_chan[..], &b_minus_y_chan[..]]
                .into_iter()
                .enumerate()
            {
                let expected =
                    gather_clamped_block(source, width, height, bx * 8, by * 8, block_w, block_h);

                let mut lf = vec![0.0f32; cx * cy];
                for ly in 0..cy {
                    for lx in 0..cx {
                        let sub = gather_clamped_block(
                            source,
                            width,
                            height,
                            bx * 8 + lx * 8,
                            by * 8 + ly * 8,
                            8,
                            8,
                        );
                        let mut dct = [0.0f32; 64];
                        dct.copy_from_slice(&sub);
                        dct2d_8_scalar(&mut dct);
                        lf[ly * cx + lx] = dct[0];
                    }
                }

                let mut pixels = coeffs[chan_idx].clone();
                transform_to_pixels(transform_type, &mut lf, &mut pixels);

                let max_err = expected
                    .iter()
                    .zip(pixels.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < err_limit,
                    "rectangular compute_forward mismatch for id {} channel {}: max_err={}",
                    transform_id,
                    chan_idx,
                    max_err
                );
            }
        }
    }

    #[test]
    fn test_square_transform_ignored_coeff_indices_match_canonical_prefix() {
        for &(transform_id, transform_type) in &[
            (DCT16_TRANSFORM_ID, HfTransformType::DCT16X16),
            (DCT32_TRANSFORM_ID, HfTransformType::DCT32X32),
        ] {
            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            let lf_count = cx * cy;
            let coeff_count = cx * cy * 64;

            let mut ignored = Vec::new();
            for coeff_idx in 0..coeff_count {
                let mut lf = vec![0.0f32; lf_count];
                let mut coeffs = vec![0.0f32; coeff_count];
                coeffs[coeff_idx] = 1.0;
                transform_to_pixels(transform_type, &mut lf, &mut coeffs);
                let energy: f32 = coeffs.iter().map(|v| v.abs()).sum();
                if energy < 1e-6 {
                    ignored.push(coeff_idx);
                }
            }

            ignored.sort_unstable();
            assert_eq!(
                ignored.len(),
                lf_count,
                "unexpected ignored count for transform id {}",
                transform_id
            );

            let canonical =
                canonical_transform_for_shape_id(block_shape_id(transform_type) as usize).unwrap();
            let mut lowfreq = natural_coeff_order_for_transform(canonical)[..lf_count].to_vec();
            lowfreq.sort_unstable();
            assert_eq!(
                ignored, lowfreq,
                "canonical shape order prefix mismatch for transform id {}",
                transform_id
            );
        }
    }

    #[test]
    fn test_rectangular_transform_ignored_coeff_indices() {
        for &(transform_id, transform_type, expected) in &[
            (
                DCT16X8_TRANSFORM_ID,
                HfTransformType::DCT16X8,
                &[0usize, 1usize][..],
            ),
            (
                DCT8X16_TRANSFORM_ID,
                HfTransformType::DCT8X16,
                &[0usize, 1usize][..],
            ),
            (
                DCT32X8_TRANSFORM_ID,
                HfTransformType::DCT32X8,
                &[0usize, 1usize, 2usize, 3usize][..],
            ),
            (
                DCT8X32_TRANSFORM_ID,
                HfTransformType::DCT8X32,
                &[0usize, 1usize, 2usize, 3usize][..],
            ),
            (
                DCT32X16_TRANSFORM_ID,
                HfTransformType::DCT32X16,
                &[
                    0usize, 1usize, 2usize, 3usize, 32usize, 33usize, 34usize, 35usize,
                ][..],
            ),
            (
                DCT16X32_TRANSFORM_ID,
                HfTransformType::DCT16X32,
                &[
                    0usize, 1usize, 2usize, 3usize, 32usize, 33usize, 34usize, 35usize,
                ][..],
            ),
        ] {
            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            let lf_count = cx * cy;
            let coeff_count = cx * cy * 64;

            let mut ignored = Vec::new();
            for coeff_idx in 0..coeff_count {
                let mut lf = vec![0.0f32; lf_count];
                let mut coeffs = vec![0.0f32; coeff_count];
                coeffs[coeff_idx] = 1.0;
                transform_to_pixels(transform_type, &mut lf, &mut coeffs);
                let energy: f32 = coeffs.iter().map(|v| v.abs()).sum();
                if energy < 1e-6 {
                    ignored.push(coeff_idx);
                }
            }

            ignored.sort_unstable();
            assert_eq!(
                ignored.len(),
                lf_count,
                "unexpected ignored count for transform id {}",
                transform_id
            );
            assert_eq!(
                ignored, expected,
                "unexpected ignored coeff indices for transform id {}",
                transform_id
            );

            let canonical = match block_shape_id(transform_type) {
                4 => HfTransformType::DCT8X16,
                5 => HfTransformType::DCT8X32,
                6 => HfTransformType::DCT16X32,
                _ => unreachable!(),
            };
            let mut lowfreq = natural_coeff_order_for_transform(canonical)[..lf_count].to_vec();
            lowfreq.sort_unstable();
            assert_eq!(
                lowfreq, expected,
                "canonical shape order prefix mismatch for transform id {}",
                transform_id
            );
        }
    }

    #[test]
    fn test_forward_afv_from_8x8_roundtrip_via_decoder_transform() {
        let mut block = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                block[y * 8 + x] = (x as f32 * 0.21 + y as f32 * 0.17).sin() * 6.0;
            }
        }

        for &(transform_id, transform_type) in &[
            (AFV0_TRANSFORM_ID, HfTransformType::AFV0),
            (AFV1_TRANSFORM_ID, HfTransformType::AFV1),
            (AFV2_TRANSFORM_ID, HfTransformType::AFV2),
            (AFV3_TRANSFORM_ID, HfTransformType::AFV3),
        ] {
            let coeffs = forward_afv_from_8x8(&block, transform_id);
            let mut lf = vec![coeffs[0]];
            let mut pixels = coeffs.clone();
            transform_to_pixels(transform_type, &mut lf, &mut pixels);

            let max_err = block
                .iter()
                .zip(pixels.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 0.01,
                "AFV roundtrip mismatch for {:?}: max_err={}",
                transform_type,
                max_err
            );
        }
    }

    #[test]
    fn test_tokenize_hf_region_dct16_nonzero_supported() {
        let width = 16;
        let height = 16;
        let bw = 2;
        let bh = 2;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = x as f32 * 0.5 + y as f32 * 0.25;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT16_TRANSFORM_ID;
        transform_map[1] = DCT16_TRANSFORM_ID;
        transform_map[bw] = DCT16_TRANSFORM_ID;
        transform_map[bw + 1] = DCT16_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct8x16_nonzero_supported() {
        let width = 16;
        let height = 8;
        let bw = 2;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.35 + y as f32 * 0.2).sin() * 5.0;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT8X16_TRANSFORM_ID;
        transform_map[1] = DCT8X16_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct16x8_nonzero_supported() {
        let width = 8;
        let height = 16;
        let bw = 1;
        let bh = 2;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.1 + y as f32 * 0.45).cos() * 6.0;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT16X8_TRANSFORM_ID;
        transform_map[1] = DCT16X8_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_identity_nonzero_supported() {
        let width = 8;
        let height = 8;
        let bw = 1;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.13 + y as f32 * 0.29).sin() * 5.0;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | IDENTITY_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct2x2_nonzero_supported() {
        let width = 8;
        let height = 8;
        let bw = 1;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.29 + y as f32 * 0.13).cos() * 5.0;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT2X2_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct4x4_nonzero_supported() {
        let width = 8;
        let height = 8;
        let bw = 1;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.17 + y as f32 * 0.23).sin() * 5.0;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT4X4_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct4x8_nonzero_supported() {
        let width = 8;
        let height = 8;
        let bw = 1;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.33 + y as f32 * 0.27).sin() * 5.3;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT4X8_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct8x4_nonzero_supported() {
        let width = 8;
        let height = 8;
        let bw = 1;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.27 + y as f32 * 0.33).cos() * 5.3;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | DCT8X4_TRANSFORM_ID;

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_afv_nonzero_supported() {
        let width = 8;
        let height = 8;
        let bw = 1;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.19 + y as f32 * 0.37).sin() * 5.3;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let raw_quant_map = vec![1u8; bw * bh];
        for afv_id in [
            AFV0_TRANSFORM_ID,
            AFV1_TRANSFORM_ID,
            AFV2_TRANSFORM_ID,
            AFV3_TRANSFORM_ID,
        ] {
            let ac_x = vec![0i32; bw * bh * 64];
            let ac_y = vec![0i32; bw * bh * 64];
            let ac_b = vec![0i32; bw * bh * 64];

            let mut transform_map = build_default_transform_map(bw, bh);
            transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | afv_id;

            let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
                &ac_x,
                &ac_y,
                &ac_b,
                &x_chan,
                &y_chan,
                &b_minus_y_chan,
                width,
                height,
                bw,
                bh,
                16384,
                &raw_quant_map,
                &transform_map,
                0.8,
                1.0,
            )
            .unwrap();

            let tokens = tokenize_hf_region(
                &ac_x,
                &ac_y,
                &ac_b,
                &transform_map,
                bw,
                0,
                0,
                bw,
                bh,
                15,
                0,
                None,
                None,
                None,
            )
            .unwrap();
            assert!(tokens.iter().any(|t| t.value > 0));
        }
    }

    #[test]
    #[ignore] // ~26s: encodes+decodes 24 transform types, run explicitly with --ignored
    fn test_forced_non8x8_transform_maps_decode() {
        for &(width, height, transform_id) in &[
            (16usize, 8usize, DCT8X16_TRANSFORM_ID),
            (8usize, 16usize, DCT16X8_TRANSFORM_ID),
            (32usize, 8usize, DCT8X32_TRANSFORM_ID),
            (8usize, 32usize, DCT32X8_TRANSFORM_ID),
            (16usize, 32usize, DCT32X16_TRANSFORM_ID),
            (32usize, 16usize, DCT16X32_TRANSFORM_ID),
            (8usize, 8usize, IDENTITY_TRANSFORM_ID),
            (8usize, 8usize, DCT2X2_TRANSFORM_ID),
            (8usize, 8usize, DCT4X4_TRANSFORM_ID),
            (8usize, 8usize, DCT4X8_TRANSFORM_ID),
            (8usize, 8usize, DCT8X4_TRANSFORM_ID),
            (8usize, 8usize, AFV0_TRANSFORM_ID),
            (8usize, 8usize, AFV1_TRANSFORM_ID),
            (8usize, 8usize, AFV2_TRANSFORM_ID),
            (8usize, 8usize, AFV3_TRANSFORM_ID),
            (32usize, 32usize, DCT32_TRANSFORM_ID),
            (64usize, 64usize, DCT64_TRANSFORM_ID),
            (32usize, 64usize, DCT64X32_TRANSFORM_ID),
            (64usize, 32usize, DCT32X64_TRANSFORM_ID),
            (128usize, 128usize, DCT128_TRANSFORM_ID),
            (64usize, 128usize, DCT128X64_TRANSFORM_ID),
            (128usize, 64usize, DCT64X128_TRANSFORM_ID),
            (256usize, 256usize, DCT256_TRANSFORM_ID),
            (128usize, 256usize, DCT256X128_TRANSFORM_ID),
            (256usize, 128usize, DCT128X256_TRANSFORM_ID),
        ] {
            let mut rgb = vec![0u8; width * height * 3];
            for y in 0..height {
                for x in 0..width {
                    let i = (y * width + x) * 3;
                    rgb[i] = ((x * 17 + y * 13) & 255) as u8;
                    rgb[i + 1] = ((x * 7 + y * 29) & 255) as u8;
                    rgb[i + 2] = ((x * 11 + y * 5) & 255) as u8;
                }
            }

            let npixels = width * height;
            let mut x_chan = vec![0.0f32; npixels];
            let mut y_chan = vec![0.0f32; npixels];
            let mut b_chan = vec![0.0f32; npixels];
            srgb_u8_to_xyb(&rgb, width, height, &mut x_chan, &mut y_chan, &mut b_chan).unwrap();
            let b_minus_y_chan: Vec<f32> =
                b_chan.iter().zip(&y_chan).map(|(&b, &y)| b - y).collect();

            let bw = width.div_ceil(8);
            let bh = height.div_ceil(8);
            let mut dct_x = vec![0.0f32; bw * bh * 64];
            let mut dct_y = vec![0.0f32; bw * bh * 64];
            let mut dct_b = vec![0.0f32; bw * bh * 64];
            forward_dct_channel(&x_chan, width, height, bw, bh, &mut dct_x);
            forward_dct_channel(&y_chan, width, height, bw, bh, &mut dct_y);
            forward_dct_channel(&b_chan, width, height, bw, bh, &mut dct_b);

            let (global_scale, quant_lf) = distance_to_quant_params(2.5);
            let raw_quant_map = vec![1u8; bw * bh];
            let cr_size = bw.div_ceil(8) * bh.div_ceil(8);
            let ytox_map = vec![0i32; cr_size];
            let ytob_map = vec![0i32; cr_size];
            let quantized = quantize_vardct_blocks(
                &dct_x,
                &dct_y,
                &dct_b,
                global_scale,
                quant_lf,
                &raw_quant_map,
                &[4096.0f32, 512.0, 256.0],
                default_dct8x8_dequant_weights(),
                0.8,
                1.0,
                bw,
                &ytox_map,
                &ytob_map,
            );

            let mut transform_map = build_default_transform_map(bw, bh);
            transform_map[0] = TRANSFORM_FIRST_BLOCK_FLAG | transform_id;
            let transform_type = HfTransformType::from_usize(transform_id as usize).unwrap();
            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            for yb in 0..cy {
                for xb in 0..cx {
                    if xb == 0 && yb == 0 {
                        continue;
                    }
                    transform_map[yb * bw + xb] = transform_id;
                }
            }

            let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
                &quantized.ac_x,
                &quantized.ac_y,
                &quantized.ac_b,
                &x_chan,
                &y_chan,
                &b_minus_y_chan,
                width,
                height,
                bw,
                bh,
                global_scale,
                &raw_quant_map,
                &transform_map,
                0.8,
                1.0,
            )
            .unwrap();

            let cs = encode_vardct_frame(
                width,
                height,
                bw,
                bh,
                global_scale,
                quant_lf,
                &quantized.dc_y,
                &quantized.dc_x,
                &quantized.dc_b,
                &ac_x,
                &ac_y,
                &ac_b,
                &raw_quant_map,
                &transform_map,
                &ytox_map,
                &ytob_map,
                true, // use_gab
            )
            .unwrap();

            let (_n, frames) = crate::api::tests::decode(&cs, usize::MAX, true, false, None)
                .expect("decode should succeed for forced rectangular transform");
            assert!(!frames.is_empty());
        }
    }

    #[test]
    fn test_tokenize_hf_region_dct32_nonzero_supported() {
        let width = 32;
        let height = 32;
        let bw = 4;
        let bh = 4;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.21 + y as f32 * 0.17).sin() * 8.0;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..4 {
            for x in 0..4 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT32_TRANSFORM_ID
                } else {
                    DCT32_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct8x32_nonzero_supported() {
        let width = 32;
        let height = 8;
        let bw = 4;
        let bh = 1;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.13 + y as f32 * 0.41).cos() * 6.8;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for x in 0..4 {
            transform_map[x] = if x == 0 {
                TRANSFORM_FIRST_BLOCK_FLAG | DCT8X32_TRANSFORM_ID
            } else {
                DCT8X32_TRANSFORM_ID
            };
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct32x8_nonzero_supported() {
        let width = 8;
        let height = 32;
        let bw = 1;
        let bh = 4;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.41 + y as f32 * 0.13).sin() * 6.8;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..4 {
            transform_map[y * bw] = if y == 0 {
                TRANSFORM_FIRST_BLOCK_FLAG | DCT32X8_TRANSFORM_ID
            } else {
                DCT32X8_TRANSFORM_ID
            };
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct16x32_nonzero_supported() {
        let width = 32;
        let height = 16;
        let bw = 4;
        let bh = 2;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.19 + y as f32 * 0.27).cos() * 7.5;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..2 {
            for x in 0..4 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT16X32_TRANSFORM_ID
                } else {
                    DCT16X32_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct32x16_nonzero_supported() {
        let width = 16;
        let height = 32;
        let bw = 2;
        let bh = 4;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = (x as f32 * 0.27 + y as f32 * 0.19).sin() * 7.5;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..4 {
            for x in 0..2 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT32X16_TRANSFORM_ID
                } else {
                    DCT32X16_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct64_nonzero_supported() {
        let width = 64;
        let height = 64;
        let bw = 8;
        let bh = 8;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = ((x as f32 * 0.07).sin() + (y as f32 * 0.11).cos()) * 5.5;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..8 {
            for x in 0..8 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT64_TRANSFORM_ID
                } else {
                    DCT64_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct64x32_nonzero_supported() {
        let width = 32;
        let height = 64;
        let bw = 4;
        let bh = 8;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = ((x as f32 * 0.21).sin() + (y as f32 * 0.05).cos()) * 5.2;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..8 {
            for x in 0..4 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT64X32_TRANSFORM_ID
                } else {
                    DCT64X32_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct32x64_nonzero_supported() {
        let width = 64;
        let height = 32;
        let bw = 8;
        let bh = 4;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = ((x as f32 * 0.05).sin() + (y as f32 * 0.21).cos()) * 5.2;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..4 {
            for x in 0..8 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT32X64_TRANSFORM_ID
                } else {
                    DCT32X64_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct128_nonzero_supported() {
        let width = 128;
        let height = 128;
        let bw = 16;
        let bh = 16;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = ((x as f32 * 0.03).sin() + (y as f32 * 0.05).cos()) * 4.8;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..16 {
            for x in 0..16 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT128_TRANSFORM_ID
                } else {
                    DCT128_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct128x64_nonzero_supported() {
        let width = 64;
        let height = 128;
        let bw = 8;
        let bh = 16;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = ((x as f32 * 0.07).sin() + (y as f32 * 0.03).cos()) * 4.8;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..16 {
            for x in 0..8 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT128X64_TRANSFORM_ID
                } else {
                    DCT128X64_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct64x128_nonzero_supported() {
        let width = 128;
        let height = 64;
        let bw = 16;
        let bh = 8;

        let mut x_chan = vec![0.0f32; width * height];
        let mut y_chan = vec![0.0f32; width * height];
        let mut b_minus_y_chan = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let v = ((x as f32 * 0.03).sin() + (y as f32 * 0.07).cos()) * 4.8;
                x_chan[idx] = v * 0.9;
                y_chan[idx] = v;
                b_minus_y_chan[idx] = v * 0.1;
            }
        }

        let ac_x = vec![0i32; bw * bh * 64];
        let ac_y = vec![0i32; bw * bh * 64];
        let ac_b = vec![0i32; bw * bh * 64];
        let raw_quant_map = vec![1u8; bw * bh];

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..8 {
            for x in 0..16 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT64X128_TRANSFORM_ID
                } else {
                    DCT64X128_TRANSFORM_ID
                };
            }
        }

        let (ac_x, ac_y, ac_b) = prepare_ac_for_transform_map(
            &ac_x,
            &ac_y,
            &ac_b,
            &x_chan,
            &y_chan,
            &b_minus_y_chan,
            width,
            height,
            bw,
            bh,
            16384,
            &raw_quant_map,
            &transform_map,
            0.8,
            1.0,
        )
        .unwrap();

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct256_nonzero_supported() {
        let bw = 32;
        let bh = 32;

        let mut ac_x = vec![0i32; bw * bh * 64];
        let mut ac_y = vec![0i32; bw * bh * 64];
        let mut ac_b = vec![0i32; bw * bh * 64];

        for by in 0..bh {
            for bx in 0..bw {
                let base = (by * bw + bx) * 64;
                ac_x[base + 1] = 1;
                ac_y[base + 2] = -1;
                ac_b[base + 3] = 1;
            }
        }

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..32 {
            for x in 0..32 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT256_TRANSFORM_ID
                } else {
                    DCT256_TRANSFORM_ID
                };
            }
        }

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct256x128_nonzero_supported() {
        let bw = 16;
        let bh = 32;

        let mut ac_x = vec![0i32; bw * bh * 64];
        let mut ac_y = vec![0i32; bw * bh * 64];
        let mut ac_b = vec![0i32; bw * bh * 64];

        for by in 0..bh {
            for bx in 0..bw {
                let base = (by * bw + bx) * 64;
                ac_x[base + 1] = 1;
                ac_y[base + 2] = -1;
                ac_b[base + 3] = 1;
            }
        }

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..32 {
            for x in 0..16 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT256X128_TRANSFORM_ID
                } else {
                    DCT256X128_TRANSFORM_ID
                };
            }
        }

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_tokenize_hf_region_dct128x256_nonzero_supported() {
        let bw = 32;
        let bh = 16;

        let mut ac_x = vec![0i32; bw * bh * 64];
        let mut ac_y = vec![0i32; bw * bh * 64];
        let mut ac_b = vec![0i32; bw * bh * 64];

        for by in 0..bh {
            for bx in 0..bw {
                let base = (by * bw + bx) * 64;
                ac_x[base + 1] = 1;
                ac_y[base + 2] = -1;
                ac_b[base + 3] = 1;
            }
        }

        let mut transform_map = build_default_transform_map(bw, bh);
        for y in 0..16 {
            for x in 0..32 {
                let idx = y * bw + x;
                transform_map[idx] = if x == 0 && y == 0 {
                    TRANSFORM_FIRST_BLOCK_FLAG | DCT128X256_TRANSFORM_ID
                } else {
                    DCT128X256_TRANSFORM_ID
                };
            }
        }

        let tokens = tokenize_hf_region(
            &ac_x,
            &ac_y,
            &ac_b,
            &transform_map,
            bw,
            0,
            0,
            bw,
            bh,
            15,
            0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(tokens.iter().any(|t| t.value > 0));
    }

    #[test]
    fn test_forward_dct_channel_constant() {
        let chan = vec![128.0f32; 64];
        let mut out = vec![0.0f32; 64];
        forward_dct_channel(&chan, 8, 8, 1, 1, &mut out);
        // DC = 128 (after 2D DCT normalization)
        assert!(
            (out[0] - 128.0).abs() < 0.01,
            "DC = {}, expected 128",
            out[0]
        );
        // All AC should be ~0
        for i in 1..64 {
            assert!(out[i].abs() < 0.01, "AC[{i}] = {}, expected ~0", out[i]);
        }
    }

    #[test]
    fn test_encode_vardct_produces_output() {
        // Minimal test: encode a small image and verify we get bytes back
        let width = 8;
        let height = 8;
        let rgb = vec![128u8; width * height * 3];
        let config = VarDctConfig::default();
        let result = encode_vardct_u8_rgb(&rgb, width, height, &config);
        assert!(result.is_ok(), "encode failed: {:?}", result.err());
        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
        // Should start with JXL container signature
        assert_eq!(&bytes[..2], &[0x00, 0x00]);
    }

    #[test]
    fn test_encode_vardct_codestream_structure() {
        let width = 8;
        let height = 8;
        let rgb = vec![128u8; width * height * 3];
        let config = VarDctConfig::default();

        let codestream = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();
        assert_eq!(codestream[0], 0xFF);
        assert_eq!(codestream[1], 0x0A);
        eprintln!("Codestream size: {} bytes", codestream.len());
        eprintln!(
            "Hex: {}",
            codestream
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        let container = encode_vardct_u8_rgb(&rgb, width, height, &config).unwrap();
        let path = "/tmp/test_vardct_8x8.jxl";
        std::fs::write(path, &container).unwrap();
        eprintln!("Written to {path} ({} bytes)", container.len());
    }

    #[test]
    fn test_decode_vardct_jxlrs() {
        let width = 8;
        let height = 8;
        let rgb = vec![128u8; width * height * 3];
        let config = VarDctConfig::default();
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();

        // Decode with the jxl-rs test helper (includes NaN check)
        let (num_frames, frames) = crate::api::tests::decode(&cs, usize::MAX, true, false, None)
            .expect("jxl-rs decode should succeed");
        assert_eq!(num_frames, 1, "should have 1 frame");

        // Check output pixels: f32 interleaved RGB, 3 channels
        let frame = &frames[0];
        let buf = &frame[0]; // interleaved color channels
        let (bw, bh) = buf.size();
        eprintln!("Decoded buffer: {}x{}", bw, bh);
        assert_eq!(bw, width * 3, "buffer width = width * 3 channels");
        assert_eq!(bh, height);

        // Print first few decoded pixels
        for y in 0..2 {
            let row = buf.row(y);
            for x in 0..2 {
                let r = row[x * 3];
                let g = row[x * 3 + 1];
                let b = row[x * 3 + 2];
                eprintln!("  pixel({},{}) = ({:.4}, {:.4}, {:.4})", x, y, r, g, b);
            }
        }
    }

    #[test]
    fn test_vardct_16x16_roundtrip() {
        // 16x16 = 4 blocks, still single-group
        let width = 16;
        let height = 16;
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                // Each quadrant a different color
                let qx = if x < 8 { 0 } else { 1 };
                let qy = if y < 8 { 0 } else { 1 };
                match (qx, qy) {
                    (0, 0) => {
                        rgb[i] = 255;
                        rgb[i + 1] = 0;
                        rgb[i + 2] = 0;
                    } // Red
                    (1, 0) => {
                        rgb[i] = 0;
                        rgb[i + 1] = 255;
                        rgb[i + 2] = 0;
                    } // Green
                    (0, 1) => {
                        rgb[i] = 0;
                        rgb[i + 1] = 0;
                        rgb[i + 2] = 255;
                    } // Blue
                    _ => {
                        rgb[i] = 255;
                        rgb[i + 1] = 255;
                        rgb[i + 2] = 0;
                    } // Yellow
                }
            }
        }
        let config = VarDctConfig::default();
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();

        // Decode with jxl-rs
        let (_n, frames) = crate::api::tests::decode(&cs, usize::MAX, true, false, None)
            .expect("decode should succeed");

        let buf = &frames[0][0];
        eprintln!("16x16 quad-color test: decoded OK");
        // Check corners - should roughly match input colors (DC-only = block average)
        // Verify dominant channel for each quadrant.
        // TL=Red (R high, G/B low), TR=Green (G high, R/B low),
        // BL=Blue (B high, R/G low), BR=Yellow (R+G high, B low).
        for (qx, qy, label, expect_r_hi, expect_g_hi, expect_b_hi) in [
            (0, 0, "TL-Red", true, false, false),
            (1, 0, "TR-Green", false, true, false),
            (0, 1, "BL-Blue", false, false, true),
            (1, 1, "BR-Yellow", true, true, false),
        ] {
            let x = qx * 8 + 4;
            let y = qy * 8 + 4;
            let row = buf.row(y);
            let r = (row[x * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (row[x * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (row[x * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            eprintln!("  {} center: ({},{},{})", label, r, g, b);
            if expect_r_hi {
                assert!(r > 180, "{label}: R should be high, got {r}");
            } else {
                assert!(r < 80, "{label}: R should be low, got {r}");
            }
            if expect_g_hi {
                assert!(g > 180, "{label}: G should be high, got {g}");
            } else {
                assert!(g < 80, "{label}: G should be low, got {g}");
            }
            if expect_b_hi {
                assert!(b > 180, "{label}: B should be high, got {b}");
            } else {
                assert!(b < 100, "{label}: B should be low, got {b}");
            }
        }

        // Also write to file for djxl verification
        let file_data = crate::encode::container::wrap_codestream(&cs).unwrap();
        std::fs::write("/tmp/test_vardct_16x16.jxl", &file_data).unwrap();
        eprintln!(
            "Written {} bytes to /tmp/test_vardct_16x16.jxl",
            file_data.len()
        );
    }

    #[test]
    fn test_vardct_gradient_roundtrip() {
        // Test with gradient image - more interesting than constant
        let width = 8;
        let height = 8;
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                rgb[i] = (x * 255 / 7) as u8; // R: gradient left-right
                rgb[i + 1] = (y * 255 / 7) as u8; // G: gradient top-bottom
                rgb[i + 2] = 128; // B: constant
            }
        }
        // Use low distance for AC detail
        let config = VarDctConfig {
            distance: 0.1,
            effort: 7,
            progressive: false,
        };
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();
        eprintln!("Gradient codestream: {} bytes", cs.len());

        // Write to file for djxl testing
        let file_data = crate::encode::container::wrap_codestream(&cs).unwrap();
        std::fs::write("/tmp/test_vardct_gradient.jxl", &file_data).unwrap();

        // Decode with jxl-rs
        let (_n, frames) = crate::api::tests::decode(&cs, usize::MAX, true, false, None)
            .expect("decode should succeed");

        let buf = &frames[0][0];
        eprintln!("Gradient test decoded OK");
        for y in 0..2 {
            let row = buf.row(y);
            for x in 0..2 {
                let r = (row[x * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (row[x * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (row[x * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
                eprintln!(
                    "  pixel({},{}) input=({},{},{}) decoded=({},{},{})",
                    x,
                    y,
                    rgb[(y * width + x) * 3],
                    rgb[(y * width + x) * 3 + 1],
                    rgb[(y * width + x) * 3 + 2],
                    r,
                    g,
                    b
                );
            }
        }
    }

    #[test]
    fn test_vardct_quality_levels() {
        // Test encode-decode roundtrip at different quality levels
        let width = 8;
        let height = 8;
        let mut rgb = vec![0u8; width * height * 3];
        // Checker pattern with 2x2 blocks
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                if (x / 2 + y / 2) % 2 == 0 {
                    rgb[i] = 200;
                    rgb[i + 1] = 50;
                    rgb[i + 2] = 100;
                } else {
                    rgb[i] = 50;
                    rgb[i + 1] = 200;
                    rgb[i + 2] = 150;
                }
            }
        }

        for (distance, label) in [
            (0.01, "near-lossless"),
            (0.5, "high"),
            (1.0, "default"),
            (3.0, "low"),
        ] {
            let config = VarDctConfig {
                distance,
                effort: 7,
                progressive: false,
            };
            let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();
            let (_n, frames) = crate::api::tests::decode(&cs, usize::MAX, true, false, None)
                .expect("decode should succeed");

            let buf = &frames[0][0];
            // Compute max pixel error
            let mut max_err = 0u32;
            for y in 0..height {
                let row = buf.row(y);
                for x in 0..width {
                    let i = (y * width + x) * 3;
                    let dr = ((row[x * 3].clamp(0.0, 1.0) * 255.0).round() as i32 - rgb[i] as i32)
                        .unsigned_abs();
                    let dg = ((row[x * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as i32
                        - rgb[i + 1] as i32)
                        .unsigned_abs();
                    let db = ((row[x * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as i32
                        - rgb[i + 2] as i32)
                        .unsigned_abs();
                    max_err = max_err.max(dr).max(dg).max(db);
                }
            }
            eprintln!(
                "  d={:.2} ({:14}): {} bytes, max_err={}",
                distance,
                label,
                cs.len(),
                max_err
            );

            // Note: at near-lossless, error can still be significant because
            // our simple encoder doesn't yet optimize for quality (no adaptive
            // quant, no CfL optimization, etc.)
            // At d=0.01, expect max_err < 100 for now.
            if distance <= 0.1 {
                assert!(
                    max_err <= 100,
                    "near-lossless should have reasonable error, got {max_err}"
                );
            }
        }
    }

    #[test]
    fn test_vardct_large_image() {
        // Test 64x64 image (8 groups x 8 groups of 8x8 blocks)
        // Still single-group since 64 < 256
        let width = 64;
        let height = 64;
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                // Smooth gradient with some structure
                rgb[i] = (x * 255 / (width - 1)) as u8;
                rgb[i + 1] = (y * 255 / (height - 1)) as u8;
                rgb[i + 2] = ((x + y) * 128 / (width + height - 2)) as u8;
            }
        }

        // Test multiple distances
        for distance in [1.0f32, 0.5] {
            let config = VarDctConfig {
                distance,
                effort: 7,
                progressive: false,
            };
            let cs = match encode_vardct_u8_rgb_codestream(&rgb, width, height, &config) {
                Ok(cs) => cs,
                Err(e) => panic!("Encoding failed: {e:?}"),
            };
            eprintln!("64x64 codestream: {} bytes", cs.len());

            // Write to file for visual inspection
            let file_data = crate::encode::container::wrap_codestream(&cs).unwrap();
            std::fs::write("/tmp/test_vardct_64x64.jxl", &file_data).unwrap();

            // Decode with jxl-rs
            let result = crate::api::tests::decode(&cs, usize::MAX, true, false, None);
            match result {
                Ok((_n, frames)) => {
                    let buf = &frames[0][0];
                    assert_eq!(buf.size(), (width * 3, height));
                    eprintln!("  d={distance}: {} bytes - OK", cs.len());
                }
                Err(e) => {
                    eprintln!("  d={distance}: {} bytes - FAILED: {e:?}", cs.len());
                }
            }
        }
    }

    // Large image test is in test_vardct_large_image -- uses djxl for verification

    #[test]
    fn test_write_vardct_to_file() {
        let width = 8;
        let height = 8;
        let rgb = vec![128u8; width * height * 3];
        let config = VarDctConfig::default();
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();
        let hex: String = cs
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("Actual codestream ({} bytes): {hex}", cs.len());

        let container = encode_vardct_u8_rgb(&rgb, width, height, &config).unwrap();
        std::fs::write("/tmp/test_vardct_8x8.jxl", &container).unwrap();
        eprintln!(
            "Written {} bytes to /tmp/test_vardct_8x8.jxl",
            container.len()
        );
    }

    #[test]
    fn test_trace_vardct_bitstream() {
        use crate::bit_reader::BitReader;
        use crate::headers::JxlHeader;

        // Generate the codestream
        let width = 8;
        let height = 8;
        let rgb = vec![128u8; width * height * 3];
        let config = VarDctConfig::default();
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();

        // Print hex
        let hex: String = cs
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("Codestream ({} bytes): {hex}", cs.len());

        // Pad for BitReader
        let mut padded = cs.clone();
        for _ in 0..128 {
            padded.push(0);
        }

        // Try parsing the full JxlHeader
        let mut br = BitReader::new(&padded);
        let header_result: crate::error::Result<_> =
            <crate::headers::FileHeader as JxlHeader>::read(&mut br);
        match header_result {
            Ok(h) => {
                eprintln!("File header parsed OK at bit {}", br.total_bits_read());
                eprintln!("  size: {}x{}", h.size.xsize(), h.size.ysize());
                eprintln!("  xyb_encoded: {}", h.image_metadata.xyb_encoded);
            }
            Err(e) => {
                eprintln!("File header error at bit {}: {e:?}", br.total_bits_read());
                return;
            }
        }

        // Try parsing frame header
        use crate::headers::encodings::UnconditionalCoder;
        use crate::headers::frame_header::FrameHeader;
        let fh_result: crate::error::Result<FrameHeader> = FrameHeader::read_unconditional(
            &(),
            &mut br,
            &crate::headers::frame_header::FrameHeaderNonserialized {
                xyb_encoded: true,
                num_extra_channels: 0,
                extra_channel_info: vec![],
                have_animation: false,
                have_timecode: false,
                img_width: 8,
                img_height: 8,
            },
        );
        match fh_result {
            Ok(fh) => {
                eprintln!("Frame header parsed OK at bit {}", br.total_bits_read());
                eprintln!("  encoding: {:?}", fh.encoding);
                eprintln!("  width: {}, height: {}", fh.width, fh.height);
            }
            Err(e) => {
                eprintln!("Frame header error at bit {}: {e:?}", br.total_bits_read());
            }
        }
    }

    #[test]
    fn test_encode_vardct_16x16() {
        let width = 16;
        let height = 16;
        let mut rgb = vec![0u8; width * height * 3];
        // Gradient
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                rgb[i] = (x * 16) as u8;
                rgb[i + 1] = (y * 16) as u8;
                rgb[i + 2] = 128;
            }
        }
        let config = VarDctConfig {
            distance: 2.0,
            effort: 7,
            progressive: false,
        };
        let result = encode_vardct_u8_rgb(&rgb, width, height, &config);
        assert!(result.is_ok(), "encode failed: {:?}", result.err());
    }

    #[test]
    fn test_vardct_multigroup() {
        // 512x512 image -> 2x2 = 4 groups (group_dim = 256px = 32 blocks)
        let width = 512;
        let height = 512;
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                rgb[i] = (x * 255 / (width - 1)) as u8;
                rgb[i + 1] = (y * 255 / (height - 1)) as u8;
                rgb[i + 2] = 128;
            }
        }

        let config = VarDctConfig {
            distance: 1.0,
            effort: 7,
            progressive: false,
        };
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config)
            .expect("multi-group encode should succeed");
        eprintln!("512x512 multi-group codestream: {} bytes", cs.len());

        // Write to file for djxl verification
        let container = encode_vardct_u8_rgb(&rgb, width, height, &config).unwrap();
        std::fs::write("/tmp/test_vardct_512x512.jxl", &container).unwrap();

        // Verify with jxl-rs decoder
        let result = crate::api::tests::decode(&cs, usize::MAX, true, false, None);
        match result {
            Ok((_n, frames)) => {
                let buf = &frames[0][0];
                assert_eq!(buf.size(), (width * 3, height));
                // Spot-check a few pixels
                let row0 = buf.row(0);
                let r0 = (row0[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g0 = (row0[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b0 = (row0[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                eprintln!("  pixel(0,0): ({r0},{g0},{b0}) expected ~(0,0,128)");
                // Center pixel
                let cy = height / 2;
                let cx = width / 2;
                let rowc = buf.row(cy);
                let rc = (rowc[cx * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
                let gc = (rowc[cx * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let bc = (rowc[cx * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
                eprintln!("  pixel({cx},{cy}): ({rc},{gc},{bc}) expected ~(128,128,128)");
                // Bottom-right
                let lrow = buf.row(height - 1);
                let rl = (lrow[(width - 1) * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
                let gl = (lrow[(width - 1) * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let bl = (lrow[(width - 1) * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
                eprintln!(
                    "  pixel({},{height}): ({rl},{gl},{bl}) expected ~(255,255,128)",
                    width - 1
                );
                eprintln!("  Multi-group 512x512 OK: {} bytes", cs.len());
            }
            Err(e) => {
                panic!("Multi-group 512x512 decode failed: {e:?}");
            }
        }
    }

    #[test]
    #[ignore] // ~11s: large multi-group encode/decode, run with --ignored
    fn test_vardct_multigroup_large() {
        // Test images that cross HF and LF group boundaries
        for (width, height, label) in [
            (2049, 1, "2049x1 (2 LF groups wide)"),
            (1024, 1024, "1024x1024 (16 HF groups, 1 LF group)"),
        ] {
            let mut rgb = vec![128u8; width * height * 3];
            for y in 0..height {
                for x in 0..width {
                    let i = (y * width + x) * 3;
                    rgb[i] = (x * 255 / width.max(1)) as u8;
                    rgb[i + 1] = (y * 255 / height.max(1)) as u8;
                    rgb[i + 2] = 128;
                }
            }

            let config = VarDctConfig {
                distance: 1.0,
                effort: 7,
                progressive: false,
            };
            let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config)
                .expect(&format!("{label} encode failed"));
            eprintln!("{label}: {} bytes", cs.len());

            let container = encode_vardct_u8_rgb(&rgb, width, height, &config).unwrap();
            let path = format!("/tmp/test_vardct_{width}x{height}.jxl");
            std::fs::write(&path, &container).unwrap();

            let result = crate::api::tests::decode(&cs, usize::MAX, true, false, None);
            match result {
                Ok((_n, frames)) => {
                    let buf = &frames[0][0];
                    assert_eq!(buf.size(), (width * 3, height));
                    eprintln!("  {label} OK");
                }
                Err(e) => {
                    panic!("{label} decode failed: {e:?}");
                }
            }
        }
    }

    #[test]
    fn test_vardct_multigroup_small() {
        use crate::bit_reader::BitReader;
        use crate::headers::JxlHeader;
        use crate::headers::encodings::UnconditionalCoder;
        use crate::headers::frame_header::{FrameHeader, FrameHeaderNonserialized};
        use crate::headers::toc::{Toc, TocNonserialized};

        // 257x257 -> just barely multi-group (2x2 groups, last group is tiny)
        let width = 257;
        let height = 257;
        let mut rgb = vec![128u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                rgb[i] = ((x * 3 + y) % 256) as u8;
                rgb[i + 1] = ((x + y * 2) % 256) as u8;
            }
        }

        let config = VarDctConfig {
            distance: 1.0,
            effort: 7,
            progressive: false,
        };
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config)
            .expect("257x257 multi-group encode should succeed");
        eprintln!("257x257 multi-group codestream: {} bytes", cs.len());

        // Write for djxl debugging
        let container = encode_vardct_u8_rgb(&rgb, width, height, &config).unwrap();
        std::fs::write("/tmp/test_vardct_257x257.jxl", &container).unwrap();

        // Step-by-step parse to find the exact error location
        let mut padded = cs.clone();
        padded.extend_from_slice(&[0u8; 256]);
        let mut br = BitReader::new(&padded);

        // 1. File header
        let fh = <crate::headers::FileHeader as JxlHeader>::read(&mut br)
            .expect("FileHeader parse failed");
        eprintln!("FileHeader OK at bit {}", br.total_bits_read());
        eprintln!("  size: {}x{}", fh.size.xsize(), fh.size.ysize());

        // 2. Byte-align before frame header
        br.jump_to_byte_boundary()
            .expect("byte align before frame header failed");
        eprintln!("After byte-align: bit {}", br.total_bits_read());

        // 3. Frame header
        let frame_hdr = FrameHeader::read_unconditional(
            &(),
            &mut br,
            &FrameHeaderNonserialized {
                xyb_encoded: true,
                num_extra_channels: 0,
                extra_channel_info: vec![],
                have_animation: false,
                have_timecode: false,
                img_width: width as u32,
                img_height: height as u32,
            },
        )
        .expect("FrameHeader parse failed");
        eprintln!("FrameHeader OK at bit {}", br.total_bits_read());
        eprintln!("  encoding: {:?}", frame_hdr.encoding);
        eprintln!("  num_groups: {}", frame_hdr.num_groups());
        eprintln!("  num_lf_groups: {}", frame_hdr.num_lf_groups());

        // 4. TOC
        let num_toc_entries = frame_hdr.num_toc_entries() as u32;
        eprintln!("  toc_entries: {}", num_toc_entries);
        let toc = Toc::read_unconditional(
            &(),
            &mut br,
            &TocNonserialized {
                num_entries: num_toc_entries,
            },
        );
        match &toc {
            Ok(toc) => {
                eprintln!("TOC OK at bit {}: {:?}", br.total_bits_read(), toc.entries);
            }
            Err(e) => {
                eprintln!("TOC FAILED at bit {}: {e:?}", br.total_bits_read());
                panic!("TOC parse failed: {e:?}");
            }
        }

        // 5. Full decode
        let result = crate::api::tests::decode(&cs, usize::MAX, true, false, None);
        match result {
            Ok((_n, frames)) => {
                let buf = &frames[0][0];
                assert_eq!(buf.size(), (width * 3, height));
                eprintln!("  257x257 multi-group OK: {} bytes", cs.len());
            }
            Err(e) => {
                panic!("257x257 multi-group decode failed: {e:?}");
            }
        }
    }
}

#[cfg(test)]
mod inverse_transform_tests {
    use super::*;

    #[test]
    fn test_inverse_transform_8x8_dct2x2_roundtrip() {
        let mut pixels = vec![vec![0.0f32; 64]; 3];
        for c in 0..3 {
            for y in 0..8 {
                for x in 0..8 {
                    pixels[c][y * 8 + x] = (x as f32 * 0.1 + y as f32 * 0.2 + c as f32 * 0.3).sin();
                }
            }
        }
        let coeffs = compute_forward_transform_coeffs(
            DCT2X2_TRANSFORM_ID,
            &pixels[0],
            &pixels[1],
            &pixels[2],
            8,
            8,
            0,
            0,
            8,
            8,
        );
        let recon = inverse_transform_8x8_all_channels(DCT2X2_TRANSFORM_ID, &coeffs);
        for c in 0..3 {
            let max_err = pixels[c]
                .iter()
                .zip(recon[c].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 0.05,
                "DCT2X2 roundtrip channel {}: max_err={}",
                c,
                max_err
            );
        }
    }

    #[test]
    fn test_inverse_transform_8x8_identity_roundtrip() {
        let mut pixels = vec![vec![0.0f32; 64]; 3];
        for c in 0..3 {
            for y in 0..8 {
                for x in 0..8 {
                    pixels[c][y * 8 + x] = (x as f32 * 0.1 + y as f32 * 0.2 + c as f32 * 0.3).sin();
                }
            }
        }
        let coeffs = compute_forward_transform_coeffs(
            IDENTITY_TRANSFORM_ID,
            &pixels[0],
            &pixels[1],
            &pixels[2],
            8,
            8,
            0,
            0,
            8,
            8,
        );
        let recon = inverse_transform_8x8_all_channels(IDENTITY_TRANSFORM_ID, &coeffs);
        for c in 0..3 {
            let max_err = pixels[c]
                .iter()
                .zip(recon[c].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 0.05,
                "IDENTITY roundtrip channel {}: max_err={}",
                c,
                max_err
            );
        }
    }

    #[test]
    fn test_inverse_transform_8x8_dct8_roundtrip() {
        // Create test pixels
        let mut pixels = vec![vec![0.0f32; 64]; 3];
        for c in 0..3 {
            for y in 0..8 {
                for x in 0..8 {
                    pixels[c][y * 8 + x] = (x as f32 * 0.1 + y as f32 * 0.2 + c as f32 * 0.3).sin();
                }
            }
        }

        // Forward DCT8
        let coeffs = compute_forward_transform_coeffs(
            DCT8_TRANSFORM_ID,
            &pixels[0],
            &pixels[1],
            &pixels[2],
            8,
            8,
            0,
            0,
            8,
            8,
        );

        // Inverse
        let recon = inverse_transform_8x8_all_channels(DCT8_TRANSFORM_ID, &coeffs);

        // Compare
        for c in 0..3 {
            let max_err = pixels[c]
                .iter()
                .zip(recon[c].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 0.01,
                "DCT8 roundtrip channel {}: max_err={}",
                c,
                max_err
            );
        }
    }
}

#[cfg(test)]
mod animation_tests {
    use super::*;

    #[test]
    fn test_encode_rgba_small() {
        // 16x16 RGBA: red with gradient alpha
        let w = 16usize;
        let h = 16usize;
        let mut rgba = vec![0u8; w * h * 4];
        for i in 0..w * h {
            rgba[i * 4] = 255; // R
            rgba[i * 4 + 1] = 0; // G
            rgba[i * 4 + 2] = 0; // B
            rgba[i * 4 + 3] = ((i * 255) / (w * h - 1)) as u8; // gradient alpha
        }
        let config = VarDctConfig {
            distance: 2.0,
            effort: 7,
            progressive: false,
        };
        let data = encode_vardct_u8_rgba(&rgba, w, h, &config).unwrap();
        assert!(data.len() > 50);
        std::fs::write("/tmp/test_rgba.jxl", &data).unwrap();
        eprintln!("RGBA: {} bytes, 16x16", data.len());
    }

    #[test]
    fn test_encode_rgba_dice() {
        // Load dice RGBA (800x600, multi-group)
        let bin = match std::fs::read("/tmp/dice_rgba.bin") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping test_encode_rgba_dice: /tmp/dice_rgba.bin not found");
                return;
            }
        };
        let w = 800usize;
        let h = 600usize;
        assert_eq!(bin.len(), w * h * 4);
        let config = VarDctConfig {
            distance: 1.0,
            effort: 7,
            progressive: false,
        };
        let data = encode_vardct_u8_rgba(&bin, w, h, &config).unwrap();
        std::fs::write("/tmp/dice_jxlrs.jxl", &data).unwrap();
        eprintln!("Dice RGBA: {} bytes, {}x{}", data.len(), w, h);
    }

    #[test]
    fn test_encode_animation_3_frames() {
        let w = 16usize;
        let h = 16usize;
        let mut frames = Vec::new();
        // 3 frames: red, green, blue
        for c in 0..3u8 {
            let mut rgb = vec![0u8; w * h * 3];
            for i in 0..w * h {
                rgb[i * 3 + c as usize] = 255;
            }
            frames.push((rgb, 100u32));
        }
        let frame_refs: Vec<(&[u8], u32)> =
            frames.iter().map(|(d, ms)| (d.as_slice(), *ms)).collect();
        let config = VarDctConfig {
            distance: 2.0,
            effort: 7,
            progressive: false,
        };
        let data = encode_vardct_animation_u8_rgb(&frame_refs, w, h, &config).unwrap();
        assert!(
            data.len() > 100,
            "animation too small: {} bytes",
            data.len()
        );
        std::fs::write("/tmp/anim_test.jxl", &data).unwrap();
        eprintln!("Animation: {} bytes, 3 frames 16x16", data.len());
    }

    #[test]
    #[ignore] // requires /tmp/anim_rgba_frames.bin from APNG extraction
    fn test_encode_icos_animation_rgba() {
        let bin = std::fs::read("/tmp/anim_rgba_frames.bin").unwrap();
        let w = u32::from_le_bytes(bin[0..4].try_into().unwrap()) as usize;
        let h = u32::from_le_bytes(bin[4..8].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(bin[8..12].try_into().unwrap()) as usize;
        let frame_size = w * h * 4; // RGBA
        let mut frames = Vec::new();
        for i in 0..n {
            let start = 12 + i * frame_size;
            let end = start + frame_size;
            frames.push((bin[start..end].to_vec(), 50u32));
        }
        let frame_refs: Vec<(&[u8], u32)> =
            frames.iter().map(|(d, ms)| (d.as_slice(), *ms)).collect();
        let config = VarDctConfig {
            distance: 1.0,
            effort: 7,
            progressive: false,
        };
        let data = encode_vardct_animation_u8_rgba(&frame_refs, w, h, &config).unwrap();
        std::fs::write("/tmp/anim_icos_rgba_jxlrs.jxl", &data).unwrap();
        eprintln!(
            "Icos RGBA animation: {} bytes, {} frames {}x{}",
            data.len(),
            n,
            w,
            h
        );
    }

    #[test]
    #[ignore] // requires /tmp/anim_rgb/frames.bin from APNG extraction
    fn test_encode_icos_animation() {
        let bin = std::fs::read("/tmp/anim_rgb/frames.bin").unwrap();
        let w = u32::from_le_bytes(bin[0..4].try_into().unwrap()) as usize;
        let h = u32::from_le_bytes(bin[4..8].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(bin[8..12].try_into().unwrap()) as usize;
        let frame_size = w * h * 3;
        let mut frames = Vec::new();
        for i in 0..n {
            let start = 12 + i * frame_size;
            let end = start + frame_size;
            frames.push((bin[start..end].to_vec(), 50u32)); // 50ms = 20fps
        }
        let frame_refs: Vec<(&[u8], u32)> =
            frames.iter().map(|(d, ms)| (d.as_slice(), *ms)).collect();
        let config = VarDctConfig {
            distance: 1.0,
            effort: 7,
            progressive: false,
        };
        let data = encode_vardct_animation_u8_rgb(&frame_refs, w, h, &config).unwrap();
        std::fs::write("/tmp/anim_icos_jxlrs.jxl", &data).unwrap();
        eprintln!(
            "Icos animation: {} bytes, {} frames {}x{} (ref libjxl: 341449)",
            data.len(),
            n,
            w,
            h
        );
    }
}
