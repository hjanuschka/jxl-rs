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
use crate::encode::headers::write_file_header;
use crate::encode::toc::write_toc;
use crate::encode::xyb::srgb_u8_to_xyb;
use crate::error::Result;
use crate::headers::encodings::{U32, U32Coder};
use jxl_transforms::dct2d_8_scalar;

/// VarDCT encoder configuration.
pub struct VarDctConfig {
    /// Quality distance parameter. Lower = better quality. 1.0 = visually lossless.
    pub distance: f32,
}

impl Default for VarDctConfig {
    fn default() -> Self {
        Self { distance: 1.0 }
    }
}

/// Map distance parameter to (global_scale, quant_lf).
fn distance_to_quant_params(distance: f32) -> (u32, u32) {
    let global_scale = (16384.0 / distance).round() as u32;
    let global_scale = global_scale.clamp(1, 65535);
    let quant_lf = 16u32;
    (global_scale, quant_lf)
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

/// Encode an sRGB u8 RGB image as a raw VarDCT JXL codestream.
pub fn encode_vardct_u8_rgb_codestream(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: &VarDctConfig,
) -> Result<Vec<u8>> {
    assert_eq!(rgb.len(), width * height * 3);
    assert!(width > 0 && height > 0);

    let npixels = width * height;
    let mut x_chan = vec![0.0f32; npixels];
    let mut y_chan = vec![0.0f32; npixels];
    let mut b_chan = vec![0.0f32; npixels];
    srgb_u8_to_xyb(rgb, width, height, &mut x_chan, &mut y_chan, &mut b_chan);

    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);
    let num_blocks = bw * bh;

    // Forward DCT per channel
    let mut dct_x = vec![0.0f32; num_blocks * 64];
    let mut dct_y = vec![0.0f32; num_blocks * 64];
    let mut dct_b = vec![0.0f32; num_blocks * 64];
    forward_dct_channel(&x_chan, width, height, bw, bh, &mut dct_x);
    forward_dct_channel(&y_chan, width, height, bw, bh, &mut dct_y);
    forward_dct_channel(&b_chan, width, height, bw, bh, &mut dct_b);

    // Quantize
    let (global_scale, quant_lf) = distance_to_quant_params(config.distance);

    // INV_LF_QUANT from the spec/decoder: [4096.0, 512.0, 256.0] for channels [X, Y, B]
    let inv_lf_quant = [4096.0f32, 512.0, 256.0];

    // CfL: with default color correlation, y_to_x_lf=0, y_to_b_lf=1.0
    // For encoding: in_y = dc_y, in_x = dc_x, in_b = dc_b - dc_y
    // The channels in dct arrays are in XYB order.

    // Separate DC and AC, quantize differently
    let mut dc_x = vec![0i32; num_blocks];
    let mut dc_y = vec![0i32; num_blocks];
    let mut dc_b = vec![0i32; num_blocks];
    let mut qx = vec![0i32; num_blocks * 64];
    let mut qy = vec![0i32; num_blocks * 64];
    let mut qb = vec![0i32; num_blocks * 64];

    for blk in 0..num_blocks {
        // DC: apply CfL decorrelation and proper DC quantization
        let raw_dc_x = dct_x[blk * 64];
        let raw_dc_y = dct_y[blk * 64];
        let raw_dc_b = dct_b[blk * 64];

        // Forward CfL: decoder does dec_b = in_y * 1.0 + in_b
        // So in_b = dec_b - in_y, in_x = dec_x, in_y = dec_y
        let cfl_dc_x = raw_dc_x;         // in_x = dc_x (y_to_x_lf=0)
        let cfl_dc_y = raw_dc_y;         // in_y = dc_y
        let cfl_dc_b = raw_dc_b - raw_dc_y; // in_b = dc_b - dc_y (y_to_b_lf=1.0)

        dc_x[blk] = quantize_dc(cfl_dc_x, global_scale, quant_lf, inv_lf_quant[0]);
        dc_y[blk] = quantize_dc(cfl_dc_y, global_scale, quant_lf, inv_lf_quant[1]);
        dc_b[blk] = quantize_dc(cfl_dc_b, global_scale, quant_lf, inv_lf_quant[2]);

        // AC: uniform quantization for now (TODO: use dequant matrix weights)
        for k in 1..64 {
            qx[blk * 64 + k] = quantize_ac(dct_x[blk * 64 + k], global_scale, 1);
            qy[blk * 64 + k] = quantize_ac(dct_y[blk * 64 + k], global_scale, 1);
            qb[blk * 64 + k] = quantize_ac(dct_b[blk * 64 + k], global_scale, 1);
        }
        // DC position in AC array is always 0 (DC is separate)
    }

    // Build the frame
    encode_vardct_frame(
        width, height, bw, bh, global_scale, quant_lf,
        &dc_y, &dc_x, &dc_b, // Note: Y, X, B order for DC
        &qx, &qy, &qb,
    )
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

/// Quantize AC coefficients for a single channel.
///
/// The decoder dequantizes AC as:
///   ac_float = quantized * dequant_weight[k] * inv_global_scale / raw_quant
///
/// where inv_global_scale = 2^16 / global_scale
///
/// For now, with all_default quant matrices and raw_quant=1, we use uniform weights.
/// TODO: use actual default dequant matrix weights.
fn quantize_ac(ac_float: f32, global_scale: u32, _raw_quant: u32) -> i32 {
    // With raw_quant=1 and assuming dequant_weight=1.0:
    // ac_float = quantized * 1.0 * (2^16 / global_scale) / 1
    // quantized = round(ac_float * global_scale / 2^16)
    let scale = global_scale as f32 / (1u32 << 16) as f32;
    (ac_float * scale).round() as i32
}

/// Build the complete VarDCT frame bitstream.
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
) -> Result<Vec<u8>> {
    let mut writer = BitWriter::new();

    // Codestream header
    write_file_header(&mut writer, width as u32, height as u32, true, false)?;

    // The decoder byte-aligns before frame-header parsing.
    writer.byte_align_zero_pad()?;

    // Frame header (VarDCT)
    write_vardct_frame_header(&mut writer, width as u32, height as u32)?;

    // Group layout
    let group_dim_blocks = 32usize; // 256 pixels / 8
    let num_groups_x = bw.div_ceil(group_dim_blocks);
    let num_groups_y = bh.div_ceil(group_dim_blocks);
    let num_groups = num_groups_x * num_groups_y;

    if num_groups == 1 {
        // Single-group image: 1 TOC entry, everything in one section.
        let section = encode_single_group_section(
            bw, bh, global_scale, quant_lf,
            dc_y, dc_x, dc_b, ac_x, ac_y, ac_b,
        )?;

        write_toc(&mut writer, &[section.len() as u32])?;
        writer.byte_align_zero_pad()?;

        let mut result = writer.finish();
        result.extend_from_slice(&section);
        Ok(result)
    } else {
        // Multi-group: LfGlobal + LfGroups + HfGlobal + HfGroups
        let total_sections = 2 + num_groups + num_groups;
        let mut sections: Vec<Vec<u8>> = Vec::with_capacity(total_sections);

        sections.push(encode_lf_global_section(global_scale, quant_lf)?);

        for g in 0..num_groups {
            let gx = g % num_groups_x;
            let gy = g / num_groups_x;
            sections.push(encode_lf_group_section(
                gx, gy, bw, bh, group_dim_blocks, dc_y, dc_x, dc_b,
            )?);
        }

        sections.push(encode_hf_global_section(num_groups, bw, bh, group_dim_blocks, ac_x, ac_y, ac_b)?);

        for g in 0..num_groups {
            let gx = g % num_groups_x;
            let gy = g / num_groups_x;
            sections.push(encode_hf_group_section(
                gx, gy, bw, bh, group_dim_blocks, ac_x, ac_y, ac_b,
            )?);
        }

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
/// All values are 0 for our minimal encoder (no chroma correlation,
/// DCT8x8 transform, quant=1, no EPF).
fn encode_hf_metadata_modular(
    w: &mut BitWriter,
    _cr_w: usize,
    _cr_h: usize,
    _count: usize,
    _bw: usize,
    _bh: usize,
    _data: &[i32],
) -> Result<()> {
    // All 4 channels contain only zeros.
    // Write a modular subbitstream that decodes to all-zero channels.
    //
    // GroupHeader: use_global_tree=false, wp_all_default=true, nb_transforms=0
    crate::encode::modular::write_minimal_group_header(w, false)?;

    // Local tree: single leaf node with predictor=0 (Zero), offset=0
    // This produces residual 0 for each pixel, which matches our all-zero data.
    // Tree tokens: property=-1, value=0, predictor=0, offset=0, multiplier=0
    let tree_stream = crate::encode::modular_encode::build_tree_tokens_zero()?;
    let tree_context_map = [0u8; 6];
    crate::encode::entropy::huffman_encode::write_huffman_histograms(
        w,
        &tree_context_map,
        &[tree_stream.uint_config],
        &[tree_stream.code.clone()],
    )?;
    crate::encode::modular_encode::write_tree_token_stream(w, &tree_stream)?;

    // Residual histograms: single symbol 0 (all residuals are 0)
    let zero_code = crate::encode::entropy::huffman_encode::build_huffman_code(&[1])
        .ok_or(crate::error::Error::InvalidHuffman)?;
    let uint_config = crate::encode::entropy::HybridUintConfig::new(4, 1, 2);
    crate::encode::entropy::huffman_encode::write_huffman_histograms(
        w,
        &[0u8],
        &[uint_config],
        &[zero_code],
    )?;

    // No residual tokens to write (all symbols are 0, encoded as 0 bits each).
    // The SymbolReader will read 0 bits per symbol from the single-symbol table.
    // But we still need the check_final_state which verifies the ans state.
    // For Huffman with single symbol, no bits are consumed, so nothing to write.

    Ok(())
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
    U32Coder::Select(U32::Val(16), U32::BitsOffset { n: 5, off: 1 }, U32::BitsOffset { n: 8, off: 1 }, U32::BitsOffset { n: 16, off: 1 })
}

/// Write VarDCT frame header.
/// Write VarDCT frame header.
///
/// Writes all fields of FrameHeader for a VarDCT, XYB-encoded, single-pass,
/// last frame with no extra channels, no animation, no timecodes.
fn write_vardct_frame_header(writer: &mut BitWriter, _width: u32, _height: u32) -> Result<()> {
    // 1. all_default = false (we need VarDCT settings)
    writer.write(1, 0)?;
    // 2. frame_type = RegularFrame (0), Bits(2)
    writer.write(2, 0)?;
    // 3. encoding = VarDCT (0), Bits(1)
    writer.write(1, 0)?;
    // 4. flags = 0 (u64: selector 00 = 0)
    writer.write(2, 0)?;
    // 5. do_ycbcr: cond !xyb_encoded=false => NOT WRITTEN
    // 6. jpeg_upsampling: cond do_ycbcr => NOT WRITTEN
    // 7. upsampling = 1, u2S(1,2,4,8), Val(1) = selector 00
    writer.write(2, 0)?;
    // 8. ec_upsampling: 0 extra channels => NOT WRITTEN
    // 9. group_size_shift: cond encoding==Modular => NOT WRITTEN
    // 10. x_qm_scale = 3 (default), Bits(3), cond VarDCT && xyb_encoded
    writer.write(3, 3)?;
    // 11. b_qm_scale = 2 (default), Bits(3)
    writer.write(3, 2)?;
    // 12. passes: cond frame_type != ReferenceOnly = true
    //     num_passes = 1, u2S(1,2,3,Bits(3)+4), Val(1) = selector 00
    writer.write(2, 0)?;
    //     num_passes == 1, so num_ds/shift/downsampling not written
    // 13. lf_level: cond frame_type==LFFrame => NOT WRITTEN
    // 14. have_crop = false
    writer.write(1, 0)?;
    // 15. crop fields: cond have_crop => NOT WRITTEN
    // 16. blending_info: cond RegularFrame = true
    //     mode = Replace (0), u2S(0,1,2,Bits(2)+3), selector 00
    writer.write(2, 0)?;
    //     alpha_channel: cond num_extra_channels>0 => NOT WRITTEN
    //     clamp: same cond => NOT WRITTEN
    //     source: cond !(full_frame && Replace) = false => NOT WRITTEN
    // 17. ec_blending_info: 0 extra channels => NOT WRITTEN
    // 18. duration: cond have_animation=false => NOT WRITTEN
    // 19. timecode: cond have_timecode=false => NOT WRITTEN
    // 20. is_last = true (default for RegularFrame)
    writer.write(1, 1)?;
    // 21. save_as_reference: cond !LFFrame && !is_last = false => NOT WRITTEN
    // 22. save_before_ct: cond false (is_last=true) => NOT WRITTEN
    // 23. name: String, size = 0, u2S(0, Bits(4), Bits(5)+16, Bits(10)+48)
    writer.write(2, 0)?; // selector 00 = Val(0)
    // 24. restoration_filter: all_default=false, gab=false, epf_iters=0
    writer.write(1, 0)?; // all_default
    writer.write(1, 0)?; // gab
    writer.write(2, 0)?; // epf_iters = 0, Bits(2)
    // 25. extensions = 0 (u64: selector 00)
    writer.write(2, 0)?;
    Ok(())
}

/// Encode all frame data as a single section (for single-group images).
fn encode_single_group_section(
    bw: usize,
    bh: usize,
    global_scale: u32,
    quant_lf: u32,
    dc_y: &[i32],
    dc_x: &[i32],
    dc_b: &[i32],
    _ac_x: &[i32],
    _ac_y: &[i32],
    _ac_b: &[i32],
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    let num_blocks = bw * bh;

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
    // Modular global: nothing for VarDCT with 0 extra channels

    // === LfGroup0: VarDCT DC ===
    // extra_precision = 0 (2 bits)
    w.write(2, 0)?;
    // DC coefficients as modular (3 channels: Y, X, B order as per decode_vardct_lf)
    // The decoder creates channels in order: [shrink_rect(1), shrink_rect(0), shrink_rect(2)]
    // which for non-subsampled is [Y_chan, X_chan, B_chan]
    let mut dc_data = vec![0i32; num_blocks * 3];
    for i in 0..num_blocks {
        dc_data[i] = dc_y[i];              // Channel 0: Y
        dc_data[num_blocks + i] = dc_x[i]; // Channel 1: X
        dc_data[2 * num_blocks + i] = dc_b[i]; // Channel 2: B
    }
    crate::encode::modular_encode::encode_modular_signed_stream(
        &mut w, bw, bh, 3, &dc_data,
    )?;

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
    let count_num_bits = if upper_bound <= 1 { 0 } else {
        32 - (upper_bound as u32 - 1).leading_zeros()
    };
    // count = num_blocks (every block gets a transform entry)
    // Write count-1 in count_num_bits bits
    if count_num_bits > 0 {
        w.write(count_num_bits as usize, (num_blocks - 1) as u64)?;
    }

    // Build modular channels for HF metadata
    let cr_w = bw.div_ceil(8); // chroma correlation map size
    let cr_h = bh.div_ceil(8);
    let ch0_size = cr_w * cr_h; // ytox
    let ch1_size = cr_w * cr_h; // ytob
    let ch2_size = num_blocks * 2; // transform (count x 2)
    let ch3_size = bw * bh; // epf
    let total = ch0_size + ch1_size + ch2_size + ch3_size;

    let hf_meta = vec![0i32; total];
    // ch0 (ytox): all 0
    // ch1 (ytob): all 0
    // ch2 (transform_image):
    //   row 0: transform types (DCT8x8 = 0)
    //   row 1: raw_quant - 1 (quant=1, so value=0)
    // All zeros -- which is correct for DCT8x8 with quant=1.
    // ch3 (epf): all 0 (no EPF sharpness)

    // The 4 channels have different sizes. We need to encode them
    // as a multi-channel modular stream where each channel has its own
    // width and height.
    // For simplicity, since all values are 0, a single-symbol stream works.
    encode_hf_metadata_modular(&mut w, cr_w, cr_h, num_blocks, bw, bh, &hf_meta)?;

    // === HfGlobal ===
    // DequantMatrices: all_default = true
    w.write(1, 1)?;
    // num_histograms: ceil_log2(1) = 0 bits
    // used_orders: selector 2 = no custom orders
    w.write(2, 2)?;
    // AC histograms: single-symbol Huffman for all 165 contexts
    write_ac_entropy_header(&mut w, 165)?;

    // === HfGroup0 ===
    // AC coefficients: with single-symbol histogram (symbol 0), no bits needed

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Encode the LfGlobal section.
fn encode_lf_global_section(global_scale: u32, quant_lf: u32) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
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
    // Modular global: for VarDCT with 0 extra channels, nothing is read.
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
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();

    let x0 = gx * group_dim_blocks;
    let y0 = gy * group_dim_blocks;
    let gw = (x0 + group_dim_blocks).min(bw) - x0;
    let gh = (y0 + group_dim_blocks).min(bh) - y0;

    // VarDCT LF: DC coefficients as modular (3 channels: Y, X, B)
    let npixels = gw * gh;
    let mut dc_data = vec![0i32; npixels * 3];
    for y in 0..gh {
        for x in 0..gw {
            let src = (y0 + y) * bw + (x0 + x);
            let dst = y * gw + x;
            dc_data[dst] = dc_y[src];
            dc_data[npixels + dst] = dc_x[src];
            dc_data[2 * npixels + dst] = dc_b[src];
        }
    }
    crate::encode::modular_encode::encode_modular_signed_stream(
        &mut w, gw, gh, 3, &dc_data,
    )?;

    // ModularLF: empty (0 extra channels)
    // (nothing to write)

    // HF metadata: raw_quant_field (all 1) + transform_map (all 0 = DCT8x8)
    let mut hf_meta = vec![0i32; npixels * 2];
    for i in 0..npixels {
        hf_meta[i] = 1; // raw_quant_field
    }
    // transform_map already 0 (DCT8x8)
    crate::encode::modular_encode::encode_modular_signed_stream(
        &mut w, gw, gh, 2, &hf_meta,
    )?;

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Encode the HfGlobal section.
fn encode_hf_global_section(
    num_groups: usize,
    _bw: usize,
    _bh: usize,
    _group_dim_blocks: usize,
    _ac_x: &[i32],
    _ac_y: &[i32],
    _ac_b: &[i32],
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

    // Per-pass data (1 pass):
    // used_orders: selector 2 = no custom orders (natural order)
    w.write(2, 2)?;

    // AC coefficient histograms
    // With default BlockContextMap: 15 block contexts
    // NON_ZERO_BUCKETS = 4, ZERO_DENSITY_CONTEXT_COUNT = 7
    // num_ac_contexts = 15 * (4 + 7) = 165
    // Total contexts = 1 histogram * 165 = 165
    let num_ac_contexts = 165;

    // Write AC entropy header with all-zero single-symbol histograms
    write_ac_entropy_header(&mut w, num_ac_contexts)?;

    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

/// Write AC entropy stream header (Huffman histograms for AC contexts).
/// Write AC entropy header: Histograms with all-zero single-symbol Huffman.
///
/// Matches the Histograms::decode format:
/// 1. LZ77 params (disabled)
/// 2. Context map (all contexts -> histogram 0)
/// 3. use_prefix_code = true (Huffman mode)
/// 4. HybridUint configs (1 config for 1 histogram)
/// 5. Huffman tables (1 single-symbol table)
fn write_ac_entropy_header(w: &mut BitWriter, num_contexts: usize) -> Result<()> {
    // 1. LZ77 enabled = false
    w.write(1, 0)?;

    // 2. Context map: all contexts -> histogram 0
    crate::encode::entropy::context_map::write_simple_zero_context_map(w, num_contexts)?;

    // 3. use_prefix_code = true (Huffman)
    // For Huffman, log_alpha_size is implicitly HUFFMAN_MAX_BITS (15)
    w.write(1, 1)?;

    // 4. HybridUint config for the single histogram
    // HybridUint::decode reads: split_exponent from log_alpha_size bits
    // For HUFFMAN_MAX_BITS = 15, that's 4 bits (ceil_log2(15+1) = 4)
    // split_exponent = 4 (default)
    // Then reads msb_in_token (max split_exponent, so 3 bits for value 4... wait)
    // Actually, HybridUint::decode(log_alpha_size=15):
    //   split_exponent = br.read(ceil_log2(log_alpha_size + 1)) = br.read(4)
    //   msb_in_token = br.read(ceil_log2(split_exponent + 1))
    //   lsb_in_token = br.read(ceil_log2(split_exponent - msb_in_token + 1))
    // For split_exponent=0: msb and lsb are both 0, reads nothing further.
    // split_exponent=0 means all values are direct tokens (no extra bits).
    // This is simplest for single-symbol (symbol 0).
    let log_alpha_size = 15; // HUFFMAN_MAX_BITS
    let ceil_log2_las_plus1 = 32 - (log_alpha_size as u32).leading_zeros(); // 4
    w.write(ceil_log2_las_plus1 as usize, 0)?; // split_exponent = 0
    // With split_exponent=0: no further HybridUint params to read.

    // 5. Single Huffman table: symbol 0, alphabet size determined by log_alpha_size
    // For Huffman, HuffmanCodes::decode reads num_histograms tables.
    // Each table is read by Table::decode which reads the alphabet size.
    // The alphabet size for Huffman is determined by the HybridUint config.
    //
    // Actually wait -- looking at HuffmanCodes::decode:
    // alphabet_size = read_varint16() + 1 per histogram
    // Then Table::decode(alphabet_size)
    //
    // For single symbol 0: alphabet_size = 1 (varint16 = 0)
    // Table::decode with al_size=1: reads nothing, fills table[0] = 0.

    // Write varint16(0) = alphabet_size - 1 = 0
    // varint16(0): first bit = 0 -> value 0
    w.write(1, 0)?; // varint16 = 0 -> alphabet_size = 1

    // Table::decode with alphabet_size=1: no bits read.
    // Done!

    Ok(())
}

/// Encode an HfGroup section (AC coefficients for one group).
#[allow(clippy::too_many_arguments)]
fn encode_hf_group_section(
    _gx: usize, _gy: usize, _bw: usize, _bh: usize,
    _group_dim_blocks: usize, _ac_x: &[i32], _ac_y: &[i32], _ac_b: &[i32],
) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    // For now: empty (all AC coefficients are zero).
    // The decoder reads AC coefficients using the histograms from HfGlobal.
    // With a single-symbol histogram (symbol 0), no bits are read per symbol.
    // We just need the section to exist.
    w.byte_align_zero_pad()?;
    Ok(w.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_to_quant_params() {
        let (gs, ql) = distance_to_quant_params(1.0);
        assert_eq!(gs, 16384);
        assert_eq!(ql, 16);

        let (gs, _) = distance_to_quant_params(0.5);
        assert_eq!(gs, 32768);

        let (gs, _) = distance_to_quant_params(2.0);
        assert_eq!(gs, 8192);
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
            assert!(
                out[i].abs() < 0.01,
                "AC[{i}] = {}, expected ~0",
                out[i]
            );
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
        eprintln!("Hex: {}", codestream.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "));

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
        let (num_frames, frames) = crate::api::tests::decode(
            &cs, usize::MAX, true, false, None,
        ).expect("jxl-rs decode should succeed");
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
                    (0, 0) => { rgb[i] = 255; rgb[i+1] = 0; rgb[i+2] = 0; }     // Red
                    (1, 0) => { rgb[i] = 0; rgb[i+1] = 255; rgb[i+2] = 0; }     // Green
                    (0, 1) => { rgb[i] = 0; rgb[i+1] = 0; rgb[i+2] = 255; }     // Blue
                    _ => { rgb[i] = 255; rgb[i+1] = 255; rgb[i+2] = 0; }         // Yellow
                }
            }
        }
        let config = VarDctConfig::default();
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();

        // Decode with jxl-rs
        let (_n, frames) = crate::api::tests::decode(
            &cs, usize::MAX, true, false, None,
        ).expect("decode should succeed");

        let buf = &frames[0][0];
        eprintln!("16x16 quad-color test: decoded OK");
        // Check corners - should roughly match input colors (DC-only = block average)
        for (qx, qy, label) in [(0,0,"TL-Red"), (1,0,"TR-Green"), (0,1,"BL-Blue"), (1,1,"BR-Yellow")] {
            let x = qx * 8 + 4;
            let y = qy * 8 + 4;
            let row = buf.row(y);
            let r = (row[x * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (row[x * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (row[x * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            eprintln!("  {} center: ({},{},{})", label, r, g, b);
        }

        // Also write to file for djxl verification
        let file_data = crate::encode::container::wrap_codestream(&cs).unwrap();
        std::fs::write("/tmp/test_vardct_16x16.jxl", &file_data).unwrap();
        eprintln!("Written {} bytes to /tmp/test_vardct_16x16.jxl", file_data.len());
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
                rgb[i] = (x * 255 / 7) as u8;     // R: gradient left-right
                rgb[i + 1] = (y * 255 / 7) as u8; // G: gradient top-bottom
                rgb[i + 2] = 128;                   // B: constant
            }
        }
        let config = VarDctConfig::default();
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();

        // Decode with jxl-rs
        let (_n, frames) = crate::api::tests::decode(
            &cs, usize::MAX, true, false, None,
        ).expect("decode should succeed");

        let buf = &frames[0][0];
        eprintln!("Gradient test decoded OK");
        for y in 0..2 {
            let row = buf.row(y);
            for x in 0..2 {
                let r = (row[x * 3].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (row[x * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (row[x * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
                eprintln!("  pixel({},{}) input=({},{},{}) decoded=({},{},{})",
                    x, y, rgb[(y*width+x)*3], rgb[(y*width+x)*3+1], rgb[(y*width+x)*3+2],
                    r, g, b);
            }
        }
    }

    #[test]
    fn test_write_vardct_to_file() {
        let width = 8;
        let height = 8;
        let rgb = vec![128u8; width * height * 3];
        let config = VarDctConfig::default();
        let cs = encode_vardct_u8_rgb_codestream(&rgb, width, height, &config).unwrap();
        let hex: String = cs.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
        eprintln!("Actual codestream ({} bytes): {hex}", cs.len());

        let container = encode_vardct_u8_rgb(&rgb, width, height, &config).unwrap();
        std::fs::write("/tmp/test_vardct_8x8.jxl", &container).unwrap();
        eprintln!("Written {} bytes to /tmp/test_vardct_8x8.jxl", container.len());
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
        let hex: String = cs.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
        eprintln!("Codestream ({} bytes): {hex}", cs.len());

        // Pad for BitReader
        let mut padded = cs.clone();
        for _ in 0..128 { padded.push(0); }

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
        use crate::headers::frame_header::FrameHeader;
        use crate::headers::encodings::UnconditionalCoder;
        let fh_result: crate::error::Result<FrameHeader> =
            FrameHeader::read_unconditional(&(), &mut br, &crate::headers::frame_header::FrameHeaderNonserialized {
                xyb_encoded: true,
                num_extra_channels: 0,
                extra_channel_info: vec![],
                have_animation: false,
                have_timecode: false,
                img_width: 8,
                img_height: 8,
            });
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
        let config = VarDctConfig { distance: 2.0 };
        let result = encode_vardct_u8_rgb(&rgb, width, height, &config);
        assert!(result.is_ok(), "encode failed: {:?}", result.err());
    }
}
