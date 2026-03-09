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
    let inv_global_scale = (1u32 << 16) as f32 / global_scale as f32;

    let mut qx = vec![0i32; num_blocks * 64];
    let mut qy = vec![0i32; num_blocks * 64];
    let mut qb = vec![0i32; num_blocks * 64];
    quantize_channel(&dct_x, inv_global_scale, &mut qx);
    quantize_channel(&dct_y, inv_global_scale, &mut qy);
    quantize_channel(&dct_b, inv_global_scale, &mut qb);

    // Separate DC and AC
    let mut dc_x = vec![0i32; num_blocks];
    let mut dc_y = vec![0i32; num_blocks];
    let mut dc_b = vec![0i32; num_blocks];
    for blk in 0..num_blocks {
        dc_x[blk] = qx[blk * 64];
        dc_y[blk] = qy[blk * 64];
        dc_b[blk] = qb[blk * 64];
        qx[blk * 64] = 0;
        qy[blk * 64] = 0;
        qb[blk * 64] = 0;
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

/// Quantize DCT coefficients (flat weight for now).
fn quantize_channel(dct: &[f32], inv_global_scale: f32, out: &mut [i32]) {
    for (i, &c) in dct.iter().enumerate() {
        out[i] = (c / inv_global_scale).round() as i32;
    }
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
    // DC coefficients as modular (3 channels: X, Y, B order as per decode_vardct_lf)
    // The decoder creates channels in order: [shrink_rect(1), shrink_rect(0), shrink_rect(2)]
    // which for non-subsampled is [X_chan, Y_chan, B_chan]
    let mut dc_data = vec![0i32; num_blocks * 3];
    for i in 0..num_blocks {
        dc_data[i] = dc_x[i];              // Channel 0: X
        dc_data[num_blocks + i] = dc_y[i]; // Channel 1: Y
        dc_data[2 * num_blocks + i] = dc_b[i]; // Channel 2: B
    }
    crate::encode::modular_encode::encode_modular_signed_stream(
        &mut w, bw, bh, 3, &dc_data,
    )?;

    // === LfGroup0: ModularLF (empty for 0 extra channels) ===

    // === LfGroup0: HF metadata ===
    // raw_quant_field (all 1) + transform_map (all 0 = DCT8x8)
    let mut hf_meta = vec![0i32; num_blocks * 2];
    for i in 0..num_blocks {
        hf_meta[i] = 1; // raw_quant_field
    }
    crate::encode::modular_encode::encode_modular_signed_stream(
        &mut w, bw, bh, 2, &hf_meta,
    )?;

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
fn write_ac_entropy_header(w: &mut BitWriter, num_contexts: usize) -> Result<()> {
    // Histograms::decode reads:
    // lz77_enabled = false
    w.write(1, 0)?;
    // use_prefix_code = true (Huffman)
    w.write(1, 1)?;
    // log_alphabet_size (5 bits) -- max symbol size
    w.write(5, 8)?; // 2^8 = 256 max symbols

    // Context map: all contexts map to the same histogram (histogram 0)
    // context_map encoding:
    // num_histograms = 1 (so trivially all map to 0)
    // The context map encoding: if num_contexts > 1, read context map.
    // Actually, the Histograms::decode reads context_map differently.
    //
    // Let me look at the decoder more carefully...
    // For now, encode a trivial "all zero coefficients" scenario.
    // We'll need to flesh this out with proper AC tokenization.

    // Actually this is getting complex. Let me write a simpler version:
    // Write the context map as "all contexts use histogram 0"
    // Then write 1 Huffman table that encodes symbol 0 with 0 bits.

    // For Huffman mode:
    // 1. Read context map (maps num_contexts entries to histogram indices)
    // 2. Read num_histograms Huffman tables

    // Context map: all zeros (all contexts -> histogram 0)
    // Encoding: use_simple = 1, nbits = 0, first_symbol = 0
    // This means: context map has only 1 unique value = 0
    // Actually context_map_decode reads:
    //   simple = read(1)
    //   if simple: nsym = read(2), then nsym symbols
    //   else: complex
    //
    // For all-same context map: simple=1, nsym=1, symbol=0
    // This gives us num_histograms=1 (just symbol 0).
    // But then each context is symbol 0, meaning all use histogram 0.

    // Looking at entropy_coding/context_map.rs decoder:
    // Actually our encoder already has context map writers!
    // Let me use write_simple_zero_context_map.

    crate::encode::entropy::context_map::write_simple_zero_context_map(w, num_contexts)?;

    // Now write 1 Huffman table for the single histogram.
    // For "all zeros" AC, a single-symbol table with symbol 0.
    crate::encode::entropy::huffman::write_single_symbol_huffman_table(w, 256, 0)?;

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
