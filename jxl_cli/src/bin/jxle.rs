// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use jxl::encode::{JxlEncoder, JxlEncoderImageData};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "JPEG XL encoder")]
struct Opt {
    /// Input file (PPM/PGM P6/P5 with maxval=255)
    input: Option<PathBuf>,

    /// Output .jxl filename
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Quality distance (lower = better quality, 0.0 = lossless, 1.0 = visually lossless)
    #[arg(short, long, default_value_t = 1.0)]
    distance: f32,

    /// Use modular (lossless) encoding instead of VarDCT
    #[arg(long)]
    modular: bool,

    /// Emit bare codestream instead of container
    #[arg(long)]
    bare: bool,

    // --- Legacy/advanced options ---

    /// Image width in pixels (for raw input modes)
    #[arg(long)]
    width: Option<u32>,

    /// Image height in pixels (for raw input modes)
    #[arg(long)]
    height: Option<u32>,

    /// Include minimal frame header + TOC (still no pixel section payload)
    #[arg(long)]
    with_frame_info: bool,

    /// Emit a minimal decodable modular image stream
    #[arg(long)]
    modular_image: bool,

    /// Encode raw interleaved RGB8 bytes from this file
    /// (length must be width*height*3)
    #[arg(long)]
    raw_rgb8_input: Option<PathBuf>,

    /// Encode raw Gray8 bytes from this file (length must be width*height)
    #[arg(long)]
    raw_gray8_input: Option<PathBuf>,

    /// Encode a binary PPM (P6, maxval=255) or PGM (P5, maxval=255)
    #[arg(long)]
    ppm_input: Option<PathBuf>,

    /// Row stride in bytes for raw input modes
    #[arg(long)]
    raw_stride: Option<usize>,

    /// Constant modular leaf offset for --modular-image mode
    #[arg(long, default_value_t = 0)]
    modular_offset: i32,

    /// Modular predictor id (0=Zero, 1=West, ... 13=AverageAll)
    #[arg(long, default_value_t = 0)]
    modular_predictor: u32,
}

/// Parsed PPM/PGM result.
enum PnmImage {
    Rgb8 {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    Gray8 {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
}

/// Read a binary PPM (P6) or PGM (P5) file with maxval=255.
fn read_pnm(path: &PathBuf) -> Result<PnmImage> {
    let bytes = std::fs::read(path).wrap_err_with(|| format!("Failed to read {:?}", path))?;

    let mut idx = 0usize;

    let skip_ws_and_comments = |idx: &mut usize, bytes: &[u8]| {
        while *idx < bytes.len() {
            let b = bytes[*idx];
            if b == b'#' {
                while *idx < bytes.len() && bytes[*idx] != b'\n' {
                    *idx += 1;
                }
                continue;
            }
            if b.is_ascii_whitespace() {
                *idx += 1;
                continue;
            }
            break;
        }
    };

    let next_token = |idx: &mut usize, bytes: &[u8]| -> Result<String> {
        skip_ws_and_comments(idx, bytes);
        if *idx >= bytes.len() {
            return Err(color_eyre::eyre::eyre!("Unexpected EOF in PNM header"));
        }
        let start = *idx;
        while *idx < bytes.len() && !bytes[*idx].is_ascii_whitespace() {
            *idx += 1;
        }
        Ok(String::from_utf8_lossy(&bytes[start..*idx]).to_string())
    };

    let magic = next_token(&mut idx, &bytes)?;
    let channels: usize = match magic.as_str() {
        "P6" => 3,
        "P5" => 1,
        other => {
            return Err(color_eyre::eyre::eyre!(
                "Unsupported PNM magic {:?}, expected P5 or P6",
                other
            ));
        }
    };

    let width: u32 = next_token(&mut idx, &bytes)?
        .parse()
        .wrap_err("Invalid PNM width")?;
    let height: u32 = next_token(&mut idx, &bytes)?
        .parse()
        .wrap_err("Invalid PNM height")?;
    let maxval: u32 = next_token(&mut idx, &bytes)?
        .parse()
        .wrap_err("Invalid PNM maxval")?;
    if maxval != 255 {
        return Err(color_eyre::eyre::eyre!(
            "Only maxval=255 supported, got {}",
            maxval
        ));
    }

    // Skip exactly one whitespace byte after maxval (the spec says one byte).
    if idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }

    let expected = (width as usize) * (height as usize) * channels;
    let remaining = bytes.len() - idx;
    if remaining < expected {
        return Err(color_eyre::eyre::eyre!(
            "PNM pixel data too short: expected {} bytes, got {}",
            expected,
            remaining
        ));
    }

    let data = bytes[idx..idx + expected].to_vec();
    match channels {
        3 => Ok(PnmImage::Rgb8 {
            width,
            height,
            data,
        }),
        1 => Ok(PnmImage::Gray8 {
            width,
            height,
            data,
        }),
        _ => unreachable!(),
    }
}

fn main() -> Result<()> {
    let opt = Opt::parse();
    let enc = JxlEncoder::default();

    // --- Simple mode: jxle input.ppm [-o output.jxl] [-d distance] ---
    if let Some(input_path) = &opt.input {
        let pnm = read_pnm(input_path)?;
        let output = opt.output.clone().unwrap_or_else(|| {
            let mut p = input_path.clone();
            p.set_extension("jxl");
            p
        });

        let (width, height, rgb) = match &pnm {
            PnmImage::Rgb8 { width, height, data } => (*width, *height, data.clone()),
            PnmImage::Gray8 { width, height, data } => {
                // Convert grayscale to RGB for VarDCT
                let mut rgb = vec![0u8; ((*width) as usize) * ((*height) as usize) * 3];
                for (i, &g) in data.iter().enumerate() {
                    rgb[i * 3] = g;
                    rgb[i * 3 + 1] = g;
                    rgb[i * 3 + 2] = g;
                }
                (*width, *height, rgb)
            }
        };

        let bytes = if opt.modular || opt.distance == 0.0 {
            // Lossless modular encoding
            match &pnm {
                PnmImage::Rgb8 { data, .. } => {
                    let image = JxlEncoderImageData::Rgb8Interleaved(data);
                    if opt.bare {
                        enc.encode_image_codestream((width, height), image)?
                    } else {
                        enc.encode_image((width, height), image)?
                    }
                }
                PnmImage::Gray8 { data, .. } => {
                    let image = JxlEncoderImageData::Gray8Interleaved(data);
                    if opt.bare {
                        enc.encode_image_codestream((width, height), image)?
                    } else {
                        enc.encode_image((width, height), image)?
                    }
                }
            }
        } else {
            // VarDCT lossy encoding
            use jxl::encode::vardct::{VarDctConfig, encode_vardct_u8_rgb, encode_vardct_u8_rgb_codestream};
            let config = VarDctConfig { distance: opt.distance };
            if opt.bare {
                encode_vardct_u8_rgb_codestream(&rgb, width as usize, height as usize, &config)
                    .map_err(|e| color_eyre::eyre::eyre!("VarDCT encode failed: {e:?}"))?
            } else {
                encode_vardct_u8_rgb(&rgb, width as usize, height as usize, &config)
                    .map_err(|e| color_eyre::eyre::eyre!("VarDCT encode failed: {e:?}"))?
            }
        };

        std::fs::write(&output, &bytes)
            .wrap_err_with(|| format!("Failed to write {:?}", output))?;

        let mode = if opt.modular || opt.distance == 0.0 { "modular" } else { "VarDCT" };
        let ratio = (rgb.len() as f64) / (bytes.len() as f64);
        eprintln!(
            "{:?} -> {:?}: {}x{}, {} bytes ({mode}, d={:.2}, {:.1}:1)",
            input_path, output, width, height, bytes.len(), opt.distance, ratio
        );
        return Ok(());
    }

    // --- Legacy advanced modes ---

    let width = opt.width.unwrap_or(1);
    let height = opt.height.unwrap_or(1);
    let output = opt.output.clone().unwrap_or_else(|| PathBuf::from("output.jxl"));

    if opt.modular_image && opt.with_frame_info {
        return Err(color_eyre::eyre::eyre!(
            "--modular-image and --with-frame-info are mutually exclusive"
        ));
    }

    if opt.raw_rgb8_input.is_some() && opt.raw_gray8_input.is_some() {
        return Err(color_eyre::eyre::eyre!(
            "--raw-rgb8-input and --raw-gray8-input are mutually exclusive"
        ));
    }

    let has_raw_input = opt.raw_rgb8_input.is_some() || opt.raw_gray8_input.is_some();
    let has_ppm_input = opt.ppm_input.is_some();

    if has_ppm_input && has_raw_input {
        return Err(color_eyre::eyre::eyre!(
            "--ppm-input is mutually exclusive with --raw-rgb8-input/--raw-gray8-input"
        ));
    }

    let has_image_input = has_raw_input || has_ppm_input;

    if has_image_input && (opt.modular_image || opt.with_frame_info) {
        return Err(color_eyre::eyre::eyre!(
            "image input flags are mutually exclusive with --modular-image/--with-frame-info"
        ));
    }

    if has_image_input && (opt.modular_offset != 0 || opt.modular_predictor != 0) {
        return Err(color_eyre::eyre::eyre!(
            "image input modes do not use --modular-offset/--modular-predictor"
        ));
    }

    if opt.raw_stride.is_some() && !has_raw_input {
        return Err(color_eyre::eyre::eyre!(
            "--raw-stride requires --raw-rgb8-input or --raw-gray8-input"
        ));
    }

    if (opt.modular_offset != 0 || opt.modular_predictor != 0) && !opt.modular_image {
        return Err(color_eyre::eyre::eyre!(
            "--modular-offset/--modular-predictor require --modular-image"
        ));
    }

    if opt.modular_predictor > 13 {
        return Err(color_eyre::eyre::eyre!(
            "--modular-predictor must be in [0, 13]"
        ));
    }

    let (bytes, mode) = if let Some(ppm_path) = &opt.ppm_input {
        let pnm = read_pnm(ppm_path)?;
        match pnm {
            PnmImage::Rgb8 {
                width,
                height,
                ref data,
            } => {
                let image = JxlEncoderImageData::Rgb8Interleaved(data);
                let bytes = if opt.bare {
                    enc.encode_image_codestream((width, height), image)?
                } else {
                    enc.encode_image((width, height), image)?
                };
                (bytes, "ppm-rgb8 modular stream")
            }
            PnmImage::Gray8 {
                width,
                height,
                ref data,
            } => {
                let image = JxlEncoderImageData::Gray8Interleaved(data);
                let bytes = if opt.bare {
                    enc.encode_image_codestream((width, height), image)?
                } else {
                    enc.encode_image((width, height), image)?
                };
                (bytes, "pgm-gray8 modular stream")
            }
        }
    } else if let Some(raw_rgb_path) = &opt.raw_rgb8_input {
        let rgb = std::fs::read(raw_rgb_path)
            .wrap_err_with(|| format!("Failed to read raw RGB8 input {:?}", raw_rgb_path))?;
        let image = if let Some(stride) = opt.raw_stride {
            JxlEncoderImageData::Rgb8Strided { data: &rgb, stride }
        } else {
            JxlEncoderImageData::Rgb8Interleaved(&rgb)
        };
        let bytes = if opt.bare {
            enc.encode_image_codestream((width, height), image)?
        } else {
            enc.encode_image((width, height), image)?
        };
        (bytes, "raw-rgb8 modular stream")
    } else if let Some(raw_gray_path) = &opt.raw_gray8_input {
        let gray = std::fs::read(raw_gray_path)
            .wrap_err_with(|| format!("Failed to read raw Gray8 input {:?}", raw_gray_path))?;
        let image = if let Some(stride) = opt.raw_stride {
            JxlEncoderImageData::Gray8Strided {
                data: &gray,
                stride,
            }
        } else {
            JxlEncoderImageData::Gray8Interleaved(&gray)
        };
        let bytes = if opt.bare {
            enc.encode_image_codestream((width, height), image)?
        } else {
            enc.encode_image((width, height), image)?
        };
        (bytes, "raw-gray8 modular stream")
    } else if opt.modular_image {
        let bytes = if opt.bare {
            enc.encode_minimal_modular_image_codestream_with_params(
                (width, height),
                opt.modular_offset,
                opt.modular_predictor,
            )?
        } else {
            enc.encode_minimal_modular_image_container_with_params(
                (width, height),
                opt.modular_offset,
                opt.modular_predictor,
            )?
        };
        (bytes, "minimal modular-image stream")
    } else if opt.with_frame_info {
        let bytes = if opt.bare {
            enc.encode_minimal_single_frame_codestream((width, height))?
        } else {
            enc.encode_minimal_single_frame_container((width, height))?
        };
        (bytes, "minimal frame-info stream")
    } else {
        let bytes = if opt.bare {
            enc.encode_minimal_codestream_header((width, height))?
        } else {
            enc.encode_minimal_container_header((width, height))?
        };
        (bytes, "header-only stream")
    };

    std::fs::write(&output, &bytes)
        .wrap_err_with(|| format!("Failed to write {:?}", output))?;

    if opt.modular_image {
        eprintln!(
            "Wrote {} bytes to {:?} ({mode}, offset={}, predictor={})",
            bytes.len(),
            output,
            opt.modular_offset,
            opt.modular_predictor
        );
    } else {
        eprintln!("Wrote {} bytes to {:?} ({mode})", bytes.len(), output);
    }
    Ok(())
}
