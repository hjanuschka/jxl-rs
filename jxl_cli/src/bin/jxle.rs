// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use jxl::encode::{JxlEncoder, JxlEncoderImageData};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Bootstrap JPEG XL encoder helper (header-only stream)")]
struct Opt {
    /// Output .jxl filename
    output: PathBuf,

    /// Image width in pixels
    #[arg(long)]
    width: u32,

    /// Image height in pixels
    #[arg(long)]
    height: u32,

    /// Emit bare codestream instead of container
    #[arg(long)]
    bare: bool,

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

    /// Constant modular leaf offset for --modular-image mode
    #[arg(long, default_value_t = 0)]
    modular_offset: i32,

    /// Modular predictor id (0=Zero, 1=West, ... 13=AverageAll)
    #[arg(long, default_value_t = 0)]
    modular_predictor: u32,
}

fn main() -> Result<()> {
    let opt = Opt::parse();
    let enc = JxlEncoder::default();

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

    if has_raw_input && (opt.modular_image || opt.with_frame_info) {
        return Err(color_eyre::eyre::eyre!(
            "--raw-rgb8-input/--raw-gray8-input are mutually exclusive with --modular-image/--with-frame-info"
        ));
    }

    if has_raw_input && (opt.modular_offset != 0 || opt.modular_predictor != 0) {
        return Err(color_eyre::eyre::eyre!(
            "raw input modes do not use --modular-offset/--modular-predictor"
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

    let (bytes, mode) = if let Some(raw_rgb_path) = &opt.raw_rgb8_input {
        let rgb = std::fs::read(raw_rgb_path)
            .wrap_err_with(|| format!("Failed to read raw RGB8 input {:?}", raw_rgb_path))?;
        let image = JxlEncoderImageData::Rgb8Interleaved(&rgb);
        let bytes = if opt.bare {
            enc.encode_image_codestream((opt.width, opt.height), image)?
        } else {
            enc.encode_image((opt.width, opt.height), image)?
        };
        (bytes, "raw-rgb8 modular stream")
    } else if let Some(raw_gray_path) = &opt.raw_gray8_input {
        let gray = std::fs::read(raw_gray_path)
            .wrap_err_with(|| format!("Failed to read raw Gray8 input {:?}", raw_gray_path))?;
        let image = JxlEncoderImageData::Gray8Interleaved(&gray);
        let bytes = if opt.bare {
            enc.encode_image_codestream((opt.width, opt.height), image)?
        } else {
            enc.encode_image((opt.width, opt.height), image)?
        };
        (bytes, "raw-gray8 modular stream")
    } else if opt.modular_image {
        let bytes = if opt.bare {
            enc.encode_minimal_modular_image_codestream_with_params(
                (opt.width, opt.height),
                opt.modular_offset,
                opt.modular_predictor,
            )?
        } else {
            enc.encode_minimal_modular_image_container_with_params(
                (opt.width, opt.height),
                opt.modular_offset,
                opt.modular_predictor,
            )?
        };
        (bytes, "minimal modular-image stream")
    } else if opt.with_frame_info {
        let bytes = if opt.bare {
            enc.encode_minimal_single_frame_codestream((opt.width, opt.height))?
        } else {
            enc.encode_minimal_single_frame_container((opt.width, opt.height))?
        };
        (bytes, "minimal frame-info stream")
    } else {
        let bytes = if opt.bare {
            enc.encode_minimal_codestream_header((opt.width, opt.height))?
        } else {
            enc.encode_minimal_container_header((opt.width, opt.height))?
        };
        (bytes, "header-only stream")
    };

    std::fs::write(&opt.output, &bytes)
        .wrap_err_with(|| format!("Failed to write {:?}", opt.output))?;

    if opt.modular_image {
        eprintln!(
            "Wrote {} bytes to {:?} ({mode}, offset={}, predictor={})",
            bytes.len(),
            opt.output,
            opt.modular_offset,
            opt.modular_predictor
        );
    } else {
        eprintln!("Wrote {} bytes to {:?} ({mode})", bytes.len(), opt.output);
    }
    Ok(())
}
