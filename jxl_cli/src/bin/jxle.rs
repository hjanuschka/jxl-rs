// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use jxl::encode::JxlEncoder;
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
}

fn main() -> Result<()> {
    let opt = Opt::parse();
    let enc = JxlEncoder::default();

    if opt.modular_image && opt.with_frame_info {
        return Err(color_eyre::eyre::eyre!(
            "--modular-image and --with-frame-info are mutually exclusive"
        ));
    }

    let bytes = if opt.modular_image {
        if opt.bare {
            enc.encode_minimal_modular_image_codestream((opt.width, opt.height))?
        } else {
            enc.encode_minimal_modular_image_container((opt.width, opt.height))?
        }
    } else if opt.with_frame_info {
        if opt.bare {
            enc.encode_minimal_single_frame_codestream((opt.width, opt.height))?
        } else {
            enc.encode_minimal_single_frame_container((opt.width, opt.height))?
        }
    } else if opt.bare {
        enc.encode_minimal_codestream_header((opt.width, opt.height))?
    } else {
        enc.encode_minimal_container_header((opt.width, opt.height))?
    };

    std::fs::write(&opt.output, &bytes)
        .wrap_err_with(|| format!("Failed to write {:?}", opt.output))?;

    let mode = if opt.modular_image {
        "minimal modular-image stream"
    } else if opt.with_frame_info {
        "minimal frame-info stream"
    } else {
        "header-only stream"
    };
    eprintln!("Wrote {} bytes to {:?} ({mode})", bytes.len(), opt.output);
    Ok(())
}
