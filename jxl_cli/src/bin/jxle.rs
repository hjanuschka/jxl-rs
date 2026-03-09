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
}

fn main() -> Result<()> {
    let opt = Opt::parse();
    let enc = JxlEncoder::default();

    let bytes = if opt.bare {
        enc.encode_minimal_codestream_header((opt.width, opt.height))?
    } else {
        enc.encode_minimal_container_header((opt.width, opt.height))?
    };

    std::fs::write(&opt.output, &bytes)
        .wrap_err_with(|| format!("Failed to write {:?}", opt.output))?;

    eprintln!(
        "Wrote {} bytes to {:?} (header-only stream)",
        bytes.len(),
        opt.output
    );
    Ok(())
}
