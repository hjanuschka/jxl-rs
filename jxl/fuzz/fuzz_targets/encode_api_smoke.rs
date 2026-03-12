#![no_main]

use jxl::encode::{JxlEncoder, JxlEncoderImageData, JxlEncoderMode, JxlEncoderOptions};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let w = 1u32 + (data[0] as u32 % 32);
    let h = 1u32 + (data[1] as u32 % 32);
    let pixels = (w as usize) * (h as usize);

    let mut opts = JxlEncoderOptions::default();
    opts.container = (data[2] & 1) != 0;
    opts.lossless = (data[2] & 2) == 0;
    opts.mode = if (data[2] & 4) != 0 {
        JxlEncoderMode::VarDct
    } else {
        JxlEncoderMode::Modular
    };
    opts.effort = 1 + (data[3] % 9);
    opts.distance_milli = 1 + (data[4] as u16 * 16);
    opts.near_lossless = data[5] % 101;
    opts.fast_lossless = (data[6] & 1) != 0;
    opts.jxlp_chunk_size = if (data[6] & 2) != 0 {
        Some(8 + (data[7] as usize % 128))
    } else {
        None
    };

    let mut payload = if data.len() > 8 { &data[8..] } else { data };
    if payload.is_empty() {
        payload = &[0];
    }

    let mut rgb = vec![0u8; pixels * 3];
    for (i, b) in rgb.iter_mut().enumerate() {
        *b = payload[i % payload.len()];
    }

    let enc = JxlEncoder::new(opts);
    let _ = enc.encode_image((w, h), JxlEncoderImageData::Rgb8Interleaved(&rgb));
});
