#![no_main]

use jxl::encode::{
    JxlEncoder, JxlEncoderImageData, JxlEncoderMode, JxlEncoderOptions,
    vardct::{VarDctConfig, encode_vardct_u8_rgba},
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let w = 1usize + (data[0] as usize % 32);
    let h = 1usize + (data[1] as usize % 32);
    let distance = 0.01f32 + (data[2] as f32 / 255.0) * 3.0f32;
    let effort = 1u8 + (data[3] % 9);
    let progressive = (data[4] & 1) != 0;

    let payload = if data.len() > 6 { &data[6..] } else { &data[5..] };
    if payload.is_empty() {
        return;
    }

    let npix = w * h;
    let mut rgba = vec![0u8; npix * 4];
    for (i, v) in rgba.iter_mut().enumerate() {
        *v = payload[i % payload.len()];
    }

    let cfg = VarDctConfig {
        distance,
        effort,
        progressive,
    };

    let _ = encode_vardct_u8_rgba(&rgba, w, h, &cfg);

    let mut opts = JxlEncoderOptions::default();
    opts.container = (data[5] & 1) != 0;
    opts.mode = JxlEncoderMode::VarDct;
    opts.lossless = (data[5] & 2) != 0;
    opts.distance_milli = (distance * 1000.0).round() as u16;
    opts.effort = effort;
    opts.jxlp_chunk_size = if (data[5] & 4) != 0 {
        Some(8 + (data[0] as usize % 128))
    } else {
        None
    };

    let enc = JxlEncoder::new(opts);
    let _ = enc.encode_image(
        (w as u32, h as u32),
        JxlEncoderImageData::Rgba8Interleaved(&rgba),
    );
});
