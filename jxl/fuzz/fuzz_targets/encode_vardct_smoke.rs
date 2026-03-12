#![no_main]

use jxl::encode::vardct::{VarDctConfig, encode_vardct_u8_rgb, encode_vardct_u8_rgb_codestream};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let w = 1usize + (data[0] as usize % 32);
    let h = 1usize + (data[1] as usize % 32);
    let distance = 0.1f32 + (data[2] as f32 / 255.0) * 2.9f32;
    let effort = 1u8 + (data[3] % 9);

    let npix = w * h;
    let mut rgb = vec![0u8; npix * 3];
    if data.len() > 4 {
        let payload = &data[4..];
        for (i, v) in rgb.iter_mut().enumerate() {
            *v = payload[i % payload.len()];
        }
    }

    let cfg = VarDctConfig {
        distance,
        effort,
        progressive: (effort % 2) == 0,
    };

    let _ = encode_vardct_u8_rgb_codestream(&rgb, w, h, &cfg);
    let _ = encode_vardct_u8_rgb(&rgb, w, h, &cfg);
});
