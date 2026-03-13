#![no_main]

use jxl_simd::{ScalarDescriptor, SimdDescriptor};
use jxl_transforms::idct2d_8_8;
use libfuzzer_sys::fuzz_target;

fn fill_block(data: &[u8]) -> [f32; 64] {
    let mut block = [0f32; 64];
    if data.is_empty() {
        return block;
    }
    for i in 0..64 {
        let b0 = data[(i * 2) % data.len()];
        let b1 = data[(i * 2 + 1) % data.len()];
        let v = i16::from_le_bytes([b0, b1]);
        block[i] = (v as f32) * (1.0 / 256.0);
    }
    block
}

#[allow(dead_code)]
fn assert_close(a: &[f32; 64], b: &[f32; 64]) {
    for i in 0..64 {
        assert!((a[i] - b[i]).abs() < 1e-3);
    }
}

fuzz_target!(|data: &[u8]| {
    let input = fill_block(data);

    let mut scalar = input;
    idct2d_8_8(ScalarDescriptor, &mut scalar);

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(d) = jxl_simd::Sse42Descriptor::new() {
            let mut block = input;
            d.call(|d| idct2d_8_8(d, &mut block));
            assert_close(&scalar, &block);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(d) = jxl_simd::AvxDescriptor::new() {
            let mut block = input;
            d.call(|d| idct2d_8_8(d, &mut block));
            assert_close(&scalar, &block);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(d) = jxl_simd::Avx512Descriptor::new() {
            let mut block = input;
            d.call(|d| idct2d_8_8(d, &mut block));
            assert_close(&scalar, &block);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(d) = jxl_simd::NeonDescriptor::new() {
            let mut block = input;
            d.call(|d| idct2d_8_8(d, &mut block));
            assert_close(&scalar, &block);
        }
    }
});
