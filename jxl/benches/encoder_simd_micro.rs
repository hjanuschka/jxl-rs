// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use jxl::encode::encodings::pack_signed;
use jxl::encode::entropy::huffman_encode::build_huffman_code;
use jxl::encode::simd::{EncoderSimdBackend, benchmark_force_scalar, detect_encoder_simd_backend};
use jxl::encode::xyb::{srgb_u8_to_xyb, srgb_u8_to_xyb_simd_assisted};
use jxl::encode::{JxlEncoder, JxlEncoderImageData, JxlEncoderMode, JxlEncoderOptions};
use jxl_simd::{ScalarDescriptor, SimdDescriptor};
use jxl_transforms::{dct2d_32_scalar, dct2d_32_simd, idct2d_8_8};

fn selected_backend() -> EncoderSimdBackend {
    if benchmark_force_scalar() {
        EncoderSimdBackend::Scalar
    } else {
        detect_encoder_simd_backend()
    }
}

fn bench_xyb(c: &mut Criterion) {
    let mut g = c.benchmark_group("encoder_xyb");

    let (w, h) = (1024usize, 768usize);
    let npixels = w * h;
    let mut rgb = vec![0u8; npixels * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb[i] = ((x * 13 + y * 7) & 255) as u8;
            rgb[i + 1] = ((x * 5 + y * 11) & 255) as u8;
            rgb[i + 2] = ((x * 3 + y * 17) & 255) as u8;
        }
    }

    let mut x = vec![0f32; npixels];
    let mut y = vec![0f32; npixels];
    let mut b = vec![0f32; npixels];

    g.bench_function(
        BenchmarkId::new("srgb_u8_to_xyb_scalar", format!("{}x{}", w, h)),
        |bencher| {
            bencher.iter(|| {
                srgb_u8_to_xyb(&rgb, w, h, &mut x, &mut y, &mut b).unwrap();
            })
        },
    );

    let backend = selected_backend();
    g.bench_function(
        BenchmarkId::new("srgb_u8_to_xyb_assisted", format!("{:?}", backend)),
        |bencher| {
            bencher.iter(|| match backend {
                EncoderSimdBackend::Scalar => {
                    srgb_u8_to_xyb_simd_assisted(
                        ScalarDescriptor,
                        &rgb,
                        w,
                        h,
                        &mut x,
                        &mut y,
                        &mut b,
                    )
                    .unwrap();
                }
                #[cfg(all(target_arch = "x86_64", any(feature = "sse42", feature = "all-simd")))]
                EncoderSimdBackend::Sse42 => {
                    if let Some(d) = jxl_simd::Sse42Descriptor::new() {
                        srgb_u8_to_xyb_simd_assisted(d, &rgb, w, h, &mut x, &mut y, &mut b)
                            .unwrap();
                    } else {
                        srgb_u8_to_xyb(&rgb, w, h, &mut x, &mut y, &mut b).unwrap();
                    }
                }
                #[cfg(all(target_arch = "x86_64", any(feature = "avx", feature = "all-simd")))]
                EncoderSimdBackend::Avx2 => {
                    if let Some(d) = jxl_simd::AvxDescriptor::new() {
                        srgb_u8_to_xyb_simd_assisted(d, &rgb, w, h, &mut x, &mut y, &mut b)
                            .unwrap();
                    } else {
                        srgb_u8_to_xyb(&rgb, w, h, &mut x, &mut y, &mut b).unwrap();
                    }
                }
                #[cfg(all(target_arch = "x86_64", any(feature = "avx512", feature = "all-simd")))]
                EncoderSimdBackend::Avx512 => {
                    if let Some(d) = jxl_simd::Avx512Descriptor::new() {
                        srgb_u8_to_xyb_simd_assisted(d, &rgb, w, h, &mut x, &mut y, &mut b)
                            .unwrap();
                    } else {
                        srgb_u8_to_xyb(&rgb, w, h, &mut x, &mut y, &mut b).unwrap();
                    }
                }
                #[cfg(all(target_arch = "aarch64", any(feature = "neon", feature = "all-simd")))]
                EncoderSimdBackend::Neon => {
                    if let Some(d) = jxl_simd::NeonDescriptor::new() {
                        srgb_u8_to_xyb_simd_assisted(d, &rgb, w, h, &mut x, &mut y, &mut b)
                            .unwrap();
                    } else {
                        srgb_u8_to_xyb(&rgb, w, h, &mut x, &mut y, &mut b).unwrap();
                    }
                }
            })
        },
    );

    g.finish();
}

fn bench_idct8(c: &mut Criterion) {
    let backend = selected_backend();
    let mut g = c.benchmark_group("encoder_idct8_kernel_reference");

    let mut base = [0.0f32; 64];
    for (i, v) in base.iter_mut().enumerate() {
        *v = ((i as f32 * 7.13).sin() * 40.0) + 3.0;
    }

    g.bench_function(
        BenchmarkId::new("idct2d_8_8", format!("{:?}", backend)),
        |bencher| {
            bencher.iter(|| {
                let mut block = base;
                match backend {
                    EncoderSimdBackend::Scalar => {
                        idct2d_8_8(ScalarDescriptor, &mut block);
                    }
                    #[cfg(all(
                        target_arch = "x86_64",
                        any(feature = "sse42", feature = "all-simd")
                    ))]
                    EncoderSimdBackend::Sse42 => {
                        if let Some(d) = jxl_simd::Sse42Descriptor::new() {
                            d.call(|d| idct2d_8_8(d, &mut block));
                        } else {
                            idct2d_8_8(ScalarDescriptor, &mut block);
                        }
                    }
                    #[cfg(all(target_arch = "x86_64", any(feature = "avx", feature = "all-simd")))]
                    EncoderSimdBackend::Avx2 => {
                        if let Some(d) = jxl_simd::AvxDescriptor::new() {
                            d.call(|d| idct2d_8_8(d, &mut block));
                        } else {
                            idct2d_8_8(ScalarDescriptor, &mut block);
                        }
                    }
                    #[cfg(all(
                        target_arch = "x86_64",
                        any(feature = "avx512", feature = "all-simd")
                    ))]
                    EncoderSimdBackend::Avx512 => {
                        if let Some(d) = jxl_simd::Avx512Descriptor::new() {
                            d.call(|d| idct2d_8_8(d, &mut block));
                        } else {
                            idct2d_8_8(ScalarDescriptor, &mut block);
                        }
                    }
                    #[cfg(all(
                        target_arch = "aarch64",
                        any(feature = "neon", feature = "all-simd")
                    ))]
                    EncoderSimdBackend::Neon => {
                        if let Some(d) = jxl_simd::NeonDescriptor::new() {
                            d.call(|d| idct2d_8_8(d, &mut block));
                        } else {
                            idct2d_8_8(ScalarDescriptor, &mut block);
                        }
                    }
                }
                std::hint::black_box(block[0]);
            })
        },
    );

    g.finish();
}

fn bench_end_to_end_corpus(c: &mut Criterion) {
    let mut g = c.benchmark_group("encoder_end_to_end_corpus");

    // Photo-like gradient/noise RGB.
    let (w, h) = (320usize, 240usize);
    let size = (w as u32, h as u32);
    let mut rgb_photo = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb_photo[i] = ((x * 13 + y * 7) & 255) as u8;
            rgb_photo[i + 1] = ((x * 3 + y * 11) & 255) as u8;
            rgb_photo[i + 2] = ((x * 17 + y * 5) & 255) as u8;
        }
    }

    // Flat-graphic RGB.
    let mut rgb_flat = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let c = if ((x / 16) + (y / 16)) % 2 == 0 {
                [0u8, 180, 255]
            } else {
                [255u8, 255, 255]
            };
            rgb_flat[i] = c[0];
            rgb_flat[i + 1] = c[1];
            rgb_flat[i + 2] = c[2];
        }
    }

    // RGBA alpha sample.
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            rgba[i] = ((x * 9 + y * 2) & 255) as u8;
            rgba[i + 1] = ((x * 5 + y * 13) & 255) as u8;
            rgba[i + 2] = ((x * 3 + y * 7) & 255) as u8;
            rgba[i + 3] = ((x + y) & 255) as u8;
        }
    }

    g.bench_function("encode_rgb_photo", |bencher| {
        bencher.iter(|| {
            let mut opts = JxlEncoderOptions::default();
            opts.mode = JxlEncoderMode::VarDct;
            opts.lossless = false;
            opts.distance_milli = 1000;
            let enc = JxlEncoder::new(opts);
            let out = enc
                .encode_image(size, JxlEncoderImageData::Rgb8Interleaved(&rgb_photo))
                .unwrap();
            std::hint::black_box(out.len());
        })
    });

    g.bench_function("encode_rgb_flat", |bencher| {
        bencher.iter(|| {
            let mut opts = JxlEncoderOptions::default();
            opts.mode = JxlEncoderMode::VarDct;
            opts.lossless = false;
            opts.distance_milli = 1000;
            let enc = JxlEncoder::new(opts);
            let out = enc
                .encode_image(size, JxlEncoderImageData::Rgb8Interleaved(&rgb_flat))
                .unwrap();
            std::hint::black_box(out.len());
        })
    });

    g.bench_function("encode_rgba_alpha", |bencher| {
        bencher.iter(|| {
            let mut opts = JxlEncoderOptions::default();
            opts.mode = JxlEncoderMode::VarDct;
            opts.lossless = false;
            opts.distance_milli = 1000;
            let enc = JxlEncoder::new(opts);
            let out = enc
                .encode_image(size, JxlEncoderImageData::Rgba8Interleaved(&rgba))
                .unwrap();
            std::hint::black_box(out.len());
        })
    });

    g.finish();
}

fn bench_forward_dct32(c: &mut Criterion) {
    let backend = selected_backend();
    let mut g = c.benchmark_group("encoder_forward_dct_reference");

    let mut base = [0.0f32; 1024];
    for (i, v) in base.iter_mut().enumerate() {
        *v = ((i as f32 * 4.73).sin() * 80.0) + 6.0;
    }

    g.bench_function(
        BenchmarkId::new("dct2d_32_scalar", format!("{:?}", backend)),
        |bencher| {
            bencher.iter(|| {
                let mut block = base;
                dct2d_32_scalar(&mut block);
                std::hint::black_box(block[0]);
            })
        },
    );

    g.bench_function(
        BenchmarkId::new("dct2d_32_simd", format!("{:?}", backend)),
        |bencher| {
            bencher.iter(|| {
                let mut block = base;
                match backend {
                    EncoderSimdBackend::Scalar => {
                        dct2d_32_simd(ScalarDescriptor, &mut block);
                    }
                    #[cfg(all(
                        target_arch = "x86_64",
                        any(feature = "sse42", feature = "all-simd")
                    ))]
                    EncoderSimdBackend::Sse42 => {
                        if let Some(d) = jxl_simd::Sse42Descriptor::new() {
                            d.call(|d| dct2d_32_simd(d, &mut block));
                        } else {
                            dct2d_32_simd(ScalarDescriptor, &mut block);
                        }
                    }
                    #[cfg(all(target_arch = "x86_64", any(feature = "avx", feature = "all-simd")))]
                    EncoderSimdBackend::Avx2 => {
                        if let Some(d) = jxl_simd::AvxDescriptor::new() {
                            d.call(|d| dct2d_32_simd(d, &mut block));
                        } else {
                            dct2d_32_simd(ScalarDescriptor, &mut block);
                        }
                    }
                    #[cfg(all(
                        target_arch = "x86_64",
                        any(feature = "avx512", feature = "all-simd")
                    ))]
                    EncoderSimdBackend::Avx512 => {
                        if let Some(d) = jxl_simd::Avx512Descriptor::new() {
                            d.call(|d| dct2d_32_simd(d, &mut block));
                        } else {
                            dct2d_32_simd(ScalarDescriptor, &mut block);
                        }
                    }
                    #[cfg(all(
                        target_arch = "aarch64",
                        any(feature = "neon", feature = "all-simd")
                    ))]
                    EncoderSimdBackend::Neon => {
                        if let Some(d) = jxl_simd::NeonDescriptor::new() {
                            d.call(|d| dct2d_32_simd(d, &mut block));
                        } else {
                            dct2d_32_simd(ScalarDescriptor, &mut block);
                        }
                    }
                }
                std::hint::black_box(block[0]);
            })
        },
    );

    g.finish();
}

fn bench_quant_and_token_prep(c: &mut Criterion) {
    let mut g = c.benchmark_group("encoder_quant_token_prep");

    let mut coeffs = vec![0i32; 64 * 1024];
    for (i, v) in coeffs.iter_mut().enumerate() {
        *v = ((i as i32 * 13 + 17) % 97) - 48;
    }
    let mut qf = vec![1.0f32; coeffs.len()];
    for (i, q) in qf.iter_mut().enumerate() {
        *q = 0.7 + ((i % 64) as f32) * 0.01;
    }

    g.bench_function("quantize_coeffs", |bencher| {
        bencher.iter(|| {
            let mut out = vec![0i32; coeffs.len()];
            for i in 0..coeffs.len() {
                out[i] = ((coeffs[i] as f32) * qf[i]).round() as i32;
            }
            std::hint::black_box(out[0]);
        })
    });

    g.bench_function("tokenize_pack_signed", |bencher| {
        bencher.iter(|| {
            let mut tokens = vec![0u32; coeffs.len()];
            for i in 0..coeffs.len() {
                tokens[i] = pack_signed(((coeffs[i] as f32) * qf[i]).round() as i32);
            }
            std::hint::black_box(tokens[0]);
        })
    });

    g.bench_function("huffman_from_token_histogram", |bencher| {
        bencher.iter(|| {
            let mut tokens = vec![0u32; coeffs.len()];
            for i in 0..coeffs.len() {
                tokens[i] = pack_signed(((coeffs[i] as f32) * qf[i]).round() as i32);
            }
            let max_sym = tokens.iter().copied().max().unwrap_or(0) as usize;
            let mut freq = vec![0u64; max_sym + 1];
            for t in tokens {
                freq[t as usize] += 1;
            }
            let code = build_huffman_code(&freq);
            std::hint::black_box(code.is_some());
        })
    });

    g.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_xyb, bench_idct8, bench_forward_dct32, bench_quant_and_token_prep, bench_end_to_end_corpus
);
criterion_main!(benches);
