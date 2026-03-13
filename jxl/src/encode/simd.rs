// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Encoder SIMD capability detection and benchmark controls.
//!
//! This module intentionally does not change encoder decisions yet.
//! It provides a centralized runtime view of the available SIMD backend,
//! plus a scalar-only override used by benchmark/reference runs.

use jxl_simd::SimdDescriptor;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncoderSimdMode {
    Auto,
    ScalarOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncoderSimdBackend {
    Scalar,
    #[cfg(all(target_arch = "x86_64", any(feature = "sse42", feature = "all-simd")))]
    Sse42,
    #[cfg(all(target_arch = "x86_64", any(feature = "avx", feature = "all-simd")))]
    Avx2,
    #[cfg(all(target_arch = "x86_64", any(feature = "avx512", feature = "all-simd")))]
    Avx512,
    #[cfg(all(target_arch = "aarch64", any(feature = "neon", feature = "all-simd")))]
    Neon,
}

fn parse_simd_mode(value: &str) -> EncoderSimdMode {
    if value.eq_ignore_ascii_case("scalar") || value.eq_ignore_ascii_case("off") || value == "0" {
        EncoderSimdMode::ScalarOnly
    } else {
        EncoderSimdMode::Auto
    }
}

pub fn requested_simd_mode() -> EncoderSimdMode {
    match std::env::var("JXL_ENC_SIMD") {
        Ok(v) => parse_simd_mode(v.trim()),
        Err(_) => EncoderSimdMode::Auto,
    }
}

pub fn detect_encoder_simd_backend() -> EncoderSimdBackend {
    if requested_simd_mode() == EncoderSimdMode::ScalarOnly {
        return EncoderSimdBackend::Scalar;
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(any(feature = "avx512", feature = "all-simd"))]
        if jxl_simd::Avx512Descriptor::new().is_some() {
            return EncoderSimdBackend::Avx512;
        }
        #[cfg(any(feature = "avx", feature = "all-simd"))]
        if jxl_simd::AvxDescriptor::new().is_some() {
            return EncoderSimdBackend::Avx2;
        }
        #[cfg(any(feature = "sse42", feature = "all-simd"))]
        if jxl_simd::Sse42Descriptor::new().is_some() {
            return EncoderSimdBackend::Sse42;
        }
    }

    #[cfg(all(target_arch = "aarch64", any(feature = "neon", feature = "all-simd")))]
    if jxl_simd::NeonDescriptor::new().is_some() {
        return EncoderSimdBackend::Neon;
    }

    EncoderSimdBackend::Scalar
}

pub fn benchmark_force_scalar() -> bool {
    requested_simd_mode() == EncoderSimdMode::ScalarOnly
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_simd::{ScalarDescriptor, SimdDescriptor};
    use jxl_transforms::idct2d_8_8;

    #[test]
    fn test_parse_simd_mode() {
        assert_eq!(parse_simd_mode("auto"), EncoderSimdMode::Auto);
        assert_eq!(parse_simd_mode(""), EncoderSimdMode::Auto);
        assert_eq!(parse_simd_mode("scalar"), EncoderSimdMode::ScalarOnly);
        assert_eq!(parse_simd_mode("off"), EncoderSimdMode::ScalarOnly);
        assert_eq!(parse_simd_mode("0"), EncoderSimdMode::ScalarOnly);
    }

    #[test]
    fn test_detect_encoder_simd_backend_non_panicking() {
        let backend = detect_encoder_simd_backend();
        match backend {
            EncoderSimdBackend::Scalar => {}
            #[cfg(all(target_arch = "x86_64", any(feature = "sse42", feature = "all-simd")))]
            EncoderSimdBackend::Sse42 => {}
            #[cfg(all(target_arch = "x86_64", any(feature = "avx", feature = "all-simd")))]
            EncoderSimdBackend::Avx2 => {}
            #[cfg(all(target_arch = "x86_64", any(feature = "avx512", feature = "all-simd")))]
            EncoderSimdBackend::Avx512 => {}
            #[cfg(all(target_arch = "aarch64", any(feature = "neon", feature = "all-simd")))]
            EncoderSimdBackend::Neon => {}
        }
    }

    fn idct8_scalar_reference() -> [f32; 64] {
        let mut block = [0f32; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin() * 23.0) + 2.0;
        }
        idct2d_8_8(ScalarDescriptor, &mut block);
        block
    }

    fn assert_near_eq(a: &[f32; 64], b: &[f32; 64]) {
        for i in 0..64 {
            assert!(
                (a[i] - b[i]).abs() < 1e-4,
                "lane {i} mismatch: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", any(feature = "sse42", feature = "all-simd")))]
    #[test]
    fn test_idct8_simd_scalar_equivalence_sse42() {
        let Some(d) = jxl_simd::Sse42Descriptor::new() else {
            return;
        };
        let scalar = idct8_scalar_reference();
        let mut simd_block = [0f32; 64];
        for (i, v) in simd_block.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin() * 23.0) + 2.0;
        }
        d.call(|d| idct2d_8_8(d, &mut simd_block));
        assert_near_eq(&scalar, &simd_block);
    }

    #[cfg(all(target_arch = "x86_64", any(feature = "avx", feature = "all-simd")))]
    #[test]
    fn test_idct8_simd_scalar_equivalence_avx2() {
        let Some(d) = jxl_simd::AvxDescriptor::new() else {
            return;
        };
        let scalar = idct8_scalar_reference();
        let mut simd_block = [0f32; 64];
        for (i, v) in simd_block.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin() * 23.0) + 2.0;
        }
        d.call(|d| idct2d_8_8(d, &mut simd_block));
        assert_near_eq(&scalar, &simd_block);
    }

    #[cfg(all(target_arch = "x86_64", any(feature = "avx512", feature = "all-simd")))]
    #[test]
    fn test_idct8_simd_scalar_equivalence_avx512() {
        let Some(d) = jxl_simd::Avx512Descriptor::new() else {
            return;
        };
        let scalar = idct8_scalar_reference();
        let mut simd_block = [0f32; 64];
        for (i, v) in simd_block.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin() * 23.0) + 2.0;
        }
        d.call(|d| idct2d_8_8(d, &mut simd_block));
        assert_near_eq(&scalar, &simd_block);
    }

    #[cfg(all(target_arch = "aarch64", any(feature = "neon", feature = "all-simd")))]
    #[test]
    fn test_idct8_simd_scalar_equivalence_neon() {
        let Some(d) = jxl_simd::NeonDescriptor::new() else {
            return;
        };
        let scalar = idct8_scalar_reference();
        let mut simd_block = [0f32; 64];
        for (i, v) in simd_block.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin() * 23.0) + 2.0;
        }
        d.call(|d| idct2d_8_8(d, &mut simd_block));
        assert_near_eq(&scalar, &simd_block);
    }
}
