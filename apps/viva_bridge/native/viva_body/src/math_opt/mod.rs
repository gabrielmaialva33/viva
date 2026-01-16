//! Low-level mathematical optimizations (SIMD/AVX2/NEON)
//! for VIVA's sensory system.
//!
//! Goal: Reduce neuro-function calculation time (Sigmoid/Tanh)
//! from nanoseconds (ns) to sub-nanoseconds.

#![allow(dead_code)] // TODO: Remove once fully integrated into batch processing

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Calculates sigmoid (Logistic Function) using AVX2 instructions if available.
/// 1 / (1 + exp(-k(x - x0)))
///
/// Vectorized implementation to process 8 floats (f32) simultaneously.
/// Extremely useful for large sensor arrays.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sigmoid_avx2(x: __m256, k: __m256, x0: __m256) -> __m256 {
    // 1. (x - x0)
    let diff = _mm256_sub_ps(x, x0);

    // 2. -k * (diff)
    // XOR with -0.0 to invert sign (faster than mul by -1.0)
    let neg_k_diff = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), k), diff);

    // 3. exp(val) - Fast polynomial approximation for exp()
    // (Simplified implementation for maximum performance in biological range)
    let exp_val = fast_exp_avx2(neg_k_diff);

    // 4. 1 + exp
    let one = _mm256_set1_ps(1.0);
    let den = _mm256_add_ps(one, exp_val);

    // 5. 1 / den
    _mm256_div_ps(one, den)
}

/// Fast EXP approximation based on Schraudolph / 4th degree polynomial
/// Sufficient precision for digital biology, extreme performance.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    // Magic constants for approximation
    let a = _mm256_set1_ps(12102203.0);
    let b = _mm256_set1_ps(1064986820.0); // Offset float->int

    // Fast conversion (Schraudolph method adapted for AVX)
    // y = a * x + b
    let val = _mm256_fmadd_ps(a, x, b); // Fused Multiply-Add

    // Bitwise cast to int and back (fast exp hack)
    _mm256_castsi256_ps(_mm256_cvttps_epi32(val))
    // Note: This is a VERY aggressive approximation.
    // In production, we might want a Taylor/Padé series for better precision.
}

/// Scalar Version (Standard) - Safe Fallback
#[inline(always)]
pub fn sigmoid_scalar(x: f32, k: f32, x0: f32) -> f32 {
    1.0 / (1.0 + (-k * (x - x0)).exp())
}

/// Public wrapper that chooses the best implementation at runtime
/// (Currently defaults to scalar for single values, AVX2 intended for batch ops)
pub fn sigmoid_optimized(x: f32, k: f32, x0: f32) -> f32 {
    // For single values, the overhead of loading AVX registers might not be worth it.
    // TODO: Create `batch_sigmoid` API for array processing.
    sigmoid_scalar(x, k, x0)
}
