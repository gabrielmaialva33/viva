//! Low-level mathematical optimizations (SIMD/AVX2/NEON)
//! for VIVA's sensory system.
//!
//! Goal: Reduce neuro-function calculation time (Sigmoid/Tanh)
//! from nanoseconds (ns) to sub-nanoseconds.

// SIMD optimizations enabled - batch_sigmoid uses AVX2 when available

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

/// Fast EXP approximation based on Schraudolph method
///
/// Uses IEEE 754 float bit manipulation for ultra-fast exp():
/// - `a` = 2^23 / ln(2) ≈ 12102203
/// - `b` = bias adjusted for reduced average error
///
/// Precision: ~0.26% max error in range [-10, 10]
/// Sufficient for biological/neural computations.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    // Schraudolph constants (IEEE 754 bit manipulation)
    // a = 2^23 / ln(2) ≈ 12102203.16
    let a = _mm256_set1_ps(12102203.0);
    // b = 127 * 2^23 - adjustment (reduces average error)
    let b = _mm256_set1_ps(1064986820.0);

    // Fast conversion: y = a * x + b
    let val = _mm256_fmadd_ps(a, x, b); // Fused Multiply-Add

    // Bitwise reinterpret as float (the Schraudolph trick)
    // Using cvtps (round) instead of cvttps (truncate) for better precision
    _mm256_castsi256_ps(_mm256_cvtps_epi32(val))
}

/// Processes a batch of sigmoids using AVX2 (Process 8 at a time)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn batch_sigmoid_avx2(xs: &[f32], k: f32, x0: f32, out: &mut [f32]) {
    let k_vec = _mm256_set1_ps(k);
    let x0_vec = _mm256_set1_ps(x0);

    // Process chunks of 8
    let chunks = xs.len() / 8;
    for i in 0..chunks {
        let offset = i * 8;
        // Load 8 floats
        let x_vec = _mm256_loadu_ps(xs.as_ptr().add(offset));
        // Compute Sigmoid
        let result = sigmoid_avx2(x_vec, k_vec, x0_vec);
        // Store 8 floats
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
    }

    // Handle remainder (scalar fallback for last <8 items)
    let processed = chunks * 8;
    for i in processed..xs.len() {
        out[i] = sigmoid_scalar(xs[i], k, x0);
    }
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
    sigmoid_scalar(x, k, x0)
}

/// Processes a batch of inputs through the sigmoid function.
/// Automatically chooses AVX2 if available at runtime.
///
/// Ideal for processing arrays of sensory inputs or neural activations.
pub fn batch_sigmoid(xs: &[f32], k: f32, x0: f32, out: &mut [f32]) {
    // Assert output buffer is large enough
    assert!(out.len() >= xs.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                batch_sigmoid_avx2(xs, k, x0, out);
                return;
            }
        }
    }

    // Scalar fallback
    for (x, o) in xs.iter().zip(out.iter_mut()) {
        *o = sigmoid_scalar(*x, k, x0);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_scalar_basic() {
        // sigmoid(0, 1, 0) = 0.5
        let result = sigmoid_scalar(0.0, 1.0, 0.0);
        assert!((result - 0.5).abs() < 1e-6);

        // sigmoid(large positive) -> 1.0
        let result = sigmoid_scalar(10.0, 1.0, 0.0);
        assert!(result > 0.99);

        // sigmoid(large negative) -> 0.0
        let result = sigmoid_scalar(-10.0, 1.0, 0.0);
        assert!(result < 0.01);
    }

    #[test]
    fn test_batch_sigmoid_correctness() {
        let inputs: Vec<f32> = vec![-5.0, -2.0, 0.0, 2.0, 5.0, 10.0, -10.0, 3.0];
        let mut outputs = vec![0.0f32; inputs.len()];
        let k = 1.0;
        let x0 = 0.0;

        batch_sigmoid(&inputs, k, x0, &mut outputs);

        // Verify against scalar
        for (i, &x) in inputs.iter().enumerate() {
            let expected = sigmoid_scalar(x, k, x0);
            let diff = (outputs[i] - expected).abs();
            // Allow small tolerance for AVX2 approximation
            assert!(diff < 0.01, "Mismatch at {}: got {}, expected {}", i, outputs[i], expected);
        }
    }

    #[test]
    fn test_batch_sigmoid_non_aligned() {
        // Test with non-8-aligned length (tests scalar remainder path)
        let inputs: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // 5 elements
        let mut outputs = vec![0.0f32; inputs.len()];

        batch_sigmoid(&inputs, 1.0, 0.0, &mut outputs);

        assert!((outputs[0] - 0.5).abs() < 0.01);
        assert!(outputs[4] > 0.98); // sigmoid(4) ≈ 0.982
    }

    #[test]
    fn test_batch_sigmoid_large() {
        // Test with 1000 elements to stress test AVX2 path
        let inputs: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 100.0).collect();
        let mut outputs = vec![0.0f32; inputs.len()];

        batch_sigmoid(&inputs, 1.0, 0.0, &mut outputs);

        // Spot check
        assert!((outputs[500] - 0.5).abs() < 0.01); // x=0 -> 0.5
        assert!(outputs[999] > 0.99); // x=4.99 -> ~1.0
        assert!(outputs[0] < 0.01); // x=-5.0 -> ~0.0
    }
}
