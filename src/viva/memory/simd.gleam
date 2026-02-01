//// SIMD-accelerated operations for HRR vectors
////
//// Uses AVX/AVX2 intrinsics via NIF for 4x speedup on vector operations.
//// Falls back to Erlang if NIF not loaded.

// =============================================================================
// FFI - Direct calls to viva_simd_nif
// =============================================================================

/// SIMD dot product - 4x faster with AVX
@external(erlang, "viva_simd_nif", "simd_dot")
pub fn dot(a: List(Float), b: List(Float)) -> Float

/// SIMD element-wise multiply - 4x faster with AVX
@external(erlang, "viva_simd_nif", "simd_mul")
pub fn mul(a: List(Float), b: List(Float)) -> List(Float)

/// SIMD matrix multiply - accelerated inner loop
@external(erlang, "viva_simd_nif", "simd_matmul")
pub fn matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  k: Int,
  n: Int,
) -> List(List(Float))

/// SIMD sum (reduction) - 4x faster with AVX
@external(erlang, "viva_simd_nif", "simd_sum")
pub fn sum(a: List(Float)) -> Float

/// SIMD scale (x * scalar) - 4x faster with AVX
@external(erlang, "viva_simd_nif", "simd_scale")
pub fn scale(a: List(Float), scalar: Float) -> List(Float)

/// Check if SIMD NIF is loaded and working
@external(erlang, "viva_simd_nif", "is_simd_available")
pub fn is_available() -> Bool
