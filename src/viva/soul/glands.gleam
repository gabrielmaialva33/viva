//// VIVA Glands - GPU-accelerated tensor operations
////
//// NIF bindings for viva_glands (Rust + CUDA + Candle)
//// The "endocrine system" of VIVA - secretes intelligence from LLMs
//// and projects it into Holographic Reduced Representations (HRR).
////
//// ## Features
//// - cuFFT for HRR bind/unbind (GPU)
//// - Candle CUDA for matmul projection
//// - SIMD AVX2 for CPU fallback
//// - Auto-detection GPU/CPU
////
//// ## Architecture
////
//// ```
//// LLM (4096 dim) ──► Projection Matrix ──► HRR (8192 dim)
////                           │
////                    Circular Conv (cuFFT)
////                           │
////                    Holographic Trace
//// ```

import gleam/dynamic.{type Dynamic}

// ============================================================================
// TYPES
// ============================================================================

/// Configuration for Glands initialization
pub type GlandsConfig {
  GlandsConfig(
    llm_dim: Int,    // Input embedding dimension (e.g., 4096)
    hrr_dim: Int,    // HRR vector dimension (e.g., 8192)
    seed: Int,       // Random seed for projection matrix
    gpu_layers: Int, // GPU layers for LLM (99 = all on GPU)
  )
}

/// Legacy config alias
pub type HRRConfig = GlandsConfig

/// Result from distillation process
pub type DistillationResult {
  DistillationResult(
    text: String,
    embedding: List(Float),      // Raw LLM embedding
    hrr_vector: List(Float),     // Projected HRR vector
    dimensions: Int,
  )
}

/// Opaque handle to the Glands native resource (GPU/CPU)
pub opaque type GlandsHandle {
  GlandsHandle(resource: Dynamic)
}

// ============================================================================
// DEFAULT CONFIG
// ============================================================================

/// Default configuration matching VIVA's HRR architecture
/// - LLM dim: 4096 (Llama/Qwen standard)
/// - HRR dim: 8192 (VIVA holographic space)
/// - Seed: 42 (reproducible projections)
/// - GPU layers: 99 (all on GPU)
pub fn default_config() -> GlandsConfig {
  GlandsConfig(llm_dim: 4096, hrr_dim: 8192, seed: 42, gpu_layers: 99)
}

/// Config for Qwen 2.5 14B (larger hidden dim)
pub fn qwen_config() -> GlandsConfig {
  GlandsConfig(llm_dim: 5120, hrr_dim: 8192, seed: 42, gpu_layers: 99)
}

/// Config for SmolLM/Phi (smaller hidden dim)
pub fn small_config() -> GlandsConfig {
  GlandsConfig(llm_dim: 2048, hrr_dim: 4096, seed: 42, gpu_layers: 99)
}

/// Config for NEAT neural networks (smaller dimensions)
pub fn neat_config() -> GlandsConfig {
  GlandsConfig(llm_dim: 64, hrr_dim: 128, seed: 42, gpu_layers: 99)
}

// ============================================================================
// NATIVE INTERFACE (Erlang FFI)
// ============================================================================

/// Initialize the Glands system with HRR configuration
/// Creates the orthogonal projection matrix on GPU/CPU
@external(erlang, "Elixir.Viva.Glands.Native", "glands_init_gleam")
fn native_init(
  llm_dim: Int,
  hrr_dim: Int,
  seed: Int,
  gpu_layers: Int,
) -> Result(Dynamic, String)

/// Project LLM embedding to HRR space
@external(erlang, "Elixir.Viva.Glands.Native", "glands_project")
fn native_project(
  handle: Dynamic,
  embedding: List(Float),
) -> Result(List(Float), String)

/// Bind two HRR vectors via circular convolution (cuFFT on GPU)
@external(erlang, "Elixir.Viva.Glands.Native", "glands_bind")
fn native_bind(
  handle: Dynamic,
  a: List(Float),
  b: List(Float),
) -> Result(List(Float), String)

/// Unbind (retrieve) from HRR trace via circular correlation
@external(erlang, "Elixir.Viva.Glands.Native", "glands_unbind")
fn native_unbind(
  handle: Dynamic,
  trace: List(Float),
  key: List(Float),
) -> Result(List(Float), String)

/// Cosine similarity between two vectors (SIMD AVX2)
@external(erlang, "Elixir.Viva.Glands.Native", "glands_similarity")
fn native_similarity(a: List(Float), b: List(Float)) -> Result(Float, String)

/// Batch cosine similarity (parallel with Rayon)
@external(erlang, "Elixir.Viva.Glands.Native", "glands_batch_similarity")
fn native_batch_similarity(
  vectors: List(List(Float)),
  query: List(Float),
) -> Result(List(Float), String)

/// Superpose vectors (normalized sum, parallel)
@external(erlang, "Elixir.Viva.Glands.Native", "glands_superpose")
fn native_superpose(vectors: List(List(Float))) -> Result(List(Float), String)

/// Health check - returns "GLANDS_ULTRA_OK (SIMD: AVX2, GPU: CUDA, Threads: N)"
@external(erlang, "Elixir.Viva.Glands.Native", "glands_check")
fn native_check() -> String

/// Benchmark GPU/CPU performance
@external(erlang, "Elixir.Viva.Glands.Native", "glands_benchmark")
fn native_benchmark(handle: Dynamic, iterations: Int) -> String

// ============================================================================
// PUBLIC API
// ============================================================================

/// Initialize the Glands neural extraction system
///
/// ## Example
/// ```gleam
/// let assert Ok(glands) = glands.init(glands.default_config())
/// ```
pub fn init(config: GlandsConfig) -> Result(GlandsHandle, String) {
  case native_init(config.llm_dim, config.hrr_dim, config.seed, config.gpu_layers) {
    Ok(resource) -> Ok(GlandsHandle(resource))
    Error(msg) -> Error("Failed to initialize Glands: " <> msg)
  }
}

/// Project an LLM embedding into HRR space (GPU-accelerated)
///
/// Takes a raw embedding from an LLM (e.g., 4096 dimensions)
/// and projects it into VIVA's holographic space (e.g., 8192 dimensions)
/// using an orthogonal projection matrix (Candle CUDA matmul).
pub fn project(
  handle: GlandsHandle,
  embedding: List(Float),
) -> Result(List(Float), String) {
  native_project(handle.resource, embedding)
}

/// Bind two HRR vectors together (cuFFT on GPU)
///
/// Uses circular convolution via FFT to create an associative binding.
/// This is the core operation for storing role-filler pairs in holographic memory.
///
/// ## Example
/// ```gleam
/// // Bind a "name" role with a "VIVA" concept
/// let assert Ok(bound) = glands.bind(handle, name_role_vector, viva_concept_vector)
/// ```
pub fn bind(
  handle: GlandsHandle,
  a: List(Float),
  b: List(Float),
) -> Result(List(Float), String) {
  native_bind(handle.resource, a, b)
}

/// Unbind (retrieve) a value from an HRR trace (cuFFT on GPU)
///
/// Uses circular correlation to approximately retrieve the original
/// binding partner when given one of the bound vectors.
///
/// ## Example
/// ```gleam
/// // Retrieve the concept bound to "name" role
/// let assert Ok(retrieved) = glands.unbind(handle, memory_trace, name_role_vector)
/// // retrieved ≈ viva_concept_vector (with some noise)
/// ```
pub fn unbind(
  handle: GlandsHandle,
  trace: List(Float),
  key: List(Float),
) -> Result(List(Float), String) {
  native_unbind(handle.resource, trace, key)
}

/// Calculate cosine similarity between two vectors (SIMD AVX2)
///
/// Returns a value between -1.0 and 1.0, where:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal
/// - -1.0 = opposite direction
pub fn similarity(a: List(Float), b: List(Float)) -> Result(Float, String) {
  native_similarity(a, b)
}

/// Batch cosine similarity (parallel with Rayon)
/// Compares multiple vectors against a single query
pub fn batch_similarity(
  vectors: List(List(Float)),
  query: List(Float),
) -> Result(List(Float), String) {
  native_batch_similarity(vectors, query)
}

/// Health check for the native Glands backend
/// Returns "GLANDS_ULTRA_OK (SIMD: AVX2, GPU: CUDA, Threads: N)"
pub fn check() -> String {
  native_check()
}

/// Benchmark GPU/CPU performance
pub fn benchmark(handle: GlandsHandle, iterations: Int) -> String {
  native_benchmark(handle.resource, iterations)
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Superposition: add multiple HRR vectors (bundling operation, parallel)
///
/// Creates a distributed representation containing all inputs.
/// Note: Unlike binding, this is NOT reversible.
/// Uses Rayon parallel processing on GPU/CPU.
pub fn superpose(vectors: List(List(Float))) -> Result(List(Float), String) {
  native_superpose(vectors)
}
