//// VIVA Glands - Neural Extraction & HRR Projection
////
//// The "endocrine system" of VIVA - secretes intelligence from LLMs
//// and projects it into Holographic Reduced Representations (HRR).
////
//// ## Architecture
////
//// ```
//// LLM (4096 dim) ──► Projection Matrix ──► HRR (8192 dim)
////                           │
////                    Circular Conv (FFT)
////                           │
////                    Holographic Trace
//// ```

import gleam/dynamic.{type Dynamic}
import gleam/result

// ============================================================================
// TYPES
// ============================================================================

/// Configuration for HRR projection
pub type HRRConfig {
  HRRConfig(
    hrr_dim: Int,    // Target HRR dimensions (default: 8192)
    llm_dim: Int,    // Source LLM dimensions (default: 4096)
    seed: Int,       // Seed for reproducible projection
  )
}

/// Result from distillation process
pub type DistillationResult {
  DistillationResult(
    text: String,
    embedding: List(Float),      // Raw LLM embedding
    hrr_vector: List(Float),     // Projected HRR vector
    dimensions: Int,
  )
}

/// Opaque handle to the Glands native resource
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
pub fn default_config() -> HRRConfig {
  HRRConfig(hrr_dim: 8192, llm_dim: 4096, seed: 42)
}

/// Config for Qwen 2.5 14B (larger hidden dim)
pub fn qwen_config() -> HRRConfig {
  HRRConfig(hrr_dim: 8192, llm_dim: 5120, seed: 42)
}

/// Config for SmolLM/Phi (smaller hidden dim)
pub fn small_config() -> HRRConfig {
  HRRConfig(hrr_dim: 4096, llm_dim: 2048, seed: 42)
}

// ============================================================================
// NATIVE INTERFACE (Erlang FFI)
// ============================================================================

/// Initialize the Glands system with HRR configuration
/// Creates the orthogonal projection matrix on GPU/CPU
@external(erlang, "Elixir.Viva.Glands.Native", "glands_init")
fn native_init(config: HRRConfig) -> Result(Dynamic, Dynamic)

/// Get device info (CPU/CUDA/Metal)
@external(erlang, "Elixir.Viva.Glands.Native", "glands_device_info")
fn native_device_info(handle: Dynamic) -> String

/// Project LLM embedding to HRR space
@external(erlang, "Elixir.Viva.Glands.Native", "glands_project")
fn native_project(
  handle: Dynamic,
  embedding: List(Float),
) -> Result(List(Float), Dynamic)

/// Bind two HRR vectors via circular convolution
@external(erlang, "Elixir.Viva.Glands.Native", "glands_bind")
fn native_bind(a: List(Float), b: List(Float)) -> Result(List(Float), Dynamic)

/// Unbind (retrieve) from HRR trace via circular correlation
@external(erlang, "Elixir.Viva.Glands.Native", "glands_unbind")
fn native_unbind(
  trace: List(Float),
  key: List(Float),
) -> Result(List(Float), Dynamic)

/// Cosine similarity between two vectors
@external(erlang, "Elixir.Viva.Glands.Native", "glands_cosine_similarity")
fn native_cosine_similarity(
  a: List(Float),
  b: List(Float),
) -> Result(Float, Dynamic)

/// Health check - returns "CANDLE_CUDA_OK" or "CANDLE_CPU_OK"
@external(erlang, "Elixir.Viva.Glands.Native", "glands_check")
fn native_check() -> String

// ============================================================================
// PUBLIC API
// ============================================================================

/// Initialize the Glands neural extraction system
///
/// ## Example
/// ```gleam
/// let assert Ok(glands) = glands.init(glands.default_config())
/// ```
pub fn init(config: HRRConfig) -> Result(GlandsHandle, String) {
  case native_init(config) {
    Ok(resource) -> Ok(GlandsHandle(resource))
    Error(_) -> Error("Failed to initialize Glands: check CUDA/GPU")
  }
}

/// Get the compute device being used
pub fn device_info(handle: GlandsHandle) -> String {
  native_device_info(handle.resource)
}

/// Project an LLM embedding into HRR space
///
/// Takes a raw embedding from an LLM (e.g., 4096 dimensions)
/// and projects it into VIVA's holographic space (e.g., 8192 dimensions)
/// using an orthogonal projection matrix.
pub fn project(
  handle: GlandsHandle,
  embedding: List(Float),
) -> Result(List(Float), String) {
  native_project(handle.resource, embedding)
  |> result.map_error(fn(_) { "Projection failed: dimension mismatch?" })
}

/// Bind two HRR vectors together
///
/// Uses circular convolution via FFT to create an associative binding.
/// This is the core operation for storing role-filler pairs in holographic memory.
///
/// ## Example
/// ```gleam
/// // Bind a "name" role with a "VIVA" concept
/// let bound = glands.bind(name_role_vector, viva_concept_vector)
/// ```
pub fn bind(a: List(Float), b: List(Float)) -> Result(List(Float), String) {
  native_bind(a, b)
  |> result.map_error(fn(_) { "Bind failed: vectors must have same dimension" })
}

/// Unbind (retrieve) a value from an HRR trace
///
/// Uses circular correlation to approximately retrieve the original
/// binding partner when given one of the bound vectors.
///
/// ## Example
/// ```gleam
/// // Retrieve the concept bound to "name" role
/// let retrieved = glands.unbind(memory_trace, name_role_vector)
/// // retrieved ≈ viva_concept_vector (with some noise)
/// ```
pub fn unbind(
  trace: List(Float),
  key: List(Float),
) -> Result(List(Float), String) {
  native_unbind(trace, key)
  |> result.map_error(fn(_) { "Unbind failed: vectors must have same dimension" })
}

/// Calculate cosine similarity between two vectors
///
/// Returns a value between -1.0 and 1.0, where:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal
/// - -1.0 = opposite direction
pub fn similarity(a: List(Float), b: List(Float)) -> Result(Float, String) {
  native_cosine_similarity(a, b)
  |> result.map_error(fn(_) { "Similarity failed: dimension mismatch" })
}

/// Health check for the native Candle backend
/// Returns "CANDLE_CUDA_OK", "CANDLE_CPU_OK", or "CANDLE_METAL_OK"
pub fn check() -> String {
  native_check()
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Superposition: add multiple HRR vectors (bundling operation)
///
/// Creates a distributed representation containing all inputs.
/// Note: Unlike binding, this is NOT reversible.
pub fn superpose(vectors: List(List(Float))) -> Result(List(Float), String) {
  case vectors {
    [] -> Error("Cannot superpose empty list")
    [first, ..rest] -> {
      let sum = list_sum(first, rest)
      Ok(normalize(sum))
    }
  }
}

fn list_sum(acc: List(Float), rest: List(List(Float))) -> List(Float) {
  case rest {
    [] -> acc
    [vec, ..remaining] -> {
      let new_acc = list_add(acc, vec)
      list_sum(new_acc, remaining)
    }
  }
}

fn list_add(a: List(Float), b: List(Float)) -> List(Float) {
  case a, b {
    [], [] -> []
    [x, ..xs], [y, ..ys] -> [x +. y, ..list_add(xs, ys)]
    _, _ -> []
  }
}

fn normalize(vec: List(Float)) -> List(Float) {
  let norm = list_norm(vec)
  case norm >. 0.0 {
    True -> list_scale(vec, 1.0 /. norm)
    False -> vec
  }
}

fn list_norm(vec: List(Float)) -> Float {
  let sum_squares = list_fold(vec, 0.0, fn(acc, x) { acc +. x *. x })
  float_sqrt(sum_squares)
}

fn list_scale(vec: List(Float), s: Float) -> List(Float) {
  list_map(vec, fn(x) { x *. s })
}

fn list_fold(list: List(a), acc: b, f: fn(b, a) -> b) -> b {
  case list {
    [] -> acc
    [x, ..xs] -> list_fold(xs, f(acc, x), f)
  }
}

fn list_map(list: List(a), f: fn(a) -> b) -> List(b) {
  case list {
    [] -> []
    [x, ..xs] -> [f(x), ..list_map(xs, f)]
  }
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
