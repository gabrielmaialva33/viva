//// HRR - Holographic Reduced Representations (Plate, 1995)
////
//// Vector algebra for representing symbolic structures in dense vectors.
//// Supports binding (association), unbinding (retrieval), and superposition.
////
//// Theory: A holographic vector distributes information across all dimensions.
//// Cutting the vector in half loses resolution, not data.
////
//// Usage:
////   let agent = hrr.random(512)
////   let action = hrr.random(512)
////   let memory = hrr.bind(agent, action)  // Agent * Action
////   let recovered = hrr.unbind(memory, agent)  // ~Action

import gleam/float
import gleam/list
import viva/neural/tensor.{type Tensor}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// HRR Vector - wrapper around tensor with HRR semantics
pub type HRR {
  HRR(
    /// The underlying vector (always 1D)
    vector: Tensor,
    /// Dimensionality
    dim: Int,
  )
}

/// HRR Error types
pub type HRRError {
  DimensionMismatch(expected: Int, got: Int)
  InvalidDimension(reason: String)
  FFTError(reason: String)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create random HRR vector (unit hypersphere)
/// Each element drawn from N(0, 1/sqrt(dim)) for unit norm expectation
pub fn random(dim: Int) -> HRR {
  let data =
    list.range(1, dim)
    |> list.map(fn(_) { random_gaussian() /. float_sqrt(int_to_float(dim)) })

  HRR(vector: tensor.Tensor(data: data, shape: [dim]), dim: dim)
}

/// Create zero HRR vector
pub fn zeros(dim: Int) -> HRR {
  HRR(vector: tensor.zeros([dim]), dim: dim)
}

/// Create HRR from existing tensor
pub fn from_tensor(t: Tensor) -> Result(HRR, HRRError) {
  case t.shape {
    [dim] -> Ok(HRR(vector: t, dim: dim))
    _ -> Error(InvalidDimension("HRR requires 1D tensor"))
  }
}

/// Create HRR from list of floats
pub fn from_list(data: List(Float)) -> HRR {
  let dim = list.length(data)
  HRR(vector: tensor.Tensor(data: data, shape: [dim]), dim: dim)
}

// =============================================================================
// CORE OPERATIONS (Plate 1995)
// =============================================================================

/// Binding (*): Associates two concepts
/// Mathematically: circular convolution
/// bind(A, B) creates a new vector that "contains" the association A*B
pub fn bind(a: HRR, b: HRR) -> Result(HRR, HRRError) {
  case a.dim == b.dim {
    False -> Error(DimensionMismatch(expected: a.dim, got: b.dim))
    True -> {
      let result = circular_convolution(td(a.vector), td(b.vector))
      Ok(HRR(vector: tensor.Tensor(data: result, shape: [a.dim]), dim: a.dim))
    }
  }
}

/// Unbinding (#): Recovers associated concept
/// Mathematically: circular correlation (convolution with inverse)
/// unbind(A*B, A) ≈ B (approximate recovery)
pub fn unbind(trace: HRR, cue: HRR) -> Result(HRR, HRRError) {
  case trace.dim == cue.dim {
    False -> Error(DimensionMismatch(expected: trace.dim, got: cue.dim))
    True -> {
      let cue_inv = approximate_inverse(td(cue.vector))
      let result = circular_convolution(td(trace.vector), cue_inv)
      Ok(HRR(
        vector: tensor.Tensor(data: result, shape: [trace.dim]),
        dim: trace.dim,
      ))
    }
  }
}

/// Superposition (+): Combines multiple memories
/// Simply adds vectors (memories coexist in superposition)
pub fn superpose(vectors: List(HRR)) -> Result(HRR, HRRError) {
  case vectors {
    [] -> Error(InvalidDimension("Cannot superpose empty list"))
    [first, ..rest] -> {
      let dim = first.dim
      let valid = list.all(rest, fn(h) { h.dim == dim })
      case valid {
        False -> Error(DimensionMismatch(expected: dim, got: 0))
        True -> {
          let sum_data =
            list.fold(rest, td(first.vector), fn(acc, h) {
              list.map2(acc, td(h.vector), fn(a, b) { a +. b })
            })
          Ok(HRR(vector: tensor.Tensor(data: sum_data, shape: [dim]), dim: dim))
        }
      }
    }
  }
}

/// Normalize to unit length
pub fn normalize(h: HRR) -> HRR {
  let norm = vector_norm(td(h.vector))
  case norm >. 0.0001 {
    True -> {
      let normalized = list.map(td(h.vector), fn(x) { x /. norm })
      HRR(vector: tensor.Tensor(data: normalized, shape: [h.dim]), dim: h.dim)
    }
    False -> h
  }
}

// =============================================================================
// SIMILARITY
// =============================================================================

/// Cosine similarity between two HRR vectors
/// Returns value in [-1, 1], where 1 = identical, 0 = orthogonal
pub fn similarity(a: HRR, b: HRR) -> Float {
  case a.dim == b.dim {
    False -> 0.0
    True -> {
      let dot_product =
        list.map2(td(a.vector), td(b.vector), fn(x, y) { x *. y })
        |> list.fold(0.0, fn(acc, x) { acc +. x })
      let norm_a = vector_norm(td(a.vector))
      let norm_b = vector_norm(td(b.vector))
      case norm_a *. norm_b >. 0.0001 {
        True -> dot_product /. { norm_a *. norm_b }
        False -> 0.0
      }
    }
  }
}

/// Dot product (unnormalized similarity)
pub fn dot(a: HRR, b: HRR) -> Float {
  case a.dim == b.dim {
    False -> 0.0
    True -> {
      list.map2(td(a.vector), td(b.vector), fn(x, y) { x *. y })
      |> list.fold(0.0, fn(acc, x) { acc +. x })
    }
  }
}

// =============================================================================
// MEMORY OPERATIONS
// =============================================================================

/// Create a role-filler binding (structured memory)
/// Example: bind_role_filler(role_agent, viva_vector)
pub fn bind_role_filler(role: HRR, filler: HRR) -> Result(HRR, HRRError) {
  bind(role, filler)
}

/// Create a sequence memory (ordered items)
/// Uses positional encoding via repeated binding
pub fn encode_sequence(
  items: List(HRR),
  position_base: HRR,
) -> Result(HRR, HRRError) {
  case items {
    [] -> Error(InvalidDimension("Cannot encode empty sequence"))
    [first, ..rest] -> {
      let initial = #(first, position_base, [first])

      let #(_, _, encoded) =
        list.fold(rest, initial, fn(state, item) {
          let #(_, current_pos, acc) = state
          case bind(item, current_pos) {
            Ok(positioned_item) -> {
              case bind(current_pos, position_base) {
                Ok(next_pos) -> #(item, next_pos, [positioned_item, ..acc])
                Error(_) -> state
              }
            }
            Error(_) -> state
          }
        })

      superpose(list.reverse(encoded))
    }
  }
}

/// Query a composite memory with a cue
/// Returns similarity scores for potential matches (sorted descending)
pub fn query(
  memory: HRR,
  cue: HRR,
  candidates: List(HRR),
) -> List(#(Int, Float)) {
  case unbind(memory, cue) {
    Ok(retrieved) -> {
      list.index_map(candidates, fn(candidate, idx) {
        #(idx, similarity(retrieved, candidate))
      })
      |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
    }
    Error(_) -> []
  }
}

// =============================================================================
// CIRCULAR CONVOLUTION (Pure Gleam - O(n²))
// =============================================================================

/// Circular convolution (naive O(n²) implementation)
/// For production, use FFT via Nx: O(n log n)
fn circular_convolution(a: List(Float), b: List(Float)) -> List(Float) {
  let n = list.length(a)

  list.range(0, n - 1)
  |> list.map(fn(k) {
    list.range(0, n - 1)
    |> list.fold(0.0, fn(acc, j) {
      let a_j = list_get(a, j, 0.0)
      let b_idx = mod(k - j + n, n)
      let b_kj = list_get(b, b_idx, 0.0)
      acc +. { a_j *. b_kj }
    })
  })
}

/// Approximate inverse for unbinding
/// For random vectors: inverse ≈ reverse (except element 0)
fn approximate_inverse(v: List(Float)) -> List(Float) {
  case v {
    [] -> []
    [first, ..rest] -> [first, ..list.reverse(rest)]
  }
}

// =============================================================================
// FFT-ACCELERATED OPERATIONS (via Nx)
// =============================================================================

/// Binding using FFT (O(n log n)) - requires Nx
/// circular_conv(a, b) = ifft(fft(a) * fft(b))
pub fn bind_fft(a: HRR, b: HRR) -> Result(HRR, HRRError) {
  case a.dim == b.dim {
    False -> Error(DimensionMismatch(expected: a.dim, got: b.dim))
    True -> {
      let result = nx_circular_conv(td(a.vector), td(b.vector))
      Ok(HRR(vector: tensor.Tensor(data: result, shape: [a.dim]), dim: a.dim))
    }
  }
}

/// Unbinding using FFT (O(n log n)) - requires Nx
pub fn unbind_fft(trace: HRR, cue: HRR) -> Result(HRR, HRRError) {
  case trace.dim == cue.dim {
    False -> Error(DimensionMismatch(expected: trace.dim, got: cue.dim))
    True -> {
      let cue_inv = approximate_inverse(td(cue.vector))
      let result = nx_circular_conv(td(trace.vector), cue_inv)
      Ok(HRR(
        vector: tensor.Tensor(data: result, shape: [trace.dim]),
        dim: trace.dim,
      ))
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn vector_norm(v: List(Float)) -> Float {
  v
  |> list.map(fn(x) { x *. x })
  |> list.fold(0.0, fn(acc, x) { acc +. x })
  |> float_sqrt
}

fn list_get(l: List(a), idx: Int, default: a) -> a {
  case list.drop(l, idx) {
    [x, ..] -> x
    [] -> default
  }
}

fn mod(a: Int, b: Int) -> Int {
  let r = a % b
  case r < 0 {
    True -> r + b
    False -> r
  }
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "rand", "normal")
fn random_gaussian() -> Float

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "erlang", "float")
fn int_to_float(i: Int) -> Float

/// FFT-based circular convolution (via Nx)
/// Falls back to naive if Nx not available
@external(erlang, "viva_hrr_fft", "circular_conv")
fn nx_circular_conv(a: List(Float), b: List(Float)) -> List(Float)
