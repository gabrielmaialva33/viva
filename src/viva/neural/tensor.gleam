//// Tensor - N-dimensional arrays for neural operations
////
//// Design: NumPy-inspired with strides for zero-copy views.
//// Uses Erlang :array for O(1) access + strides for efficient transpose/reshape.
//// Inspired by NumPy ndarray and PyTorch ATen internals.

import gleam/float
import gleam/int
import gleam/list
import gleam/result

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

/// Access list element by index (Gleam doesn't have list.at)
fn list_at_int(lst: List(Int), index: Int) -> Result(Int, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}

fn list_at_float(lst: List(Float), index: Int) -> Result(Float, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}

/// O(1) array access using Erlang :array
fn array_get(arr: ErlangArray, index: Int) -> Float {
  array_get_ffi(arr, index)
}

/// Convert list to Erlang array for O(1) access
fn list_to_array(lst: List(Float)) -> ErlangArray {
  list_to_array_ffi(lst)
}

// array_to_list removed - unused private function

/// Get array size
fn array_size(arr: ErlangArray) -> Int {
  array_size_ffi(arr)
}

// =============================================================================
// TYPES
// =============================================================================

/// Opaque type for Erlang :array
pub type ErlangArray

/// Tensor with NumPy-style strides for zero-copy views
/// - storage: contiguous data buffer (Erlang array for O(1) access)
/// - shape: dimensions [d0, d1, ..., dn]
/// - strides: bytes to skip for each dimension [s0, s1, ..., sn]
/// - offset: starting position in storage (for views/slices)
pub type Tensor {
  Tensor(data: List(Float), shape: List(Int))
  // V2: Strided tensor with O(1) access
  StridedTensor(
    storage: ErlangArray,
    shape: List(Int),
    strides: List(Int),
    offset: Int,
  )
}

/// Tensor operation errors
pub type TensorError {
  ShapeMismatch(expected: List(Int), got: List(Int))
  InvalidShape(reason: String)
  DimensionError(reason: String)
}

// =============================================================================
// AUTOMATIC BACKEND SELECTOR (Qwen3-235B recommendation)
// =============================================================================

/// Operation type for backend selection
pub type OperationType {
  /// Sequential operations - better with lists (dot, sum, reduce)
  Sequential
  /// Random access operations - better with strided (get, get2d, indexing)
  RandomAccess
  /// Matrix operations - strided for large, lists for small (matmul)
  MatrixOp
}

/// Configuration for automatic backend selection
pub type TensorConfig {
  TensorConfig(
    /// Minimum size to use strided for random access (default: 500)
    strided_threshold_random: Int,
    /// Minimum size to use strided for matmul (default: 100 = 10x10)
    strided_threshold_matmul: Int,
    /// Force strided for all operations (override)
    force_strided: Bool,
    /// Force list for all operations (override)
    force_list: Bool,
  )
}

/// Default configuration based on benchmarks
pub fn default_config() -> TensorConfig {
  TensorConfig(
    strided_threshold_random: 500,
    // 500+ elements → strided ~4x faster
    strided_threshold_matmul: 64,
    // 8x8+ matrices → strided ~4x faster
    force_strided: False,
    force_list: False,
  )
}

/// High-performance config (always strided for large tensors)
pub fn performance_config() -> TensorConfig {
  TensorConfig(
    strided_threshold_random: 100,
    strided_threshold_matmul: 32,
    force_strided: False,
    force_list: False,
  )
}

/// Memory-efficient config (prefer lists)
pub fn memory_config() -> TensorConfig {
  TensorConfig(
    strided_threshold_random: 5000,
    strided_threshold_matmul: 256,
    force_strided: False,
    force_list: False,
  )
}

/// Check if should use strided backend for given operation
pub fn should_use_strided(
  t: Tensor,
  op: OperationType,
  config: TensorConfig,
) -> Bool {
  // Override checks
  case config.force_strided, config.force_list {
    True, _ -> True
    _, True -> False
    False, False -> {
      let tensor_size = size(t)

      case op {
        // Sequential ops (dot, sum) - NEVER use strided (0.7x slower)
        Sequential -> False

        // Random access (get, get2d) - strided if large enough
        RandomAccess -> tensor_size >= config.strided_threshold_random

        // Matrix ops - strided if matrices are large enough
        MatrixOp -> {
          case t.shape {
            [rows, cols] -> rows * cols >= config.strided_threshold_matmul
            _ -> tensor_size >= config.strided_threshold_matmul
          }
        }
      }
    }
  }
}

/// Ensure tensor is in optimal format for operation
pub fn ensure_optimal(
  t: Tensor,
  op: OperationType,
  config: TensorConfig,
) -> Tensor {
  let use_strided = should_use_strided(t, op, config)

  case t, use_strided {
    // Already strided and should be strided - keep it
    StridedTensor(..), True -> t

    // Already list and should be list - keep it
    Tensor(..), False -> t

    // Need to convert to strided
    Tensor(..), True -> to_strided(t)

    // Need to convert to list
    StridedTensor(..), False -> to_contiguous(t)
  }
}

// =============================================================================
// SMART OPERATIONS (auto-selecting backend)
// =============================================================================

/// Smart dot product - auto-selects optimal backend
/// Uses list-based for sequential access (faster for dot)
pub fn dot_smart(
  a: Tensor,
  b: Tensor,
  config: TensorConfig,
) -> Result(Float, TensorError) {
  // Dot is Sequential - always use list-based (0.7x penalty with strided)
  let a_opt = ensure_optimal(a, Sequential, config)
  let b_opt = ensure_optimal(b, Sequential, config)
  dot(a_opt, b_opt)
}

/// Smart get - auto-selects optimal backend
/// Uses strided for large tensors (up to 36x faster)
/// NOTE: For repeated access, convert once with to_strided() first!
pub fn get_smart(
  t: Tensor,
  index: Int,
  _config: TensorConfig,
) -> Result(Float, TensorError) {
  // If already strided, use fast path directly (no conversion)
  case t {
    StridedTensor(..) -> get_fast(t, index)
    Tensor(..) -> {
      // For list tensors, only use strided if worth the conversion
      // (single access is NOT worth converting - use get directly)
      get(t, index)
    }
  }
}

/// Smart get2d - auto-selects optimal backend
/// Uses strided for large matrices (up to 140x faster)
/// NOTE: For repeated access, convert once with to_strided() first!
pub fn get2d_smart(
  t: Tensor,
  row: Int,
  col: Int,
  _config: TensorConfig,
) -> Result(Float, TensorError) {
  // If already strided, use fast path directly (no conversion)
  case t {
    StridedTensor(..) -> get2d_fast(t, row, col)
    Tensor(..) -> get2d(t, row, col)
  }
}

/// Prepare tensor for repeated random access
/// Call once, then use get_fast/get2d_fast for O(1) access
pub fn prepare_for_access(t: Tensor, config: TensorConfig) -> Tensor {
  case should_use_strided(t, RandomAccess, config) {
    True -> to_strided(t)
    False -> t
  }
}

/// Prepare tensor for matrix operations
/// Call once before repeated matmul operations
pub fn prepare_for_matmul(t: Tensor, config: TensorConfig) -> Tensor {
  case should_use_strided(t, MatrixOp, config) {
    True -> to_strided(t)
    False -> t
  }
}

/// Smart matmul - auto-selects optimal backend
/// Uses strided for large matrices (up to 20x faster)
pub fn matmul_smart(
  a: Tensor,
  b: Tensor,
  config: TensorConfig,
) -> Result(Tensor, TensorError) {
  let use_strided_a = should_use_strided(a, MatrixOp, config)
  let use_strided_b = should_use_strided(b, MatrixOp, config)

  case use_strided_a || use_strided_b {
    True -> {
      let a_opt = ensure_optimal(a, MatrixOp, config)
      let b_opt = ensure_optimal(b, MatrixOp, config)
      matmul_fast(a_opt, b_opt)
    }
    False -> matmul(a, b)
  }
}

/// Smart sum - uses list-based (sequential access)
pub fn sum_smart(t: Tensor, config: TensorConfig) -> Float {
  let t_opt = ensure_optimal(t, Sequential, config)
  sum(t_opt)
}

/// Smart mean - uses list-based (sequential access)
pub fn mean_smart(t: Tensor, config: TensorConfig) -> Float {
  let t_opt = ensure_optimal(t, Sequential, config)
  mean(t_opt)
}

// =============================================================================
// CONVENIENCE: Default config operations
// =============================================================================

/// Dot with default config
pub fn dot_auto(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  dot_smart(a, b, default_config())
}

/// Get with default config
pub fn get_auto(t: Tensor, index: Int) -> Result(Float, TensorError) {
  get_smart(t, index, default_config())
}

/// Get2d with default config
pub fn get2d_auto(t: Tensor, row: Int, col: Int) -> Result(Float, TensorError) {
  get2d_smart(t, row, col, default_config())
}

/// Matmul with default config
pub fn matmul_auto(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  matmul_smart(a, b, default_config())
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create tensor of zeros
pub fn zeros(shape: List(Int)) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  Tensor(data: list.repeat(0.0, size), shape: shape)
}

/// Create tensor of ones
pub fn ones(shape: List(Int)) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  Tensor(data: list.repeat(1.0, size), shape: shape)
}

/// Create tensor filled with value
pub fn fill(shape: List(Int), value: Float) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  Tensor(data: list.repeat(value, size), shape: shape)
}

/// Create tensor from list (1D)
pub fn from_list(data: List(Float)) -> Tensor {
  Tensor(data: data, shape: [list.length(data)])
}

/// Create 2D tensor (matrix) from list of lists
pub fn from_list2d(rows: List(List(Float))) -> Result(Tensor, TensorError) {
  case rows {
    [] -> Ok(Tensor(data: [], shape: [0, 0]))
    [first, ..rest] -> {
      let cols = list.length(first)
      let valid = list.all(rest, fn(row) { list.length(row) == cols })

      case valid {
        False -> Error(InvalidShape("Rows have different lengths"))
        True -> {
          let data = list.flatten(rows)
          let num_rows = list.length(rows)
          Ok(Tensor(data: data, shape: [num_rows, cols]))
        }
      }
    }
  }
}

/// Create vector (1D tensor)
pub fn vector(data: List(Float)) -> Tensor {
  from_list(data)
}

/// Create matrix (2D tensor) with explicit dimensions
pub fn matrix(
  rows: Int,
  cols: Int,
  data: List(Float),
) -> Result(Tensor, TensorError) {
  let expected_size = rows * cols
  let actual_size = list.length(data)

  case expected_size == actual_size {
    True -> Ok(Tensor(data: data, shape: [rows, cols]))
    False ->
      Error(InvalidShape(
        "Expected "
        <> int.to_string(expected_size)
        <> " elements, got "
        <> int.to_string(actual_size),
      ))
  }
}

// =============================================================================
// PROPERTIES
// =============================================================================

/// Extract data as list from any tensor variant
fn get_data(t: Tensor) -> List(Float) {
  case t {
    Tensor(data, _) -> data
    StridedTensor(storage, shape, strides, offset) -> {
      // Materialize strided data to list
      let total_size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
      list.range(0, total_size - 1)
      |> list.map(fn(flat_idx) {
        let indices = flat_to_multi(flat_idx, shape)
        let idx =
          list.zip(indices, strides)
          |> list.fold(offset, fn(acc, pair) {
            let #(i, s) = pair
            acc + i * s
          })
        array_get(storage, idx)
      })
    }
  }
}

/// Total number of elements
pub fn size(t: Tensor) -> Int {
  case t {
    Tensor(data, _) -> list.length(data)
    StridedTensor(_, shape, _, _) ->
      list.fold(shape, 1, fn(acc, dim) { acc * dim })
  }
}

/// Number of dimensions (rank)
pub fn rank(t: Tensor) -> Int {
  list.length(t.shape)
}

/// Specific dimension
pub fn dim(t: Tensor, axis: Int) -> Result(Int, TensorError) {
  list_at_int(t.shape, axis)
  |> result.map_error(fn(_) {
    DimensionError("Axis " <> int.to_string(axis) <> " out of bounds")
  })
}

/// Return number of rows (for matrices)
pub fn rows(t: Tensor) -> Int {
  case t.shape {
    [r, ..] -> r
    [] -> 0
  }
}

/// Return number of columns (for matrices)
pub fn cols(t: Tensor) -> Int {
  case t.shape {
    [_, c, ..] -> c
    [n] -> n
    [] -> 0
  }
}

// =============================================================================
// ELEMENT ACCESS
// =============================================================================

/// Access element by linear index
pub fn get(t: Tensor, index: Int) -> Result(Float, TensorError) {
  case t {
    Tensor(data, _) ->
      list_at_float(data, index)
      |> result.map_error(fn(_) {
        DimensionError("Index " <> int.to_string(index) <> " out of bounds")
      })
    StridedTensor(storage, shape, strides, offset) -> {
      let indices = flat_to_multi(index, shape)
      let flat_idx =
        list.zip(indices, strides)
        |> list.fold(offset, fn(acc, pair) {
          let #(i, s) = pair
          acc + i * s
        })
      Ok(array_get(storage, flat_idx))
    }
  }
}

/// Access 2D element
pub fn get2d(t: Tensor, row: Int, col: Int) -> Result(Float, TensorError) {
  case t.shape {
    [_rows, num_cols] -> {
      let index = row * num_cols + col
      get(t, index)
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

/// Get matrix row as vector
pub fn get_row(t: Tensor, row_idx: Int) -> Result(Tensor, TensorError) {
  case t.shape {
    [num_rows, num_cols] -> {
      case row_idx >= 0 && row_idx < num_rows {
        True -> {
          let data = get_data(t)
          let start = row_idx * num_cols
          let row_data =
            data
            |> list.drop(start)
            |> list.take(num_cols)
          Ok(from_list(row_data))
        }
        False -> Error(DimensionError("Row index out of bounds"))
      }
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

/// Get matrix column as vector
pub fn get_col(t: Tensor, col_idx: Int) -> Result(Tensor, TensorError) {
  case t.shape {
    [num_rows, num_cols] -> {
      case col_idx >= 0 && col_idx < num_cols {
        True -> {
          let col_data =
            list.range(0, num_rows - 1)
            |> list.filter_map(fn(row) { get2d(t, row, col_idx) })
          Ok(from_list(col_data))
        }
        False -> Error(DimensionError("Column index out of bounds"))
      }
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================

/// Apply function to each element
pub fn map(t: Tensor, f: fn(Float) -> Float) -> Tensor {
  let data = get_data(t)
  Tensor(data: list.map(data, f), shape: t.shape)
}

/// Apply function with index
pub fn map_indexed(t: Tensor, f: fn(Float, Int) -> Float) -> Tensor {
  let data = get_data(t)
  Tensor(data: list.index_map(data, fn(x, i) { f(x, i) }), shape: t.shape)
}

/// Element-wise addition
pub fn add(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x +. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise subtraction
pub fn sub(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x -. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise multiplication (Hadamard)
pub fn mul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x *. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise division
pub fn div(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x /. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Scale by constant
pub fn scale(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x *. s })
}

/// Add constant
pub fn add_scalar(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x +. s })
}

/// Negation
pub fn negate(t: Tensor) -> Tensor {
  scale(t, -1.0)
}

// =============================================================================
// REDUCTION OPERATIONS
// =============================================================================

/// Sum all elements
pub fn sum(t: Tensor) -> Float {
  let data = get_data(t)
  list.fold(data, 0.0, fn(acc, x) { acc +. x })
}

/// Product of all elements
pub fn product(t: Tensor) -> Float {
  let data = get_data(t)
  list.fold(data, 1.0, fn(acc, x) { acc *. x })
}

/// Mean
pub fn mean(t: Tensor) -> Float {
  let s = sum(t)
  let n = int.to_float(size(t))
  case n >. 0.0 {
    True -> s /. n
    False -> 0.0
  }
}

/// Maximum value
pub fn max(t: Tensor) -> Float {
  let data = get_data(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.max(acc, x) })
  }
}

/// Minimum value
pub fn min(t: Tensor) -> Float {
  let data = get_data(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.min(acc, x) })
  }
}

/// Argmax - index of largest element
pub fn argmax(t: Tensor) -> Int {
  let data = get_data(t)
  case data {
    [] -> 0
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x >. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      idx
    }
  }
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

/// Dot product of two vectors
pub fn dot(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case rank(a) == 1 && rank(b) == 1 && size(a) == size(b) {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let products = list.map2(a_data, b_data, fn(x, y) { x *. y })
      Ok(list.fold(products, 0.0, fn(acc, x) { acc +. x }))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Matrix-vector multiplication: [m, n] @ [n] -> [m]
pub fn matmul_vec(mat: Tensor, vec: Tensor) -> Result(Tensor, TensorError) {
  case mat.shape, vec.shape {
    [m, n], [vec_n] if n == vec_n -> {
      let mat_data = get_data(mat)
      let vec_data = get_data(vec)
      let result_data =
        list.range(0, m - 1)
        |> list.map(fn(row_idx) {
          let start = row_idx * n
          let row =
            mat_data
            |> list.drop(start)
            |> list.take(n)
          list.map2(row, vec_data, fn(a, b) { a *. b })
          |> list.fold(0.0, fn(acc, x) { acc +. x })
        })
      Ok(Tensor(data: result_data, shape: [m]))
    }
    [_m, n], [vec_n] -> Error(ShapeMismatch(expected: [n], got: [vec_n]))
    _, _ -> Error(DimensionError("Expected matrix and vector"))
  }
}

/// Matrix-matrix multiplication: [m, n] @ [n, p] -> [m, p]
pub fn matmul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape, b.shape {
    [m, n], [n2, p] if n == n2 -> {
      let result_data =
        list.range(0, m - 1)
        |> list.flat_map(fn(i) {
          list.range(0, p - 1)
          |> list.map(fn(j) {
            // Compute element [i, j]
            list.range(0, n - 1)
            |> list.fold(0.0, fn(acc, k) {
              let a_ik = case get2d(a, i, k) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
              let b_kj = case get2d(b, k, j) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
              acc +. a_ik *. b_kj
            })
          })
        })
      Ok(Tensor(data: result_data, shape: [m, p]))
    }
    [_m, n], [n2, _p] -> Error(ShapeMismatch(expected: [n, -1], got: [n2, -1]))
    _, _ -> Error(DimensionError("Expected two matrices"))
  }
}

/// Matrix transpose
pub fn transpose(t: Tensor) -> Result(Tensor, TensorError) {
  case t.shape {
    [m, n] -> {
      let result_data =
        list.range(0, n - 1)
        |> list.flat_map(fn(j) {
          list.range(0, m - 1)
          |> list.filter_map(fn(i) { get2d(t, i, j) })
        })
      Ok(Tensor(data: result_data, shape: [n, m]))
    }
    _ -> Error(DimensionError("Transpose requires 2D tensor"))
  }
}

/// Outer product: [m] @ [n] -> [m, n]
pub fn outer(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case rank(a) == 1 && rank(b) == 1 {
    True -> {
      let m = size(a)
      let n = size(b)
      let a_data = get_data(a)
      let b_data = get_data(b)
      let result_data =
        list.flat_map(a_data, fn(ai) { list.map(b_data, fn(bj) { ai *. bj }) })
      Ok(Tensor(data: result_data, shape: [m, n]))
    }
    False -> Error(DimensionError("Outer product requires two vectors"))
  }
}

// =============================================================================
// UTILITY
// =============================================================================

/// Convert to list
pub fn to_list(t: Tensor) -> List(Float) {
  get_data(t)
}

/// Convert matrix to list of lists
pub fn to_list2d(t: Tensor) -> Result(List(List(Float)), TensorError) {
  case t.shape {
    [num_rows, num_cols] -> {
      let data = get_data(t)
      let rows =
        list.range(0, num_rows - 1)
        |> list.map(fn(i) {
          let start = i * num_cols
          data
          |> list.drop(start)
          |> list.take(num_cols)
        })
      Ok(rows)
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

/// Clone tensor
pub fn clone(t: Tensor) -> Tensor {
  let data = get_data(t)
  Tensor(data: data, shape: t.shape)
}

/// Reshape tensor
pub fn reshape(t: Tensor, new_shape: List(Int)) -> Result(Tensor, TensorError) {
  let old_size = size(t)
  let new_size = list.fold(new_shape, 1, fn(acc, dim) { acc * dim })

  case old_size == new_size {
    True -> {
      let data = get_data(t)
      Ok(Tensor(data: data, shape: new_shape))
    }
    False ->
      Error(InvalidShape(
        "Cannot reshape: size mismatch ("
        <> int.to_string(old_size)
        <> " vs "
        <> int.to_string(new_size)
        <> ")",
      ))
  }
}

/// Flatten to 1D
pub fn flatten(t: Tensor) -> Tensor {
  let data = get_data(t)
  Tensor(data: data, shape: [size(t)])
}

/// Concatenate vectors
pub fn concat(tensors: List(Tensor)) -> Tensor {
  let data = list.flat_map(tensors, fn(t) { get_data(t) })
  from_list(data)
}

/// L2 norm
pub fn norm(t: Tensor) -> Float {
  let data = get_data(t)
  let sum_sq = list.fold(data, 0.0, fn(acc, x) { acc +. x *. x })
  float_sqrt(sum_sq)
}

/// Normalize to unit length
pub fn normalize(t: Tensor) -> Tensor {
  let n = norm(t)
  case n >. 0.0001 {
    True -> scale(t, 1.0 /. n)
    False -> t
  }
}

/// Clamp values
pub fn clamp(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  map(t, fn(x) { float.min(float.max(x, min_val), max_val) })
}

// =============================================================================
// RANDOM (using Erlang externals)
// =============================================================================

/// Tensor with uniform random values [0, 1)
pub fn random_uniform(shape: List(Int)) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let data =
    list.range(1, size)
    |> list.map(fn(_) { random_float() })
  Tensor(data: data, shape: shape)
}

/// Tensor with normal random values (approx via Box-Muller)
pub fn random_normal(shape: List(Int), mean: Float, std: Float) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let data =
    list.range(1, size)
    |> list.map(fn(_) {
      // Approximate Box-Muller transform
      let u1 = float.max(random_float(), 0.0001)
      let u2 = random_float()
      let z =
        float_sqrt(-2.0 *. float_log(u1))
        *. float_cos(2.0 *. 3.14159265359 *. u2)
      mean +. z *. std
    })
  Tensor(data: data, shape: shape)
}

/// Xavier initialization for weights
pub fn xavier_init(fan_in: Int, fan_out: Int) -> Tensor {
  let limit = float_sqrt(6.0 /. int.to_float(fan_in + fan_out))
  let data =
    list.range(1, fan_in * fan_out)
    |> list.map(fn(_) {
      let r = random_float()
      r *. 2.0 *. limit -. limit
    })
  Tensor(data: data, shape: [fan_in, fan_out])
}

/// He initialization (for ReLU)
pub fn he_init(fan_in: Int, fan_out: Int) -> Tensor {
  let std = float_sqrt(2.0 /. int.to_float(fan_in))
  random_normal([fan_in, fan_out], 0.0, std)
}

// =============================================================================
// EXTERNAL FUNCTIONS
// =============================================================================

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "log")
fn float_log(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_float() -> Float

// =============================================================================
// ERLANG ARRAY FFI - O(1) ACCESS (NumPy-style)
// =============================================================================

@external(erlang, "viva_tensor_ffi", "list_to_array")
fn list_to_array_ffi(lst: List(Float)) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_get")
fn array_get_ffi(arr: ErlangArray, index: Int) -> Float

@external(erlang, "viva_tensor_ffi", "array_size")
fn array_size_ffi(arr: ErlangArray) -> Int

// =============================================================================
// SIMD NIFs - AVX/SSE ACCELERATION
// =============================================================================

/// SIMD-accelerated dot product (AVX on x86_64)
@external(erlang, "viva_simd_nif", "simd_dot")
pub fn simd_dot(a: List(Float), b: List(Float)) -> Float

/// SIMD-accelerated element-wise multiply
@external(erlang, "viva_simd_nif", "simd_mul")
pub fn simd_mul(a: List(Float), b: List(Float)) -> List(Float)

/// SIMD-accelerated matrix multiply
/// A: MxK, B: KxN -> Result: MxN
@external(erlang, "viva_simd_nif", "simd_matmul")
pub fn simd_matmul_raw(
  a: List(Float),
  b: List(Float),
  m: Int,
  k: Int,
  n: Int,
) -> List(List(Float))

/// SIMD-accelerated sum
@external(erlang, "viva_simd_nif", "simd_sum")
pub fn simd_sum(a: List(Float)) -> Float

/// SIMD-accelerated scale
@external(erlang, "viva_simd_nif", "simd_scale")
pub fn simd_scale_raw(a: List(Float), scalar: Float) -> List(Float)

/// Check if SIMD is available
@external(erlang, "viva_simd_nif", "is_simd_available")
pub fn is_simd_available() -> Bool

/// SIMD-accelerated dot product for tensors
pub fn dot_simd(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case size(a) == size(b) {
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
    True -> Ok(simd_dot(to_list(a), to_list(b)))
  }
}

/// SIMD-accelerated tensor scaling
pub fn scale_simd(t: Tensor, scalar: Float) -> Tensor {
  let data = simd_scale_raw(to_list(t), scalar)
  Tensor(data: data, shape: t.shape)
}

/// SIMD-accelerated tensor sum
pub fn sum_simd(t: Tensor) -> Float {
  simd_sum(to_list(t))
}

// =============================================================================
// BROADCASTING (inspired by Nx)
// =============================================================================

/// Check if two shapes can be broadcast together
pub fn can_broadcast(a: List(Int), b: List(Int)) -> Bool {
  let #(longer, shorter) = case list.length(a) >= list.length(b) {
    True -> #(a, b)
    False -> #(b, a)
  }

  // Pad shorter with 1s on the left
  let diff = list.length(longer) - list.length(shorter)
  let padded = list.append(list.repeat(1, diff), shorter)

  // Check each dimension: must be equal or one of them is 1
  list.zip(longer, padded)
  |> list.all(fn(pair) {
    let #(dim_a, dim_b) = pair
    dim_a == dim_b || dim_a == 1 || dim_b == 1
  })
}

/// Compute broadcast shape
pub fn broadcast_shape(
  a: List(Int),
  b: List(Int),
) -> Result(List(Int), TensorError) {
  case can_broadcast(a, b) {
    False -> Error(ShapeMismatch(expected: a, got: b))
    True -> {
      let max_rank = int.max(list.length(a), list.length(b))
      let diff_a = max_rank - list.length(a)
      let diff_b = max_rank - list.length(b)
      let padded_a = list.append(list.repeat(1, diff_a), a)
      let padded_b = list.append(list.repeat(1, diff_b), b)

      let result_shape =
        list.zip(padded_a, padded_b)
        |> list.map(fn(pair) {
          let #(dim_a, dim_b) = pair
          int.max(dim_a, dim_b)
        })

      Ok(result_shape)
    }
  }
}

/// Broadcast tensor to target shape
pub fn broadcast_to(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  case can_broadcast(t.shape, target_shape) {
    False -> Error(ShapeMismatch(expected: target_shape, got: t.shape))
    True -> {
      // Simple case: shapes are equal
      case t.shape == target_shape {
        True -> Ok(t)
        False -> {
          // Compute strides for broadcasting
          let data = broadcast_data(t, target_shape)
          Ok(Tensor(data: data, shape: target_shape))
        }
      }
    }
  }
}

/// Internal: broadcast data to target shape
fn broadcast_data(t: Tensor, target_shape: List(Int)) -> List(Float) {
  let target_size = list.fold(target_shape, 1, fn(acc, dim) { acc * dim })
  let src_shape = t.shape
  let src_rank = list.length(src_shape)
  let target_rank = list.length(target_shape)
  let data = get_data(t)

  // Pad source shape with 1s
  let diff = target_rank - src_rank
  let padded_shape = list.append(list.repeat(1, diff), src_shape)

  // Generate indices and map to source
  list.range(0, target_size - 1)
  |> list.map(fn(flat_idx) {
    // Convert flat index to multi-dimensional index
    let target_indices = flat_to_multi(flat_idx, target_shape)

    // Map to source indices (clamp to 0 for broadcast dimensions)
    let src_indices =
      list.zip(target_indices, padded_shape)
      |> list.map(fn(pair) {
        let #(idx, dim) = pair
        case dim == 1 {
          True -> 0
          False -> idx
        }
      })
      |> list.drop(diff)
    // Remove padding

    // Convert back to flat index in source
    let src_flat = multi_to_flat(src_indices, src_shape)
    case list_at_float(data, src_flat) {
      Ok(v) -> v
      Error(_) -> 0.0
    }
  })
}

/// Convert flat index to multi-dimensional indices
fn flat_to_multi(flat: Int, shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(indices, _) =
    list.fold(reversed, #([], flat), fn(acc, dim) {
      let #(idxs, remaining) = acc
      let idx = remaining % dim
      let next = remaining / dim
      #([idx, ..idxs], next)
    })
  indices
}

/// Convert multi-dimensional indices to flat index
fn multi_to_flat(indices: List(Int), shape: List(Int)) -> Int {
  let strides = compute_strides(shape)

  list.zip(indices, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(idx, stride) = pair
    acc + idx * stride
  })
}

/// Compute strides for a shape
fn compute_strides(shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(strides, _) =
    list.fold(reversed, #([], 1), fn(acc, dim) {
      let #(s, running) = acc
      #([running, ..s], running * dim)
    })
  strides
}

/// Element-wise addition with broadcasting
pub fn add_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(a.shape, b.shape))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  add(a_bc, b_bc)
}

/// Element-wise multiplication with broadcasting
pub fn mul_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(a.shape, b.shape))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  mul(a_bc, b_bc)
}

// =============================================================================
// SHAPE MANIPULATION (inspired by Nx/PyTorch)
// =============================================================================

/// Remove dimensions of size 1
pub fn squeeze(t: Tensor) -> Tensor {
  let data = get_data(t)
  let new_shape = list.filter(t.shape, fn(dim) { dim != 1 })
  let final_shape = case new_shape {
    [] -> [1]
    // Scalar becomes 1D
    _ -> new_shape
  }
  Tensor(data: data, shape: final_shape)
}

/// Remove dimension at specific axis if it's 1
pub fn squeeze_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  case list_at_int(t.shape, axis) {
    Error(_) -> Error(DimensionError("Axis out of bounds"))
    Ok(dim) -> {
      case dim == 1 {
        False -> Error(InvalidShape("Dimension at axis is not 1"))
        True -> {
          let data = get_data(t)
          let new_shape =
            t.shape
            |> list.index_map(fn(d, i) { #(d, i) })
            |> list.filter(fn(pair) { pair.1 != axis })
            |> list.map(fn(pair) { pair.0 })
          Ok(Tensor(data: data, shape: new_shape))
        }
      }
    }
  }
}

/// Add dimension of size 1 at specified axis
pub fn unsqueeze(t: Tensor, axis: Int) -> Tensor {
  let data = get_data(t)
  let rnk = list.length(t.shape)
  let insert_at = case axis < 0 {
    True -> rnk + axis + 1
    False -> axis
  }

  let #(before, after) = list.split(t.shape, insert_at)
  let new_shape = list.flatten([before, [1], after])
  Tensor(data: data, shape: new_shape)
}

/// Expand tensor to add batch dimension
pub fn expand_dims(t: Tensor, axis: Int) -> Tensor {
  unsqueeze(t, axis)
}

// =============================================================================
// STATISTICS (inspired by Nx)
// =============================================================================

/// Variance of all elements
pub fn variance(t: Tensor) -> Float {
  let data = get_data(t)
  let m = mean(t)
  let squared_diffs =
    list.map(data, fn(x) {
      let diff = x -. m
      diff *. diff
    })
  let n = int.to_float(size(t))
  case n >. 0.0 {
    True -> list.fold(squared_diffs, 0.0, fn(acc, x) { acc +. x }) /. n
    False -> 0.0
  }
}

/// Standard deviation
pub fn std(t: Tensor) -> Float {
  float_sqrt(variance(t))
}

/// Argmin - index of smallest element
pub fn argmin(t: Tensor) -> Int {
  let data = get_data(t)
  case data {
    [] -> 0
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x <. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      idx
    }
  }
}

/// Sum along axis
pub fn sum_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  case t.shape {
    [_n] if axis == 0 -> Ok(Tensor(data: [sum(t)], shape: [1]))
    [rows, cols] if axis == 0 -> {
      // Sum columns
      let col_sums =
        list.range(0, cols - 1)
        |> list.map(fn(col) {
          list.range(0, rows - 1)
          |> list.fold(0.0, fn(acc, row) {
            case get2d(t, row, col) {
              Ok(v) -> acc +. v
              Error(_) -> acc
            }
          })
        })
      Ok(Tensor(data: col_sums, shape: [cols]))
    }
    [rows, cols] if axis == 1 -> {
      // Sum rows
      let row_sums =
        list.range(0, rows - 1)
        |> list.map(fn(row) {
          list.range(0, cols - 1)
          |> list.fold(0.0, fn(acc, col) {
            case get2d(t, row, col) {
              Ok(v) -> acc +. v
              Error(_) -> acc
            }
          })
        })
      Ok(Tensor(data: row_sums, shape: [rows]))
    }
    _ -> Error(DimensionError("sum_axis only supports 1D and 2D tensors"))
  }
}

/// Mean along axis
pub fn mean_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  case t.shape {
    [_n] if axis == 0 -> Ok(Tensor(data: [mean(t)], shape: [1]))
    [rows, _cols] if axis == 0 -> {
      use sums <- result.try(sum_axis(t, 0))
      Ok(scale(sums, 1.0 /. int.to_float(rows)))
    }
    [_rows, cols] if axis == 1 -> {
      use sums <- result.try(sum_axis(t, 1))
      Ok(scale(sums, 1.0 /. int.to_float(cols)))
    }
    _ -> Error(DimensionError("mean_axis only supports 1D and 2D tensors"))
  }
}

// =============================================================================
// STACKING (inspired by Nx/NumPy)
// =============================================================================

/// Stack tensors along new axis
pub fn stack(tensors: List(Tensor), axis: Int) -> Result(Tensor, TensorError) {
  case tensors {
    [] -> Error(InvalidShape("Cannot stack empty list"))
    [first, ..rest] -> {
      // Check all shapes are equal
      let all_same = list.all(rest, fn(t) { t.shape == first.shape })
      case all_same {
        False ->
          Error(InvalidShape("All tensors must have same shape to stack"))
        True -> {
          let n = list.length(tensors)
          let new_shape = case axis {
            0 -> [n, ..first.shape]
            _ -> {
              // Insert n at axis position
              let #(before, after) = list.split(first.shape, axis)
              list.flatten([before, [n], after])
            }
          }
          let data = list.flat_map(tensors, fn(t) { get_data(t) })
          Ok(Tensor(data: data, shape: new_shape))
        }
      }
    }
  }
}

/// Concatenate tensors along existing axis
pub fn concat_axis(
  tensors: List(Tensor),
  axis: Int,
) -> Result(Tensor, TensorError) {
  case tensors {
    [] -> Error(InvalidShape("Cannot concat empty list"))
    [first] -> Ok(first)
    [first, ..rest] -> {
      // For axis 0 concatenation of 1D tensors
      case axis == 0 && list.length(first.shape) == 1 {
        True -> {
          let data = list.flat_map(tensors, fn(t) { get_data(t) })
          Ok(from_list(data))
        }
        False -> {
          // For 2D tensors along axis 0 (vertical stack)
          case first.shape {
            [_rows, cols] if axis == 0 -> {
              let all_same_cols =
                list.all(rest, fn(t) {
                  case t.shape {
                    [_, c] -> c == cols
                    _ -> False
                  }
                })
              case all_same_cols {
                False -> Error(InvalidShape("Column dimensions must match"))
                True -> {
                  let total_rows =
                    list.fold(tensors, 0, fn(acc, t) {
                      case t.shape {
                        [r, _] -> acc + r
                        _ -> acc
                      }
                    })
                  let data = list.flat_map(tensors, fn(t) { get_data(t) })
                  Ok(Tensor(data: data, shape: [total_rows, cols]))
                }
              }
            }
            _ ->
              Error(DimensionError(
                "concat_axis: unsupported shape/axis combination",
              ))
          }
        }
      }
    }
  }
}

// =============================================================================
// SLICING (inspired by Nx)
// =============================================================================

/// Slice tensor: get subset of data
pub fn slice(
  t: Tensor,
  starts: List(Int),
  lengths: List(Int),
) -> Result(Tensor, TensorError) {
  let tdata = get_data(t)
  case t.shape {
    [_n] -> {
      // 1D slice
      case starts, lengths {
        [start], [len] -> {
          let data =
            tdata
            |> list.drop(start)
            |> list.take(len)
          Ok(Tensor(data: data, shape: [len]))
        }
        _, _ -> Error(InvalidShape("Slice params don't match tensor rank"))
      }
    }
    [_rows, cols] -> {
      // 2D slice
      case starts, lengths {
        [row_start, col_start], [row_len, col_len] -> {
          let data =
            list.range(row_start, row_start + row_len - 1)
            |> list.flat_map(fn(r) {
              let row_offset = r * cols
              list.range(col_start, col_start + col_len - 1)
              |> list.filter_map(fn(c) { list_at_float(tdata, row_offset + c) })
            })
          Ok(Tensor(data: data, shape: [row_len, col_len]))
        }
        _, _ -> Error(InvalidShape("Slice params don't match tensor rank"))
      }
    }
    _ -> Error(DimensionError("Slice only supports 1D and 2D tensors"))
  }
}

/// Get first n elements along first axis
pub fn take_first(t: Tensor, n: Int) -> Tensor {
  let tdata = get_data(t)
  case t.shape {
    [_size] -> {
      let data = list.take(tdata, n)
      Tensor(data: data, shape: [n])
    }
    [_rows, cols] -> {
      let data = list.take(tdata, n * cols)
      Tensor(data: data, shape: [n, cols])
    }
    _ -> t
  }
}

/// Get last n elements along first axis
pub fn take_last(t: Tensor, n: Int) -> Tensor {
  let tdata = get_data(t)
  case t.shape {
    [_size] -> {
      let data =
        tdata
        |> list.reverse
        |> list.take(n)
        |> list.reverse
      Tensor(data: data, shape: [n])
    }
    [rows, cols] -> {
      let skip = { rows - n } * cols
      let data = list.drop(tdata, skip)
      Tensor(data: data, shape: [n, cols])
    }
    _ -> t
  }
}

// =============================================================================
// STRIDED TENSOR - NumPy-style O(1) operations
// The magic: transpose/reshape are ZERO-COPY (just change metadata)
// =============================================================================

/// Convert regular tensor to strided (O(n) once, then O(1) access)
pub fn to_strided(t: Tensor) -> Tensor {
  case t {
    StridedTensor(_, _, _, _) -> t
    Tensor(data, shape) -> {
      let storage = list_to_array(data)
      let strides = compute_strides(shape)
      StridedTensor(storage: storage, shape: shape, strides: strides, offset: 0)
    }
  }
}

/// Convert strided tensor back to regular (materializes the view)
pub fn to_contiguous(t: Tensor) -> Tensor {
  case t {
    Tensor(_, _) -> t
    StridedTensor(storage, shape, strides, offset) -> {
      let total_size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
      let data =
        list.range(0, total_size - 1)
        |> list.map(fn(flat_idx) {
          let indices = flat_to_multi(flat_idx, shape)
          strided_index_get(storage, offset, strides, indices)
        })
      Tensor(data: data, shape: shape)
    }
  }
}

/// Get element from strided tensor - O(1)!
fn strided_index_get(
  storage: ErlangArray,
  offset: Int,
  strides: List(Int),
  indices: List(Int),
) -> Float {
  let flat_idx =
    list.zip(indices, strides)
    |> list.fold(offset, fn(acc, pair) {
      let #(idx, stride) = pair
      acc + idx * stride
    })
  array_get(storage, flat_idx)
}

/// ZERO-COPY TRANSPOSE - NumPy's killer feature!
/// Just swap strides and shape, no data movement
pub fn transpose_strided(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    Tensor(_, shape) -> {
      case shape {
        [_m, _n] -> {
          // Convert to strided first, then transpose
          let strided = to_strided(t)
          transpose_strided(strided)
        }
        _ -> Error(DimensionError("Transpose requires 2D tensor"))
      }
    }
    StridedTensor(storage, shape, strides, offset) -> {
      case shape, strides {
        [m, n], [s0, s1] -> {
          // THE MAGIC: just swap strides and shape!
          Ok(StridedTensor(
            storage: storage,
            shape: [n, m],
            strides: [s1, s0],
            offset: offset,
          ))
        }
        _, _ -> Error(DimensionError("Transpose requires 2D tensor"))
      }
    }
  }
}

/// ZERO-COPY RESHAPE - another killer feature!
/// Only works if tensor is contiguous
pub fn reshape_strided(
  t: Tensor,
  new_shape: List(Int),
) -> Result(Tensor, TensorError) {
  let old_size = case t {
    Tensor(data, _) -> list.length(data)
    StridedTensor(storage, _, _, _) -> array_size(storage)
  }
  let new_size = list.fold(new_shape, 1, fn(acc, dim) { acc * dim })

  case old_size == new_size {
    False ->
      Error(InvalidShape(
        "Cannot reshape: size mismatch ("
        <> int.to_string(old_size)
        <> " vs "
        <> int.to_string(new_size)
        <> ")",
      ))
    True -> {
      case t {
        Tensor(data, _) -> {
          let storage = list_to_array(data)
          let strides = compute_strides(new_shape)
          Ok(StridedTensor(
            storage: storage,
            shape: new_shape,
            strides: strides,
            offset: 0,
          ))
        }
        StridedTensor(storage, _, _, offset) -> {
          let strides = compute_strides(new_shape)
          Ok(StridedTensor(
            storage: storage,
            shape: new_shape,
            strides: strides,
            offset: offset,
          ))
        }
      }
    }
  }
}

/// Fast dot product using Erlang FFI - O(n) but optimized
pub fn dot_fast(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case rank(a) == 1 && rank(b) == 1 && size(a) == size(b) {
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
    True -> {
      // Convert to arrays and use optimized FFI
      let arr_a = case a {
        Tensor(data, _) -> list_to_array(data)
        StridedTensor(storage, _, _, _) -> storage
      }
      let arr_b = case b {
        Tensor(data, _) -> list_to_array(data)
        StridedTensor(storage, _, _, _) -> storage
      }
      Ok(array_dot_ffi(arr_a, arr_b))
    }
  }
}

/// Fast matrix multiplication using Erlang FFI
pub fn matmul_fast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape, b.shape {
    [m, n], [n2, p] if n == n2 -> {
      // Convert to arrays
      let arr_a = case a {
        Tensor(data, _) -> list_to_array(data)
        StridedTensor(storage, _, _, _) -> storage
      }
      let arr_b = case b {
        Tensor(data, _) -> list_to_array(data)
        StridedTensor(storage, _, _, _) -> storage
      }
      // Use optimized FFI matmul
      let result_arr = array_matmul_ffi(arr_a, arr_b, m, p, n)
      let strides = compute_strides([m, p])
      Ok(StridedTensor(
        storage: result_arr,
        shape: [m, p],
        strides: strides,
        offset: 0,
      ))
    }
    [_m, n], [n2, _p] -> Error(ShapeMismatch(expected: [n, -1], got: [n2, -1]))
    _, _ -> Error(DimensionError("Expected two matrices"))
  }
}

/// Get element with O(1) access for StridedTensor
pub fn get_fast(t: Tensor, index: Int) -> Result(Float, TensorError) {
  case t {
    Tensor(data, _) ->
      list_at_float(data, index)
      |> result.map_error(fn(_) {
        DimensionError("Index " <> int.to_string(index) <> " out of bounds")
      })
    StridedTensor(storage, shape, strides, offset) -> {
      let indices = flat_to_multi(index, shape)
      let flat_idx =
        list.zip(indices, strides)
        |> list.fold(offset, fn(acc, pair) {
          let #(idx, stride) = pair
          acc + idx * stride
        })
      Ok(array_get(storage, flat_idx))
    }
  }
}

/// Get 2D element with O(1) access
pub fn get2d_fast(t: Tensor, row: Int, col: Int) -> Result(Float, TensorError) {
  case t {
    Tensor(_, _) -> get2d(t, row, col)
    StridedTensor(storage, shape, strides, offset) -> {
      case shape, strides {
        [_rows, _cols], [s0, s1] -> {
          let flat_idx = offset + row * s0 + col * s1
          Ok(array_get(storage, flat_idx))
        }
        _, _ -> Error(DimensionError("Tensor is not 2D"))
      }
    }
  }
}

/// Create strided tensor directly (for advanced users)
pub fn strided(
  data: List(Float),
  shape: List(Int),
  strides: List(Int),
  offset: Int,
) -> Tensor {
  let storage = list_to_array(data)
  StridedTensor(
    storage: storage,
    shape: shape,
    strides: strides,
    offset: offset,
  )
}

/// Create strided tensor from existing storage (for views)
pub fn view(
  storage: ErlangArray,
  shape: List(Int),
  strides: List(Int),
  offset: Int,
) -> Tensor {
  StridedTensor(
    storage: storage,
    shape: shape,
    strides: strides,
    offset: offset,
  )
}

/// Check if tensor is contiguous in memory
pub fn is_contiguous(t: Tensor) -> Bool {
  case t {
    Tensor(_, _) -> True
    StridedTensor(_, shape, strides, _) -> {
      let expected_strides = compute_strides(shape)
      strides == expected_strides
    }
  }
}

/// Slice with zero-copy (returns view into same storage)
pub fn slice_strided(
  t: Tensor,
  starts: List(Int),
  lengths: List(Int),
) -> Result(Tensor, TensorError) {
  case t {
    Tensor(_, _) -> {
      // Convert to strided first
      let strided = to_strided(t)
      slice_strided(strided, starts, lengths)
    }
    StridedTensor(storage, shape, strides, offset) -> {
      // Validate dimensions match
      case
        list.length(starts) == list.length(shape)
        && list.length(lengths) == list.length(shape)
      {
        False -> Error(InvalidShape("Slice params don't match tensor rank"))
        True -> {
          // Compute new offset (move to start position)
          let new_offset =
            list.zip(starts, strides)
            |> list.fold(offset, fn(acc, pair) {
              let #(start, stride) = pair
              acc + start * stride
            })
          // Strides stay the same, just change shape and offset!
          Ok(StridedTensor(
            storage: storage,
            shape: lengths,
            strides: strides,
            offset: new_offset,
          ))
        }
      }
    }
  }
}

// =============================================================================
// ADDITIONAL FFI DECLARATIONS
// =============================================================================

@external(erlang, "viva_tensor_ffi", "array_dot")
fn array_dot_ffi(a: ErlangArray, b: ErlangArray) -> Float

@external(erlang, "viva_tensor_ffi", "array_matmul")
fn array_matmul_ffi(
  a: ErlangArray,
  b: ErlangArray,
  m: Int,
  n: Int,
  k: Int,
) -> ErlangArray

// Unused FFI functions removed:
// array_map_ffi, array_map2_ffi, array_fold_ffi
// Can be re-added when needed for strided tensor operations
