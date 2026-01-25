//// Tensor - N-dimensional arrays for neural operations
////
//// Design: Simplicity over performance (BEAM is not a GPU).
//// Uses List(Float) with explicit shape. Future optimizations via NIFs.

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

// =============================================================================
// TYPES
// =============================================================================

/// Tensor: data + shape
pub type Tensor {
  Tensor(data: List(Float), shape: List(Int))
}

/// Tensor operation errors
pub type TensorError {
  ShapeMismatch(expected: List(Int), got: List(Int))
  InvalidShape(reason: String)
  DimensionError(reason: String)
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

/// Total number of elements
pub fn size(t: Tensor) -> Int {
  list.length(t.data)
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
  list_at_float(t.data, index)
  |> result.map_error(fn(_) {
    DimensionError("Index " <> int.to_string(index) <> " out of bounds")
  })
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
          let start = row_idx * num_cols
          let row_data =
            t.data
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
  Tensor(data: list.map(t.data, f), shape: t.shape)
}

/// Apply function with index
pub fn map_indexed(t: Tensor, f: fn(Float, Int) -> Float) -> Tensor {
  Tensor(data: list.index_map(t.data, fn(x, i) { f(x, i) }), shape: t.shape)
}

/// Element-wise addition
pub fn add(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let data = list.map2(a.data, b.data, fn(x, y) { x +. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise subtraction
pub fn sub(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let data = list.map2(a.data, b.data, fn(x, y) { x -. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise multiplication (Hadamard)
pub fn mul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let data = list.map2(a.data, b.data, fn(x, y) { x *. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise division
pub fn div(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let data = list.map2(a.data, b.data, fn(x, y) { x /. y })
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
  list.fold(t.data, 0.0, fn(acc, x) { acc +. x })
}

/// Product of all elements
pub fn product(t: Tensor) -> Float {
  list.fold(t.data, 1.0, fn(acc, x) { acc *. x })
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
  case t.data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.max(acc, x) })
  }
}

/// Minimum value
pub fn min(t: Tensor) -> Float {
  case t.data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.min(acc, x) })
  }
}

/// Argmax - index of largest element
pub fn argmax(t: Tensor) -> Int {
  case t.data {
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
      let products = list.map2(a.data, b.data, fn(x, y) { x *. y })
      Ok(list.fold(products, 0.0, fn(acc, x) { acc +. x }))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Matrix-vector multiplication: [m, n] @ [n] -> [m]
pub fn matmul_vec(mat: Tensor, vec: Tensor) -> Result(Tensor, TensorError) {
  case mat.shape, vec.shape {
    [m, n], [vec_n] if n == vec_n -> {
      let result_data =
        list.range(0, m - 1)
        |> list.map(fn(row_idx) {
          let start = row_idx * n
          let row =
            mat.data
            |> list.drop(start)
            |> list.take(n)
          list.map2(row, vec.data, fn(a, b) { a *. b })
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
      let result_data =
        list.flat_map(a.data, fn(ai) { list.map(b.data, fn(bj) { ai *. bj }) })
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
  t.data
}

/// Convert matrix to list of lists
pub fn to_list2d(t: Tensor) -> Result(List(List(Float)), TensorError) {
  case t.shape {
    [num_rows, num_cols] -> {
      let rows =
        list.range(0, num_rows - 1)
        |> list.map(fn(i) {
          let start = i * num_cols
          t.data
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
  Tensor(data: t.data, shape: t.shape)
}

/// Reshape tensor
pub fn reshape(t: Tensor, new_shape: List(Int)) -> Result(Tensor, TensorError) {
  let old_size = size(t)
  let new_size = list.fold(new_shape, 1, fn(acc, dim) { acc * dim })

  case old_size == new_size {
    True -> Ok(Tensor(data: t.data, shape: new_shape))
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
  Tensor(data: t.data, shape: [size(t)])
}

/// Concatenate vectors
pub fn concat(tensors: List(Tensor)) -> Tensor {
  let data = list.flat_map(tensors, fn(t) { t.data })
  from_list(data)
}

/// L2 norm
pub fn norm(t: Tensor) -> Float {
  let sum_sq = list.fold(t.data, 0.0, fn(acc, x) { acc +. x *. x })
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
    case list_at_float(t.data, src_flat) {
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
  let new_shape = list.filter(t.shape, fn(dim) { dim != 1 })
  let final_shape = case new_shape {
    [] -> [1]
    // Scalar becomes 1D
    _ -> new_shape
  }
  Tensor(data: t.data, shape: final_shape)
}

/// Remove dimension at specific axis if it's 1
pub fn squeeze_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  case list_at_int(t.shape, axis) {
    Error(_) -> Error(DimensionError("Axis out of bounds"))
    Ok(dim) -> {
      case dim == 1 {
        False -> Error(InvalidShape("Dimension at axis is not 1"))
        True -> {
          let new_shape =
            t.shape
            |> list.index_map(fn(d, i) { #(d, i) })
            |> list.filter(fn(pair) { pair.1 != axis })
            |> list.map(fn(pair) { pair.0 })
          Ok(Tensor(data: t.data, shape: new_shape))
        }
      }
    }
  }
}

/// Add dimension of size 1 at specified axis
pub fn unsqueeze(t: Tensor, axis: Int) -> Tensor {
  let rank = list.length(t.shape)
  let insert_at = case axis < 0 {
    True -> rank + axis + 1
    False -> axis
  }

  let #(before, after) = list.split(t.shape, insert_at)
  let new_shape = list.flatten([before, [1], after])
  Tensor(data: t.data, shape: new_shape)
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
  let m = mean(t)
  let squared_diffs =
    list.map(t.data, fn(x) {
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
  case t.data {
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
          let data = list.flat_map(tensors, fn(t) { t.data })
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
          let data = list.flat_map(tensors, fn(t) { t.data })
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
                  let data = list.flat_map(tensors, fn(t) { t.data })
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
  case t.shape {
    [_n] -> {
      // 1D slice
      case starts, lengths {
        [start], [len] -> {
          let data =
            t.data
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
              |> list.filter_map(fn(c) { list_at_float(t.data, row_offset + c) })
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
  case t.shape {
    [_size] -> {
      let data = list.take(t.data, n)
      Tensor(data: data, shape: [n])
    }
    [_rows, cols] -> {
      let data = list.take(t.data, n * cols)
      Tensor(data: data, shape: [n, cols])
    }
    _ -> t
  }
}

/// Get last n elements along first axis
pub fn take_last(t: Tensor, n: Int) -> Tensor {
  case t.shape {
    [_size] -> {
      let data =
        t.data
        |> list.reverse
        |> list.take(n)
        |> list.reverse
      Tensor(data: data, shape: [n])
    }
    [rows, cols] -> {
      let skip = { rows - n } * cols
      let data = list.drop(t.data, skip)
      Tensor(data: data, shape: [n, cols])
    }
    _ -> t
  }
}
