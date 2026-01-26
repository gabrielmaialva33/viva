//// Neural Utils - Shared helpers for neural network operations
////
//// Provides utilities for tensor manipulation, random generation,
//// and axis-aware operations needed by advanced layers.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva/neural/tensor.{type Tensor, type TensorError, Tensor}

// =============================================================================
// AXIS-AWARE OPERATIONS
// =============================================================================

/// Split tensor into n equal parts along axis
/// Returns list of tensors, each with shape reduced along axis
pub fn split(t: Tensor, n: Int, axis: Int) -> Result(List(Tensor), TensorError) {
  let tdata = tensor.to_list(t)
  case t.shape {
    [size] if axis == 0 -> {
      // 1D split
      let chunk_size = size / n
      let chunks = chunk_list(tdata, chunk_size)
      Ok(list.map(chunks, fn(data) { Tensor(data: data, shape: [chunk_size]) }))
    }
    [rows, cols] if axis == 0 -> {
      // Split rows
      let rows_per_chunk = rows / n
      let elements_per_chunk = rows_per_chunk * cols
      let chunks = chunk_list(tdata, elements_per_chunk)
      Ok(
        list.map(chunks, fn(data) {
          Tensor(data: data, shape: [rows_per_chunk, cols])
        }),
      )
    }
    [rows, cols] if axis == 1 -> {
      // Split columns (more complex - need to gather non-contiguous data)
      let cols_per_chunk = cols / n
      let chunks =
        list.range(0, n - 1)
        |> list.map(fn(chunk_idx) {
          let start_col = chunk_idx * cols_per_chunk
          let data =
            list.range(0, rows - 1)
            |> list.flat_map(fn(row) {
              let row_start = row * cols
              list.range(start_col, start_col + cols_per_chunk - 1)
              |> list.filter_map(fn(col) { list_at(tdata, row_start + col) })
            })
          Tensor(data: data, shape: [rows, cols_per_chunk])
        })
      Ok(chunks)
    }
    _ -> Error(tensor.DimensionError("split only supports 1D and 2D tensors"))
  }
}

/// Softmax along specific axis
/// For 2D tensor with axis=1, applies softmax to each row
pub fn softmax_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  let tdata = tensor.to_list(t)
  case t.shape {
    [_n] if axis == 0 -> {
      // 1D softmax - same as regular softmax
      Ok(softmax(t))
    }
    [rows, cols] if axis == 1 -> {
      // Softmax per row
      let data =
        list.range(0, rows - 1)
        |> list.flat_map(fn(row_idx) {
          let start = row_idx * cols
          let row_data =
            tdata
            |> list.drop(start)
            |> list.take(cols)
          let row_tensor = Tensor(data: row_data, shape: [cols])
          let softmax_row = softmax(row_tensor)
          tensor.to_list(softmax_row)
        })
      Ok(Tensor(data: data, shape: [rows, cols]))
    }
    [rows, cols] if axis == 0 -> {
      // Softmax per column
      let data =
        list.range(0, cols - 1)
        |> list.flat_map(fn(col_idx) {
          // Extract column
          let col_data =
            list.range(0, rows - 1)
            |> list.filter_map(fn(row) { list_at(tdata, row * cols + col_idx) })
          let col_tensor = Tensor(data: col_data, shape: [rows])
          tensor.to_list(softmax(col_tensor))
        })
      // Need to transpose result back - data is column-major
      let transposed =
        list.range(0, rows - 1)
        |> list.flat_map(fn(row) {
          list.range(0, cols - 1)
          |> list.filter_map(fn(col) { list_at(data, col * rows + row) })
        })
      Ok(Tensor(data: transposed, shape: [rows, cols]))
    }
    _ -> Error(tensor.DimensionError("softmax_axis only supports 1D and 2D"))
  }
}

/// Variance along axis (for BatchNorm)
pub fn variance_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  let tdata = tensor.to_list(t)
  case t.shape {
    [_n] if axis == 0 -> {
      Ok(Tensor(data: [tensor.variance(t)], shape: [1]))
    }
    [rows, cols] if axis == 0 -> {
      // Variance per column
      let vars =
        list.range(0, cols - 1)
        |> list.map(fn(col_idx) {
          let col_data =
            list.range(0, rows - 1)
            |> list.filter_map(fn(row) { list_at(tdata, row * cols + col_idx) })
          let col_mean =
            list.fold(col_data, 0.0, fn(a, x) { a +. x }) /. int.to_float(rows)
          let squared_diffs =
            list.map(col_data, fn(x) {
              let diff = x -. col_mean
              diff *. diff
            })
          list.fold(squared_diffs, 0.0, fn(a, x) { a +. x })
          /. int.to_float(rows)
        })
      Ok(Tensor(data: vars, shape: [cols]))
    }
    [rows, cols] if axis == 1 -> {
      // Variance per row
      let vars =
        list.range(0, rows - 1)
        |> list.map(fn(row_idx) {
          let start = row_idx * cols
          let row_data =
            tdata
            |> list.drop(start)
            |> list.take(cols)
          let row_mean =
            list.fold(row_data, 0.0, fn(a, x) { a +. x }) /. int.to_float(cols)
          let squared_diffs =
            list.map(row_data, fn(x) {
              let diff = x -. row_mean
              diff *. diff
            })
          list.fold(squared_diffs, 0.0, fn(a, x) { a +. x })
          /. int.to_float(cols)
        })
      Ok(Tensor(data: vars, shape: [rows]))
    }
    _ -> Error(tensor.DimensionError("variance_axis only supports 1D and 2D"))
  }
}

/// Element-wise square root
pub fn sqrt(t: Tensor) -> Tensor {
  tensor.map(t, float_sqrt)
}

/// Add epsilon and take sqrt (for numerical stability in normalization)
pub fn sqrt_eps(t: Tensor, epsilon: Float) -> Tensor {
  tensor.map(t, fn(x) { float_sqrt(x +. epsilon) })
}

// =============================================================================
// RANDOM GENERATION
// =============================================================================

/// Bernoulli random tensor (0.0 or 1.0 based on probability p)
/// p = probability of 1.0
pub fn random_bernoulli(shape: List(Int), p: Float) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let data =
    list.range(1, size)
    |> list.map(fn(_) {
      case random_float() <. p {
        True -> 1.0
        False -> 0.0
      }
    })
  Tensor(data: data, shape: shape)
}

/// Create random dropout mask with keep probability
/// Returns mask and inverse scale factor for training
pub fn dropout_mask(shape: List(Int), keep_prob: Float) -> #(Tensor, Float) {
  let mask = random_bernoulli(shape, keep_prob)
  let scale = case keep_prob >. 0.0 {
    True -> 1.0 /. keep_prob
    False -> 0.0
  }
  #(mask, scale)
}

// =============================================================================
// TENSOR UTILITIES
// =============================================================================

/// Concatenate tensors along axis
pub fn concat_along_axis(
  tensors: List(Tensor),
  axis: Int,
) -> Result(Tensor, TensorError) {
  case tensors {
    [] -> Error(tensor.InvalidShape("Cannot concat empty list"))
    [single] -> Ok(single)
    [first, ..rest] -> {
      let _ = rest
      case axis == 0 {
        True -> tensor.concat_axis(tensors, 0)
        False -> {
          // For axis 1, need special handling
          case first.shape {
            [rows, _cols] -> {
              let total_cols =
                list.fold(tensors, 0, fn(acc, t) {
                  case t.shape {
                    [_, c] -> acc + c
                    _ -> acc
                  }
                })
              // Interleave data from all tensors row by row
              let data =
                list.range(0, rows - 1)
                |> list.flat_map(fn(row) {
                  list.flat_map(tensors, fn(t) {
                    let td = tensor.to_list(t)
                    case t.shape {
                      [_, cols] -> {
                        let start = row * cols
                        td
                        |> list.drop(start)
                        |> list.take(cols)
                      }
                      _ -> []
                    }
                  })
                })
              Ok(Tensor(data: data, shape: [rows, total_cols]))
            }
            _ ->
              Error(tensor.DimensionError("concat axis 1 requires 2D tensors"))
          }
        }
      }
    }
  }
}

/// Element-wise maximum of two tensors
pub fn maximum(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { float.max(x, y) })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(tensor.ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise minimum of two tensors
pub fn minimum(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { float.min(x, y) })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(tensor.ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Repeat tensor n times along axis 0
pub fn repeat(t: Tensor, n: Int) -> Tensor {
  let tdata = tensor.to_list(t)
  let data = list.flatten(list.repeat(tdata, n))
  case t.shape {
    [size] -> Tensor(data: data, shape: [size * n])
    [rows, cols] -> Tensor(data: data, shape: [rows * n, cols])
    shape -> Tensor(data: data, shape: shape)
  }
}

/// Tile tensor to match target shape (broadcasting helper)
pub fn tile_to_shape(
  t: Tensor,
  target: List(Int),
) -> Result(Tensor, TensorError) {
  tensor.broadcast_to(t, target)
}

// =============================================================================
// ACTIVATION HELPERS
// =============================================================================

/// Sigmoid function
pub fn sigmoid(x: Float) -> Float {
  1.0 /. { 1.0 +. float_exp(0.0 -. x) }
}

/// Tanh function
pub fn tanh(x: Float) -> Float {
  let e2x = float_exp(2.0 *. x)
  { e2x -. 1.0 } /. { e2x +. 1.0 }
}

/// Apply sigmoid to tensor
pub fn sigmoid_tensor(t: Tensor) -> Tensor {
  tensor.map(t, sigmoid)
}

/// Apply tanh to tensor
pub fn tanh_tensor(t: Tensor) -> Tensor {
  tensor.map(t, tanh)
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

/// Softmax helper (numerically stable)
fn softmax(t: Tensor) -> Tensor {
  let max_val = tensor.max(t)
  let shifted = tensor.add_scalar(t, 0.0 -. max_val)
  let exp_vals = tensor.map(shifted, float_exp)
  let sum_exp = tensor.sum(exp_vals)
  tensor.scale(exp_vals, 1.0 /. sum_exp)
}

/// Chunk list into sublists of size n
fn chunk_list(items: List(a), size: Int) -> List(List(a)) {
  case items {
    [] -> []
    _ -> {
      let #(head, tail) = list.split(items, size)
      case head {
        [] -> []
        _ -> [head, ..chunk_list(tail, size)]
      }
    }
  }
}

/// List access by index
fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  lst
  |> list.drop(index)
  |> list.first
}

// =============================================================================
// EXTERNAL FUNCTIONS
// =============================================================================

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "exp")
fn float_exp(x: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_float() -> Float
