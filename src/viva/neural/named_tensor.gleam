//// NamedTensor - Tensors with named axes (inspired by Nx/xarray)
////
//// Named axes provide semantic meaning to dimensions, enabling:
//// - Self-documenting code: `tensor |> sum(axis: Batch)` vs `sum(axis: 0)`
//// - Safe broadcasting: only compatible names can be broadcast
//// - Flexible operations: reference axes by name, not position
////
//// Design principles:
//// - Zero runtime overhead for basic operations (names are metadata)
//// - Compile-time safety through Gleam's type system
//// - Fluent API with pipelines
//// - Interop with plain Tensor when needed

import gleam/int
import gleam/list
import gleam/option.{None, Some}
import gleam/result
import gleam/string
import viva/neural/tensor.{type Tensor, type TensorError}

// =============================================================================
// AXIS TYPES - Semantic dimension names
// =============================================================================

/// Named axis - gives semantic meaning to a dimension
pub type Axis {
  /// Batch dimension (samples in mini-batch)
  Batch
  /// Sequence/time dimension (for RNNs, transformers)
  Seq
  /// Feature/channel dimension
  Feature
  /// Spatial height
  Height
  /// Spatial width
  Width
  /// Input dimension (for weight matrices)
  Input
  /// Output dimension (for weight matrices)
  Output
  /// Head dimension (for multi-head attention)
  Head
  /// Embedding dimension
  Embed
  /// Custom named axis
  Named(String)
  /// Anonymous axis (unnamed, referenced by position)
  Anon
}

/// Axis with its size
pub type AxisSpec {
  AxisSpec(name: Axis, size: Int)
}

// =============================================================================
// NAMED TENSOR TYPE
// =============================================================================

/// Tensor with named axes
pub type NamedTensor {
  NamedTensor(
    /// Underlying data tensor
    data: Tensor,
    /// Axis specifications (names + sizes, in order)
    axes: List(AxisSpec),
  )
}

/// Error types for named tensor operations
pub type NamedTensorError {
  /// Axis not found
  AxisNotFound(name: Axis)
  /// Duplicate axis name
  DuplicateAxis(name: Axis)
  /// Axis mismatch in operation
  AxisMismatch(expected: Axis, got: Axis)
  /// Size mismatch for same axis
  SizeMismatch(axis: Axis, expected: Int, got: Int)
  /// Cannot broadcast axes
  BroadcastError(reason: String)
  /// Underlying tensor error
  TensorErr(TensorError)
  /// Invalid operation
  InvalidOp(reason: String)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create named tensor from data and axis specs
pub fn new(
  data: Tensor,
  axes: List(AxisSpec),
) -> Result(NamedTensor, NamedTensorError) {
  // Validate: axis count matches rank
  let data_rank = tensor.rank(data)
  let axes_count = list.length(axes)

  case data_rank == axes_count {
    False ->
      Error(InvalidOp(
        "Axis count ("
        <> int.to_string(axes_count)
        <> ") doesn't match tensor rank ("
        <> int.to_string(data_rank)
        <> ")",
      ))
    True -> {
      // Validate: sizes match
      case validate_sizes(data.shape, axes) {
        Error(e) -> Error(e)
        Ok(_) -> {
          // Validate: no duplicate names (except Anon)
          case validate_unique_names(axes) {
            Error(e) -> Error(e)
            Ok(_) -> Ok(NamedTensor(data: data, axes: axes))
          }
        }
      }
    }
  }
}

/// Create from tensor with inferred anonymous axes
pub fn from_tensor(t: Tensor) -> NamedTensor {
  let axes = list.map(t.shape, fn(size) { AxisSpec(name: Anon, size: size) })
  NamedTensor(data: t, axes: axes)
}

/// Create named tensor of zeros
pub fn zeros(axes: List(AxisSpec)) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.zeros(shape)
  NamedTensor(data: data, axes: axes)
}

/// Create named tensor of ones
pub fn ones(axes: List(AxisSpec)) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.ones(shape)
  NamedTensor(data: data, axes: axes)
}

/// Create named tensor with random values
pub fn random(axes: List(AxisSpec)) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.random_uniform(shape)
  NamedTensor(data: data, axes: axes)
}

/// Create named tensor with normal distribution
pub fn randn(axes: List(AxisSpec), mean: Float, std: Float) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.random_normal(shape, mean, std)
  NamedTensor(data: data, axes: axes)
}

// =============================================================================
// AXIS SPEC HELPERS
// =============================================================================

/// Create axis spec with shorthand
pub fn axis(name: Axis, size: Int) -> AxisSpec {
  AxisSpec(name: name, size: size)
}

/// Shorthand constructors for common axes
pub fn batch(size: Int) -> AxisSpec {
  AxisSpec(name: Batch, size: size)
}

pub fn seq(size: Int) -> AxisSpec {
  AxisSpec(name: Seq, size: size)
}

pub fn feature(size: Int) -> AxisSpec {
  AxisSpec(name: Feature, size: size)
}

pub fn height(size: Int) -> AxisSpec {
  AxisSpec(name: Height, size: size)
}

pub fn width(size: Int) -> AxisSpec {
  AxisSpec(name: Width, size: size)
}

pub fn input(size: Int) -> AxisSpec {
  AxisSpec(name: Input, size: size)
}

pub fn output(size: Int) -> AxisSpec {
  AxisSpec(name: Output, size: size)
}

pub fn head(size: Int) -> AxisSpec {
  AxisSpec(name: Head, size: size)
}

pub fn embed(size: Int) -> AxisSpec {
  AxisSpec(name: Embed, size: size)
}

pub fn named(name: String, size: Int) -> AxisSpec {
  AxisSpec(name: Named(name), size: size)
}

// =============================================================================
// AXIS LOOKUP & MANIPULATION
// =============================================================================

/// Find axis index by name
pub fn find_axis(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  find_axis_in_list(t.axes, name, 0)
}

fn find_axis_in_list(
  axes: List(AxisSpec),
  name: Axis,
  idx: Int,
) -> Result(Int, NamedTensorError) {
  case axes {
    [] -> Error(AxisNotFound(name))
    [first, ..rest] -> {
      case axis_equals(first.name, name) {
        True -> Ok(idx)
        False -> find_axis_in_list(rest, name, idx + 1)
      }
    }
  }
}

/// Get axis size by name
pub fn axis_size(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  case find_axis(t, name) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case list_at(t.axes, idx) {
        Error(_) -> Error(AxisNotFound(name))
        Ok(spec) -> Ok(spec.size)
      }
    }
  }
}

/// Check if tensor has axis
pub fn has_axis(t: NamedTensor, name: Axis) -> Bool {
  case find_axis(t, name) {
    Ok(_) -> True
    Error(_) -> False
  }
}

/// Get all axis names
pub fn axis_names(t: NamedTensor) -> List(Axis) {
  list.map(t.axes, fn(a) { a.name })
}

/// Get shape as list
pub fn shape(t: NamedTensor) -> List(Int) {
  t.data.shape
}

/// Get rank (number of dimensions)
pub fn rank(t: NamedTensor) -> Int {
  list.length(t.axes)
}

/// Total number of elements
pub fn size(t: NamedTensor) -> Int {
  tensor.size(t.data)
}

// =============================================================================
// AXIS OPERATIONS
// =============================================================================

/// Rename an axis
pub fn rename_axis(
  t: NamedTensor,
  from: Axis,
  to: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, from) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      let new_axes =
        list.index_map(t.axes, fn(spec, i) {
          case i == idx {
            True -> AxisSpec(..spec, name: to)
            False -> spec
          }
        })
      Ok(NamedTensor(..t, axes: new_axes))
    }
  }
}

/// Transpose/reorder axes by names
pub fn transpose(
  t: NamedTensor,
  new_order: List(Axis),
) -> Result(NamedTensor, NamedTensorError) {
  // Find indices for each axis in new order
  let indices_result = list.try_map(new_order, fn(name) { find_axis(t, name) })

  case indices_result {
    Error(e) -> Error(e)
    Ok(indices) -> {
      // Reorder axes specs
      let new_axes = list.filter_map(indices, fn(idx) { list_at(t.axes, idx) })

      // For 2D, use tensor transpose
      case t.data.shape, indices {
        [_, _], [1, 0] -> {
          case tensor.transpose(t.data) {
            Ok(transposed) -> Ok(NamedTensor(data: transposed, axes: new_axes))
            Error(e) -> Error(TensorErr(e))
          }
        }
        _, _ -> {
          // General case: use permutation
          case permute_data(t.data, indices) {
            Ok(permuted) -> Ok(NamedTensor(data: permuted, axes: new_axes))
            Error(e) -> Error(TensorErr(e))
          }
        }
      }
    }
  }
}

/// Add a new axis of size 1
pub fn unsqueeze(t: NamedTensor, name: Axis, position: Int) -> NamedTensor {
  let new_spec = AxisSpec(name: name, size: 1)
  let #(before, after) = list.split(t.axes, position)
  let new_axes = list.flatten([before, [new_spec], after])
  let new_data = tensor.unsqueeze(t.data, position)
  NamedTensor(data: new_data, axes: new_axes)
}

/// Remove axis of size 1 by name
pub fn squeeze(
  t: NamedTensor,
  name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, name) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case list_at(t.axes, idx) {
        Error(_) -> Error(AxisNotFound(name))
        Ok(spec) -> {
          case spec.size == 1 {
            False -> Error(InvalidOp("Cannot squeeze axis with size != 1"))
            True -> {
              case tensor.squeeze_axis(t.data, idx) {
                Error(e) -> Error(TensorErr(e))
                Ok(squeezed) -> {
                  let new_axes =
                    list.filter(
                      list.index_map(t.axes, fn(a, i) { #(a, i) }),
                      fn(pair) { pair.1 != idx },
                    )
                    |> list.map(fn(pair) { pair.0 })
                  Ok(NamedTensor(data: squeezed, axes: new_axes))
                }
              }
            }
          }
        }
      }
    }
  }
}

// =============================================================================
// REDUCTION OPERATIONS (by axis name)
// =============================================================================

/// Sum along named axis
pub fn sum(t: NamedTensor, along: Axis) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, along) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case tensor.sum_axis(t.data, idx) {
        Error(e) -> Error(TensorErr(e))
        Ok(summed) -> {
          // Remove the summed axis
          let new_axes = remove_axis_at(t.axes, idx)
          // Handle scalar result
          let final_axes = case new_axes {
            [] -> [AxisSpec(name: Anon, size: 1)]
            _ -> new_axes
          }
          Ok(NamedTensor(data: summed, axes: final_axes))
        }
      }
    }
  }
}

/// Mean along named axis
pub fn mean(
  t: NamedTensor,
  along: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, along) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case tensor.mean_axis(t.data, idx) {
        Error(e) -> Error(TensorErr(e))
        Ok(meaned) -> {
          let new_axes = remove_axis_at(t.axes, idx)
          let final_axes = case new_axes {
            [] -> [AxisSpec(name: Anon, size: 1)]
            _ -> new_axes
          }
          Ok(NamedTensor(data: meaned, axes: final_axes))
        }
      }
    }
  }
}

/// Max value along named axis
pub fn max_along(
  t: NamedTensor,
  along: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  // For now, only support 2D
  case find_axis(t, along), t.data.shape {
    Error(e), _ -> Error(e)
    Ok(axis_idx), [rows, cols] -> {
      let result_data = case axis_idx {
        0 -> {
          // Max along rows -> result has `cols` elements
          list.range(0, cols - 1)
          |> list.map(fn(col) {
            list.range(0, rows - 1)
            |> list.fold(0.0 -. 999_999.0, fn(acc, row) {
              case tensor.get2d(t.data, row, col) {
                Ok(v) -> float_max(acc, v)
                Error(_) -> acc
              }
            })
          })
        }
        1 -> {
          // Max along cols -> result has `rows` elements
          list.range(0, rows - 1)
          |> list.map(fn(row) {
            list.range(0, cols - 1)
            |> list.fold(0.0 -. 999_999.0, fn(acc, col) {
              case tensor.get2d(t.data, row, col) {
                Ok(v) -> float_max(acc, v)
                Error(_) -> acc
              }
            })
          })
        }
        _ -> []
      }
      let new_axes = remove_axis_at(t.axes, axis_idx)
      let result_shape = [list.length(result_data)]
      Ok(NamedTensor(
        data: tensor.Tensor(data: result_data, shape: result_shape),
        axes: new_axes,
      ))
    }
    _, _ -> Error(InvalidOp("max_along only supports 2D tensors"))
  }
}

/// Argmax along named axis
pub fn argmax_along(
  t: NamedTensor,
  along: Axis,
) -> Result(List(Int), NamedTensorError) {
  case find_axis(t, along), t.data.shape {
    Error(e), _ -> Error(e)
    Ok(axis_idx), [rows, cols] -> {
      let result = case axis_idx {
        0 -> {
          // Argmax along rows
          list.range(0, cols - 1)
          |> list.map(fn(col) {
            let #(best_idx, _, _) =
              list.range(0, rows - 1)
              |> list.fold(#(0, 0.0 -. 999_999.0, 0), fn(acc, row) {
                let #(best, best_val, curr) = acc
                case tensor.get2d(t.data, row, col) {
                  Ok(v) if v >. best_val -> #(curr, v, curr + 1)
                  _ -> #(best, best_val, curr + 1)
                }
              })
            best_idx
          })
        }
        1 -> {
          // Argmax along cols
          list.range(0, rows - 1)
          |> list.map(fn(row) {
            let #(best_idx, _, _) =
              list.range(0, cols - 1)
              |> list.fold(#(0, 0.0 -. 999_999.0, 0), fn(acc, col) {
                let #(best, best_val, curr) = acc
                case tensor.get2d(t.data, row, col) {
                  Ok(v) if v >. best_val -> #(curr, v, curr + 1)
                  _ -> #(best, best_val, curr + 1)
                }
              })
            best_idx
          })
        }
        _ -> []
      }
      Ok(result)
    }
    _, _ -> Error(InvalidOp("argmax_along only supports 2D tensors"))
  }
}

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================

/// Element-wise addition (axes must match or be broadcastable)
pub fn add(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  case align_axes(a, b) {
    Error(e) -> Error(e)
    Ok(#(a_aligned, b_aligned, result_axes)) -> {
      case tensor.add_broadcast(a_aligned.data, b_aligned.data) {
        Error(e) -> Error(TensorErr(e))
        Ok(result) -> Ok(NamedTensor(data: result, axes: result_axes))
      }
    }
  }
}

/// Element-wise subtraction
pub fn sub(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  case align_axes(a, b) {
    Error(e) -> Error(e)
    Ok(#(a_aligned, b_aligned, result_axes)) -> {
      case tensor.sub(a_aligned.data, b_aligned.data) {
        Error(e) -> Error(TensorErr(e))
        Ok(result) -> Ok(NamedTensor(data: result, axes: result_axes))
      }
    }
  }
}

/// Element-wise multiplication
pub fn mul(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  case align_axes(a, b) {
    Error(e) -> Error(e)
    Ok(#(a_aligned, b_aligned, result_axes)) -> {
      case tensor.mul_broadcast(a_aligned.data, b_aligned.data) {
        Error(e) -> Error(TensorErr(e))
        Ok(result) -> Ok(NamedTensor(data: result, axes: result_axes))
      }
    }
  }
}

/// Scale by constant
pub fn scale(t: NamedTensor, s: Float) -> NamedTensor {
  NamedTensor(..t, data: tensor.scale(t.data, s))
}

/// Apply function to each element
pub fn map(t: NamedTensor, f: fn(Float) -> Float) -> NamedTensor {
  NamedTensor(..t, data: tensor.map(t.data, f))
}

// =============================================================================
// MATRIX OPERATIONS (with named axes)
// =============================================================================

/// Matrix multiplication with explicit contraction axis
/// matmul(a, b, contract: Input) contracts over the Input axis
pub fn matmul(
  a: NamedTensor,
  b: NamedTensor,
  contract_a: Axis,
  contract_b: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  // Find contraction axes
  use a_idx <- result.try(find_axis(a, contract_a))
  use b_idx <- result.try(find_axis(b, contract_b))

  // Get sizes
  case list_at(a.axes, a_idx), list_at(b.axes, b_idx) {
    Ok(a_spec), Ok(b_spec) -> {
      case a_spec.size == b_spec.size {
        False -> Error(SizeMismatch(contract_a, a_spec.size, b_spec.size))
        True -> {
          // For 2D tensors: standard matmul
          case a.data.shape, b.data.shape {
            [_m, k1], [k2, _n] if k1 == k2 -> {
              case tensor.matmul(a.data, b.data) {
                Error(e) -> Error(TensorErr(e))
                Ok(result) -> {
                  // Result axes: a's non-contracted + b's non-contracted
                  let a_remaining = remove_axis_at(a.axes, a_idx)
                  let b_remaining = remove_axis_at(b.axes, b_idx)
                  let result_axes = list.append(a_remaining, b_remaining)
                  Ok(NamedTensor(data: result, axes: result_axes))
                }
              }
            }
            _, _ -> Error(InvalidOp("matmul requires 2D tensors"))
          }
        }
      }
    }
    _, _ -> Error(AxisNotFound(contract_a))
  }
}

/// Dot product over named axis
pub fn dot(
  a: NamedTensor,
  b: NamedTensor,
  along: Axis,
) -> Result(Float, NamedTensorError) {
  use _a_idx <- result.try(find_axis(a, along))
  use _b_idx <- result.try(find_axis(b, along))

  case tensor.rank(a.data) == 1 && tensor.rank(b.data) == 1 {
    False -> Error(InvalidOp("dot requires 1D tensors"))
    True -> {
      case tensor.dot(a.data, b.data) {
        Error(e) -> Error(TensorErr(e))
        Ok(result) -> Ok(result)
      }
    }
  }
}

// =============================================================================
// EINSUM-LIKE API (simplified)
// =============================================================================

/// Specification for einsum-like operations
pub type EinsumSpec {
  /// Contract (sum over) these axes
  Contract(List(Axis))
  /// Keep these axes in result
  Keep(List(Axis))
  /// Batch axes (present in all inputs, kept in output)
  BatchAxes(List(Axis))
}

/// Einsum-like contraction
/// Example: einsum([a, b], Contract([Feature])) contracts over Feature axis
pub fn einsum(
  tensors: List(NamedTensor),
  spec: EinsumSpec,
) -> Result(NamedTensor, NamedTensorError) {
  case tensors, spec {
    [a, b], Contract(contract_axes) -> {
      // Binary contraction
      case contract_axes {
        [axis] -> {
          // Find axis in both tensors
          case find_axis(a, axis), find_axis(b, axis) {
            Ok(_), Ok(_) -> matmul(a, b, axis, axis)
            Error(e), _ -> Error(e)
            _, Error(e) -> Error(e)
          }
        }
        _ -> Error(InvalidOp("Only single axis contraction supported"))
      }
    }
    [t], Contract([axis]) -> {
      // Unary contraction = sum
      sum(t, axis)
    }
    _, _ -> Error(InvalidOp("Unsupported einsum configuration"))
  }
}

// =============================================================================
// SLICING WITH NAMED AXES
// =============================================================================

/// Slice specification
pub type SliceSpec {
  /// Take range [start, start+length) from axis
  SliceRange(axis: Axis, start: Int, length: Int)
  /// Take specific index (reduces rank)
  SliceIndex(axis: Axis, index: Int)
  /// Take first n elements
  SliceFirst(axis: Axis, n: Int)
  /// Take last n elements
  SliceLast(axis: Axis, n: Int)
}

/// Apply slice to tensor
pub fn slice(
  t: NamedTensor,
  spec: SliceSpec,
) -> Result(NamedTensor, NamedTensorError) {
  case spec {
    SliceFirst(axis, n) -> {
      case find_axis(t, axis) {
        Error(e) -> Error(e)
        Ok(idx) if idx == 0 -> {
          let new_data = tensor.take_first(t.data, n)
          let new_axes =
            list.index_map(t.axes, fn(a, i) {
              case i == idx {
                True -> AxisSpec(..a, size: n)
                False -> a
              }
            })
          Ok(NamedTensor(data: new_data, axes: new_axes))
        }
        Ok(_) -> Error(InvalidOp("SliceFirst only supported on first axis"))
      }
    }
    SliceLast(axis, n) -> {
      case find_axis(t, axis) {
        Error(e) -> Error(e)
        Ok(idx) if idx == 0 -> {
          let new_data = tensor.take_last(t.data, n)
          let new_axes =
            list.index_map(t.axes, fn(a, i) {
              case i == idx {
                True -> AxisSpec(..a, size: n)
                False -> a
              }
            })
          Ok(NamedTensor(data: new_data, axes: new_axes))
        }
        Ok(_) -> Error(InvalidOp("SliceLast only supported on first axis"))
      }
    }
    SliceRange(axis, start, length) -> {
      case find_axis(t, axis) {
        Error(e) -> Error(e)
        Ok(idx) -> {
          // Build starts and lengths lists
          let starts =
            list.index_map(t.axes, fn(_, i) {
              case i == idx {
                True -> start
                False -> 0
              }
            })
          let lengths =
            list.index_map(t.axes, fn(a, i) {
              case i == idx {
                True -> length
                False -> a.size
              }
            })
          case tensor.slice(t.data, starts, lengths) {
            Error(e) -> Error(TensorErr(e))
            Ok(sliced) -> {
              let new_axes =
                list.index_map(t.axes, fn(a, i) {
                  case i == idx {
                    True -> AxisSpec(..a, size: length)
                    False -> a
                  }
                })
              Ok(NamedTensor(data: sliced, axes: new_axes))
            }
          }
        }
      }
    }
    SliceIndex(axis, index) -> {
      // This reduces rank - select single element along axis
      case find_axis(t, axis) {
        Error(e) -> Error(e)
        Ok(idx) -> {
          let starts =
            list.index_map(t.axes, fn(_, i) {
              case i == idx {
                True -> index
                False -> 0
              }
            })
          let lengths =
            list.index_map(t.axes, fn(a, i) {
              case i == idx {
                True -> 1
                False -> a.size
              }
            })
          case tensor.slice(t.data, starts, lengths) {
            Error(e) -> Error(TensorErr(e))
            Ok(sliced) -> {
              // Squeeze out the indexed axis
              let new_axes = remove_axis_at(t.axes, idx)
              let squeezed = tensor.squeeze(sliced)
              Ok(NamedTensor(data: squeezed, axes: new_axes))
            }
          }
        }
      }
    }
  }
}

// =============================================================================
// STACKING & CONCATENATION
// =============================================================================

/// Stack tensors along new axis
pub fn stack(
  tensors: List(NamedTensor),
  new_axis: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case tensors {
    [] -> Error(InvalidOp("Cannot stack empty list"))
    [first, ..rest] -> {
      // Check all have same axes
      let all_same = list.all(rest, fn(t) { axes_equal(t.axes, first.axes) })
      case all_same {
        False -> Error(InvalidOp("All tensors must have same axes to stack"))
        True -> {
          let data_list = list.map(tensors, fn(t) { t.data })
          case tensor.stack(data_list, 0) {
            Error(e) -> Error(TensorErr(e))
            Ok(stacked) -> {
              let n = list.length(tensors)
              let new_axes = [AxisSpec(name: new_axis, size: n), ..first.axes]
              Ok(NamedTensor(data: stacked, axes: new_axes))
            }
          }
        }
      }
    }
  }
}

/// Concatenate tensors along existing axis
pub fn concat(
  tensors: List(NamedTensor),
  along: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case tensors {
    [] -> Error(InvalidOp("Cannot concat empty list"))
    [first, ..rest] -> {
      // Find axis in first tensor
      case find_axis(first, along) {
        Error(e) -> Error(e)
        Ok(axis_idx) -> {
          // Check all have compatible axes (same names, same sizes except concat axis)
          let check_compatible = fn(t: NamedTensor) -> Bool {
            case list.length(t.axes) == list.length(first.axes) {
              False -> False
              True -> {
                list.zip(t.axes, first.axes)
                |> list.index_fold(True, fn(acc, pair, i) {
                  let #(ta, fa) = pair
                  case i == axis_idx {
                    True -> acc && axis_equals(ta.name, fa.name)
                    False ->
                      acc && axis_equals(ta.name, fa.name) && ta.size == fa.size
                  }
                })
              }
            }
          }

          let all_compatible = list.all(rest, check_compatible)
          case all_compatible {
            False ->
              Error(InvalidOp("Tensors have incompatible axes for concat"))
            True -> {
              let data_list = list.map(tensors, fn(t) { t.data })
              case tensor.concat_axis(data_list, axis_idx) {
                Error(e) -> Error(TensorErr(e))
                Ok(concatenated) -> {
                  // Sum sizes along concat axis
                  let total_size =
                    list.fold(tensors, 0, fn(acc, t) {
                      case list_at(t.axes, axis_idx) {
                        Ok(spec) -> acc + spec.size
                        Error(_) -> acc
                      }
                    })
                  let new_axes =
                    list.index_map(first.axes, fn(a, i) {
                      case i == axis_idx {
                        True -> AxisSpec(..a, size: total_size)
                        False -> a
                      }
                    })
                  Ok(NamedTensor(data: concatenated, axes: new_axes))
                }
              }
            }
          }
        }
      }
    }
  }
}

// =============================================================================
// CONVERSION & INSPECTION
// =============================================================================

/// Convert to plain tensor (drop names)
pub fn to_tensor(t: NamedTensor) -> Tensor {
  t.data
}

/// Pretty print tensor info
pub fn describe(t: NamedTensor) -> String {
  let axes_str =
    t.axes
    |> list.map(fn(a) { axis_to_string(a.name) <> ":" <> int.to_string(a.size) })
    |> string.join(", ")

  "NamedTensor[" <> axes_str <> "]"
}

/// Get human-readable axis name
pub fn axis_to_string(a: Axis) -> String {
  case a {
    Batch -> "batch"
    Seq -> "seq"
    Feature -> "feature"
    Height -> "height"
    Width -> "width"
    Input -> "input"
    Output -> "output"
    Head -> "head"
    Embed -> "embed"
    Named(s) -> s
    Anon -> "_"
  }
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

fn validate_sizes(
  shape: List(Int),
  axes: List(AxisSpec),
) -> Result(Nil, NamedTensorError) {
  case shape, axes {
    [], [] -> Ok(Nil)
    [s, ..s_rest], [a, ..a_rest] -> {
      case s == a.size {
        True -> validate_sizes(s_rest, a_rest)
        False -> Error(SizeMismatch(a.name, a.size, s))
      }
    }
    _, _ -> Error(InvalidOp("Shape and axes length mismatch"))
  }
}

fn validate_unique_names(axes: List(AxisSpec)) -> Result(Nil, NamedTensorError) {
  let named_axes =
    list.filter(axes, fn(a) {
      case a.name {
        Anon -> False
        _ -> True
      }
    })
  let names = list.map(named_axes, fn(a) { a.name })
  case has_duplicates(names) {
    True -> Error(DuplicateAxis(Anon))
    // Simplified error
    False -> Ok(Nil)
  }
}

fn has_duplicates(items: List(Axis)) -> Bool {
  case items {
    [] -> False
    [first, ..rest] -> {
      case list.any(rest, fn(x) { axis_equals(x, first) }) {
        True -> True
        False -> has_duplicates(rest)
      }
    }
  }
}

fn axis_equals(a: Axis, b: Axis) -> Bool {
  case a, b {
    Anon, Anon -> True
    Batch, Batch -> True
    Seq, Seq -> True
    Feature, Feature -> True
    Height, Height -> True
    Width, Width -> True
    Input, Input -> True
    Output, Output -> True
    Head, Head -> True
    Embed, Embed -> True
    Named(s1), Named(s2) -> s1 == s2
    _, _ -> False
  }
}

fn axes_equal(a: List(AxisSpec), b: List(AxisSpec)) -> Bool {
  case a, b {
    [], [] -> True
    [a1, ..a_rest], [b1, ..b_rest] ->
      axis_equals(a1.name, b1.name)
      && a1.size == b1.size
      && axes_equal(a_rest, b_rest)
    _, _ -> False
  }
}

fn remove_axis_at(axes: List(AxisSpec), idx: Int) -> List(AxisSpec) {
  axes
  |> list.index_map(fn(a, i) { #(a, i) })
  |> list.filter(fn(pair) { pair.1 != idx })
  |> list.map(fn(pair) { pair.0 })
}

fn list_at(lst: List(a), idx: Int) -> Result(a, Nil) {
  lst
  |> list.drop(idx)
  |> list.first
}

/// Align axes for broadcasting
fn align_axes(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(#(NamedTensor, NamedTensor, List(AxisSpec)), NamedTensorError) {
  // Simple case: same axes
  case axes_equal(a.axes, b.axes) {
    True -> Ok(#(a, b, a.axes))
    False -> {
      // Try to broadcast
      // For now, just check if shapes are broadcastable
      case tensor.can_broadcast(a.data.shape, b.data.shape) {
        False -> Error(BroadcastError("Axes not compatible for broadcasting"))
        True -> {
          case tensor.broadcast_shape(a.data.shape, b.data.shape) {
            Error(_) -> Error(BroadcastError("Cannot compute broadcast shape"))
            Ok(result_shape) -> {
              // Create result axes (use a's names where possible, else b's)
              let result_axes =
                result_shape
                |> list.index_map(fn(size, i) {
                  let a_name = case list_at(a.axes, i) {
                    Ok(spec) -> Some(spec.name)
                    Error(_) -> None
                  }
                  let b_name = case list_at(b.axes, i) {
                    Ok(spec) -> Some(spec.name)
                    Error(_) -> None
                  }
                  let name = case a_name, b_name {
                    Some(n), _ -> n
                    None, Some(n) -> n
                    None, None -> Anon
                  }
                  AxisSpec(name: name, size: size)
                })
              Ok(#(a, b, result_axes))
            }
          }
        }
      }
    }
  }
}

/// Permute tensor data according to axis indices
fn permute_data(t: Tensor, indices: List(Int)) -> Result(Tensor, TensorError) {
  // For now, only support 2D transpose
  case indices {
    [1, 0] -> tensor.transpose(t)
    [0, 1] -> Ok(t)
    _ -> Error(tensor.InvalidShape("Permutation not supported for this rank"))
  }
}

fn float_max(a: Float, b: Float) -> Float {
  case a >. b {
    True -> a
    False -> b
  }
}
