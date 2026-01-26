//// Named Tensor Core - Core types and basic operations
////
//// Core type definitions and fundamental operations for named tensors.

import gleam/int
import gleam/list
import gleam/string
import viva/neural/axis.{
  type Axis, type AxisSpec, Anon, AxisSpec, axis_equals,
  axis_to_string,
}
import viva/neural/tensor.{type Tensor, type TensorError}

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
      case validate_sizes(data.shape, axes) {
        Error(e) -> Error(e)
        Ok(_) -> {
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
  let indices_result = list.try_map(new_order, fn(name) { find_axis(t, name) })

  case indices_result {
    Error(e) -> Error(e)
    Ok(indices) -> {
      let new_axes = list.filter_map(indices, fn(idx) { list_at(t.axes, idx) })

      case t.data.shape, indices {
        [_, _], [1, 0] -> {
          case tensor.transpose(t.data) {
            Ok(transposed) -> Ok(NamedTensor(data: transposed, axes: new_axes))
            Error(e) -> Error(TensorErr(e))
          }
        }
        _, _ -> {
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

// =============================================================================
// INTERNAL HELPERS (pub for use by other modules)
// =============================================================================

pub fn validate_sizes(
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

pub fn validate_unique_names(
  axes: List(AxisSpec),
) -> Result(Nil, NamedTensorError) {
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
    False -> Ok(Nil)
  }
}

pub fn has_duplicates(items: List(Axis)) -> Bool {
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

pub fn remove_axis_at(axes: List(AxisSpec), idx: Int) -> List(AxisSpec) {
  axes
  |> list.index_map(fn(a, i) { #(a, i) })
  |> list.filter(fn(pair) { pair.1 != idx })
  |> list.map(fn(pair) { pair.0 })
}

pub fn list_at(lst: List(a), idx: Int) -> Result(a, Nil) {
  lst
  |> list.drop(idx)
  |> list.first
}

fn permute_data(t: Tensor, indices: List(Int)) -> Result(Tensor, TensorError) {
  case indices {
    [1, 0] -> tensor.transpose(t)
    [0, 1] -> Ok(t)
    _ -> Error(tensor.InvalidShape("Permutation not supported for this rank"))
  }
}
