//// Named Tensor Core - Core types and basic operations
////
//// Core type definitions and fundamental operations for named tensors.

import gleam/int
import gleam/list
import gleam/string
import viva_tensor/axis.{
  type Axis, type AxisSpec, Anon, equals as axis_equals,
  to_string as axis_to_string,
}
import viva_tensor/core/error
import viva_tensor/named as tensor_named
import viva_tensor/tensor.{type Tensor, type TensorError}

// =============================================================================
// NAMED TENSOR TYPE
// =============================================================================

/// Tensor with named axes
pub type NamedTensor =
  tensor_named.NamedTensor

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
  case tensor_named.new(data, axes) {
    Ok(tensor) -> Ok(tensor)
    Error(err) -> Error(normalize_error(err))
  }
}

/// Create from tensor with inferred anonymous axes
pub fn from_tensor(t: Tensor) -> NamedTensor {
  tensor_named.from_tensor(t)
}

/// Create named tensor of zeros
pub fn zeros(axes: List(AxisSpec)) -> NamedTensor {
  tensor_named.zeros(axes)
}

/// Create named tensor of ones
pub fn ones(axes: List(AxisSpec)) -> NamedTensor {
  tensor_named.ones(axes)
}

/// Create named tensor with random values
pub fn random(axes: List(AxisSpec)) -> NamedTensor {
  tensor_named.random(axes)
}

/// Create named tensor with normal distribution
pub fn randn(axes: List(AxisSpec), mean: Float, std: Float) -> NamedTensor {
  tensor_named.randn(axes, mean, std)
}

// =============================================================================
// AXIS LOOKUP & MANIPULATION
// =============================================================================

/// Find axis index by name
pub fn find_axis(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  case tensor_named.find_axis(t, name) {
    Ok(idx) -> Ok(idx)
    Error(err) -> Error(normalize_error(err))
  }
}

/// Get axis size by name
pub fn axis_size(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  case tensor_named.axis_size(t, name) {
    Ok(size) -> Ok(size)
    Error(err) -> Error(normalize_error(err))
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
  case tensor_named.rename_axis(t, from, to) {
    Ok(named_tensor) -> Ok(named_tensor)
    Error(err) -> Error(normalize_error(err))
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
            Ok(transposed) ->
              Ok(tensor_named.NamedTensor(data: transposed, axes: new_axes))
            Error(e) -> Error(TensorErr(e))
          }
        }
        _, _ -> {
          case permute_data(t.data, indices) {
            Ok(permuted) ->
              Ok(tensor_named.NamedTensor(data: permuted, axes: new_axes))
            Error(e) -> Error(TensorErr(e))
          }
        }
      }
    }
  }
}

/// Add a new axis of size 1
pub fn unsqueeze(t: NamedTensor, name: Axis, position: Int) -> NamedTensor {
  tensor_named.unsqueeze(t, name, position)
}

/// Remove axis of size 1 by name
pub fn squeeze(
  t: NamedTensor,
  name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case tensor_named.squeeze(t, name) {
    Ok(named_tensor) -> Ok(named_tensor)
    Error(err) -> Error(normalize_error(err))
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

/// Sum along named axis (compat alias for upstream `sum_along`)
pub fn sum_along(
  t: NamedTensor,
  axis_name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case tensor_named.sum_along(t, axis_name) {
    Ok(tensor) -> Ok(tensor)
    Error(err) -> Error(normalize_error(err))
  }
}

/// Mean along named axis (compat alias for upstream `mean_along`)
pub fn mean_along(
  t: NamedTensor,
  axis_name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case tensor_named.mean_along(t, axis_name) {
    Ok(tensor) -> Ok(tensor)
    Error(err) -> Error(normalize_error(err))
  }
}

/// Convert upstream `viva_tensor` error into local compatibility error
fn normalize_error(error: tensor_named.NamedTensorError) -> NamedTensorError {
  case error {
    tensor_named.AxisNotFound(name) -> AxisNotFound(name)
    tensor_named.DuplicateAxis(name) -> DuplicateAxis(name)
    tensor_named.AxisMismatch(expected, got) -> AxisMismatch(expected, got)
    tensor_named.SizeMismatch(axis, expected, got) ->
      SizeMismatch(axis, expected, got)
    tensor_named.BroadcastErr(reason) -> BroadcastError(reason)
    tensor_named.TensorErr(err) -> TensorErr(err)
    tensor_named.InvalidOp(reason) -> InvalidOp(reason)
  }
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
    _ -> Error(error.InvalidShape("Permutation not supported for this rank"))
  }
}
