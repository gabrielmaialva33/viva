//// Named Operations - Operations on named tensors
////
//// Reduction, element-wise, and matrix operations using named axes.

import gleam/int
import gleam/list
import gleam/option.{None, Some}
import gleam/result
import viva/neural/axis.{
  type Axis, type AxisSpec, Anon, AxisSpec, axes_equal, axis_equals,
}
import viva/neural/tensor.{type Tensor}

// Re-import from named_tensor_core
import viva/neural/named_tensor_core.{
  type NamedTensor, type NamedTensorError, AxisNotFound, InvalidOp,
  NamedTensor, SizeMismatch, TensorErr, find_axis, list_at, remove_axis_at,
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
          let new_axes = remove_axis_at(t.axes, idx)
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
  case find_axis(t, along), t.data.shape {
    Error(e), _ -> Error(e)
    Ok(axis_idx), [rows, cols] -> {
      let result_data = case axis_idx {
        0 -> {
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
pub fn matmul(
  a: NamedTensor,
  b: NamedTensor,
  contract_a: Axis,
  contract_b: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  use a_idx <- result.try(find_axis(a, contract_a))
  use b_idx <- result.try(find_axis(b, contract_b))

  case list_at(a.axes, a_idx), list_at(b.axes, b_idx) {
    Ok(a_spec), Ok(b_spec) -> {
      case a_spec.size == b_spec.size {
        False -> Error(SizeMismatch(contract_a, a_spec.size, b_spec.size))
        True -> {
          case a.data.shape, b.data.shape {
            [_m, k1], [k2, _n] if k1 == k2 -> {
              case tensor.matmul(a.data, b.data) {
                Error(e) -> Error(TensorErr(e))
                Ok(result) -> {
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
      case find_axis(first, along) {
        Error(e) -> Error(e)
        Ok(axis_idx) -> {
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
// INTERNAL HELPERS
// =============================================================================

/// Align axes for broadcasting
fn align_axes(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(#(NamedTensor, NamedTensor, List(AxisSpec)), NamedTensorError) {
  case axes_equal(a.axes, b.axes) {
    True -> Ok(#(a, b, a.axes))
    False -> {
      case tensor.can_broadcast(a.data.shape, b.data.shape) {
        False ->
          Error(named_tensor_core.BroadcastError(
            "Axes not compatible for broadcasting",
          ))
        True -> {
          case tensor.broadcast_shape(a.data.shape, b.data.shape) {
            Error(_) ->
              Error(named_tensor_core.BroadcastError(
                "Cannot compute broadcast shape",
              ))
            Ok(result_shape) -> {
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

fn float_max(a: Float, b: Float) -> Float {
  case a >. b {
    True -> a
    False -> b
  }
}
