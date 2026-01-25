//// Einsum - Einstein summation-like API for named tensors
////
//// Simplified einsum-style contractions using named axes.

import viva/neural/axis.{type Axis}
import viva/neural/named_ops
import viva/neural/named_tensor_core.{
  type NamedTensor, type NamedTensorError, InvalidOp, find_axis,
}

// =============================================================================
// EINSUM-LIKE API
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
          case find_axis(a, axis), find_axis(b, axis) {
            Ok(_), Ok(_) -> named_ops.matmul(a, b, axis, axis)
            Error(e), _ -> Error(e)
            _, Error(e) -> Error(e)
          }
        }
        _ -> Error(InvalidOp("Only single axis contraction supported"))
      }
    }
    [t], Contract([axis]) -> {
      // Unary contraction = sum
      named_ops.sum(t, axis)
    }
    _, _ -> Error(InvalidOp("Unsupported einsum configuration"))
  }
}
