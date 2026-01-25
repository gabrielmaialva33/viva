//// Activation - Activation functions for neural networks
////
//// Each function returns both the value and derivative for backprop.
//// Functional design: pure functions, no side effects.

import gleam/list
import viva/neural/tensor.{type Tensor}

// =============================================================================
// TYPES
// =============================================================================

/// Activation function type
pub type ActivationType {
  Sigmoid
  Tanh
  ReLU
  LeakyReLU(alpha: Float)
  ELU(alpha: Float)
  Softmax
  Linear
  Swish
  GELU
}

/// Activation result with value and derivative
pub type ActivationResult {
  ActivationResult(
    /// Activated value
    value: Float,
    /// Derivative at point (for backprop)
    derivative: Float,
  )
}

// =============================================================================
// SCALAR ACTIVATIONS
// =============================================================================

/// Sigmoid: 1 / (1 + e^(-x))
pub fn sigmoid(x: Float) -> ActivationResult {
  let s = 1.0 /. { 1.0 +. float_exp(0.0 -. x) }
  ActivationResult(value: s, derivative: s *. { 1.0 -. s })
}

/// Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
pub fn tanh(x: Float) -> ActivationResult {
  let t = float_tanh(x)
  ActivationResult(value: t, derivative: 1.0 -. t *. t)
}

/// ReLU: max(0, x)
pub fn relu(x: Float) -> ActivationResult {
  case x >. 0.0 {
    True -> ActivationResult(value: x, derivative: 1.0)
    False -> ActivationResult(value: 0.0, derivative: 0.0)
  }
}

/// Leaky ReLU: x if x > 0, else alpha * x
pub fn leaky_relu(x: Float, alpha: Float) -> ActivationResult {
  case x >. 0.0 {
    True -> ActivationResult(value: x, derivative: 1.0)
    False -> ActivationResult(value: alpha *. x, derivative: alpha)
  }
}

/// ELU: x if x > 0, else alpha * (e^x - 1)
pub fn elu(x: Float, alpha: Float) -> ActivationResult {
  case x >. 0.0 {
    True -> ActivationResult(value: x, derivative: 1.0)
    False -> {
      let exp_x = float_exp(x)
      ActivationResult(
        value: alpha *. { exp_x -. 1.0 },
        derivative: alpha *. exp_x,
      )
    }
  }
}

/// Linear (identity)
pub fn linear(x: Float) -> ActivationResult {
  ActivationResult(value: x, derivative: 1.0)
}

/// Swish: x * sigmoid(x)
pub fn swish(x: Float) -> ActivationResult {
  let s = 1.0 /. { 1.0 +. float_exp(0.0 -. x) }
  let value = x *. s
  // Derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
  let derivative = s +. x *. s *. { 1.0 -. s }
  ActivationResult(value: value, derivative: derivative)
}

/// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: Float) -> ActivationResult {
  let sqrt_2_pi = 0.7978845608
  let a = 0.044715
  let x3 = x *. x *. x
  let inner = sqrt_2_pi *. { x +. a *. x3 }
  let t = float_tanh(inner)
  let value = 0.5 *. x *. { 1.0 +. t }

  // Approximate derivative
  let sech2 = 1.0 -. t *. t
  let d_inner = sqrt_2_pi *. { 1.0 +. 3.0 *. a *. x *. x }
  let derivative = 0.5 *. { 1.0 +. t } +. 0.5 *. x *. sech2 *. d_inner
  ActivationResult(value: value, derivative: derivative)
}

// =============================================================================
// TENSOR ACTIVATIONS
// =============================================================================

/// Apply activation to tensor, returns values and derivatives separately
pub fn apply(t: Tensor, activation: ActivationType) -> #(Tensor, Tensor) {
  let results =
    t.data
    |> list.map(fn(x) { apply_scalar(x, activation) })

  let values = list.map(results, fn(r) { r.value })
  let derivatives = list.map(results, fn(r) { r.derivative })

  #(
    tensor.Tensor(data: values, shape: t.shape),
    tensor.Tensor(data: derivatives, shape: t.shape),
  )
}

/// Apply activation to tensor (values only, no derivatives)
pub fn forward(t: Tensor, activation: ActivationType) -> Tensor {
  let #(values, _) = apply(t, activation)
  values
}

/// Apply scalar activation function based on type
fn apply_scalar(x: Float, activation: ActivationType) -> ActivationResult {
  case activation {
    Sigmoid -> sigmoid(x)
    Tanh -> tanh(x)
    ReLU -> relu(x)
    LeakyReLU(alpha) -> leaky_relu(x, alpha)
    ELU(alpha) -> elu(x, alpha)
    Linear -> linear(x)
    Swish -> swish(x)
    GELU -> gelu(x)
    Softmax -> linear(x)
    // Softmax is handled separately (needs entire vector)
  }
}

// =============================================================================
// SOFTMAX (special - needs entire vector)
// =============================================================================

/// Softmax: e^xi / sum(e^xj) - normalizes to probability distribution
pub fn softmax(t: Tensor) -> Tensor {
  // Subtract max for numerical stability
  let max_val = tensor.max(t)
  let shifted = tensor.add_scalar(t, 0.0 -. max_val)

  // Compute exponentials
  let exp_vals = tensor.map(shifted, float_exp)

  // Normalize
  let sum_exp = tensor.sum(exp_vals)
  tensor.scale(exp_vals, 1.0 /. sum_exp)
}

/// Softmax derivative (simplified Jacobian for backprop)
/// Returns d_softmax/d_input given the upstream gradient
pub fn softmax_backward(softmax_output: Tensor, upstream_grad: Tensor) -> Tensor {
  // For each element i: sum_j(upstream_j * softmax_j * (delta_ij - softmax_i))
  // Simplified: s * (upstream - sum(upstream * s))
  case tensor.mul(upstream_grad, softmax_output) {
    Ok(weighted) -> {
      let sum_weighted = tensor.sum(weighted)
      let subtracted = tensor.add_scalar(upstream_grad, 0.0 -. sum_weighted)
      case tensor.mul(softmax_output, subtracted) {
        Ok(result) -> result
        Error(_) -> upstream_grad
      }
    }
    Error(_) -> upstream_grad
  }
}

// =============================================================================
// UTILITY
// =============================================================================

/// Returns activation name
pub fn name(activation: ActivationType) -> String {
  case activation {
    Sigmoid -> "sigmoid"
    Tanh -> "tanh"
    ReLU -> "relu"
    LeakyReLU(_) -> "leaky_relu"
    ELU(_) -> "elu"
    Softmax -> "softmax"
    Linear -> "linear"
    Swish -> "swish"
    GELU -> "gelu"
  }
}

/// Parse name to type
pub fn from_name(name: String) -> ActivationType {
  case name {
    "sigmoid" -> Sigmoid
    "tanh" -> Tanh
    "relu" -> ReLU
    "leaky_relu" -> LeakyReLU(0.01)
    "elu" -> ELU(1.0)
    "softmax" -> Softmax
    "linear" -> Linear
    "swish" -> Swish
    "gelu" -> GELU
    _ -> ReLU
    // default
  }
}

/// Returns default activation for hidden layers
pub fn default_hidden() -> ActivationType {
  ReLU
}

/// Returns default activation for output layer (classification)
pub fn default_output_classification() -> ActivationType {
  Softmax
}

/// Returns default activation for output layer (regression)
pub fn default_output_regression() -> ActivationType {
  Linear
}

// =============================================================================
// EXTERNAL FUNCTIONS
// =============================================================================

@external(erlang, "math", "exp")
fn float_exp(x: Float) -> Float

@external(erlang, "math", "tanh")
fn float_tanh(x: Float) -> Float
