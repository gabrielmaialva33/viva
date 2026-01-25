//// Layer - Layer types for neural networks
////
//// Supports Dense (fully connected) as base.
//// Each layer maintains weights, biases, and metadata for forward/backward.

import gleam/result
import viva/neural/activation.{type ActivationType}
import viva/neural/tensor.{type Tensor, type TensorError}

// =============================================================================
// TYPES
// =============================================================================

/// Dense layer (fully connected)
pub type DenseLayer {
  DenseLayer(
    /// Weights: [input_size, output_size]
    weights: Tensor,
    /// Biases: [output_size]
    biases: Tensor,
    /// Activation function
    activation: ActivationType,
    /// Input size
    input_size: Int,
    /// Output size
    output_size: Int,
  )
}

/// Gradients of a Dense layer
pub type DenseGradients {
  DenseGradients(
    /// Weights gradient
    d_weights: Tensor,
    /// Biases gradient
    d_biases: Tensor,
    /// Gradient to propagate to previous layer
    d_input: Tensor,
  )
}

/// Forward pass cache (for backprop)
pub type DenseCache {
  DenseCache(
    /// Original input
    input: Tensor,
    /// Output before activation (z = Wx + b)
    pre_activation: Tensor,
    /// Output after activation
    output: Tensor,
    /// Activation derivatives
    activation_derivatives: Tensor,
  )
}

/// Union type for different layer types
pub type Layer {
  Dense(DenseLayer)
  // Future: Dropout, BatchNorm, Conv2D, etc.
}

/// Generic cache
pub type LayerCache {
  DenseLayerCache(DenseCache)
}

/// Generic gradients
pub type LayerGradients {
  DenseLayerGradients(DenseGradients)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create Dense layer with Xavier initialization
pub fn dense(
  input_size: Int,
  output_size: Int,
  activation: ActivationType,
) -> DenseLayer {
  let weights = tensor.xavier_init(input_size, output_size)
  let biases = tensor.zeros([output_size])

  DenseLayer(
    weights: weights,
    biases: biases,
    activation: activation,
    input_size: input_size,
    output_size: output_size,
  )
}

/// Create Dense layer with He initialization (good for ReLU)
pub fn dense_he(
  input_size: Int,
  output_size: Int,
  activation: ActivationType,
) -> DenseLayer {
  let weights = tensor.he_init(input_size, output_size)
  let biases = tensor.zeros([output_size])

  DenseLayer(
    weights: weights,
    biases: biases,
    activation: activation,
    input_size: input_size,
    output_size: output_size,
  )
}

/// Create Dense layer with custom weights
pub fn dense_with_weights(
  weights: Tensor,
  biases: Tensor,
  activation: ActivationType,
) -> Result(DenseLayer, TensorError) {
  case weights.shape, biases.shape {
    [input_size, output_size], [bias_size] if output_size == bias_size ->
      Ok(DenseLayer(
        weights: weights,
        biases: biases,
        activation: activation,
        input_size: input_size,
        output_size: output_size,
      ))
    _, _ ->
      Error(tensor.InvalidShape(
        "Weights and biases dimensions don't match",
      ))
  }
}

/// Create Dense layer without activation (linear)
pub fn dense_linear(input_size: Int, output_size: Int) -> DenseLayer {
  dense(input_size, output_size, activation.Linear)
}

// =============================================================================
// FORWARD PASS
// =============================================================================

/// Dense layer forward pass, returns output and cache
pub fn forward(layer: DenseLayer, input: Tensor) -> Result(#(Tensor, DenseCache), TensorError) {
  // Check input dimension
  case tensor.size(input) == layer.input_size {
    False ->
      Error(tensor.ShapeMismatch(
        expected: [layer.input_size],
        got: input.shape,
      ))
    True -> {
      // z = W^T @ x + b (transpose because weights is [in, out])
      use weights_t <- result.try(tensor.transpose(layer.weights))
      use z <- result.try(tensor.matmul_vec(weights_t, input))
      use pre_activation <- result.try(tensor.add(z, layer.biases))

      // Apply activation
      let #(output, derivatives) = case layer.activation {
        activation.Softmax -> {
          let out = activation.softmax(pre_activation)
          // Softmax derivative is handled specially in backward
          #(out, tensor.ones([layer.output_size]))
        }
        _ -> activation.apply(pre_activation, layer.activation)
      }

      let cache =
        DenseCache(
          input: input,
          pre_activation: pre_activation,
          output: output,
          activation_derivatives: derivatives,
        )

      Ok(#(output, cache))
    }
  }
}

/// Simple forward pass (no cache, for inference)
pub fn predict(layer: DenseLayer, input: Tensor) -> Result(Tensor, TensorError) {
  case forward(layer, input) {
    Ok(#(output, _)) -> Ok(output)
    Error(e) -> Error(e)
  }
}

// =============================================================================
// BACKWARD PASS
// =============================================================================

/// Dense layer backward pass
pub fn backward(
  layer: DenseLayer,
  cache: DenseCache,
  upstream_gradient: Tensor,
) -> Result(DenseGradients, TensorError) {
  // 1. Multiply upstream by activation gradient
  let delta = case layer.activation {
    activation.Softmax ->
      activation.softmax_backward(cache.output, upstream_gradient)
    _ ->
      case tensor.mul(upstream_gradient, cache.activation_derivatives) {
        Ok(d) -> d
        Error(_) -> upstream_gradient
      }
  }

  // 2. Biases gradient = delta
  let d_biases = delta

  // 3. Weights gradient = outer(input, delta)
  use d_weights <- result.try(tensor.outer(cache.input, delta))

  // 4. Gradient for previous layer = W @ delta
  use d_input <- result.try(tensor.matmul_vec(layer.weights, delta))

  Ok(DenseGradients(
    d_weights: d_weights,
    d_biases: d_biases,
    d_input: d_input,
  ))
}

// =============================================================================
// PARAMETER UPDATE
// =============================================================================

/// Update weights with gradients (simple SGD)
pub fn update_sgd(
  layer: DenseLayer,
  gradients: DenseGradients,
  learning_rate: Float,
) -> Result(DenseLayer, TensorError) {
  // weights = weights - lr * d_weights
  let scaled_dw = tensor.scale(gradients.d_weights, learning_rate)
  use new_weights <- result.try(tensor.sub(layer.weights, scaled_dw))

  // biases = biases - lr * d_biases
  let scaled_db = tensor.scale(gradients.d_biases, learning_rate)
  use new_biases <- result.try(tensor.sub(layer.biases, scaled_db))

  Ok(DenseLayer(..layer, weights: new_weights, biases: new_biases))
}

/// Update weights with momentum
pub fn update_momentum(
  layer: DenseLayer,
  gradients: DenseGradients,
  velocity_w: Tensor,
  velocity_b: Tensor,
  learning_rate: Float,
  momentum: Float,
) -> Result(#(DenseLayer, Tensor, Tensor), TensorError) {
  // velocity_w = momentum * velocity_w + lr * d_weights
  let scaled_dw = tensor.scale(gradients.d_weights, learning_rate)
  let momentum_vw = tensor.scale(velocity_w, momentum)
  use new_velocity_w <- result.try(tensor.add(momentum_vw, scaled_dw))

  // velocity_b = momentum * velocity_b + lr * d_biases
  let scaled_db = tensor.scale(gradients.d_biases, learning_rate)
  let momentum_vb = tensor.scale(velocity_b, momentum)
  use new_velocity_b <- result.try(tensor.add(momentum_vb, scaled_db))

  // weights = weights - velocity_w
  use new_weights <- result.try(tensor.sub(layer.weights, new_velocity_w))

  // biases = biases - velocity_b
  use new_biases <- result.try(tensor.sub(layer.biases, new_velocity_b))

  Ok(#(
    DenseLayer(..layer, weights: new_weights, biases: new_biases),
    new_velocity_w,
    new_velocity_b,
  ))
}

// =============================================================================
// UTILITY
// =============================================================================

/// Total number of trainable parameters
pub fn param_count(layer: DenseLayer) -> Int {
  tensor.size(layer.weights) + tensor.size(layer.biases)
}

/// Clone layer
pub fn clone(layer: DenseLayer) -> DenseLayer {
  DenseLayer(
    weights: tensor.clone(layer.weights),
    biases: tensor.clone(layer.biases),
    activation: layer.activation,
    input_size: layer.input_size,
    output_size: layer.output_size,
  )
}

/// Zero gradients (useful for batch accumulation)
pub fn zero_gradients(layer: DenseLayer) -> DenseGradients {
  DenseGradients(
    d_weights: tensor.zeros([layer.input_size, layer.output_size]),
    d_biases: tensor.zeros([layer.output_size]),
    d_input: tensor.zeros([layer.input_size]),
  )
}

/// Initialize velocities for momentum (zeros)
pub fn init_velocities(layer: DenseLayer) -> #(Tensor, Tensor) {
  #(
    tensor.zeros([layer.input_size, layer.output_size]),
    tensor.zeros([layer.output_size]),
  )
}

/// Returns layer description
pub fn describe(layer: DenseLayer) -> String {
  "Dense("
  <> int_to_string(layer.input_size)
  <> " -> "
  <> int_to_string(layer.output_size)
  <> ", "
  <> activation.name(layer.activation)
  <> ")"
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "erlang", "integer_to_binary")
fn int_to_string(i: Int) -> String
