//// Regularization - Dropout and other regularization techniques
////
//// Implements dropout for preventing overfitting in neural networks.
//// Inspired by Axon's dropout implementation.

import gleam/float
import gleam/list
import gleam/result
import viva/neural/tensor.{type Tensor, type TensorError, Tensor}
import viva/neural/utils

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// Dropout layer configuration
pub type DropoutLayer {
  DropoutLayer(
    /// Probability of dropping a unit (0.0 to 1.0)
    rate: Float,
    /// Current mode
    mode: DropoutMode,
  )
}

/// Dropout mode
pub type DropoutMode {
  /// Training: apply dropout with random mask
  Training
  /// Inference: pass through unchanged
  Inference
}

/// Cache for backward pass
pub type DropoutCache {
  DropoutCache(
    /// Mask used during forward pass
    mask: Tensor,
    /// Scale factor (1 / keep_prob)
    scale: Float,
    /// Input shape for reference
    input_shape: List(Int),
  )
}

/// Gradients from dropout backward
pub type DropoutGradients {
  DropoutGradients(
    /// Gradient w.r.t. input
    d_input: Tensor,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new dropout layer
/// rate: probability of dropping (e.g., 0.5 means 50% dropped)
pub fn new(rate: Float) -> DropoutLayer {
  let clamped_rate = float.min(float.max(rate, 0.0), 1.0)
  DropoutLayer(rate: clamped_rate, mode: Training)
}

/// Create dropout layer in inference mode
pub fn new_inference(rate: Float) -> DropoutLayer {
  let clamped_rate = float.min(float.max(rate, 0.0), 1.0)
  DropoutLayer(rate: clamped_rate, mode: Inference)
}

/// Set mode to training
pub fn train(layer: DropoutLayer) -> DropoutLayer {
  DropoutLayer(..layer, mode: Training)
}

/// Set mode to inference
pub fn eval(layer: DropoutLayer) -> DropoutLayer {
  DropoutLayer(..layer, mode: Inference)
}

// =============================================================================
// FORWARD PASS
// =============================================================================

/// Forward pass through dropout layer
/// Returns (output, cache) where cache is needed for backward pass
pub fn forward(
  layer: DropoutLayer,
  input: Tensor,
) -> Result(#(Tensor, DropoutCache), TensorError) {
  case layer.mode {
    Inference -> {
      // During inference, just pass through unchanged
      let cache =
        DropoutCache(
          mask: tensor.ones(input.shape),
          scale: 1.0,
          input_shape: input.shape,
        )
      Ok(#(input, cache))
    }
    Training -> {
      forward_training(layer, input)
    }
  }
}

/// Training mode forward pass
fn forward_training(
  layer: DropoutLayer,
  input: Tensor,
) -> Result(#(Tensor, DropoutCache), TensorError) {
  let keep_prob = 1.0 -. layer.rate

  // Handle edge cases
  case keep_prob <=. 0.0 {
    True -> {
      // Drop everything
      let zeros = tensor.zeros(input.shape)
      let cache =
        DropoutCache(mask: zeros, scale: 0.0, input_shape: input.shape)
      Ok(#(zeros, cache))
    }
    False -> {
      case keep_prob >=. 1.0 {
        True -> {
          // Keep everything
          let cache =
            DropoutCache(
              mask: tensor.ones(input.shape),
              scale: 1.0,
              input_shape: input.shape,
            )
          Ok(#(input, cache))
        }
        False -> {
          // Normal dropout: mask * scale
          let #(mask, scale) = utils.dropout_mask(input.shape, keep_prob)

          // Apply mask and scale: output = input * mask * scale
          use masked <- result.try(tensor.mul(input, mask))
          let output = tensor.scale(masked, scale)

          let cache =
            DropoutCache(mask: mask, scale: scale, input_shape: input.shape)
          Ok(#(output, cache))
        }
      }
    }
  }
}

/// Simple forward without cache (for inference)
pub fn apply(layer: DropoutLayer, input: Tensor) -> Tensor {
  case layer.mode {
    Inference -> input
    Training -> {
      let keep_prob = 1.0 -. layer.rate
      case keep_prob <=. 0.0 {
        True -> tensor.zeros(input.shape)
        False -> {
          case keep_prob >=. 1.0 {
            True -> input
            False -> {
              let #(mask, scale) = utils.dropout_mask(input.shape, keep_prob)
              case tensor.mul(input, mask) {
                Ok(masked) -> tensor.scale(masked, scale)
                Error(_) -> input
              }
            }
          }
        }
      }
    }
  }
}

// =============================================================================
// BACKWARD PASS
// =============================================================================

/// Backward pass through dropout layer
/// Gradient flows through where mask is 1, scaled appropriately
pub fn backward(
  _layer: DropoutLayer,
  cache: DropoutCache,
  upstream: Tensor,
) -> Result(DropoutGradients, TensorError) {
  // d_input = upstream * mask * scale
  // Same operation as forward - mask gates gradient flow
  use masked <- result.try(tensor.mul(upstream, cache.mask))
  let d_input = tensor.scale(masked, cache.scale)

  Ok(DropoutGradients(d_input: d_input))
}

// =============================================================================
// SPATIAL DROPOUT (for Conv2D)
// =============================================================================

/// Spatial Dropout - drops entire feature maps
/// Useful for convolutional networks
pub type SpatialDropoutLayer {
  SpatialDropoutLayer(rate: Float, mode: DropoutMode)
}

/// Create spatial dropout layer
pub fn spatial_new(rate: Float) -> SpatialDropoutLayer {
  let clamped_rate = float.min(float.max(rate, 0.0), 1.0)
  SpatialDropoutLayer(rate: clamped_rate, mode: Training)
}

/// Spatial dropout forward
/// For 4D input [batch, channels, height, width], drops entire channels
pub fn spatial_forward(
  layer: SpatialDropoutLayer,
  input: Tensor,
) -> Result(#(Tensor, DropoutCache), TensorError) {
  case layer.mode {
    Inference -> {
      let cache =
        DropoutCache(
          mask: tensor.ones(input.shape),
          scale: 1.0,
          input_shape: input.shape,
        )
      Ok(#(input, cache))
    }
    Training -> {
      case input.shape {
        [batch, channels, height, width] -> {
          // Create mask for [batch, channels] and broadcast to full shape
          let keep_prob = 1.0 -. layer.rate
          let #(channel_mask, scale) =
            utils.dropout_mask([batch, channels], keep_prob)

          // Broadcast mask to full spatial dimensions
          let full_mask_data =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              list.range(0, channels - 1)
              |> list.flat_map(fn(c) {
                // Get mask value for this batch/channel
                let idx = b * channels + c
                let mask_val = case list_at(td(channel_mask), idx) {
                  Ok(v) -> v
                  Error(_) -> 1.0
                }
                // Repeat for all spatial positions
                list.repeat(mask_val, height * width)
              })
            })

          let full_mask =
            Tensor(data: full_mask_data, shape: [batch, channels, height, width])

          use masked <- result.try(tensor.mul(input, full_mask))
          let output = tensor.scale(masked, scale)

          let cache =
            DropoutCache(
              mask: full_mask,
              scale: scale,
              input_shape: input.shape,
            )
          Ok(#(output, cache))
        }
        _ -> {
          // Fallback to regular dropout for non-4D tensors
          forward_training(
            DropoutLayer(rate: layer.rate, mode: Training),
            input,
          )
        }
      }
    }
  }
}

// =============================================================================
// ALPHA DROPOUT (for SELU networks)
// =============================================================================

/// Alpha Dropout - maintains mean and variance for self-normalizing networks
/// Used with SELU activation
pub type AlphaDropoutLayer {
  AlphaDropoutLayer(rate: Float, mode: DropoutMode)
}

/// Alpha dropout constants (for SELU self-normalization)
const alpha_p: Float = -1.7580993408473766
// a_prime constant removed - unused

/// Create alpha dropout layer
pub fn alpha_new(rate: Float) -> AlphaDropoutLayer {
  let clamped_rate = float.min(float.max(rate, 0.0), 1.0)
  AlphaDropoutLayer(rate: clamped_rate, mode: Training)
}

/// Alpha dropout forward
/// Maintains self-normalizing property when used with SELU
pub fn alpha_forward(
  layer: AlphaDropoutLayer,
  input: Tensor,
) -> Result(#(Tensor, DropoutCache), TensorError) {
  case layer.mode {
    Inference -> {
      let cache =
        DropoutCache(
          mask: tensor.ones(input.shape),
          scale: 1.0,
          input_shape: input.shape,
        )
      Ok(#(input, cache))
    }
    Training -> {
      let keep_prob = 1.0 -. layer.rate

      // Compute affine transformation parameters
      let a =
        float_pow(
          keep_prob +. alpha_p *. alpha_p *. keep_prob *. { 1.0 -. keep_prob },
          -0.5,
        )
      let b = 0.0 -. a *. { 1.0 -. keep_prob } *. alpha_p

      // Generate mask
      let mask = utils.random_bernoulli(input.shape, keep_prob)

      // Where mask is 0, replace with alpha_p (saturation value)
      let dropout_data =
        list.map2(td(input), td(mask), fn(x, m) {
          case m >. 0.5 {
            True -> x
            False -> alpha_p
          }
        })

      let dropout_tensor = Tensor(data: dropout_data, shape: input.shape)

      // Apply affine transformation: a * x + b
      let output = tensor.add_scalar(tensor.scale(dropout_tensor, a), b)

      let cache = DropoutCache(mask: mask, scale: a, input_shape: input.shape)
      Ok(#(output, cache))
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  lst
  |> list.drop(index)
  |> list.first
}

@external(erlang, "math", "pow")
fn float_pow(base: Float, exp: Float) -> Float
