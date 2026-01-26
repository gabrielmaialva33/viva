//// Normalization - BatchNorm, LayerNorm, and other normalization techniques
////
//// Implements normalization layers essential for deep networks.
//// Based on the formula: y = (x - mean) / sqrt(var + eps) * gamma + beta

import gleam/int
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

/// Batch Normalization layer
pub type BatchNormLayer {
  BatchNormLayer(
    /// Learnable scale parameter [num_features]
    gamma: Tensor,
    /// Learnable shift parameter [num_features]
    beta: Tensor,
    /// Running mean for inference (EMA)
    running_mean: Tensor,
    /// Running variance for inference (EMA)
    running_var: Tensor,
    /// Number of features (channels)
    num_features: Int,
    /// Momentum for running stats update (default 0.1)
    momentum: Float,
    /// Small constant for numerical stability (default 1e-5)
    epsilon: Float,
    /// Current mode
    mode: NormMode,
  )
}

/// Normalization mode
pub type NormMode {
  /// Training: compute stats from batch, update running stats
  Training
  /// Inference: use running stats
  Inference
}

/// Cache for backward pass
pub type BatchNormCache {
  BatchNormCache(
    /// Normalized values (before gamma/beta)
    normalized: Tensor,
    /// Batch mean
    mean: Tensor,
    /// Batch variance
    var: Tensor,
    /// Inverse std: 1/sqrt(var + eps)
    inv_std: Tensor,
    /// Original input
    input: Tensor,
    /// Centered input (x - mean)
    centered: Tensor,
  )
}

/// Gradients from BatchNorm backward
pub type BatchNormGradients {
  BatchNormGradients(
    /// Gradient w.r.t. input
    d_input: Tensor,
    /// Gradient w.r.t. gamma
    d_gamma: Tensor,
    /// Gradient w.r.t. beta
    d_beta: Tensor,
  )
}

/// Layer Normalization
pub type LayerNormLayer {
  LayerNormLayer(
    /// Learnable scale
    gamma: Tensor,
    /// Learnable shift
    beta: Tensor,
    /// Normalized shape
    normalized_shape: List(Int),
    /// Epsilon
    epsilon: Float,
  )
}

// =============================================================================
// BATCH NORM CONSTRUCTORS
// =============================================================================

/// Create new batch normalization layer
pub fn batch_norm_new(num_features: Int) -> BatchNormLayer {
  BatchNormLayer(
    gamma: tensor.ones([num_features]),
    beta: tensor.zeros([num_features]),
    running_mean: tensor.zeros([num_features]),
    running_var: tensor.ones([num_features]),
    num_features: num_features,
    momentum: 0.1,
    epsilon: 0.00001,
    mode: Training,
  )
}

/// Create batch norm with custom momentum and epsilon
pub fn batch_norm_with_opts(
  num_features: Int,
  momentum: Float,
  epsilon: Float,
) -> BatchNormLayer {
  BatchNormLayer(
    gamma: tensor.ones([num_features]),
    beta: tensor.zeros([num_features]),
    running_mean: tensor.zeros([num_features]),
    running_var: tensor.ones([num_features]),
    num_features: num_features,
    momentum: momentum,
    epsilon: epsilon,
    mode: Training,
  )
}

/// Set mode to training
pub fn batch_norm_train(layer: BatchNormLayer) -> BatchNormLayer {
  BatchNormLayer(..layer, mode: Training)
}

/// Set mode to evaluation
pub fn batch_norm_eval(layer: BatchNormLayer) -> BatchNormLayer {
  BatchNormLayer(..layer, mode: Inference)
}

// =============================================================================
// BATCH NORM FORWARD
// =============================================================================

/// Forward pass through batch normalization
/// Input shape: [batch_size, num_features] for 1D
/// or [batch_size, num_features, ...] for higher dims
pub fn batch_norm_forward(
  layer: BatchNormLayer,
  input: Tensor,
) -> Result(#(Tensor, BatchNormCache, BatchNormLayer), TensorError) {
  case layer.mode {
    Training -> batch_norm_forward_training(layer, input)
    Inference -> batch_norm_forward_inference(layer, input)
  }
}

/// Training mode forward pass
fn batch_norm_forward_training(
  layer: BatchNormLayer,
  input: Tensor,
) -> Result(#(Tensor, BatchNormCache, BatchNormLayer), TensorError) {
  case input.shape {
    [batch_size, num_features] -> {
      // 2D input: [batch, features]
      // Compute mean and variance along batch dimension (axis 0)
      use mean <- result.try(tensor.mean_axis(input, 0))
      use var_tensor <- result.try(utils.variance_axis(input, 0))

      // Normalize: (x - mean) / sqrt(var + eps)
      use mean_broadcast <- result.try(broadcast_to_batch(
        mean,
        batch_size,
        num_features,
      ))
      use centered <- result.try(tensor.sub(input, mean_broadcast))

      let inv_std =
        tensor.map(var_tensor, fn(v) { 1.0 /. float_sqrt(v +. layer.epsilon) })

      use inv_std_broadcast <- result.try(broadcast_to_batch(
        inv_std,
        batch_size,
        num_features,
      ))
      use normalized <- result.try(tensor.mul(centered, inv_std_broadcast))

      // Apply affine transform: gamma * normalized + beta
      use gamma_broadcast <- result.try(broadcast_to_batch(
        layer.gamma,
        batch_size,
        num_features,
      ))
      use beta_broadcast <- result.try(broadcast_to_batch(
        layer.beta,
        batch_size,
        num_features,
      ))

      use scaled <- result.try(tensor.mul(normalized, gamma_broadcast))
      use output <- result.try(tensor.add(scaled, beta_broadcast))

      // Update running stats with EMA
      let new_running_mean =
        update_ema(mean, layer.running_mean, layer.momentum)
      let new_running_var =
        update_ema(var_tensor, layer.running_var, layer.momentum)

      let new_layer =
        BatchNormLayer(
          ..layer,
          running_mean: new_running_mean,
          running_var: new_running_var,
        )

      let cache =
        BatchNormCache(
          normalized: normalized,
          mean: mean,
          var: var_tensor,
          inv_std: inv_std,
          input: input,
          centered: centered,
        )

      Ok(#(output, cache, new_layer))
    }
    _ ->
      Error(tensor.DimensionError("BatchNorm expects [batch, features] input"))
  }
}

/// Inference mode forward pass
fn batch_norm_forward_inference(
  layer: BatchNormLayer,
  input: Tensor,
) -> Result(#(Tensor, BatchNormCache, BatchNormLayer), TensorError) {
  case input.shape {
    [batch_size, num_features] -> {
      // Use running stats
      use mean_broadcast <- result.try(broadcast_to_batch(
        layer.running_mean,
        batch_size,
        num_features,
      ))
      use centered <- result.try(tensor.sub(input, mean_broadcast))

      let inv_std =
        tensor.map(layer.running_var, fn(v) {
          1.0 /. float_sqrt(v +. layer.epsilon)
        })

      use inv_std_broadcast <- result.try(broadcast_to_batch(
        inv_std,
        batch_size,
        num_features,
      ))
      use normalized <- result.try(tensor.mul(centered, inv_std_broadcast))

      // Apply affine transform
      use gamma_broadcast <- result.try(broadcast_to_batch(
        layer.gamma,
        batch_size,
        num_features,
      ))
      use beta_broadcast <- result.try(broadcast_to_batch(
        layer.beta,
        batch_size,
        num_features,
      ))

      use scaled <- result.try(tensor.mul(normalized, gamma_broadcast))
      use output <- result.try(tensor.add(scaled, beta_broadcast))

      // Dummy cache for inference (not used in backward)
      let cache =
        BatchNormCache(
          normalized: normalized,
          mean: layer.running_mean,
          var: layer.running_var,
          inv_std: inv_std,
          input: input,
          centered: centered,
        )

      Ok(#(output, cache, layer))
    }
    _ ->
      Error(tensor.DimensionError("BatchNorm expects [batch, features] input"))
  }
}

// =============================================================================
// BATCH NORM BACKWARD
// =============================================================================

/// Backward pass through batch normalization
/// Complex due to batch statistics dependency
pub fn batch_norm_backward(
  layer: BatchNormLayer,
  cache: BatchNormCache,
  upstream: Tensor,
) -> Result(BatchNormGradients, TensorError) {
  case upstream.shape {
    [batch_size, num_features] -> {
      let n = int.to_float(batch_size)

      // d_beta = sum(upstream, axis=0)
      use d_beta <- result.try(tensor.sum_axis(upstream, 0))

      // d_gamma = sum(upstream * normalized, axis=0)
      use upstream_normalized <- result.try(tensor.mul(
        upstream,
        cache.normalized,
      ))
      use d_gamma <- result.try(tensor.sum_axis(upstream_normalized, 0))

      // d_normalized = upstream * gamma
      use gamma_broadcast <- result.try(broadcast_to_batch(
        layer.gamma,
        batch_size,
        num_features,
      ))
      use d_normalized <- result.try(tensor.mul(upstream, gamma_broadcast))

      // d_var = sum(d_normalized * centered * -0.5 * (var + eps)^(-1.5), axis=0)
      use d_norm_centered <- result.try(tensor.mul(d_normalized, cache.centered))
      let inv_var_cubed =
        tensor.map(cache.var, fn(v) {
          -0.5 *. float_pow(v +. layer.epsilon, -1.5)
        })
      use inv_var_cubed_broadcast <- result.try(broadcast_to_batch(
        inv_var_cubed,
        batch_size,
        num_features,
      ))
      use d_var_per_sample <- result.try(tensor.mul(
        d_norm_centered,
        inv_var_cubed_broadcast,
      ))
      use d_var <- result.try(tensor.sum_axis(d_var_per_sample, 0))

      // d_mean = sum(d_normalized * -inv_std, axis=0) + d_var * mean(-2 * centered) / n
      use inv_std_broadcast <- result.try(broadcast_to_batch(
        cache.inv_std,
        batch_size,
        num_features,
      ))
      let neg_inv_std = tensor.negate(inv_std_broadcast)
      use d_mean_term1 <- result.try(tensor.mul(d_normalized, neg_inv_std))
      use d_mean_part1 <- result.try(tensor.sum_axis(d_mean_term1, 0))

      let neg_2_centered = tensor.scale(cache.centered, -2.0)
      use sum_neg_2_centered <- result.try(tensor.sum_axis(neg_2_centered, 0))
      let mean_neg_2_centered = tensor.scale(sum_neg_2_centered, 1.0 /. n)
      use d_mean_part2 <- result.try(tensor.mul(d_var, mean_neg_2_centered))
      use d_mean <- result.try(tensor.add(d_mean_part1, d_mean_part2))

      // d_input = d_normalized * inv_std + d_var * 2 * centered / n + d_mean / n
      use d_input_term1 <- result.try(tensor.mul(
        d_normalized,
        inv_std_broadcast,
      ))

      use d_var_broadcast <- result.try(broadcast_to_batch(
        d_var,
        batch_size,
        num_features,
      ))
      let centered_scaled = tensor.scale(cache.centered, 2.0 /. n)
      use d_input_term2 <- result.try(tensor.mul(
        d_var_broadcast,
        centered_scaled,
      ))

      use d_mean_broadcast <- result.try(broadcast_to_batch(
        d_mean,
        batch_size,
        num_features,
      ))
      let d_input_term3 = tensor.scale(d_mean_broadcast, 1.0 /. n)

      use d_input_partial <- result.try(tensor.add(d_input_term1, d_input_term2))
      use d_input <- result.try(tensor.add(d_input_partial, d_input_term3))

      Ok(BatchNormGradients(d_input: d_input, d_gamma: d_gamma, d_beta: d_beta))
    }
    _ -> Error(tensor.DimensionError("Upstream shape mismatch"))
  }
}

// =============================================================================
// LAYER NORMALIZATION
// =============================================================================

/// Create new layer normalization
pub fn layer_norm_new(normalized_shape: List(Int)) -> LayerNormLayer {
  let size = list.fold(normalized_shape, 1, fn(acc, dim) { acc * dim })
  LayerNormLayer(
    gamma: tensor.ones([size]),
    beta: tensor.zeros([size]),
    normalized_shape: normalized_shape,
    epsilon: 0.00001,
  )
}

/// Layer norm forward pass
/// Normalizes over the last dimensions (unlike batch norm which normalizes over batch)
pub fn layer_norm_forward(
  layer: LayerNormLayer,
  input: Tensor,
) -> Result(#(Tensor, LayerNormCache), TensorError) {
  case input.shape {
    [batch_size, features] -> {
      // Normalize each sample independently
      let result_data =
        list.range(0, batch_size - 1)
        |> list.flat_map(fn(b) {
          let start = b * features
          let sample_data =
            td(input)
            |> list.drop(start)
            |> list.take(features)

          // Compute mean and variance for this sample
          let mean =
            list.fold(sample_data, 0.0, fn(acc, x) { acc +. x })
            /. int.to_float(features)
          let variance =
            list.fold(sample_data, 0.0, fn(acc, x) {
              let diff = x -. mean
              acc +. diff *. diff
            })
            /. int.to_float(features)
          let inv_std = 1.0 /. float_sqrt(variance +. layer.epsilon)

          // Normalize and apply gamma/beta
          list.index_map(sample_data, fn(x, i) {
            let normalized = { x -. mean } *. inv_std
            let gamma_i = case list_at(td(layer.gamma), i) {
              Ok(g) -> g
              Error(_) -> 1.0
            }
            let beta_i = case list_at(td(layer.beta), i) {
              Ok(b) -> b
              Error(_) -> 0.0
            }
            normalized *. gamma_i +. beta_i
          })
        })

      let output = Tensor(data: result_data, shape: [batch_size, features])
      let cache = LayerNormCache(input: input, mean: tensor.zeros([batch_size]))
      Ok(#(output, cache))
    }
    _ ->
      Error(tensor.DimensionError("LayerNorm expects [batch, features] input"))
  }
}

/// Cache for layer norm backward
pub type LayerNormCache {
  LayerNormCache(input: Tensor, mean: Tensor)
}

// =============================================================================
// GROUP NORMALIZATION
// =============================================================================

/// Group Normalization layer
pub type GroupNormLayer {
  GroupNormLayer(
    gamma: Tensor,
    beta: Tensor,
    num_groups: Int,
    num_channels: Int,
    epsilon: Float,
  )
}

/// Create group normalization layer
pub fn group_norm_new(num_groups: Int, num_channels: Int) -> GroupNormLayer {
  GroupNormLayer(
    gamma: tensor.ones([num_channels]),
    beta: tensor.zeros([num_channels]),
    num_groups: num_groups,
    num_channels: num_channels,
    epsilon: 0.00001,
  )
}

/// Group norm forward
/// Splits channels into groups and normalizes each group independently
pub fn group_norm_forward(
  layer: GroupNormLayer,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  case input.shape {
    [batch_size, num_channels] -> {
      let channels_per_group = num_channels / layer.num_groups

      let result_data =
        list.range(0, batch_size - 1)
        |> list.flat_map(fn(b) {
          let sample_start = b * num_channels
          list.range(0, layer.num_groups - 1)
          |> list.flat_map(fn(g) {
            let group_start = sample_start + g * channels_per_group
            let group_data =
              td(input)
              |> list.drop(group_start)
              |> list.take(channels_per_group)

            // Compute mean and variance for this group
            let mean =
              list.fold(group_data, 0.0, fn(acc, x) { acc +. x })
              /. int.to_float(channels_per_group)
            let variance =
              list.fold(group_data, 0.0, fn(acc, x) {
                let diff = x -. mean
                acc +. diff *. diff
              })
              /. int.to_float(channels_per_group)
            let inv_std = 1.0 /. float_sqrt(variance +. layer.epsilon)

            // Normalize and apply gamma/beta
            list.index_map(group_data, fn(x, i) {
              let global_channel = g * channels_per_group + i
              let normalized = { x -. mean } *. inv_std
              let gamma_c = case list_at(td(layer.gamma), global_channel) {
                Ok(g) -> g
                Error(_) -> 1.0
              }
              let beta_c = case list_at(td(layer.beta), global_channel) {
                Ok(b) -> b
                Error(_) -> 0.0
              }
              normalized *. gamma_c +. beta_c
            })
          })
        })

      Ok(Tensor(data: result_data, shape: [batch_size, num_channels]))
    }
    _ ->
      Error(tensor.DimensionError("GroupNorm expects [batch, channels] input"))
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Update exponential moving average
fn update_ema(current: Tensor, running: Tensor, momentum: Float) -> Tensor {
  // new_running = (1 - momentum) * running + momentum * current
  let scaled_running = tensor.scale(running, 1.0 -. momentum)
  let scaled_current = tensor.scale(current, momentum)
  case tensor.add(scaled_running, scaled_current) {
    Ok(result) -> result
    Error(_) -> running
  }
}

/// Broadcast 1D tensor to 2D [batch, features]
fn broadcast_to_batch(
  t: Tensor,
  batch_size: Int,
  num_features: Int,
) -> Result(Tensor, TensorError) {
  let tdata = td(t)
  case t.shape {
    [n] if n == num_features -> {
      // Repeat data for each batch
      let data = list.flatten(list.repeat(tdata, batch_size))
      Ok(Tensor(data: data, shape: [batch_size, num_features]))
    }
    [1] -> {
      // Single value, broadcast to all
      let value = case list.first(tdata) {
        Ok(v) -> v
        Error(_) -> 0.0
      }
      let data = list.repeat(value, batch_size * num_features)
      Ok(Tensor(data: data, shape: [batch_size, num_features]))
    }
    _ -> Error(tensor.ShapeMismatch(expected: [num_features], got: t.shape))
  }
}

fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  lst
  |> list.drop(index)
  |> list.first
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "pow")
fn float_pow(base: Float, exp: Float) -> Float
