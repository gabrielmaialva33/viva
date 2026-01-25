//// Train - Neural network training
////
//// Implements backpropagation, loss functions, and training loops.
//// Functional design: no mutation, returns newly trained networks.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva/neural/layer.{DenseGradients}
import viva/neural/network.{
  type MomentumState, type Network, type NetworkGradients,
}
import viva/neural/tensor.{type Tensor, type TensorError}

// =============================================================================
// TYPES
// =============================================================================

/// Loss function type
pub type LossType {
  MSE
  CrossEntropy
  BinaryCrossEntropy
  MAE
}

/// Training configuration
pub type TrainConfig {
  TrainConfig(
    /// Learning rate
    learning_rate: Float,
    /// Momentum (0 = no momentum)
    momentum: Float,
    /// Number of epochs
    epochs: Int,
    /// Batch size
    batch_size: Int,
    /// Loss function
    loss: LossType,
    /// L2 regularization (weight decay)
    l2_lambda: Float,
    /// Gradient clipping (0 = disabled)
    gradient_clip: Float,
    /// Log every N epochs (0 = disabled)
    log_interval: Int,
  )
}

/// Loss result + gradient
pub type LossResult {
  LossResult(
    /// Loss value
    loss: Float,
    /// Loss gradient with respect to output
    gradient: Tensor,
  )
}

/// Training metrics
pub type TrainMetrics {
  TrainMetrics(
    /// Average epoch loss
    epoch_loss: Float,
    /// Number of samples processed
    samples: Int,
    /// Current epoch
    epoch: Int,
  )
}

/// Input/target pair for training
pub type Sample {
  Sample(input: Tensor, target: Tensor)
}

// =============================================================================
// DEFAULT CONFIG
// =============================================================================

/// Default training config
pub fn default_config() -> TrainConfig {
  TrainConfig(
    learning_rate: 0.01,
    momentum: 0.9,
    epochs: 100,
    batch_size: 32,
    loss: MSE,
    l2_lambda: 0.0,
    gradient_clip: 0.0,
    log_interval: 10,
  )
}

/// Config for classification
pub fn classification_config() -> TrainConfig {
  TrainConfig(..default_config(), loss: CrossEntropy, learning_rate: 0.001)
}

/// Config for regression
pub fn regression_config() -> TrainConfig {
  TrainConfig(..default_config(), loss: MSE, learning_rate: 0.01)
}

// =============================================================================
// LOSS FUNCTIONS
// =============================================================================

/// Compute loss and gradient
pub fn compute_loss(
  predicted: Tensor,
  target: Tensor,
  loss_type: LossType,
) -> Result(LossResult, TensorError) {
  case loss_type {
    MSE -> mse_loss(predicted, target)
    CrossEntropy -> cross_entropy_loss(predicted, target)
    BinaryCrossEntropy -> binary_cross_entropy_loss(predicted, target)
    MAE -> mae_loss(predicted, target)
  }
}

/// Mean Squared Error: (1/n) * sum((pred - target)^2)
fn mse_loss(
  predicted: Tensor,
  target: Tensor,
) -> Result(LossResult, TensorError) {
  use diff <- result.try(tensor.sub(predicted, target))

  // Loss
  let squared = tensor.map(diff, fn(x) { x *. x })
  let loss = tensor.mean(squared)

  // Gradient: 2 * (pred - target) / n
  let n = int.to_float(tensor.size(predicted))
  let gradient = tensor.scale(diff, 2.0 /. n)

  Ok(LossResult(loss: loss, gradient: gradient))
}

/// Mean Absolute Error: (1/n) * sum(|pred - target|)
fn mae_loss(
  predicted: Tensor,
  target: Tensor,
) -> Result(LossResult, TensorError) {
  use diff <- result.try(tensor.sub(predicted, target))

  // Loss
  let abs_diff = tensor.map(diff, fn(x) { float.absolute_value(x) })
  let loss = tensor.mean(abs_diff)

  // Gradient: sign(pred - target) / n
  let n = int.to_float(tensor.size(predicted))
  let gradient =
    tensor.map(diff, fn(x) {
      case x >. 0.0 {
        True -> 1.0 /. n
        False ->
          case x <. 0.0 {
            True -> -1.0 /. n
            False -> 0.0
          }
      }
    })

  Ok(LossResult(loss: loss, gradient: gradient))
}

/// Cross Entropy: -sum(target * log(pred))
fn cross_entropy_loss(
  predicted: Tensor,
  target: Tensor,
) -> Result(LossResult, TensorError) {
  // Clip for numerical stability
  let eps = 0.0000001
  let clipped = tensor.clamp(predicted, eps, 1.0 -. eps)

  // Loss
  let log_pred = tensor.map(clipped, float_log)
  use product <- result.try(tensor.mul(target, log_pred))
  let loss = 0.0 -. tensor.sum(product)

  // Gradient: -target / pred
  let gradient =
    tensor.map_indexed(clipped, fn(p, i) {
      case tensor.get(target, i) {
        Ok(t) -> 0.0 -. t /. p
        Error(_) -> 0.0
      }
    })

  Ok(LossResult(loss: loss, gradient: gradient))
}

/// Binary Cross Entropy: -sum(target * log(pred) + (1-target) * log(1-pred))
fn binary_cross_entropy_loss(
  predicted: Tensor,
  target: Tensor,
) -> Result(LossResult, TensorError) {
  let eps = 0.0000001
  let clipped = tensor.clamp(predicted, eps, 1.0 -. eps)

  // Loss for each element
  let losses =
    list.map2(clipped.data, target.data, fn(p, t) {
      0.0 -. { t *. float_log(p) +. { 1.0 -. t } *. float_log(1.0 -. p) }
    })
  let loss =
    list.fold(losses, 0.0, fn(a, b) { a +. b })
    /. int.to_float(list.length(losses))

  // Gradient
  let gradient_data =
    list.map2(clipped.data, target.data, fn(p, t) {
      { p -. t } /. { p *. { 1.0 -. p } }
    })
  let n = int.to_float(list.length(gradient_data))
  let scaled = list.map(gradient_data, fn(g) { g /. n })

  Ok(LossResult(
    loss: loss,
    gradient: tensor.Tensor(data: scaled, shape: predicted.shape),
  ))
}

// =============================================================================
// SINGLE STEP TRAINING
// =============================================================================

/// Single training step (forward + backward + update)
pub fn train_step(
  net: Network,
  input: Tensor,
  target: Tensor,
  config: TrainConfig,
) -> Result(#(Network, Float), TensorError) {
  // Forward
  use #(output, cache) <- result.try(network.forward(net, input))

  // Compute loss
  use loss_result <- result.try(compute_loss(output, target, config.loss))

  // Backward
  use gradients <- result.try(network.backward(net, cache, loss_result.gradient))

  // Apply gradient clipping if configured
  let gradients = case config.gradient_clip >. 0.0 {
    True -> clip_gradients(gradients, config.gradient_clip)
    False -> gradients
  }

  // Apply L2 regularization if configured
  let gradients = case config.l2_lambda >. 0.0 {
    True -> add_l2_gradients(net, gradients, config.l2_lambda)
    False -> gradients
  }

  // Update
  use updated_net <- result.try(network.update_sgd(
    net,
    gradients,
    config.learning_rate,
  ))

  Ok(#(updated_net, loss_result.loss))
}

/// Single step with momentum
pub fn train_step_momentum(
  net: Network,
  momentum_state: MomentumState,
  input: Tensor,
  target: Tensor,
  config: TrainConfig,
) -> Result(#(Network, MomentumState, Float), TensorError) {
  // Forward
  use #(output, cache) <- result.try(network.forward(net, input))

  // Compute loss
  use loss_result <- result.try(compute_loss(output, target, config.loss))

  // Backward
  use gradients <- result.try(network.backward(net, cache, loss_result.gradient))

  // Gradient processing
  let gradients = case config.gradient_clip >. 0.0 {
    True -> clip_gradients(gradients, config.gradient_clip)
    False -> gradients
  }

  // Update with momentum
  use #(updated_net, new_momentum) <- result.try(network.update_momentum(
    net,
    gradients,
    momentum_state,
    config.learning_rate,
    config.momentum,
  ))

  Ok(#(updated_net, new_momentum, loss_result.loss))
}

// =============================================================================
// BATCH TRAINING
// =============================================================================

/// Train on a batch of samples
pub fn train_batch(
  net: Network,
  samples: List(Sample),
  config: TrainConfig,
) -> Result(#(Network, Float), TensorError) {
  case samples {
    [] -> Ok(#(net, 0.0))
    _ -> {
      // Train sequentially and accumulate loss
      let result =
        list.try_fold(samples, #(net, 0.0), fn(acc, sample) {
          let #(current_net, total_loss) = acc
          case train_step(current_net, sample.input, sample.target, config) {
            Ok(#(updated, loss)) -> Ok(#(updated, total_loss +. loss))
            Error(e) -> Error(e)
          }
        })

      case result {
        Ok(#(final_net, total_loss)) -> {
          let avg_loss = total_loss /. int.to_float(list.length(samples))
          Ok(#(final_net, avg_loss))
        }
        Error(e) -> Error(e)
      }
    }
  }
}

// =============================================================================
// FULL TRAINING LOOP
// =============================================================================

/// Train for multiple epochs
pub fn fit(
  net: Network,
  samples: List(Sample),
  config: TrainConfig,
) -> Result(#(Network, List(TrainMetrics)), TensorError) {
  fit_loop(net, samples, config, 0, [])
}

fn fit_loop(
  net: Network,
  samples: List(Sample),
  config: TrainConfig,
  epoch: Int,
  metrics: List(TrainMetrics),
) -> Result(#(Network, List(TrainMetrics)), TensorError) {
  case epoch >= config.epochs {
    True -> Ok(#(net, list.reverse(metrics)))
    False -> {
      // Shuffle and split into batches
      let shuffled = shuffle_samples(samples)
      let batches = split_chunks(shuffled, config.batch_size)

      // Train on each batch
      let epoch_result =
        list.try_fold(batches, #(net, 0.0, 0), fn(acc, batch) {
          let #(current_net, total_loss, count) = acc
          case train_batch(current_net, batch, config) {
            Ok(#(updated, batch_loss)) -> {
              let batch_count = list.length(batch)
              Ok(#(
                updated,
                total_loss +. batch_loss *. int.to_float(batch_count),
                count + batch_count,
              ))
            }
            Error(e) -> Error(e)
          }
        })

      case epoch_result {
        Ok(#(updated_net, total_loss, sample_count)) -> {
          let avg_loss = total_loss /. int.to_float(sample_count)
          let epoch_metrics =
            TrainMetrics(
              epoch_loss: avg_loss,
              samples: sample_count,
              epoch: epoch + 1,
            )

          fit_loop(updated_net, samples, config, epoch + 1, [
            epoch_metrics,
            ..metrics
          ])
        }
        Error(e) -> Error(e)
      }
    }
  }
}

// =============================================================================
// EVALUATION
// =============================================================================

/// Evaluate network on test dataset
pub fn evaluate(
  net: Network,
  samples: List(Sample),
  loss_type: LossType,
) -> Result(Float, TensorError) {
  case samples {
    [] -> Ok(0.0)
    _ -> {
      let result =
        list.try_fold(samples, 0.0, fn(total_loss, sample) {
          case network.predict(net, sample.input) {
            Ok(output) -> {
              case compute_loss(output, sample.target, loss_type) {
                Ok(loss_result) -> Ok(total_loss +. loss_result.loss)
                Error(e) -> Error(e)
              }
            }
            Error(e) -> Error(e)
          }
        })

      case result {
        Ok(total) -> Ok(total /. int.to_float(list.length(samples)))
        Error(e) -> Error(e)
      }
    }
  }
}

/// Compute accuracy for classification
pub fn accuracy(
  net: Network,
  samples: List(Sample),
) -> Result(Float, TensorError) {
  case samples {
    [] -> Ok(0.0)
    _ -> {
      let result =
        list.try_fold(samples, 0, fn(correct, sample) {
          case network.predict(net, sample.input) {
            Ok(output) -> {
              let pred_class = tensor.argmax(output)
              let true_class = tensor.argmax(sample.target)
              case pred_class == true_class {
                True -> Ok(correct + 1)
                False -> Ok(correct)
              }
            }
            Error(e) -> Error(e)
          }
        })

      case result {
        Ok(correct) ->
          Ok(int.to_float(correct) /. int.to_float(list.length(samples)))
        Error(e) -> Error(e)
      }
    }
  }
}

// =============================================================================
// GRADIENT UTILITIES
// =============================================================================

/// Clip gradients to avoid explosion
fn clip_gradients(grads: NetworkGradients, max_norm: Float) -> NetworkGradients {
  let clipped =
    list.map(grads.layer_gradients, fn(lg) {
      let w_norm = tensor.norm(lg.d_weights)
      let b_norm = tensor.norm(lg.d_biases)

      let d_weights = case w_norm >. max_norm {
        True -> tensor.scale(lg.d_weights, max_norm /. w_norm)
        False -> lg.d_weights
      }

      let d_biases = case b_norm >. max_norm {
        True -> tensor.scale(lg.d_biases, max_norm /. b_norm)
        False -> lg.d_biases
      }

      DenseGradients(
        d_weights: d_weights,
        d_biases: d_biases,
        d_input: lg.d_input,
      )
    })

  network.NetworkGradients(layer_gradients: clipped)
}

/// Add L2 regularization to gradients
fn add_l2_gradients(
  net: Network,
  grads: NetworkGradients,
  lambda: Float,
) -> NetworkGradients {
  let paired = list.zip(net.layers, grads.layer_gradients)

  let updated =
    list.map(paired, fn(pair) {
      let #(layer, lg) = pair
      // d_weights += lambda * weights
      let l2_term = tensor.scale(layer.weights, lambda)
      let d_weights = case tensor.add(lg.d_weights, l2_term) {
        Ok(w) -> w
        Error(_) -> lg.d_weights
      }

      DenseGradients(
        d_weights: d_weights,
        d_biases: lg.d_biases,
        d_input: lg.d_input,
      )
    })

  network.NetworkGradients(layer_gradients: updated)
}

// =============================================================================
// DATA UTILITIES
// =============================================================================

/// Create sample from lists
pub fn sample(input: List(Float), target: List(Float)) -> Sample {
  Sample(input: tensor.from_list(input), target: tensor.from_list(target))
}

/// Split list into chunks
fn split_chunks(items: List(a), chunk_size: Int) -> List(List(a)) {
  case items {
    [] -> []
    _ -> {
      let #(head, tail) = list.split(items, chunk_size)
      [head, ..split_chunks(tail, chunk_size)]
    }
  }
}

/// Simple shuffle (simplified Fisher-Yates)
fn shuffle_samples(samples: List(Sample)) -> List(Sample) {
  samples
  |> list.map(fn(s) { #(random_float(), s) })
  |> list.sort(fn(a, b) { float.compare(a.0, b.0) })
  |> list.map(fn(pair) { pair.1 })
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "math", "log")
fn float_log(x: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_float() -> Float
