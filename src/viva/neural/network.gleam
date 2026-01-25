//// Network - Layer composition for neural networks
////
//// A network is a sequence of layers that process data from input to output.
//// Supports forward pass, backpropagation, and parameter updates.

import gleam/int
import gleam/list
import viva/neural/activation.{type ActivationType}
import viva/neural/layer.{type DenseCache, type DenseGradients, type DenseLayer}
import viva/neural/tensor.{type Tensor, type TensorError}

// =============================================================================
// TYPES
// =============================================================================

/// Sequential neural network
pub type Network {
  Network(
    /// List of layers (from input to output)
    layers: List(DenseLayer),
    /// Input size
    input_size: Int,
    /// Output size
    output_size: Int,
  )
}

/// Complete forward pass cache (for backprop)
pub type NetworkCache {
  NetworkCache(
    /// Cache for each layer
    layer_caches: List(DenseCache),
  )
}

/// Gradients for entire network
pub type NetworkGradients {
  NetworkGradients(
    /// Gradients for each layer
    layer_gradients: List(DenseGradients),
  )
}

/// Builder for fluent network construction
pub type NetworkBuilder {
  NetworkBuilder(layers: List(DenseLayer), last_size: Int)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create network from list of layer sizes
/// Ex: new([784, 128, 64, 10], activation.ReLU, activation.Softmax)
pub fn new(
  layer_sizes: List(Int),
  hidden_activation: ActivationType,
  output_activation: ActivationType,
) -> Result(Network, TensorError) {
  case layer_sizes {
    [] | [_] -> Error(tensor.InvalidShape("Need at least 2 layer sizes"))
    [input_size, ..rest] -> {
      let layers =
        build_layers(input_size, rest, hidden_activation, output_activation)
      let output_size = case list.last(layer_sizes) {
        Ok(size) -> size
        Error(_) -> 0
      }
      Ok(Network(
        layers: layers,
        input_size: input_size,
        output_size: output_size,
      ))
    }
  }
}

/// Build layers recursively
fn build_layers(
  prev_size: Int,
  remaining: List(Int),
  hidden_activation: ActivationType,
  output_activation: ActivationType,
) -> List(DenseLayer) {
  case remaining {
    [] -> []
    [size] -> {
      // Last layer uses output_activation
      [layer.dense(prev_size, size, output_activation)]
    }
    [size, ..rest] -> {
      // Intermediate layers use hidden_activation
      let l = layer.dense(prev_size, size, hidden_activation)
      [l, ..build_layers(size, rest, hidden_activation, output_activation)]
    }
  }
}

/// Create network from list of pre-built layers
pub fn from_layers(layers: List(DenseLayer)) -> Result(Network, TensorError) {
  case layers {
    [] -> Error(tensor.InvalidShape("Need at least one layer"))
    [first, ..] -> {
      let input_size = first.input_size
      let output_size = case list.last(layers) {
        Ok(last) -> last.output_size
        Error(_) -> 0
      }

      // Validate that layers are compatible
      case validate_layers(layers) {
        True ->
          Ok(Network(
            layers: layers,
            input_size: input_size,
            output_size: output_size,
          ))
        False -> Error(tensor.InvalidShape("Layer dimensions don't match"))
      }
    }
  }
}

/// Validate that output of each layer == input of next
fn validate_layers(layers: List(DenseLayer)) -> Bool {
  case layers {
    [] | [_] -> True
    [first, second, ..rest] ->
      first.output_size == second.input_size
      && validate_layers([second, ..rest])
  }
}

// =============================================================================
// BUILDER PATTERN
// =============================================================================

/// Start builder with input size
pub fn builder(input_size: Int) -> NetworkBuilder {
  NetworkBuilder(layers: [], last_size: input_size)
}

/// Add Dense layer
pub fn add_dense(
  b: NetworkBuilder,
  output_size: Int,
  activation: ActivationType,
) -> NetworkBuilder {
  let new_layer = layer.dense(b.last_size, output_size, activation)
  NetworkBuilder(
    layers: list.append(b.layers, [new_layer]),
    last_size: output_size,
  )
}

/// Add Dense layer with He initialization
pub fn add_dense_he(
  b: NetworkBuilder,
  output_size: Int,
  activation: ActivationType,
) -> NetworkBuilder {
  let new_layer = layer.dense_he(b.last_size, output_size, activation)
  NetworkBuilder(
    layers: list.append(b.layers, [new_layer]),
    last_size: output_size,
  )
}

/// Finalize and create the network
pub fn build(b: NetworkBuilder) -> Result(Network, TensorError) {
  from_layers(b.layers)
}

// =============================================================================
// FORWARD PASS
// =============================================================================

/// Complete forward pass, returns output and cache
pub fn forward(
  net: Network,
  input: Tensor,
) -> Result(#(Tensor, NetworkCache), TensorError) {
  forward_layers(net.layers, input, [])
}

/// Recursive forward through layers
fn forward_layers(
  layers: List(DenseLayer),
  current_input: Tensor,
  caches: List(DenseCache),
) -> Result(#(Tensor, NetworkCache), TensorError) {
  case layers {
    [] -> Ok(#(current_input, NetworkCache(layer_caches: list.reverse(caches))))
    [l, ..rest] -> {
      case layer.forward(l, current_input) {
        Ok(#(output, cache)) -> forward_layers(rest, output, [cache, ..caches])
        Error(e) -> Error(e)
      }
    }
  }
}

/// Simple forward pass (no cache, for inference)
pub fn predict(net: Network, input: Tensor) -> Result(Tensor, TensorError) {
  case forward(net, input) {
    Ok(#(output, _)) -> Ok(output)
    Error(e) -> Error(e)
  }
}

/// Batch prediction
pub fn predict_batch(
  net: Network,
  inputs: List(Tensor),
) -> Result(List(Tensor), TensorError) {
  list.try_map(inputs, fn(input) { predict(net, input) })
}

// =============================================================================
// BACKWARD PASS
// =============================================================================

/// Complete backward pass
pub fn backward(
  net: Network,
  cache: NetworkCache,
  loss_gradient: Tensor,
) -> Result(NetworkGradients, TensorError) {
  // Combine layers with caches (both in original order)
  let paired = list.zip(net.layers, cache.layer_caches)

  // Backward is from last to first layer
  let reversed = list.reverse(paired)

  backward_layers(reversed, loss_gradient, [])
}

/// Recursive backward through layers (reverse order)
fn backward_layers(
  layers_with_caches: List(#(DenseLayer, DenseCache)),
  upstream_grad: Tensor,
  gradients: List(DenseGradients),
) -> Result(NetworkGradients, TensorError) {
  case layers_with_caches {
    [] -> Ok(NetworkGradients(layer_gradients: gradients))
    [#(l, cache), ..rest] -> {
      case layer.backward(l, cache, upstream_grad) {
        Ok(grads) -> {
          // Propagate gradient to previous layer
          backward_layers(rest, grads.d_input, [grads, ..gradients])
        }
        Error(e) -> Error(e)
      }
    }
  }
}

// =============================================================================
// PARAMETER UPDATE
// =============================================================================

/// Update all weights with SGD
pub fn update_sgd(
  net: Network,
  gradients: NetworkGradients,
  learning_rate: Float,
) -> Result(Network, TensorError) {
  let paired = list.zip(net.layers, gradients.layer_gradients)

  let new_layers =
    list.try_map(paired, fn(pair) {
      let #(l, grads) = pair
      layer.update_sgd(l, grads, learning_rate)
    })

  case new_layers {
    Ok(layers) -> Ok(Network(..net, layers: layers))
    Error(e) -> Error(e)
  }
}

/// Momentum state for each layer
pub type MomentumState {
  MomentumState(velocity_weights: List(Tensor), velocity_biases: List(Tensor))
}

/// Initialize momentum state
pub fn init_momentum(net: Network) -> MomentumState {
  let #(vw, vb) =
    list.map(net.layers, layer.init_velocities)
    |> list.unzip

  MomentumState(velocity_weights: vw, velocity_biases: vb)
}

/// Update with momentum
pub fn update_momentum(
  net: Network,
  gradients: NetworkGradients,
  momentum_state: MomentumState,
  learning_rate: Float,
  momentum: Float,
) -> Result(#(Network, MomentumState), TensorError) {
  let zipped =
    list.zip(net.layers, gradients.layer_gradients)
    |> list.zip(momentum_state.velocity_weights)
    |> list.zip(momentum_state.velocity_biases)

  let results =
    list.try_map(zipped, fn(item) {
      let #(#(#(l, grads), vw), vb) = item
      layer.update_momentum(l, grads, vw, vb, learning_rate, momentum)
    })

  case results {
    Ok(updates) -> {
      let #(new_layers, new_vw, new_vb) =
        list.fold(updates, #([], [], []), fn(acc, update) {
          let #(layers, vws, vbs) = acc
          let #(layer, vw, vb) = update
          #([layer, ..layers], [vw, ..vws], [vb, ..vbs])
        })

      Ok(#(
        Network(..net, layers: list.reverse(new_layers)),
        MomentumState(
          velocity_weights: list.reverse(new_vw),
          velocity_biases: list.reverse(new_vb),
        ),
      ))
    }
    Error(e) -> Error(e)
  }
}

// =============================================================================
// UTILITY
// =============================================================================

/// Total number of trainable parameters
pub fn param_count(net: Network) -> Int {
  list.fold(net.layers, 0, fn(acc, l) { acc + layer.param_count(l) })
}

/// Number of layers
pub fn depth(net: Network) -> Int {
  list.length(net.layers)
}

/// Clone network
pub fn clone(net: Network) -> Network {
  Network(
    layers: list.map(net.layers, layer.clone),
    input_size: net.input_size,
    output_size: net.output_size,
  )
}

/// Architecture description
pub fn describe(net: Network) -> String {
  let layer_descs =
    list.map(net.layers, layer.describe)
    |> list.intersperse(" -> ")
    |> list.fold("", fn(acc, s) { acc <> s })

  "Network["
  <> int.to_string(depth(net))
  <> " layers, "
  <> int.to_string(param_count(net))
  <> " params]: "
  <> layer_descs
}

/// Returns sizes of all layers
pub fn layer_sizes(net: Network) -> List(Int) {
  [net.input_size, ..list.map(net.layers, fn(l) { l.output_size })]
}

// =============================================================================
// CONVENIENCE CONSTRUCTORS
// =============================================================================

/// Simple MLP for classification
pub fn mlp_classifier(
  input_size: Int,
  hidden_sizes: List(Int),
  num_classes: Int,
) -> Result(Network, TensorError) {
  let all_sizes = [input_size, ..list.append(hidden_sizes, [num_classes])]
  new(all_sizes, activation.ReLU, activation.Softmax)
}

/// MLP for regression
pub fn mlp_regressor(
  input_size: Int,
  hidden_sizes: List(Int),
  output_size: Int,
) -> Result(Network, TensorError) {
  let all_sizes = [input_size, ..list.append(hidden_sizes, [output_size])]
  new(all_sizes, activation.ReLU, activation.Linear)
}

/// Small network for VIVA (PAD -> decision)
pub fn viva_brain(
  input_size: Int,
  output_size: Int,
) -> Result(Network, TensorError) {
  builder(input_size)
  |> add_dense(16, activation.Tanh)
  |> add_dense(8, activation.Tanh)
  |> add_dense(output_size, activation.Sigmoid)
  |> build
}
