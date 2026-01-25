//// NetworkAccelerated - Auto-switching entre Pure e Nx
////
//// Detecta automaticamente se Nx/EXLA está disponível e usa.
//// Fallback transparente para Gleam puro se não estiver.
////
//// Uso:
////   import viva/neural/network_accelerated as net
////   let output = net.predict(network, input)  // auto-detecta backend

import gleam/list
import gleam/result
import viva/neural/activation
import viva/neural/layer.{type DenseCache, type DenseGradients, type DenseLayer}
import viva/neural/network.{
  type Network, type NetworkCache, type NetworkGradients,
}
import viva/neural/nx_backend.{type Backend, CUDA, Nx, Pure}
import viva/neural/tensor.{type Tensor, type TensorError}

// =============================================================================
// RUNTIME DETECTION
// =============================================================================

/// Check if Nx is available at runtime
pub fn nx_available() -> Bool {
  // Try to call Nx - if it fails, not available
  case try_nx_check() {
    Ok(_) -> True
    Error(_) -> False
  }
}

/// Get best available backend
pub fn auto_backend() -> Backend {
  case nx_available() {
    True -> Nx
    False -> Pure
  }
}

/// Check with GPU preference
pub fn best_backend() -> Backend {
  case nx_available() {
    True ->
      case nx_backend.cuda_available() {
        True -> Nx
        // GPU available
        False -> Nx
        // CPU EXLA still faster than pure Gleam
      }
    False -> Pure
  }
}

// =============================================================================
// ACCELERATED FORWARD PASS
// =============================================================================

/// Forward pass with auto-detection
pub fn forward(
  net: Network,
  input: Tensor,
) -> Result(#(Tensor, NetworkCache), TensorError) {
  forward_with_backend(net, input, auto_backend())
}

/// Forward pass with explicit backend
pub fn forward_with_backend(
  net: Network,
  input: Tensor,
  backend: Backend,
) -> Result(#(Tensor, NetworkCache), TensorError) {
  case backend {
    Pure -> network.forward(net, input)
    Nx | CUDA(_) -> forward_nx(net, input)
  }
}

/// Nx-accelerated forward pass
fn forward_nx(
  net: Network,
  input: Tensor,
) -> Result(#(Tensor, NetworkCache), TensorError) {
  forward_layers_nx(net.layers, input, [])
}

fn forward_layers_nx(
  layers: List(DenseLayer),
  current_input: Tensor,
  caches: List(DenseCache),
) -> Result(#(Tensor, NetworkCache), TensorError) {
  case layers {
    [] ->
      Ok(#(
        current_input,
        network.NetworkCache(layer_caches: list.reverse(caches)),
      ))
    [l, ..rest] -> {
      case forward_layer_nx(l, current_input) {
        Ok(#(output, cache)) ->
          forward_layers_nx(rest, output, [cache, ..caches])
        Error(e) -> Error(e)
      }
    }
  }
}

fn forward_layer_nx(
  l: DenseLayer,
  input: Tensor,
) -> Result(#(Tensor, DenseCache), TensorError) {
  // z = W^T @ x + b
  use weights_t <- result.try(nx_backend.transpose(l.weights, Nx))
  use z <- result.try(nx_backend.matmul_vec(weights_t, input, Nx))
  use pre_activation <- result.try(nx_backend.add(z, l.biases, Nx))

  // Apply activation
  let #(output, derivatives) = case l.activation {
    activation.Softmax -> {
      let out = nx_backend.softmax(pre_activation, Nx)
      #(out, tensor.ones([l.output_size]))
    }
    _ -> activation.apply(pre_activation, l.activation)
  }

  let cache =
    layer.DenseCache(
      input: input,
      pre_activation: pre_activation,
      output: output,
      activation_derivatives: derivatives,
    )

  Ok(#(output, cache))
}

// =============================================================================
// ACCELERATED INFERENCE
// =============================================================================

/// Simple predict with auto-detection
pub fn predict(net: Network, input: Tensor) -> Result(Tensor, TensorError) {
  predict_with_backend(net, input, auto_backend())
}

/// Predict with explicit backend
pub fn predict_with_backend(
  net: Network,
  input: Tensor,
  backend: Backend,
) -> Result(Tensor, TensorError) {
  case forward_with_backend(net, input, backend) {
    Ok(#(output, _)) -> Ok(output)
    Error(e) -> Error(e)
  }
}

/// Batch predict (where Nx really shines)
pub fn predict_batch(
  net: Network,
  inputs: List(Tensor),
) -> Result(List(Tensor), TensorError) {
  predict_batch_with_backend(net, inputs, auto_backend())
}

/// Batch predict with explicit backend
pub fn predict_batch_with_backend(
  net: Network,
  inputs: List(Tensor),
  backend: Backend,
) -> Result(List(Tensor), TensorError) {
  case backend {
    Pure -> network.predict_batch(net, inputs)
    Nx | CUDA(_) -> Ok(predict_batch_nx(net, inputs))
  }
}

/// Nx batch prediction (vectorized)
fn predict_batch_nx(net: Network, inputs: List(Tensor)) -> List(Tensor) {
  // Process through each layer with batch operations
  let current = inputs

  list.fold(net.layers, current, fn(batch, layer) {
    // Batch matmul for all inputs at once
    let outputs = nx_backend.batch_matmul(batch, layer.weights, Nx)

    // Add biases and apply activation
    list.map(outputs, fn(out) {
      case nx_backend.add(out, layer.biases, Nx) {
        Ok(with_bias) ->
          case layer.activation {
            activation.Softmax -> nx_backend.softmax(with_bias, Nx)
            _ -> activation.forward(with_bias, layer.activation)
          }
        Error(_) -> out
      }
    })
  })
}

// =============================================================================
// ACCELERATED BACKWARD PASS
// =============================================================================

/// Backward pass with auto-detection
pub fn backward(
  net: Network,
  cache: NetworkCache,
  loss_gradient: Tensor,
) -> Result(NetworkGradients, TensorError) {
  backward_with_backend(net, cache, loss_gradient, auto_backend())
}

/// Backward with explicit backend
pub fn backward_with_backend(
  net: Network,
  cache: NetworkCache,
  loss_gradient: Tensor,
  backend: Backend,
) -> Result(NetworkGradients, TensorError) {
  case backend {
    Pure -> network.backward(net, cache, loss_gradient)
    Nx | CUDA(_) -> backward_nx(net, cache, loss_gradient)
  }
}

fn backward_nx(
  net: Network,
  cache: NetworkCache,
  loss_gradient: Tensor,
) -> Result(NetworkGradients, TensorError) {
  let paired = list.zip(net.layers, cache.layer_caches)
  let reversed = list.reverse(paired)
  backward_layers_nx(reversed, loss_gradient, [])
}

fn backward_layers_nx(
  layers_with_caches: List(#(DenseLayer, DenseCache)),
  upstream_grad: Tensor,
  gradients: List(DenseGradients),
) -> Result(NetworkGradients, TensorError) {
  case layers_with_caches {
    [] -> Ok(network.NetworkGradients(layer_gradients: gradients))
    [#(l, cache), ..rest] -> {
      case backward_layer_nx(l, cache, upstream_grad) {
        Ok(grads) ->
          backward_layers_nx(rest, grads.d_input, [grads, ..gradients])
        Error(e) -> Error(e)
      }
    }
  }
}

fn backward_layer_nx(
  l: DenseLayer,
  cache: DenseCache,
  upstream_gradient: Tensor,
) -> Result(DenseGradients, TensorError) {
  // Delta = upstream * activation_derivative
  let delta = case l.activation {
    activation.Softmax ->
      activation.softmax_backward(cache.output, upstream_gradient)
    _ ->
      case nx_backend.mul(upstream_gradient, cache.activation_derivatives, Nx) {
        Ok(d) -> d
        Error(_) -> upstream_gradient
      }
  }

  // d_biases = delta
  let d_biases = delta

  // d_weights = outer(input, delta)
  use d_weights <- result.try(tensor.outer(cache.input, delta))

  // d_input = W @ delta
  use d_input <- result.try(nx_backend.matmul_vec(l.weights, delta, Nx))

  Ok(layer.DenseGradients(
    d_weights: d_weights,
    d_biases: d_biases,
    d_input: d_input,
  ))
}

// =============================================================================
// ACCELERATED UPDATE
// =============================================================================

/// SGD update with auto-detection
pub fn update_sgd(
  net: Network,
  gradients: NetworkGradients,
  learning_rate: Float,
) -> Result(Network, TensorError) {
  update_sgd_with_backend(net, gradients, learning_rate, auto_backend())
}

/// SGD with explicit backend
pub fn update_sgd_with_backend(
  net: Network,
  gradients: NetworkGradients,
  learning_rate: Float,
  backend: Backend,
) -> Result(Network, TensorError) {
  case backend {
    Pure -> network.update_sgd(net, gradients, learning_rate)
    Nx | CUDA(_) -> update_sgd_nx(net, gradients, learning_rate)
  }
}

fn update_sgd_nx(
  net: Network,
  gradients: NetworkGradients,
  learning_rate: Float,
) -> Result(Network, TensorError) {
  let paired = list.zip(net.layers, gradients.layer_gradients)

  let new_layers =
    list.try_map(paired, fn(pair) {
      let #(l, grads) = pair

      // weights = weights - lr * d_weights
      let scaled_dw = nx_backend.scale(grads.d_weights, learning_rate, Nx)
      use new_weights <- result.try(nx_backend.sub(l.weights, scaled_dw, Nx))

      // biases = biases - lr * d_biases
      let scaled_db = nx_backend.scale(grads.d_biases, learning_rate, Nx)
      use new_biases <- result.try(nx_backend.sub(l.biases, scaled_db, Nx))

      Ok(layer.DenseLayer(..l, weights: new_weights, biases: new_biases))
    })

  case new_layers {
    Ok(layers) -> Ok(network.Network(..net, layers: layers))
    Error(e) -> Error(e)
  }
}

// =============================================================================
// BENCHMARKING
// =============================================================================

/// Compare backends performance
pub fn benchmark(net: Network, input: Tensor, iterations: Int) -> #(Int, Int) {
  let pure_time = benchmark_backend(net, input, iterations, Pure)
  let nx_time = case nx_available() {
    True -> benchmark_backend(net, input, iterations, Nx)
    False -> 0
  }
  #(pure_time, nx_time)
}

fn benchmark_backend(
  net: Network,
  input: Tensor,
  iterations: Int,
  backend: Backend,
) -> Int {
  let start = erlang_monotonic_time()

  list.range(1, iterations)
  |> list.each(fn(_) {
    let _ = predict_with_backend(net, input, backend)
    Nil
  })

  let end = erlang_monotonic_time()
  end - start
}

// =============================================================================
// VIVA INTEGRATION
// =============================================================================

/// Create VIVA brain with best backend
pub fn viva_brain(
  input_size: Int,
  output_size: Int,
) -> Result(Network, TensorError) {
  network.viva_brain(input_size, output_size)
}

/// VIVA decision with auto-acceleration
pub fn viva_decide(
  net: Network,
  pad_input: Tensor,
) -> Result(Tensor, TensorError) {
  // VIVA brain is small, but we still use acceleration if available
  predict(net, pad_input)
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "erlang", "monotonic_time")
fn erlang_monotonic_time() -> Int

/// Try to check if Nx module exists
fn try_nx_check() -> Result(Bool, Nil) {
  case nx_check_module() {
    True -> Ok(True)
    False -> Error(Nil)
  }
}

@external(erlang, "viva_nx_check", "available")
fn nx_check_module() -> Bool
