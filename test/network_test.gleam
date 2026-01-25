//// Neural Network Tests
////
//// Tests for network construction, forward/backward pass, and training.

import gleam/float
import gleam/list
import gleeunit/should
import viva/neural/activation
import viva/neural/network
import viva/neural/tensor

// =============================================================================
// CONSTRUCTOR TESTS
// =============================================================================

pub fn network_new_simple_test() {
  let result = network.new([2, 3, 1], activation.ReLU, activation.Sigmoid)
  should.be_ok(result)

  let assert Ok(net) = result
  should.equal(net.input_size, 2)
  should.equal(net.output_size, 1)
  should.equal(network.depth(net), 2)
}

pub fn network_new_multiple_layers_test() {
  let result = network.new([4, 8, 4, 2], activation.Tanh, activation.Softmax)
  should.be_ok(result)

  let assert Ok(net) = result
  should.equal(net.input_size, 4)
  should.equal(net.output_size, 2)
  should.equal(network.depth(net), 3)
}

pub fn network_new_invalid_single_layer_test() {
  let result = network.new([4], activation.ReLU, activation.Sigmoid)
  should.be_error(result)
}

pub fn network_new_invalid_empty_test() {
  let result = network.new([], activation.ReLU, activation.Sigmoid)
  should.be_error(result)
}

pub fn network_builder_test() {
  let result =
    network.builder(4)
    |> network.add_dense(8, activation.ReLU)
    |> network.add_dense(4, activation.ReLU)
    |> network.add_dense(2, activation.Sigmoid)
    |> network.build

  should.be_ok(result)

  let assert Ok(net) = result
  should.equal(net.input_size, 4)
  should.equal(net.output_size, 2)
  should.equal(network.depth(net), 3)
}

pub fn network_builder_he_init_test() {
  let result =
    network.builder(10)
    |> network.add_dense_he(32, activation.ReLU)
    |> network.add_dense_he(16, activation.ReLU)
    |> network.add_dense(5, activation.Softmax)
    |> network.build

  should.be_ok(result)
}

// =============================================================================
// CONVENIENCE CONSTRUCTOR TESTS
// =============================================================================

pub fn network_mlp_classifier_test() {
  let result = network.mlp_classifier(784, [128, 64], 10)
  should.be_ok(result)

  let assert Ok(net) = result
  should.equal(net.input_size, 784)
  should.equal(net.output_size, 10)
}

pub fn network_mlp_regressor_test() {
  let result = network.mlp_regressor(10, [32, 16], 1)
  should.be_ok(result)

  let assert Ok(net) = result
  should.equal(net.input_size, 10)
  should.equal(net.output_size, 1)
}

pub fn network_viva_brain_test() {
  let result = network.viva_brain(4, 3)
  should.be_ok(result)

  let assert Ok(net) = result
  should.equal(net.input_size, 4)
  should.equal(net.output_size, 3)
  // Should have 3 layers: 4->16->8->3
  should.equal(network.depth(net), 3)
}

// =============================================================================
// FORWARD PASS TESTS
// =============================================================================

pub fn network_forward_simple_test() {
  let assert Ok(net) = network.new([2, 3, 1], activation.Linear, activation.Linear)
  let input = tensor.from_list([1.0, 2.0])

  let result = network.forward(net, input)
  should.be_ok(result)

  let assert Ok(#(output, cache)) = result
  // Output shape should be [1]
  should.equal(output.shape, [1])
  // Cache should have 2 layer caches
  should.equal(list.length(cache.layer_caches), 2)
}

pub fn network_forward_multi_layer_test() {
  let assert Ok(net) = network.new([4, 8, 4, 2], activation.ReLU, activation.Sigmoid)
  let input = tensor.from_list([0.1, 0.2, 0.3, 0.4])

  let result = network.forward(net, input)
  should.be_ok(result)

  let assert Ok(#(output, cache)) = result
  should.equal(output.shape, [2])
  should.equal(list.length(cache.layer_caches), 3)
}

pub fn network_predict_test() {
  let assert Ok(net) = network.new([3, 5, 2], activation.Tanh, activation.Softmax)
  let input = tensor.from_list([0.5, -0.5, 0.0])

  let result = network.predict(net, input)
  should.be_ok(result)

  let assert Ok(output) = result
  should.equal(output.shape, [2])

  // Softmax outputs should sum to ~1
  let sum = tensor.sum(output)
  should.be_true(float.loosely_equals(sum, 1.0, 0.0001))
}

pub fn network_predict_batch_test() {
  let assert Ok(net) = network.new([2, 4, 2], activation.ReLU, activation.Softmax)

  let inputs = [
    tensor.from_list([1.0, 0.0]),
    tensor.from_list([0.0, 1.0]),
    tensor.from_list([0.5, 0.5]),
  ]

  let result = network.predict_batch(net, inputs)
  should.be_ok(result)

  let assert Ok(outputs) = result
  should.equal(list.length(outputs), 3)
}

// =============================================================================
// BACKWARD PASS TESTS
// =============================================================================

pub fn network_backward_shapes_test() {
  let assert Ok(net) = network.new([3, 4, 2], activation.ReLU, activation.Sigmoid)
  let input = tensor.from_list([0.1, 0.2, 0.3])

  let assert Ok(#(_output, cache)) = network.forward(net, input)

  // Fake loss gradient (same shape as output)
  let loss_grad = tensor.from_list([0.5, -0.5])

  let result = network.backward(net, cache, loss_grad)
  should.be_ok(result)

  let assert Ok(gradients) = result
  should.equal(list.length(gradients.layer_gradients), 2)
}

pub fn network_backward_gradients_nonzero_test() {
  let assert Ok(net) = network.new([2, 3, 1], activation.Linear, activation.Linear)
  let input = tensor.from_list([1.0, 2.0])

  let assert Ok(#(_output, cache)) = network.forward(net, input)
  let loss_grad = tensor.from_list([1.0])

  let assert Ok(gradients) = network.backward(net, cache, loss_grad)

  // Gradients should not be all zero
  let first_grads = case list.first(gradients.layer_gradients) {
    Ok(g) -> g
    Error(_) -> panic as "No gradients"
  }

  // Use map with absolute_value instead of tensor.abs
  let abs_weights = tensor.map(first_grads.d_weights, float.absolute_value)
  let weight_sum = tensor.sum(abs_weights)
  should.be_true(weight_sum >. 0.0)
}

// =============================================================================
// UPDATE TESTS
// =============================================================================

pub fn network_update_sgd_test() {
  let assert Ok(net) = network.new([2, 3, 1], activation.Linear, activation.Linear)
  let input = tensor.from_list([1.0, 2.0])

  let assert Ok(#(_output, cache)) = network.forward(net, input)
  let loss_grad = tensor.from_list([1.0])

  let assert Ok(gradients) = network.backward(net, cache, loss_grad)
  let result = network.update_sgd(net, gradients, 0.01)

  should.be_ok(result)
}

pub fn network_momentum_init_test() {
  let assert Ok(net) = network.new([4, 8, 2], activation.ReLU, activation.Softmax)
  let momentum_state = network.init_momentum(net)

  should.equal(list.length(momentum_state.velocity_weights), 2)
  should.equal(list.length(momentum_state.velocity_biases), 2)
}

pub fn network_update_momentum_test() {
  let assert Ok(net) = network.new([2, 4, 2], activation.ReLU, activation.Sigmoid)
  let input = tensor.from_list([0.5, 0.5])

  let momentum_state = network.init_momentum(net)

  let assert Ok(#(_output, cache)) = network.forward(net, input)
  let loss_grad = tensor.from_list([0.1, -0.1])

  let assert Ok(gradients) = network.backward(net, cache, loss_grad)
  let result = network.update_momentum(net, gradients, momentum_state, 0.01, 0.9)

  should.be_ok(result)
}

// =============================================================================
// UTILITY TESTS
// =============================================================================

pub fn network_param_count_test() {
  // 2->3: weights=6, biases=3 = 9
  // 3->1: weights=3, biases=1 = 4
  // Total: 13
  let assert Ok(net) = network.new([2, 3, 1], activation.Linear, activation.Linear)
  should.equal(network.param_count(net), 13)
}

pub fn network_depth_test() {
  let assert Ok(net) = network.new([10, 20, 30, 40, 5], activation.ReLU, activation.Softmax)
  should.equal(network.depth(net), 4)
}

pub fn network_clone_test() {
  let assert Ok(net) = network.new([2, 4, 2], activation.Tanh, activation.Sigmoid)
  let cloned = network.clone(net)

  should.equal(network.param_count(cloned), network.param_count(net))
  should.equal(network.depth(cloned), network.depth(net))
}

pub fn network_describe_test() {
  let assert Ok(net) = network.new([4, 8, 2], activation.ReLU, activation.Softmax)
  let desc = network.describe(net)

  // Should contain layer info
  should.be_true(desc != "")
}

pub fn network_layer_sizes_test() {
  let assert Ok(net) = network.new([4, 8, 4, 2], activation.ReLU, activation.Softmax)
  let sizes = network.layer_sizes(net)

  should.equal(sizes, [4, 8, 4, 2])
}
