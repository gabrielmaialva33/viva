import gleam/float
import gleam/list
import gleam/option
import gleeunit/should
import viva/neural/activation
import viva/neural/attention
import viva/neural/conv
import viva/neural/normalization
import viva/neural/recurrent
import viva/neural/regularization
import viva/neural/tensor
import viva/neural/utils

// =============================================================================
// UTILS TESTS
// =============================================================================

pub fn utils_sqrt_test() {
  let t = tensor.from_list([4.0, 9.0, 16.0, 25.0])
  let result = utils.sqrt(t)
  let expected = [2.0, 3.0, 4.0, 5.0]
  list.map2(result.data, expected, fn(a, b) {
    should.be_true(float.loosely_equals(a, b, 0.001))
  })
}

pub fn utils_sigmoid_test() {
  // sigmoid(0) = 0.5
  should.be_true(float.loosely_equals(utils.sigmoid(0.0), 0.5, 0.001))
  // sigmoid approaches 1 for large positive
  should.be_true(utils.sigmoid(10.0) >. 0.99)
  // sigmoid approaches 0 for large negative
  should.be_true(utils.sigmoid(-10.0) <. 0.01)
}

pub fn utils_tanh_test() {
  // tanh(0) = 0
  should.be_true(float.loosely_equals(utils.tanh(0.0), 0.0, 0.001))
  // tanh approaches 1 for large positive
  should.be_true(utils.tanh(10.0) >. 0.99)
  // tanh approaches -1 for large negative
  should.be_true(utils.tanh(-10.0) <. -0.99)
}

pub fn utils_random_bernoulli_shape_test() {
  let t = utils.random_bernoulli([3, 4], 0.5)
  should.equal(t.shape, [3, 4])
  should.equal(tensor.size(t), 12)
}

pub fn utils_random_bernoulli_values_test() {
  // All values should be 0.0 or 1.0
  let t = utils.random_bernoulli([100], 0.5)
  let all_binary = list.all(t.data, fn(x) { x == 0.0 || x == 1.0 })
  should.be_true(all_binary)
}

pub fn utils_softmax_axis_1d_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let result = utils.softmax_axis(t, 0)
  should.be_ok(result)
  let assert Ok(soft) = result
  // Sum should be 1.0
  let sum = tensor.sum(soft)
  should.be_true(float.loosely_equals(sum, 1.0, 0.001))
}

pub fn utils_softmax_axis_2d_test() {
  let assert Ok(t) = tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let result = utils.softmax_axis(t, 1)
  should.be_ok(result)
  let assert Ok(soft) = result
  should.equal(soft.shape, [2, 3])
  // Each row should sum to 1.0
  let assert Ok(row0) = tensor.get_row(soft, 0)
  let assert Ok(row1) = tensor.get_row(soft, 1)
  should.be_true(float.loosely_equals(tensor.sum(row0), 1.0, 0.001))
  should.be_true(float.loosely_equals(tensor.sum(row1), 1.0, 0.001))
}

// =============================================================================
// DROPOUT TESTS
// =============================================================================

pub fn dropout_new_test() {
  let layer = regularization.new(0.5)
  should.be_true(float.loosely_equals(layer.rate, 0.5, 0.001))
}

pub fn dropout_inference_passthrough_test() {
  let layer = regularization.new_inference(0.5)
  let input = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let result = regularization.forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  // In inference mode, output should equal input
  should.equal(output.data, input.data)
}

pub fn dropout_training_drops_test() {
  let layer = regularization.new(0.5)
  let input = tensor.ones([1000])
  let result = regularization.forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  // Not all values should be the same (some dropped, some scaled)
  let unique_values =
    list.unique(list.map(output.data, fn(x) {
      float.round(x *. 100.0)  // Round to avoid float comparison issues
    }))
  should.be_true(list.length(unique_values) > 1)
}

pub fn dropout_zero_rate_passthrough_test() {
  let layer = regularization.new(0.0)  // No dropout
  let input = tensor.from_list([1.0, 2.0, 3.0])
  let result = regularization.forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  should.equal(output.data, input.data)
}

pub fn dropout_backward_test() {
  let layer = regularization.new(0.5)
  let input = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let result = regularization.forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(_output, cache)) = result

  let upstream = tensor.from_list([1.0, 1.0, 1.0, 1.0])
  let grad_result = regularization.backward(layer, cache, upstream)
  should.be_ok(grad_result)
  let assert Ok(grads) = grad_result
  should.equal(grads.d_input.shape, [4])
}

// =============================================================================
// BATCH NORM TESTS
// =============================================================================

pub fn batchnorm_new_test() {
  let layer = normalization.batch_norm_new(10)
  should.equal(layer.num_features, 10)
  should.equal(layer.gamma.shape, [10])
  should.equal(layer.beta.shape, [10])
}

pub fn batchnorm_normalizes_test() {
  let layer = normalization.batch_norm_new(3)
  // Create input with known mean and variance
  let assert Ok(input) = tensor.matrix(2, 3, [
    1.0, 2.0, 3.0,
    5.0, 6.0, 7.0,
  ])
  let result = normalization.batch_norm_forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache, _new_layer)) = result
  should.equal(output.shape, [2, 3])
}

pub fn batchnorm_inference_uses_running_stats_test() {
  let layer = normalization.batch_norm_new(3)
  let layer = normalization.batch_norm_eval(layer)
  let assert Ok(input) = tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let result = normalization.batch_norm_forward(layer, input)
  should.be_ok(result)
}

// =============================================================================
// LAYER NORM TESTS
// =============================================================================

pub fn layernorm_new_test() {
  let layer = normalization.layer_norm_new([10])
  should.equal(layer.normalized_shape, [10])
}

pub fn layernorm_forward_test() {
  let layer = normalization.layer_norm_new([4])
  let assert Ok(input) = tensor.matrix(2, 4, [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
  ])
  let result = normalization.layer_norm_forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  should.equal(output.shape, [2, 4])
}

// =============================================================================
// GROUP NORM TESTS
// =============================================================================

pub fn groupnorm_forward_test() {
  let layer = normalization.group_norm_new(2, 4)  // 2 groups, 4 channels
  let assert Ok(input) = tensor.matrix(2, 4, [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
  ])
  let result = normalization.group_norm_forward(layer, input)
  should.be_ok(result)
  let assert Ok(output) = result
  should.equal(output.shape, [2, 4])
}

// =============================================================================
// CONV2D TESTS
// =============================================================================

pub fn conv2d_new_test() {
  let layer = conv.new(3, 16, 3, 1, conv.Valid, activation.ReLU)
  should.equal(layer.in_channels, 3)
  should.equal(layer.out_channels, 16)
  should.equal(layer.kernel_h, 3)
  should.equal(layer.kernel_w, 3)
}

pub fn conv2d_output_shape_valid_test() {
  // Input: 1 batch, 1 channel, 5x5
  // Kernel: 3x3, stride 1, valid padding
  // Output: 1 batch, 1 channel, 3x3 (5 - 3 + 1 = 3)
  let layer = conv.new(1, 1, 3, 1, conv.Valid, activation.Linear)
  let input = tensor.ones([1, 1, 5, 5])
  let result = conv.forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  should.equal(output.shape, [1, 1, 3, 3])
}

pub fn conv2d_output_shape_same_test() {
  // Input: 1 batch, 1 channel, 5x5
  // Kernel: 3x3, stride 1, same padding
  // Output: 1 batch, 1 channel, 5x5 (same as input)
  let layer = conv.new(1, 1, 3, 1, conv.Same, activation.Linear)
  let input = tensor.ones([1, 1, 5, 5])
  let result = conv.forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  should.equal(output.shape, [1, 1, 5, 5])
}

pub fn conv2d_identity_kernel_test() {
  // A 1x1 convolution with identity-like weights
  let filters = tensor.Tensor(data: [1.0], shape: [1, 1, 1, 1])
  let biases = tensor.zeros([1])
  let layer = conv.from_weights(
    filters, biases, 1, 1, 1, 1, 1, 1, conv.Valid, activation.Linear
  )
  let input = tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 1, 2, 2])
  let result = conv.forward(layer, input)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  // Output should equal input (identity convolution)
  should.equal(output.data, input.data)
}

// =============================================================================
// LSTM TESTS
// =============================================================================

pub fn lstm_new_test() {
  let cell = recurrent.lstm_new(10, 20)
  should.equal(cell.input_size, 10)
  should.equal(cell.hidden_size, 20)
  should.equal(cell.w_input.shape, [80, 10])  // 4 * 20 = 80
  should.equal(cell.w_hidden.shape, [80, 20])
}

pub fn lstm_init_state_test() {
  let state = recurrent.lstm_init_state(32)
  should.equal(state.c.shape, [32])
  should.equal(state.h.shape, [32])
  // Should be zeros
  should.be_true(list.all(state.c.data, fn(x) { x == 0.0 }))
  should.be_true(list.all(state.h.data, fn(x) { x == 0.0 }))
}

pub fn lstm_forward_shapes_test() {
  let cell = recurrent.lstm_new(4, 8)
  let state = recurrent.lstm_init_state(8)
  let input = tensor.random_uniform([4])

  let result = recurrent.lstm_forward(cell, input, state)
  should.be_ok(result)
  let assert Ok(#(h_new, new_state, _cache)) = result
  should.equal(h_new.shape, [8])
  should.equal(new_state.h.shape, [8])
  should.equal(new_state.c.shape, [8])
}

pub fn lstm_sequence_test() {
  let cell = recurrent.lstm_new(4, 8)
  let state = recurrent.lstm_init_state(8)
  let inputs = [
    tensor.random_uniform([4]),
    tensor.random_uniform([4]),
    tensor.random_uniform([4]),
  ]

  let result = recurrent.lstm_sequence(cell, inputs, state)
  should.be_ok(result)
  let assert Ok(#(outputs, final_state, caches)) = result
  should.equal(list.length(outputs), 3)
  should.equal(list.length(caches), 3)
  should.equal(final_state.h.shape, [8])
}

// =============================================================================
// GRU TESTS
// =============================================================================

pub fn gru_new_test() {
  let cell = recurrent.gru_new(10, 20)
  should.equal(cell.input_size, 10)
  should.equal(cell.hidden_size, 20)
}

pub fn gru_forward_shapes_test() {
  let cell = recurrent.gru_new(4, 8)
  let state = recurrent.gru_init_state(8)
  let input = tensor.random_uniform([4])

  let result = recurrent.gru_forward(cell, input, state)
  should.be_ok(result)
  let assert Ok(#(h_new, new_state, _cache)) = result
  should.equal(h_new.shape, [8])
  should.equal(new_state.h.shape, [8])
}

pub fn gru_sequence_test() {
  let cell = recurrent.gru_new(4, 8)
  let state = recurrent.gru_init_state(8)
  let inputs = [
    tensor.random_uniform([4]),
    tensor.random_uniform([4]),
  ]

  let result = recurrent.gru_sequence(cell, inputs, state)
  should.be_ok(result)
  let assert Ok(#(outputs, final_state, _caches)) = result
  should.equal(list.length(outputs), 2)
  should.equal(final_state.h.shape, [8])
}

// =============================================================================
// ATTENTION TESTS
// =============================================================================

pub fn attention_basic_test() {
  // Simple 3x4 attention: 3 queries, 3 keys, 4-dim values
  let assert Ok(query) = tensor.matrix(3, 4, [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
  ])
  let assert Ok(key) = tensor.matrix(3, 4, [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
  ])
  let assert Ok(value) = tensor.matrix(3, 4, [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
  ])

  let result = attention.attention(query, key, value)
  should.be_ok(result)
  let assert Ok(output) = result
  should.equal(output.shape, [3, 4])
}

pub fn attention_weights_sum_one_test() {
  let assert Ok(query) = tensor.matrix(2, 4, [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
  ])
  let assert Ok(key) = tensor.matrix(3, 4, [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
  ])
  let assert Ok(value) = tensor.matrix(3, 4, [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
  ])

  let result = attention.scaled_dot_product_attention(query, key, value, option.None)
  should.be_ok(result)
  let assert Ok(#(attn_result, _cache)) = result

  // Each row of weights should sum to 1.0
  let assert Ok(row0) = tensor.get_row(attn_result.weights, 0)
  let assert Ok(row1) = tensor.get_row(attn_result.weights, 1)
  should.be_true(float.loosely_equals(tensor.sum(row0), 1.0, 0.01))
  should.be_true(float.loosely_equals(tensor.sum(row1), 1.0, 0.01))
}

pub fn causal_mask_test() {
  let mask = attention.causal_mask(4)
  should.equal(mask.shape, [4, 4])
  // Upper triangle should be large negative (masked)
  // Lower triangle + diagonal should be 0 (not masked)
  let assert Ok(val_01) = tensor.get2d(mask, 0, 1)  // Should be masked (large neg)
  let assert Ok(val_00) = tensor.get2d(mask, 0, 0)  // Should be 0 (diagonal)
  let assert Ok(val_10) = tensor.get2d(mask, 1, 0)  // Should be 0 (below diag)
  should.be_true(val_01 <. -1000.0)
  should.be_true(float.loosely_equals(val_00, 0.0, 0.001))
  should.be_true(float.loosely_equals(val_10, 0.0, 0.001))
}

pub fn mha_new_test() {
  let mha = attention.mha_new(64, 8)  // d_model=64, 8 heads
  should.equal(mha.d_model, 64)
  should.equal(mha.num_heads, 8)
  should.equal(mha.d_k, 8)  // 64 / 8
}

pub fn mha_forward_test() {
  let mha = attention.mha_new(16, 4)  // d_model=16, 4 heads
  let assert Ok(input) = tensor.matrix(5, 16, list.repeat(0.1, 5 * 16))

  let result = attention.mha_forward(mha, input, input, input, option.None)
  should.be_ok(result)
  let assert Ok(#(output, _cache)) = result
  should.equal(output.shape, [5, 16])
}

// =============================================================================
// POSITIONAL ENCODING TESTS
// =============================================================================

pub fn positional_encoding_test() {
  let pe = attention.positional_encoding_new(100, 64)
  should.equal(pe.max_len, 100)
  should.equal(pe.d_model, 64)
  should.equal(pe.encoding.shape, [100, 64])
}

pub fn positional_encoding_add_test() {
  let pe = attention.positional_encoding_new(100, 16)
  let assert Ok(input) = tensor.matrix(10, 16, list.repeat(0.0, 10 * 16))

  let result = attention.add_positional_encoding(pe, input)
  should.be_ok(result)
  let assert Ok(output) = result
  should.equal(output.shape, [10, 16])
  // Output should not be all zeros (PE was added)
  should.be_true(tensor.sum(output) != 0.0)
}

pub fn relative_position_bias_test() {
  let rpb = attention.relative_position_bias_new(8)
  let bias = attention.compute_relative_bias(rpb, 5)
  should.equal(bias.shape, [5, 5])
}
