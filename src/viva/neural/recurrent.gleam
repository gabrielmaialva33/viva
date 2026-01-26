//// Recurrent - LSTM and GRU cells for sequence modeling
////
//// Implements recurrent neural network cells with proper gate
//// mechanisms for learning long-range dependencies.
//// Inspired by Axon's lstm_cell and gru_cell implementations.

import gleam/int
import gleam/list
import gleam/result
import viva_tensor/tensor.{type Tensor, type TensorError, Tensor}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// LSTM Cell
/// Uses 4 gates: input, forget, cell, output
pub type LSTMCell {
  LSTMCell(
    /// Input weights [4 * hidden, input_size] for i,f,g,o gates
    w_input: Tensor,
    /// Hidden weights [4 * hidden, hidden_size] for i,f,g,o gates
    w_hidden: Tensor,
    /// Biases [4 * hidden] for i,f,g,o gates
    biases: Tensor,
    /// Input dimension
    input_size: Int,
    /// Hidden state dimension
    hidden_size: Int,
  )
}

/// LSTM hidden state (cell state + hidden state)
pub type LSTMState {
  LSTMState(
    /// Cell state c
    c: Tensor,
    /// Hidden state h
    h: Tensor,
  )
}

/// LSTM cache for backward pass
pub type LSTMCache {
  LSTMCache(
    /// Input at this step
    input: Tensor,
    /// Previous state
    prev_state: LSTMState,
    /// Gate values [i, f, g, o] before activation
    gates_pre: Tensor,
    /// Gate values after activation
    i: Tensor,
    f: Tensor,
    g: Tensor,
    o: Tensor,
    /// New cell state
    c_new: Tensor,
  )
}

/// GRU Cell
/// Uses 2 gates: reset, update (simpler than LSTM)
pub type GRUCell {
  GRUCell(
    /// Reset gate weights [hidden, input_size]
    w_r_input: Tensor,
    w_r_hidden: Tensor,
    b_r: Tensor,
    /// Update gate weights
    w_z_input: Tensor,
    w_z_hidden: Tensor,
    b_z: Tensor,
    /// New hidden weights
    w_n_input: Tensor,
    w_n_hidden: Tensor,
    b_n: Tensor,
    /// Dimensions
    input_size: Int,
    hidden_size: Int,
  )
}

/// GRU state (just hidden, no cell state)
pub type GRUState {
  GRUState(h: Tensor)
}

/// GRU cache for backward pass
pub type GRUCache {
  GRUCache(input: Tensor, prev_h: Tensor, r: Tensor, z: Tensor, n: Tensor)
}

/// Gradients for LSTM
pub type LSTMGradients {
  LSTMGradients(
    d_w_input: Tensor,
    d_w_hidden: Tensor,
    d_biases: Tensor,
    d_input: Tensor,
    d_h_prev: Tensor,
    d_c_prev: Tensor,
  )
}

/// Gradients for GRU
pub type GRUGradients {
  GRUGradients(
    d_w_r_input: Tensor,
    d_w_r_hidden: Tensor,
    d_b_r: Tensor,
    d_w_z_input: Tensor,
    d_w_z_hidden: Tensor,
    d_b_z: Tensor,
    d_w_n_input: Tensor,
    d_w_n_hidden: Tensor,
    d_b_n: Tensor,
    d_input: Tensor,
    d_h_prev: Tensor,
  )
}

// =============================================================================
// LSTM CONSTRUCTORS
// =============================================================================

/// Create new LSTM cell
pub fn lstm_new(input_size: Int, hidden_size: Int) -> LSTMCell {
  // Initialize with Xavier/Glorot
  let limit_i = float_sqrt(6.0 /. int.to_float(input_size + hidden_size))
  let limit_h = float_sqrt(6.0 /. int.to_float(hidden_size * 2))

  LSTMCell(
    w_input: random_uniform_init([4 * hidden_size, input_size], limit_i),
    w_hidden: random_uniform_init([4 * hidden_size, hidden_size], limit_h),
    biases: lstm_bias_init(hidden_size),
    input_size: input_size,
    hidden_size: hidden_size,
  )
}

/// Initialize LSTM biases (forget gate bias to 1.0 for better gradient flow)
fn lstm_bias_init(hidden_size: Int) -> Tensor {
  // [i, f, g, o] - forget gate initialized to 1.0
  let i_bias = list.repeat(0.0, hidden_size)
  let f_bias = list.repeat(1.0, hidden_size)
  // Key: forget bias = 1
  let g_bias = list.repeat(0.0, hidden_size)
  let o_bias = list.repeat(0.0, hidden_size)
  let data = list.flatten([i_bias, f_bias, g_bias, o_bias])
  Tensor(data: data, shape: [4 * hidden_size])
}

/// Create initial LSTM state (zeros)
pub fn lstm_init_state(hidden_size: Int) -> LSTMState {
  LSTMState(c: tensor.zeros([hidden_size]), h: tensor.zeros([hidden_size]))
}

// =============================================================================
// LSTM FORWARD
// =============================================================================

/// LSTM forward step
/// Input: [input_size], State: LSTMState
/// Output: (new_h, new_state, cache)
pub fn lstm_forward(
  cell: LSTMCell,
  input: Tensor,
  state: LSTMState,
) -> Result(#(Tensor, LSTMState, LSTMCache), TensorError) {
  let h = cell.hidden_size

  // Compute all gates at once: W_i @ x + W_h @ h + b
  use input_contrib <- result.try(tensor.matmul_vec(cell.w_input, input))
  use hidden_contrib <- result.try(tensor.matmul_vec(cell.w_hidden, state.h))
  use sum_contrib <- result.try(tensor.add(input_contrib, hidden_contrib))
  use gates_pre <- result.try(tensor.add(sum_contrib, cell.biases))

  // Split into 4 gates
  let i_pre = slice_tensor(gates_pre, 0, h)
  let f_pre = slice_tensor(gates_pre, h, h)
  let g_pre = slice_tensor(gates_pre, 2 * h, h)
  let o_pre = slice_tensor(gates_pre, 3 * h, h)

  // Apply activations
  let i = tensor.map(i_pre, sigmoid)
  // Input gate
  let f = tensor.map(f_pre, sigmoid)
  // Forget gate
  let g = tensor.map(g_pre, tanh)
  // Cell gate (candidate)
  let o = tensor.map(o_pre, sigmoid)
  // Output gate

  // Update cell state: c_new = f * c_prev + i * g
  use f_c <- result.try(tensor.mul(f, state.c))
  use i_g <- result.try(tensor.mul(i, g))
  use c_new <- result.try(tensor.add(f_c, i_g))

  // Compute new hidden: h_new = o * tanh(c_new)
  let c_tanh = tensor.map(c_new, tanh)
  use h_new <- result.try(tensor.mul(o, c_tanh))

  let new_state = LSTMState(c: c_new, h: h_new)

  let cache =
    LSTMCache(
      input: input,
      prev_state: state,
      gates_pre: gates_pre,
      i: i,
      f: f,
      g: g,
      o: o,
      c_new: c_new,
    )

  Ok(#(h_new, new_state, cache))
}

/// Process entire sequence with LSTM
pub fn lstm_sequence(
  cell: LSTMCell,
  inputs: List(Tensor),
  initial_state: LSTMState,
) -> Result(#(List(Tensor), LSTMState, List(LSTMCache)), TensorError) {
  let #(outputs_rev, final_state, caches_rev) =
    list.fold(inputs, #([], initial_state, []), fn(acc, input) {
      let #(outs, state, caches) = acc
      case lstm_forward(cell, input, state) {
        Ok(#(h, new_state, cache)) -> {
          #([h, ..outs], new_state, [cache, ..caches])
        }
        Error(_) -> acc
      }
    })

  Ok(#(list.reverse(outputs_rev), final_state, list.reverse(caches_rev)))
}

// =============================================================================
// LSTM BACKWARD (BPTT)
// =============================================================================

/// LSTM backward for single step
pub fn lstm_backward(
  cell: LSTMCell,
  cache: LSTMCache,
  d_h_next: Tensor,
  d_c_next: Tensor,
) -> Result(LSTMGradients, TensorError) {
  let _h = cell.hidden_size

  // d_o = d_h_next * tanh(c_new) * sigmoid'(o_pre)
  let c_tanh = tensor.map(cache.c_new, tanh)
  use d_o_contrib <- result.try(tensor.mul(d_h_next, c_tanh))
  let d_o =
    map2_with(d_o_contrib, cache.o, fn(d, o) {
      d *. o *. { 1.0 -. o }
      // sigmoid derivative
    })

  // d_c = d_c_next + d_h_next * o * (1 - tanh(c_new)^2)
  use d_c_from_h <- result.try(tensor.mul(d_h_next, cache.o))
  let d_c_tanh =
    map2_with(d_c_from_h, c_tanh, fn(d, t) {
      d *. { 1.0 -. t *. t }
      // tanh derivative
    })
  use d_c <- result.try(tensor.add(d_c_next, d_c_tanh))

  // d_f = d_c * c_prev * sigmoid'(f_pre)
  use d_f_contrib <- result.try(tensor.mul(d_c, cache.prev_state.c))
  let d_f = map2_with(d_f_contrib, cache.f, fn(d, f) { d *. f *. { 1.0 -. f } })

  // d_i = d_c * g * sigmoid'(i_pre)
  use d_i_contrib <- result.try(tensor.mul(d_c, cache.g))
  let d_i = map2_with(d_i_contrib, cache.i, fn(d, i) { d *. i *. { 1.0 -. i } })

  // d_g = d_c * i * tanh'(g_pre)
  use d_g_contrib <- result.try(tensor.mul(d_c, cache.i))
  let d_g = map2_with(d_g_contrib, cache.g, fn(d, g) { d *. { 1.0 -. g *. g } })

  // Concatenate gate gradients
  let d_gates = tensor.concat([d_i, d_f, d_g, d_o])

  // d_biases = d_gates
  let d_biases = d_gates

  // d_w_input = d_gates @ input^T
  use d_w_input <- result.try(tensor.outer(d_gates, cache.input))

  // d_w_hidden = d_gates @ h_prev^T
  use d_w_hidden <- result.try(tensor.outer(d_gates, cache.prev_state.h))

  // d_input = w_input^T @ d_gates
  use w_input_t <- result.try(tensor.transpose(cell.w_input))
  use d_input <- result.try(tensor.matmul_vec(w_input_t, d_gates))

  // d_h_prev = w_hidden^T @ d_gates
  use w_hidden_t <- result.try(tensor.transpose(cell.w_hidden))
  use d_h_prev <- result.try(tensor.matmul_vec(w_hidden_t, d_gates))

  // d_c_prev = d_c * f
  use d_c_prev <- result.try(tensor.mul(d_c, cache.f))

  Ok(LSTMGradients(
    d_w_input: d_w_input,
    d_w_hidden: d_w_hidden,
    d_biases: d_biases,
    d_input: d_input,
    d_h_prev: d_h_prev,
    d_c_prev: d_c_prev,
  ))
}

// =============================================================================
// GRU CONSTRUCTORS
// =============================================================================

/// Create new GRU cell
pub fn gru_new(input_size: Int, hidden_size: Int) -> GRUCell {
  let limit_i = float_sqrt(6.0 /. int.to_float(input_size + hidden_size))
  let limit_h = float_sqrt(6.0 /. int.to_float(hidden_size * 2))

  GRUCell(
    // Reset gate
    w_r_input: random_uniform_init([hidden_size, input_size], limit_i),
    w_r_hidden: random_uniform_init([hidden_size, hidden_size], limit_h),
    b_r: tensor.zeros([hidden_size]),
    // Update gate
    w_z_input: random_uniform_init([hidden_size, input_size], limit_i),
    w_z_hidden: random_uniform_init([hidden_size, hidden_size], limit_h),
    b_z: tensor.zeros([hidden_size]),
    // New hidden
    w_n_input: random_uniform_init([hidden_size, input_size], limit_i),
    w_n_hidden: random_uniform_init([hidden_size, hidden_size], limit_h),
    b_n: tensor.zeros([hidden_size]),
    input_size: input_size,
    hidden_size: hidden_size,
  )
}

/// Create initial GRU state (zeros)
pub fn gru_init_state(hidden_size: Int) -> GRUState {
  GRUState(h: tensor.zeros([hidden_size]))
}

// =============================================================================
// GRU FORWARD
// =============================================================================

/// GRU forward step
/// Simpler than LSTM - no cell state
pub fn gru_forward(
  cell: GRUCell,
  input: Tensor,
  state: GRUState,
) -> Result(#(Tensor, GRUState, GRUCache), TensorError) {
  // Reset gate: r = sigmoid(W_r_i @ x + W_r_h @ h + b_r)
  use r_i <- result.try(tensor.matmul_vec(cell.w_r_input, input))
  use r_h <- result.try(tensor.matmul_vec(cell.w_r_hidden, state.h))
  use r_sum <- result.try(tensor.add(r_i, r_h))
  use r_biased <- result.try(tensor.add(r_sum, cell.b_r))
  let r = tensor.map(r_biased, sigmoid)

  // Update gate: z = sigmoid(W_z_i @ x + W_z_h @ h + b_z)
  use z_i <- result.try(tensor.matmul_vec(cell.w_z_input, input))
  use z_h <- result.try(tensor.matmul_vec(cell.w_z_hidden, state.h))
  use z_sum <- result.try(tensor.add(z_i, z_h))
  use z_biased <- result.try(tensor.add(z_sum, cell.b_z))
  let z = tensor.map(z_biased, sigmoid)

  // New hidden: n = tanh(W_n_i @ x + r * (W_n_h @ h) + b_n)
  use n_i <- result.try(tensor.matmul_vec(cell.w_n_input, input))
  use n_h_pre <- result.try(tensor.matmul_vec(cell.w_n_hidden, state.h))
  use n_h <- result.try(tensor.mul(r, n_h_pre))
  // r gates the hidden contribution
  use n_sum <- result.try(tensor.add(n_i, n_h))
  use n_biased <- result.try(tensor.add(n_sum, cell.b_n))
  let n = tensor.map(n_biased, tanh)

  // New hidden state: h_new = (1 - z) * n + z * h
  let one_minus_z = tensor.map(z, fn(x) { 1.0 -. x })
  use term1 <- result.try(tensor.mul(one_minus_z, n))
  use term2 <- result.try(tensor.mul(z, state.h))
  use h_new <- result.try(tensor.add(term1, term2))

  let new_state = GRUState(h: h_new)

  let cache = GRUCache(input: input, prev_h: state.h, r: r, z: z, n: n)

  Ok(#(h_new, new_state, cache))
}

/// Process entire sequence with GRU
pub fn gru_sequence(
  cell: GRUCell,
  inputs: List(Tensor),
  initial_state: GRUState,
) -> Result(#(List(Tensor), GRUState, List(GRUCache)), TensorError) {
  let #(outputs_rev, final_state, caches_rev) =
    list.fold(inputs, #([], initial_state, []), fn(acc, input) {
      let #(outs, state, caches) = acc
      case gru_forward(cell, input, state) {
        Ok(#(h, new_state, cache)) -> {
          #([h, ..outs], new_state, [cache, ..caches])
        }
        Error(_) -> acc
      }
    })

  Ok(#(list.reverse(outputs_rev), final_state, list.reverse(caches_rev)))
}

// =============================================================================
// BIDIRECTIONAL WRAPPER
// =============================================================================

/// Result from bidirectional processing
pub type BidirectionalOutput {
  BidirectionalOutput(
    /// Forward outputs
    forward: List(Tensor),
    /// Backward outputs (reversed to align with forward)
    backward: List(Tensor),
    /// Combined outputs (concatenated)
    combined: List(Tensor),
  )
}

/// Process sequence bidirectionally with LSTM
pub fn lstm_bidirectional(
  forward_cell: LSTMCell,
  backward_cell: LSTMCell,
  inputs: List(Tensor),
) -> Result(BidirectionalOutput, TensorError) {
  let h = forward_cell.hidden_size

  // Forward pass
  let init_fwd = lstm_init_state(h)
  use #(fwd_outputs, _, _) <- result.try(lstm_sequence(
    forward_cell,
    inputs,
    init_fwd,
  ))

  // Backward pass (reverse input sequence)
  let init_bwd = lstm_init_state(h)
  let reversed_inputs = list.reverse(inputs)
  use #(bwd_outputs_rev, _, _) <- result.try(lstm_sequence(
    backward_cell,
    reversed_inputs,
    init_bwd,
  ))
  let bwd_outputs = list.reverse(bwd_outputs_rev)

  // Combine outputs
  let combined =
    list.map2(fwd_outputs, bwd_outputs, fn(f, b) { tensor.concat([f, b]) })

  Ok(BidirectionalOutput(
    forward: fwd_outputs,
    backward: bwd_outputs,
    combined: combined,
  ))
}

// =============================================================================
// HELPERS
// =============================================================================

fn sigmoid(x: Float) -> Float {
  1.0 /. { 1.0 +. float_exp(0.0 -. x) }
}

fn tanh(x: Float) -> Float {
  let e2x = float_exp(2.0 *. x)
  { e2x -. 1.0 } /. { e2x +. 1.0 }
}

/// Slice tensor from start to start+length
fn slice_tensor(t: Tensor, start: Int, length: Int) -> Tensor {
  let data =
    td(t)
    |> list.drop(start)
    |> list.take(length)
  Tensor(data: data, shape: [length])
}

/// Map over two tensors with custom function
fn map2_with(a: Tensor, b: Tensor, f: fn(Float, Float) -> Float) -> Tensor {
  let data = list.map2(td(a), td(b), f)
  Tensor(data: data, shape: a.shape)
}

/// Random uniform initialization in range [-limit, limit]
fn random_uniform_init(shape: List(Int), limit: Float) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let data =
    list.range(1, size)
    |> list.map(fn(_) {
      let r = random_float()
      r *. 2.0 *. limit -. limit
    })
  Tensor(data: data, shape: shape)
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "exp")
fn float_exp(x: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_float() -> Float
