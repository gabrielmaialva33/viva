//// Attention - Self-Attention and Multi-Head Attention mechanisms
////
//// Implements the Transformer attention mechanism from
//// "Attention Is All You Need" (Vaswani et al., 2017).
////
//// Core formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import viva/neural/tensor.{type Tensor, type TensorError, Tensor}
import viva/neural/utils

// =============================================================================
// TYPES
// =============================================================================

/// Result of attention computation
pub type AttentionResult {
  AttentionResult(
    /// Output of attention [seq_len, d_v]
    output: Tensor,
    /// Attention weights [seq_q, seq_k] for visualization
    weights: Tensor,
  )
}

/// Cache for backward pass
pub type AttentionCache {
  AttentionCache(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scores: Tensor,
    weights: Tensor,
  )
}

/// Multi-Head Attention layer
pub type MultiHeadAttention {
  MultiHeadAttention(
    /// Query projection [d_model, d_model]
    w_query: Tensor,
    /// Key projection [d_model, d_model]
    w_key: Tensor,
    /// Value projection [d_model, d_model]
    w_value: Tensor,
    /// Output projection [d_model, d_model]
    w_out: Tensor,
    /// Number of attention heads
    num_heads: Int,
    /// Model dimension
    d_model: Int,
    /// Dimension per head
    d_k: Int,
  )
}

/// Positional Encoding (sinusoidal)
pub type PositionalEncoding {
  PositionalEncoding(
    /// Precomputed encoding [max_len, d_model]
    encoding: Tensor,
    /// Maximum sequence length
    max_len: Int,
    /// Model dimension
    d_model: Int,
  )
}

/// Gradients for attention
pub type AttentionGradients {
  AttentionGradients(d_query: Tensor, d_key: Tensor, d_value: Tensor)
}

/// Gradients for multi-head attention
pub type MHAGradients {
  MHAGradients(
    d_w_query: Tensor,
    d_w_key: Tensor,
    d_w_value: Tensor,
    d_w_out: Tensor,
    d_input: Tensor,
  )
}

// =============================================================================
// SCALED DOT-PRODUCT ATTENTION
// =============================================================================

/// Scaled dot-product attention
/// Q: [seq_q, d_k] - Query vectors
/// K: [seq_k, d_k] - Key vectors
/// V: [seq_k, d_v] - Value vectors
/// mask: Optional [seq_q, seq_k] - Attention mask (0 = attend, -inf = ignore)
///
/// Returns: [seq_q, d_v]
pub fn scaled_dot_product_attention(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  mask: Option(Tensor),
) -> Result(#(AttentionResult, AttentionCache), TensorError) {
  case query.shape, key.shape, value.shape {
    [_seq_q, d_k], [seq_k, d_k2], [seq_k2, _d_v]
      if d_k == d_k2 && seq_k == seq_k2
    -> {
      // 1. Compute attention scores: Q @ K^T
      use key_t <- result.try(tensor.transpose(key))
      use scores <- result.try(tensor.matmul(query, key_t))

      // 2. Scale by sqrt(d_k)
      let scale = 1.0 /. float_sqrt(int.to_float(d_k))
      let scaled_scores = tensor.scale(scores, scale)

      // 3. Apply mask if provided (add large negative values)
      let masked_scores = case mask {
        None -> scaled_scores
        Some(m) -> {
          case tensor.add(scaled_scores, m) {
            Ok(result) -> result
            Error(_) -> scaled_scores
          }
        }
      }

      // 4. Softmax to get attention weights
      use weights <- result.try(utils.softmax_axis(masked_scores, 1))

      // 5. Apply attention: weights @ V
      use output <- result.try(tensor.matmul(weights, value))

      let result = AttentionResult(output: output, weights: weights)
      let cache =
        AttentionCache(
          query: query,
          key: key,
          value: value,
          scores: scaled_scores,
          weights: weights,
        )

      Ok(#(result, cache))
    }
    _, _, _ -> Error(tensor.DimensionError("Attention shape mismatch"))
  }
}

/// Simple attention forward (no cache)
pub fn attention(
  query: Tensor,
  key: Tensor,
  value: Tensor,
) -> Result(Tensor, TensorError) {
  use #(result, _) <- result.try(scaled_dot_product_attention(
    query,
    key,
    value,
    None,
  ))
  Ok(result.output)
}

// =============================================================================
// ATTENTION BACKWARD
// =============================================================================

/// Backward pass for scaled dot-product attention
pub fn attention_backward(
  cache: AttentionCache,
  d_output: Tensor,
) -> Result(AttentionGradients, TensorError) {
  case d_output.shape, cache.value.shape, cache.query.shape {
    [seq_q, _d_v], [seq_k, _d_v2], [_seq_q2, d_k] -> {
      // d_V = weights^T @ d_output
      use weights_t <- result.try(tensor.transpose(cache.weights))
      use d_value <- result.try(tensor.matmul(weights_t, d_output))

      // d_weights = d_output @ V^T
      use value_t <- result.try(tensor.transpose(cache.value))
      use d_weights <- result.try(tensor.matmul(d_output, value_t))

      // d_scores = softmax_backward(d_weights, weights)
      // For softmax: d_scores = weights * (d_weights - sum(d_weights * weights, axis=-1))
      use d_weights_times_weights <- result.try(tensor.mul(
        d_weights,
        cache.weights,
      ))
      let dww_data = tensor.to_list(d_weights_times_weights)
      let row_sums =
        list.range(0, seq_q - 1)
        |> list.map(fn(row) {
          let start = row * seq_k
          dww_data
          |> list.drop(start)
          |> list.take(seq_k)
          |> list.fold(0.0, fn(acc, x) { acc +. x })
        })

      let weights_data = tensor.to_list(cache.weights)
      let d_weights_data = tensor.to_list(d_weights)
      let d_scores_data =
        list.range(0, seq_q - 1)
        |> list.flat_map(fn(row) {
          let row_sum = case list_at(row_sums, row) {
            Ok(s) -> s
            Error(_) -> 0.0
          }
          list.range(0, seq_k - 1)
          |> list.map(fn(col) {
            let idx = row * seq_k + col
            let w = case list_at(weights_data, idx) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            let dw = case list_at(d_weights_data, idx) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            w *. { dw -. row_sum }
          })
        })
      let d_scores = Tensor(data: d_scores_data, shape: [seq_q, seq_k])

      // Scale back
      let scale = 1.0 /. float_sqrt(int.to_float(d_k))
      let d_scores_scaled = tensor.scale(d_scores, scale)

      // d_Q = d_scores_scaled @ K
      use d_query <- result.try(tensor.matmul(d_scores_scaled, cache.key))

      // d_K = d_scores_scaled^T @ Q
      use d_scores_t <- result.try(tensor.transpose(d_scores_scaled))
      use d_key <- result.try(tensor.matmul(d_scores_t, cache.query))

      Ok(AttentionGradients(d_query: d_query, d_key: d_key, d_value: d_value))
    }
    _, _, _ -> Error(tensor.DimensionError("Backward shape mismatch"))
  }
}

// =============================================================================
// MULTI-HEAD ATTENTION
// =============================================================================

/// Create multi-head attention layer
pub fn mha_new(d_model: Int, num_heads: Int) -> MultiHeadAttention {
  let d_k = d_model / num_heads

  // Xavier initialization
  let limit = float_sqrt(6.0 /. int.to_float(d_model * 2))

  MultiHeadAttention(
    w_query: random_uniform_init([d_model, d_model], limit),
    w_key: random_uniform_init([d_model, d_model], limit),
    w_value: random_uniform_init([d_model, d_model], limit),
    w_out: random_uniform_init([d_model, d_model], limit),
    num_heads: num_heads,
    d_model: d_model,
    d_k: d_k,
  )
}

/// Multi-head attention forward pass
/// Input: [seq_len, d_model]
/// For self-attention, query = key = value = input
pub fn mha_forward(
  layer: MultiHeadAttention,
  query: Tensor,
  key: Tensor,
  value: Tensor,
  mask: Option(Tensor),
) -> Result(#(Tensor, MHACache), TensorError) {
  case query.shape, key.shape, value.shape {
    [seq_q, d_model], [seq_k, _d_model2], [_seq_k2, _d_model3]
      if d_model == layer.d_model
    -> {
      // 1. Project to Q, K, V
      use q_proj <- result.try(tensor.matmul(query, layer.w_query))
      use k_proj <- result.try(tensor.matmul(key, layer.w_key))
      use v_proj <- result.try(tensor.matmul(value, layer.w_value))

      // 2. Split into heads
      // Reshape [seq, d_model] -> [seq, num_heads, d_k] -> process each head
      let head_outputs =
        list.range(0, layer.num_heads - 1)
        |> list.filter_map(fn(h) {
          // Extract head h: columns [h * d_k, (h+1) * d_k)
          let q_head = extract_head(q_proj, h, layer.d_k, seq_q)
          let k_head = extract_head(k_proj, h, layer.d_k, seq_k)
          let v_head = extract_head(v_proj, h, layer.d_k, seq_k)

          // Apply attention for this head
          case scaled_dot_product_attention(q_head, k_head, v_head, mask) {
            Ok(#(result, _cache)) -> Ok(result.output)
            Error(_) -> Error(Nil)
          }
        })

      // 3. Concatenate heads
      let concat_data =
        list.range(0, seq_q - 1)
        |> list.flat_map(fn(row) {
          list.flat_map(head_outputs, fn(head) {
            let head_data = tensor.to_list(head)
            let start = row * layer.d_k
            head_data
            |> list.drop(start)
            |> list.take(layer.d_k)
          })
        })
      let concat = Tensor(data: concat_data, shape: [seq_q, layer.d_model])

      // 4. Output projection
      use output <- result.try(tensor.matmul(concat, layer.w_out))

      let cache =
        MHACache(
          query: query,
          key: key,
          value: value,
          q_proj: q_proj,
          k_proj: k_proj,
          v_proj: v_proj,
          concat: concat,
        )

      Ok(#(output, cache))
    }
    _, _, _ -> Error(tensor.DimensionError("MHA shape mismatch"))
  }
}

/// Self-attention (Q = K = V)
pub fn self_attention(
  layer: MultiHeadAttention,
  input: Tensor,
  mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  use #(output, _) <- result.try(mha_forward(layer, input, input, input, mask))
  Ok(output)
}

/// Cache for MHA backward
pub type MHACache {
  MHACache(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    concat: Tensor,
  )
}

// =============================================================================
// ATTENTION MASKS
// =============================================================================

/// Create causal mask (for autoregressive models)
/// Prevents attending to future positions
pub fn causal_mask(seq_len: Int) -> Tensor {
  let data =
    list.range(0, seq_len - 1)
    |> list.flat_map(fn(i) {
      list.range(0, seq_len - 1)
      |> list.map(fn(j) {
        case j > i {
          True -> -1_000_000_000.0
          // Large negative (will become 0 after softmax)
          False -> 0.0
        }
      })
    })
  Tensor(data: data, shape: [seq_len, seq_len])
}

/// Create padding mask
/// mask_positions: list of indices to mask
pub fn padding_mask(seq_len: Int, mask_positions: List(Int)) -> Tensor {
  let data =
    list.range(0, seq_len - 1)
    |> list.map(fn(i) {
      case list.contains(mask_positions, i) {
        True -> -1_000_000_000.0
        False -> 0.0
      }
    })
  // Expand to [1, seq_len] for broadcasting
  Tensor(data: data, shape: [1, seq_len])
}

/// Combine causal and padding masks
pub fn combine_masks(
  causal: Tensor,
  padding: Tensor,
) -> Result(Tensor, TensorError) {
  // Broadcast padding [1, seq] to [seq, seq]
  case causal.shape, padding.shape {
    [seq, seq2], [1, seq3] if seq == seq2 && seq == seq3 -> {
      let causal_data = tensor.to_list(causal)
      let padding_data = tensor.to_list(padding)
      let data =
        list.range(0, seq - 1)
        |> list.flat_map(fn(i) {
          list.range(0, seq - 1)
          |> list.map(fn(j) {
            let causal_val = case list_at(causal_data, i * seq + j) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            let padding_val = case list_at(padding_data, j) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            float.min(causal_val, padding_val)
          })
        })
      Ok(Tensor(data: data, shape: [seq, seq]))
    }
    _, _ -> Error(tensor.DimensionError("Mask shape mismatch"))
  }
}

// =============================================================================
// POSITIONAL ENCODING
// =============================================================================

/// Create sinusoidal positional encoding
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
pub fn positional_encoding_new(max_len: Int, d_model: Int) -> PositionalEncoding {
  let data =
    list.range(0, max_len - 1)
    |> list.flat_map(fn(pos) {
      list.range(0, d_model - 1)
      |> list.map(fn(i) {
        let div_term =
          float_pow(
            10_000.0,
            int.to_float(2 * { i / 2 }) /. int.to_float(d_model),
          )
        let angle = int.to_float(pos) /. div_term
        case i % 2 == 0 {
          True -> float_sin(angle)
          False -> float_cos(angle)
        }
      })
    })

  PositionalEncoding(
    encoding: Tensor(data: data, shape: [max_len, d_model]),
    max_len: max_len,
    d_model: d_model,
  )
}

/// Add positional encoding to input embeddings
pub fn add_positional_encoding(
  pe: PositionalEncoding,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  case input.shape {
    [seq_len, d_model] if seq_len <= pe.max_len && d_model == pe.d_model -> {
      // Slice encoding to match sequence length
      let encoding_data = tensor.to_list(pe.encoding)
      let pe_data = list.take(encoding_data, seq_len * d_model)
      let pe_slice = Tensor(data: pe_data, shape: [seq_len, d_model])
      tensor.add(input, pe_slice)
    }
    _ -> Error(tensor.DimensionError("Positional encoding size mismatch"))
  }
}

/// Learned positional embeddings (alternative to sinusoidal)
pub type LearnedPositionalEmbedding {
  LearnedPositionalEmbedding(
    /// Learnable embeddings [max_len, d_model]
    embeddings: Tensor,
    max_len: Int,
    d_model: Int,
  )
}

/// Create learned positional embeddings
pub fn learned_pe_new(max_len: Int, d_model: Int) -> LearnedPositionalEmbedding {
  let limit = float_sqrt(6.0 /. int.to_float(d_model))
  LearnedPositionalEmbedding(
    embeddings: random_uniform_init([max_len, d_model], limit),
    max_len: max_len,
    d_model: d_model,
  )
}

// =============================================================================
// RELATIVE POSITIONAL ENCODING
// =============================================================================

/// Relative position bias
pub type RelativePositionBias {
  RelativePositionBias(
    /// Bias table [2 * max_distance - 1]
    bias_table: Tensor,
    /// Maximum relative distance
    max_distance: Int,
  )
}

/// Create relative position bias (as in T5, BERT)
pub fn relative_position_bias_new(max_distance: Int) -> RelativePositionBias {
  let limit = 0.02
  // Small initialization
  let table_size = 2 * max_distance - 1
  RelativePositionBias(
    bias_table: random_uniform_init([table_size], limit),
    max_distance: max_distance,
  )
}

/// Compute relative position bias matrix
pub fn compute_relative_bias(rpb: RelativePositionBias, seq_len: Int) -> Tensor {
  let bias_data = tensor.to_list(rpb.bias_table)
  let data =
    list.range(0, seq_len - 1)
    |> list.flat_map(fn(i) {
      list.range(0, seq_len - 1)
      |> list.map(fn(j) {
        let relative_pos = j - i
        let clamped =
          int.clamp(relative_pos, -rpb.max_distance + 1, rpb.max_distance - 1)
        let table_idx = clamped + rpb.max_distance - 1
        case list_at(bias_data, table_idx) {
          Ok(v) -> v
          Error(_) -> 0.0
        }
      })
    })
  Tensor(data: data, shape: [seq_len, seq_len])
}

// =============================================================================
// HELPERS
// =============================================================================

/// Extract head from projected tensor
fn extract_head(proj: Tensor, head_idx: Int, d_k: Int, seq_len: Int) -> Tensor {
  let proj_data = tensor.to_list(proj)
  let start_col = head_idx * d_k
  let data =
    list.range(0, seq_len - 1)
    |> list.flat_map(fn(row) {
      case proj.shape {
        [_seq, d_model] -> {
          let row_start = row * d_model
          proj_data
          |> list.drop(row_start + start_col)
          |> list.take(d_k)
        }
        _ -> list.repeat(0.0, d_k)
      }
    })
  Tensor(data: data, shape: [seq_len, d_k])
}

fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  lst
  |> list.drop(index)
  |> list.first
}

/// Random uniform initialization
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

@external(erlang, "math", "pow")
fn float_pow(base: Float, exp: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_float() -> Float
