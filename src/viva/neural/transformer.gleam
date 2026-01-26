//// Transformer - Complete Transformer architecture
////
//// Implements "Attention Is All You Need" (Vaswani et al., 2017)
//// with encoder, decoder, and encoder-decoder configurations.

import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import viva/neural/activation.{type ActivationType, ReLU}
import viva/neural/attention.{
  type MHACache, type MHAGradients, type MultiHeadAttention, type PositionalEncoding,
  causal_mask, mha_forward, mha_new, positional_encoding_new,
}
import viva/neural/normalization.{
  type LayerNormLayer, layer_norm_forward, layer_norm_new,
}
import viva/neural/tensor.{type Tensor, type TensorError, Tensor}

// =============================================================================
// TYPES
// =============================================================================

/// Feed-Forward Network (2 linear layers with activation)
pub type FeedForward {
  FeedForward(
    /// First linear: [d_model, d_ff]
    w1: Tensor,
    b1: Tensor,
    /// Second linear: [d_ff, d_model]
    w2: Tensor,
    b2: Tensor,
    /// Hidden dimension
    d_ff: Int,
    /// Model dimension
    d_model: Int,
    /// Activation function
    activation: ActivationType,
  )
}

/// Transformer Encoder Block
pub type EncoderBlock {
  EncoderBlock(
    /// Self-attention layer
    self_attn: MultiHeadAttention,
    /// Layer norm after attention
    norm1: LayerNormLayer,
    /// Feed-forward network
    ffn: FeedForward,
    /// Layer norm after FFN
    norm2: LayerNormLayer,
    /// Dropout rate
    dropout_rate: Float,
  )
}

/// Transformer Decoder Block
pub type DecoderBlock {
  DecoderBlock(
    /// Masked self-attention
    self_attn: MultiHeadAttention,
    /// Cross-attention (attends to encoder output)
    cross_attn: MultiHeadAttention,
    /// Layer norm after self-attention
    norm1: LayerNormLayer,
    /// Layer norm after cross-attention
    norm2: LayerNormLayer,
    /// Feed-forward network
    ffn: FeedForward,
    /// Layer norm after FFN
    norm3: LayerNormLayer,
    /// Dropout rate
    dropout_rate: Float,
  )
}

/// Full Transformer Encoder
pub type TransformerEncoder {
  TransformerEncoder(
    /// Stack of encoder blocks
    blocks: List(EncoderBlock),
    /// Positional encoding
    pos_encoding: PositionalEncoding,
    /// Model dimension
    d_model: Int,
    /// Number of layers
    num_layers: Int,
  )
}

/// Full Transformer Decoder
pub type TransformerDecoder {
  TransformerDecoder(
    /// Stack of decoder blocks
    blocks: List(DecoderBlock),
    /// Positional encoding
    pos_encoding: PositionalEncoding,
    /// Model dimension
    d_model: Int,
    /// Number of layers
    num_layers: Int,
  )
}

/// Complete Encoder-Decoder Transformer
pub type Transformer {
  Transformer(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    /// Output projection [d_model, vocab_size]
    output_proj: Tensor,
    /// Vocabulary size
    vocab_size: Int,
  )
}

/// Cache for Transformer forward pass
pub type TransformerCache {
  TransformerCache(
    encoder_outputs: List(Tensor),
    decoder_outputs: List(Tensor),
    attn_caches: List(MHACache),
  )
}

/// Gradients for Transformer
pub type TransformerGradients {
  TransformerGradients(
    encoder_grads: List(EncoderBlockGradients),
    decoder_grads: List(DecoderBlockGradients),
    d_output_proj: Tensor,
  )
}

/// Gradients for encoder block
pub type EncoderBlockGradients {
  EncoderBlockGradients(
    d_self_attn: MHAGradients,
    d_ffn: FFNGradients,
    d_norm1: Tensor,
    d_norm2: Tensor,
  )
}

/// Gradients for decoder block
pub type DecoderBlockGradients {
  DecoderBlockGradients(
    d_self_attn: MHAGradients,
    d_cross_attn: MHAGradients,
    d_ffn: FFNGradients,
  )
}

/// Gradients for FFN
pub type FFNGradients {
  FFNGradients(d_w1: Tensor, d_b1: Tensor, d_w2: Tensor, d_b2: Tensor)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create feed-forward network
pub fn ffn_new(d_model: Int, d_ff: Int) -> FeedForward {
  let limit1 = float_sqrt(6.0 /. int.to_float(d_model + d_ff))
  let limit2 = float_sqrt(6.0 /. int.to_float(d_ff + d_model))

  FeedForward(
    w1: random_uniform_init([d_model, d_ff], limit1),
    b1: tensor.zeros([d_ff]),
    w2: random_uniform_init([d_ff, d_model], limit2),
    b2: tensor.zeros([d_model]),
    d_ff: d_ff,
    d_model: d_model,
    activation: ReLU,
  )
}

/// Create encoder block
pub fn encoder_block_new(
  d_model: Int,
  num_heads: Int,
  d_ff: Int,
  dropout_rate: Float,
) -> EncoderBlock {
  EncoderBlock(
    self_attn: mha_new(d_model, num_heads),
    norm1: layer_norm_new([d_model]),
    ffn: ffn_new(d_model, d_ff),
    norm2: layer_norm_new([d_model]),
    dropout_rate: dropout_rate,
  )
}

/// Create decoder block
pub fn decoder_block_new(
  d_model: Int,
  num_heads: Int,
  d_ff: Int,
  dropout_rate: Float,
) -> DecoderBlock {
  DecoderBlock(
    self_attn: mha_new(d_model, num_heads),
    cross_attn: mha_new(d_model, num_heads),
    norm1: layer_norm_new([d_model]),
    norm2: layer_norm_new([d_model]),
    ffn: ffn_new(d_model, d_ff),
    norm3: layer_norm_new([d_model]),
    dropout_rate: dropout_rate,
  )
}

/// Create transformer encoder
pub fn encoder_new(
  d_model: Int,
  num_heads: Int,
  d_ff: Int,
  num_layers: Int,
  max_seq_len: Int,
  dropout_rate: Float,
) -> TransformerEncoder {
  let blocks =
    list.range(1, num_layers)
    |> list.map(fn(_) { encoder_block_new(d_model, num_heads, d_ff, dropout_rate) })

  TransformerEncoder(
    blocks: blocks,
    pos_encoding: positional_encoding_new(max_seq_len, d_model),
    d_model: d_model,
    num_layers: num_layers,
  )
}

/// Create transformer decoder
pub fn decoder_new(
  d_model: Int,
  num_heads: Int,
  d_ff: Int,
  num_layers: Int,
  max_seq_len: Int,
  dropout_rate: Float,
) -> TransformerDecoder {
  let blocks =
    list.range(1, num_layers)
    |> list.map(fn(_) { decoder_block_new(d_model, num_heads, d_ff, dropout_rate) })

  TransformerDecoder(
    blocks: blocks,
    pos_encoding: positional_encoding_new(max_seq_len, d_model),
    d_model: d_model,
    num_layers: num_layers,
  )
}

/// Create full encoder-decoder transformer
pub fn transformer_new(
  d_model: Int,
  num_heads: Int,
  d_ff: Int,
  num_encoder_layers: Int,
  num_decoder_layers: Int,
  max_seq_len: Int,
  vocab_size: Int,
  dropout_rate: Float,
) -> Transformer {
  let limit = float_sqrt(6.0 /. int.to_float(d_model + vocab_size))

  Transformer(
    encoder: encoder_new(d_model, num_heads, d_ff, num_encoder_layers, max_seq_len, dropout_rate),
    decoder: decoder_new(d_model, num_heads, d_ff, num_decoder_layers, max_seq_len, dropout_rate),
    output_proj: random_uniform_init([d_model, vocab_size], limit),
    vocab_size: vocab_size,
  )
}

/// Create encoder-only transformer (like BERT)
pub fn encoder_only(
  d_model: Int,
  num_heads: Int,
  d_ff: Int,
  num_layers: Int,
  max_seq_len: Int,
  dropout_rate: Float,
) -> TransformerEncoder {
  encoder_new(d_model, num_heads, d_ff, num_layers, max_seq_len, dropout_rate)
}

/// Create decoder-only transformer (like GPT)
pub fn decoder_only(
  d_model: Int,
  num_heads: Int,
  d_ff: Int,
  num_layers: Int,
  max_seq_len: Int,
  dropout_rate: Float,
) -> TransformerDecoder {
  decoder_new(d_model, num_heads, d_ff, num_layers, max_seq_len, dropout_rate)
}

// =============================================================================
// FORWARD PASS
// =============================================================================

/// Feed-forward forward pass
/// x: [seq_len, d_model] -> [seq_len, d_model]
pub fn ffn_forward(ffn: FeedForward, x: Tensor) -> Result(Tensor, TensorError) {
  // Linear 1: x @ W1 + b1
  use h <- result.try(tensor.matmul(x, ffn.w1))
  use h <- result.try(add_bias_2d(h, ffn.b1))

  // Activation (ReLU)
  let h = apply_activation_2d(h, ffn.activation)

  // Linear 2: h @ W2 + b2
  use out <- result.try(tensor.matmul(h, ffn.w2))
  add_bias_2d(out, ffn.b2)
}

/// Encoder block forward pass
/// Pre-LN Transformer variant (norm before attention/ffn)
pub fn encoder_block_forward(
  block: EncoderBlock,
  x: Tensor,
  mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  let seq_len = case x.shape {
    [s, _] -> s
    _ -> 0
  }

  // 1. Self-attention with residual
  // Pre-norm: norm(x) -> attention -> add x
  use x_norm1 <- result.try(layer_norm_forward_2d(block.norm1, x, seq_len))
  use #(attn_out, _cache) <- result.try(mha_forward(
    block.self_attn,
    x_norm1,
    x_norm1,
    x_norm1,
    mask,
  ))
  use x <- result.try(tensor.add(x, attn_out))

  // 2. FFN with residual
  use x_norm2 <- result.try(layer_norm_forward_2d(block.norm2, x, seq_len))
  use ffn_out <- result.try(ffn_forward(block.ffn, x_norm2))
  tensor.add(x, ffn_out)
}

/// Decoder block forward pass
pub fn decoder_block_forward(
  block: DecoderBlock,
  x: Tensor,
  encoder_output: Tensor,
  self_mask: Option(Tensor),
  cross_mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  let seq_len = case x.shape {
    [s, _] -> s
    _ -> 0
  }

  // 1. Masked self-attention with residual
  use x_norm1 <- result.try(layer_norm_forward_2d(block.norm1, x, seq_len))
  use #(self_attn_out, _) <- result.try(mha_forward(
    block.self_attn,
    x_norm1,
    x_norm1,
    x_norm1,
    self_mask,
  ))
  use x <- result.try(tensor.add(x, self_attn_out))

  // 2. Cross-attention with residual (attends to encoder output)
  use x_norm2 <- result.try(layer_norm_forward_2d(block.norm2, x, seq_len))
  use #(cross_attn_out, _) <- result.try(mha_forward(
    block.cross_attn,
    x_norm2,           // Query from decoder
    encoder_output,    // Key from encoder
    encoder_output,    // Value from encoder
    cross_mask,
  ))
  use x <- result.try(tensor.add(x, cross_attn_out))

  // 3. FFN with residual
  use x_norm3 <- result.try(layer_norm_forward_2d(block.norm3, x, seq_len))
  use ffn_out <- result.try(ffn_forward(block.ffn, x_norm3))
  tensor.add(x, ffn_out)
}

/// Transformer encoder forward pass
/// Input: [seq_len, d_model] (after embedding)
pub fn encoder_forward(
  encoder: TransformerEncoder,
  x: Tensor,
  mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  // Add positional encoding
  use x <- result.try(add_positional_encoding(x, encoder.pos_encoding))

  // Pass through all encoder blocks
  list.fold(encoder.blocks, Ok(x), fn(result, block) {
    case result {
      Ok(h) -> encoder_block_forward(block, h, mask)
      Error(e) -> Error(e)
    }
  })
}

/// Transformer decoder forward pass
/// tgt: [tgt_seq_len, d_model] - Target sequence
/// memory: [src_seq_len, d_model] - Encoder output
pub fn decoder_forward(
  decoder: TransformerDecoder,
  tgt: Tensor,
  memory: Tensor,
  tgt_mask: Option(Tensor),
  memory_mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  // Add positional encoding
  use x <- result.try(add_positional_encoding(tgt, decoder.pos_encoding))

  // Pass through all decoder blocks
  list.fold(decoder.blocks, Ok(x), fn(result, block) {
    case result {
      Ok(h) -> decoder_block_forward(block, h, memory, tgt_mask, memory_mask)
      Error(e) -> Error(e)
    }
  })
}

/// Full transformer forward pass (for sequence-to-sequence)
/// src: [src_seq_len, d_model] - Source sequence (embedded)
/// tgt: [tgt_seq_len, d_model] - Target sequence (embedded)
pub fn transformer_forward(
  transformer: Transformer,
  src: Tensor,
  tgt: Tensor,
  src_mask: Option(Tensor),
  tgt_mask: Option(Tensor),
  memory_mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  // 1. Encode source
  use memory <- result.try(encoder_forward(transformer.encoder, src, src_mask))

  // 2. Decode target with encoder memory
  use decoder_out <- result.try(decoder_forward(
    transformer.decoder,
    tgt,
    memory,
    tgt_mask,
    memory_mask,
  ))

  // 3. Project to vocabulary
  tensor.matmul(decoder_out, transformer.output_proj)
}

/// Encoder-only forward (for BERT-like models)
pub fn encoder_only_forward(
  encoder: TransformerEncoder,
  x: Tensor,
  mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  encoder_forward(encoder, x, mask)
}

/// Decoder-only forward with causal mask (for GPT-like models)
pub fn decoder_only_forward(
  decoder: TransformerDecoder,
  x: Tensor,
) -> Result(Tensor, TensorError) {
  let seq_len = case x.shape {
    [s, _] -> s
    _ -> 0
  }

  // Create causal mask to prevent attending to future positions
  let mask = causal_mask(seq_len)

  // Use x as both target and memory (self-attention only)
  // Add positional encoding
  use x <- result.try(add_positional_encoding(x, decoder.pos_encoding))

  // Pass through decoder blocks (self-attention only mode)
  list.fold(decoder.blocks, Ok(x), fn(result, block) {
    case result {
      Ok(h) -> decoder_block_self_attn_only(block, h, Some(mask))
      Error(e) -> Error(e)
    }
  })
}

/// Decoder block with only self-attention (for decoder-only models)
fn decoder_block_self_attn_only(
  block: DecoderBlock,
  x: Tensor,
  mask: Option(Tensor),
) -> Result(Tensor, TensorError) {
  let seq_len = case x.shape {
    [s, _] -> s
    _ -> 0
  }

  // 1. Masked self-attention with residual
  use x_norm1 <- result.try(layer_norm_forward_2d(block.norm1, x, seq_len))
  use #(self_attn_out, _) <- result.try(mha_forward(
    block.self_attn,
    x_norm1,
    x_norm1,
    x_norm1,
    mask,
  ))
  use x <- result.try(tensor.add(x, self_attn_out))

  // 2. FFN with residual (skip cross-attention)
  use x_norm3 <- result.try(layer_norm_forward_2d(block.norm3, x, seq_len))
  use ffn_out <- result.try(ffn_forward(block.ffn, x_norm3))
  tensor.add(x, ffn_out)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Add bias to 2D tensor [seq_len, dim]
fn add_bias_2d(x: Tensor, bias: Tensor) -> Result(Tensor, TensorError) {
  case x.shape, bias.shape {
    [seq_len, dim], [dim2] if dim == dim2 -> {
      let xdata = tensor.to_list(x)
      let bdata = tensor.to_list(bias)
      let result =
        list.range(0, seq_len - 1)
        |> list.flat_map(fn(row) {
          let start = row * dim
          list.range(0, dim - 1)
          |> list.map(fn(col) {
            let x_val = case list_at(xdata, start + col) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            let b_val = case list_at(bdata, col) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            x_val +. b_val
          })
        })
      Ok(Tensor(data: result, shape: [seq_len, dim]))
    }
    _, _ -> Error(tensor.DimensionError("Bias shape mismatch"))
  }
}

/// Apply activation to 2D tensor
fn apply_activation_2d(x: Tensor, activation: ActivationType) -> Tensor {
  let data = tensor.to_list(x)
  let new_data = case activation {
    ReLU -> list.map(data, fn(v) { float_max(0.0, v) })
    _ -> data
  }
  Tensor(data: new_data, shape: x.shape)
}

/// Layer norm for 2D tensor [seq_len, d_model]
fn layer_norm_forward_2d(
  norm: LayerNormLayer,
  x: Tensor,
  seq_len: Int,
) -> Result(Tensor, TensorError) {
  // Apply layer norm to each row (position)
  let xdata = tensor.to_list(x)
  let d_model = case x.shape {
    [_, d] -> d
    _ -> 0
  }

  let gamma_data = tensor.to_list(norm.gamma)
  let beta_data = tensor.to_list(norm.beta)

  let result =
    list.range(0, seq_len - 1)
    |> list.flat_map(fn(row) {
      let start = row * d_model
      let row_data =
        list.range(0, d_model - 1)
        |> list.filter_map(fn(col) { list_at(xdata, start + col) })

      // Compute mean
      let sum = list.fold(row_data, 0.0, fn(acc, v) { acc +. v })
      let mean = sum /. int.to_float(d_model)

      // Compute variance
      let var_sum =
        list.fold(row_data, 0.0, fn(acc, v) {
          let diff = v -. mean
          acc +. diff *. diff
        })
      let var = var_sum /. int.to_float(d_model)

      // Normalize and scale
      let inv_std = 1.0 /. float_sqrt(var +. norm.epsilon)
      list.range(0, d_model - 1)
      |> list.map(fn(col) {
        let x_val = case list_at(row_data, col) {
          Ok(v) -> v
          Error(_) -> 0.0
        }
        let gamma = case list_at(gamma_data, col) {
          Ok(v) -> v
          Error(_) -> 1.0
        }
        let beta = case list_at(beta_data, col) {
          Ok(v) -> v
          Error(_) -> 0.0
        }
        { x_val -. mean } *. inv_std *. gamma +. beta
      })
    })

  Ok(Tensor(data: result, shape: [seq_len, d_model]))
}

/// Add positional encoding to input
fn add_positional_encoding(
  x: Tensor,
  pos_enc: PositionalEncoding,
) -> Result(Tensor, TensorError) {
  case x.shape {
    [seq_len, d_model] if d_model == pos_enc.d_model && seq_len <= pos_enc.max_len -> {
      let xdata = tensor.to_list(x)
      let pos_data = tensor.to_list(pos_enc.encoding)

      let result =
        list.range(0, seq_len - 1)
        |> list.flat_map(fn(pos) {
          list.range(0, d_model - 1)
          |> list.map(fn(dim) {
            let x_val = case list_at(xdata, pos * d_model + dim) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            let pos_val = case list_at(pos_data, pos * d_model + dim) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
            x_val +. pos_val
          })
        })

      Ok(Tensor(data: result, shape: [seq_len, d_model]))
    }
    _ -> Error(tensor.DimensionError("Position encoding size mismatch"))
  }
}

/// Random uniform initialization
fn random_uniform_init(shape: List(Int), limit: Float) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, x) { acc * x })
  let data =
    list.range(1, size)
    |> list.map(fn(i) {
      let x = int.to_float(i * 1103515245 + 12345)
      let normalized = float_mod(x, 1000000.0) /. 1000000.0
      { normalized *. 2.0 -. 1.0 } *. limit
    })
  Tensor(data: data, shape: shape)
}

fn list_at(lst: List(Float), index: Int) -> Result(Float, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}

// =============================================================================
// UTILITY
// =============================================================================

/// Get total parameter count
pub fn transformer_param_count(transformer: Transformer) -> Int {
  let encoder_params = encoder_param_count(transformer.encoder)
  let decoder_params = decoder_param_count(transformer.decoder)
  let output_params = tensor.size(transformer.output_proj)
  encoder_params + decoder_params + output_params
}

/// Encoder parameter count
pub fn encoder_param_count(encoder: TransformerEncoder) -> Int {
  list.fold(encoder.blocks, 0, fn(acc, block) { acc + encoder_block_params(block) })
}

/// Decoder parameter count
pub fn decoder_param_count(decoder: TransformerDecoder) -> Int {
  list.fold(decoder.blocks, 0, fn(acc, block) { acc + decoder_block_params(block) })
}

fn encoder_block_params(block: EncoderBlock) -> Int {
  let attn = mha_params(block.self_attn)
  let norm1 = tensor.size(block.norm1.gamma) + tensor.size(block.norm1.beta)
  let norm2 = tensor.size(block.norm2.gamma) + tensor.size(block.norm2.beta)
  let ffn = ffn_params(block.ffn)
  attn + norm1 + norm2 + ffn
}

fn decoder_block_params(block: DecoderBlock) -> Int {
  let self_attn = mha_params(block.self_attn)
  let cross_attn = mha_params(block.cross_attn)
  let norm1 = tensor.size(block.norm1.gamma) + tensor.size(block.norm1.beta)
  let norm2 = tensor.size(block.norm2.gamma) + tensor.size(block.norm2.beta)
  let norm3 = tensor.size(block.norm3.gamma) + tensor.size(block.norm3.beta)
  let ffn = ffn_params(block.ffn)
  self_attn + cross_attn + norm1 + norm2 + norm3 + ffn
}

fn mha_params(mha: MultiHeadAttention) -> Int {
  tensor.size(mha.w_query) + tensor.size(mha.w_key) + tensor.size(mha.w_value) + tensor.size(mha.w_out)
}

fn ffn_params(ffn: FeedForward) -> Int {
  tensor.size(ffn.w1) + tensor.size(ffn.b1) + tensor.size(ffn.w2) + tensor.size(ffn.b2)
}

/// Describe transformer architecture
pub fn describe(transformer: Transformer) -> String {
  "Transformer(enc="
  <> int_to_string(transformer.encoder.num_layers)
  <> ", dec="
  <> int_to_string(transformer.decoder.num_layers)
  <> ", d="
  <> int_to_string(transformer.encoder.d_model)
  <> ", vocab="
  <> int_to_string(transformer.vocab_size)
  <> ", params="
  <> int_to_string(transformer_param_count(transformer))
  <> ")"
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "math", "sqrt")
fn float_sqrt(f: Float) -> Float

@external(erlang, "erlang", "max")
fn float_max(a: Float, b: Float) -> Float

fn float_mod(a: Float, b: Float) -> Float {
  a -. float_floor(a /. b) *. b
}

@external(erlang, "math", "floor")
fn float_floor(f: Float) -> Float

@external(erlang, "erlang", "integer_to_binary")
fn int_to_string(i: Int) -> String
