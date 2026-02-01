//// VIVA LLM - llama.cpp integration for knowledge distillation
////
//// Extracts hidden states and logits from teacher models (Qwen3-32B)
//// for DHC (Distilled Holographic Compression) training.
////
//// Key functions:
//// - load_teacher(): Load Qwen3-32B as teacher
//// - get_hidden_states(): Extract embeddings from all layers
//// - get_logits(): Get soft labels for distillation
////

import gleam/list
import gleam/int

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// Opaque reference to loaded LLM model
pub type LlmModel

/// Model info: embedding dimension and vocabulary size
pub type ModelInfo {
  ModelInfo(n_embd: Int, n_vocab: Int)
}

/// Hidden states extracted from teacher model
pub type HiddenStates {
  HiddenStates(
    embeddings: List(Float),  // Flattened [n_tokens, n_embd]
    n_tokens: Int,
    n_embd: Int,
  )
}

/// Error types for LLM operations
pub type LlmError {
  ModelNotFound(path: String)
  LoadFailed(reason: String)
  InferenceFailed(reason: String)
  TokenizationFailed(reason: String)
}

// -----------------------------------------------------------------------------
// FFI External Functions
// -----------------------------------------------------------------------------

@external(erlang, "Elixir.Viva.Llm.Native", "llm_load_model")
fn ffi_load_model(path: String, gpu_layers: Int) -> LlmModel

@external(erlang, "Elixir.Viva.Llm.Native", "llm_model_info")
fn ffi_model_info(model: LlmModel) -> #(Int, Int)

@external(erlang, "Elixir.Viva.Llm.Native", "llm_get_hidden_states")
fn ffi_get_hidden_states(
  model: LlmModel,
  prompt: String,
  ctx_size: Int,
) -> #(List(Float), Int)

@external(erlang, "Elixir.Viva.Llm.Native", "llm_get_logits")
fn ffi_get_logits(model: LlmModel, prompt: String, ctx_size: Int) -> List(Float)

@external(erlang, "Elixir.Viva.Llm.Native", "llm_predict")
fn ffi_predict(model: LlmModel, prompt: String) -> #(String, List(Float))

@external(erlang, "Elixir.Viva.Llm.Native", "llm_tokenize")
fn ffi_tokenize(model: LlmModel, text: String) -> List(Int)

@external(erlang, "Elixir.Viva.Llm.Native", "llm_detokenize")
fn ffi_detokenize(model: LlmModel, tokens: List(Int)) -> String

@external(erlang, "Elixir.Viva.Llm.Native", "llm_memory_status")
fn ffi_memory_status() -> #(Int, Int)

@external(erlang, "Elixir.Viva.Llm.Native", "llm_native_check")
fn ffi_native_check() -> String

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// Load teacher model (Qwen3-32B) from GGUF file
///
/// ## Arguments
/// - `path`: Absolute path to .gguf model file
/// - `gpu_layers`: Number of layers to offload to GPU (65 for full Qwen3-32B)
///
/// ## Returns
/// - `Ok(LlmModel)`: Handle to loaded model
/// - `Error(LlmError)`: Load failed
pub fn load_model(path: String, gpu_layers: Int) -> LlmModel {
  ffi_load_model(path, gpu_layers)
}

/// Load Qwen3-32B teacher model with default settings
/// Uses models/Qwen3-32B-Q4_K_M.gguf with 65 GPU layers
pub fn load_teacher() -> LlmModel {
  load_model("models/Qwen3-32B-Q4_K_M.gguf", 65)
}

/// Get model dimensions
pub fn model_info(model: LlmModel) -> ModelInfo {
  let #(n_embd, n_vocab) = ffi_model_info(model)
  ModelInfo(n_embd: n_embd, n_vocab: n_vocab)
}

/// Extract hidden states (embeddings) from teacher model
///
/// This is the KEY function for DHC distillation - it extracts
/// the internal representations that will be compressed into HRR.
///
/// ## Arguments
/// - `model`: Loaded teacher model
/// - `prompt`: Input text to process
/// - `ctx_size`: Context window size (default: 2048)
///
/// ## Returns
/// HiddenStates with flattened embeddings [n_tokens, n_embd]
pub fn get_hidden_states(
  model: LlmModel,
  prompt: String,
  ctx_size: Int,
) -> HiddenStates {
  let info = model_info(model)
  let #(embeddings, n_tokens) = ffi_get_hidden_states(model, prompt, ctx_size)

  HiddenStates(
    embeddings: embeddings,
    n_tokens: n_tokens,
    n_embd: info.n_embd,
  )
}

/// Get logits (output probabilities) for soft-label distillation
///
/// Returns logits for the LAST token, useful for:
/// - Temperature-scaled soft labels
/// - KL divergence loss computation
///
/// ## Returns
/// List of logits with length = n_vocab
pub fn get_logits(
  model: LlmModel,
  prompt: String,
  ctx_size: Int,
) -> List(Float) {
  ffi_get_logits(model, prompt, ctx_size)
}

/// Apply softmax with temperature scaling
/// Used for knowledge distillation soft labels
pub fn softmax_with_temperature(
  logits: List(Float),
  temperature: Float,
) -> List(Float) {
  // Scale by temperature
  let scaled = list.map(logits, fn(x) { x /. temperature })

  // Find max for numerical stability
  let max_val = list.fold(scaled, -1_000_000.0, fn(acc, x) {
    case x >. acc {
      True -> x
      False -> acc
    }
  })

  // Compute exp(x - max)
  let exps = list.map(scaled, fn(x) {
    let diff = x -. max_val
    exp(diff)
  })

  // Sum of exps
  let sum = list.fold(exps, 0.0, fn(acc, x) { acc +. x })

  // Normalize
  list.map(exps, fn(x) { x /. sum })
}

/// Tokenize text to token IDs
pub fn tokenize(model: LlmModel, text: String) -> List(Int) {
  ffi_tokenize(model, text)
}

/// Detokenize token IDs back to text
pub fn detokenize(model: LlmModel, tokens: List(Int)) -> String {
  ffi_detokenize(model, tokens)
}

/// Get embedding for last token (legacy/convenience)
pub fn get_last_embedding(model: LlmModel, prompt: String) -> List(Float) {
  let #(_, embedding) = ffi_predict(model, prompt)
  embedding
}

/// Get system memory status
/// Returns (total_memory, free_memory) in bytes
pub fn memory_status() -> #(Int, Int) {
  ffi_memory_status()
}

/// Check native capabilities (AVX512, AVX2, SSE)
pub fn native_check() -> String {
  ffi_native_check()
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Reshape flattened embeddings to 2D [n_tokens][n_embd]
pub fn reshape_hidden_states(
  states: HiddenStates,
) -> List(List(Float)) {
  chunk_list(states.embeddings, states.n_embd)
}

/// Get mean pooled embedding (average of all token embeddings)
pub fn mean_pooled_embedding(states: HiddenStates) -> List(Float) {
  let token_embeddings = reshape_hidden_states(states)
  let n_tokens = int.to_float(states.n_tokens)

  // Sum across tokens
  let summed = list.fold(token_embeddings, list.repeat(0.0, states.n_embd), fn(acc, emb) {
    list.map2(acc, emb, fn(a, e) { a +. e })
  })

  // Average
  list.map(summed, fn(x) { x /. n_tokens })
}

// Chunk a list into sublists of given size
fn chunk_list(lst: List(a), size: Int) -> List(List(a)) {
  case lst {
    [] -> []
    _ -> {
      let #(chunk, rest) = list.split(lst, size)
      [chunk, ..chunk_list(rest, size)]
    }
  }
}

// FFI for exp function
@external(erlang, "math", "exp")
fn exp(x: Float) -> Float
