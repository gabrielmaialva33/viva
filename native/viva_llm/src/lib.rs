use rustler::{Env, ResourceArc, Term};
use std::path::Path;
use std::num::NonZeroU32;
use std::sync::Mutex;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use sysinfo::{System, RefreshKind, MemoryRefreshKind};

// --- Resources ---
struct LlmResource {
    model: LlamaModel,
    backend: LlamaBackend,
    n_embd: i32,      // embedding dimension
    n_vocab: i32,     // vocabulary size
}

// Thread-safe wrapper for context (contexts are not Send)
struct ContextResource {
    // We'll create context per-call since llama contexts aren't easily shareable
    model_path: String,
    gpu_layers: i32,
}

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

// --- NIFs ---

/// Load a GGUF model into GPU memory
/// Returns: ResourceArc with model info
#[rustler::nif(schedule = "DirtyIo")]
fn llm_load_model(path: String, gpu_layers: i32) -> Result<ResourceArc<LlmResource>, rustler::Error> {
    let backend = LlamaBackend::init()
        .map_err(|e| rustler::Error::Term(Box::new(format!("backend_init: {}", e))))?;

    let model_path = Path::new(&path);
    if !model_path.exists() {
        return Err(rustler::Error::Term(Box::new(format!("file_not_found: {}", path))));
    }

    let model_params = llama_cpp_2::model::params::LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers as u32);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .map_err(|e| rustler::Error::Term(Box::new(format!("load_failed: {}", e))))?;

    // Get model dimensions
    let n_embd = model.n_embd();
    let n_vocab = model.n_vocab();

    Ok(ResourceArc::new(LlmResource {
        model,
        backend,
        n_embd,
        n_vocab,
    }))
}

/// Get model info: (n_embd, n_vocab)
#[rustler::nif]
fn llm_model_info(resource: ResourceArc<LlmResource>) -> (i32, i32) {
    (resource.n_embd, resource.n_vocab)
}

/// Extract hidden states (embeddings) from the last layer for a given prompt
/// This is the key function for knowledge distillation
/// Returns: Vec<f32> of shape [n_tokens, n_embd] flattened
#[rustler::nif(schedule = "DirtyCpu")]
fn llm_get_hidden_states(
    resource: ResourceArc<LlmResource>,
    prompt: String,
    ctx_size: u32,
) -> Result<(Vec<f32>, i32), rustler::Error> {
    // Create context with embeddings enabled
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size))
        .with_embeddings(true);  // CRITICAL: Enable embeddings extraction

    let mut ctx = resource.model.new_context(&resource.backend, ctx_params)
        .map_err(|e| rustler::Error::Term(Box::new(format!("context_failed: {}", e))))?;

    // Tokenize
    let tokens = resource.model.str_to_token(&prompt, AddBos::Always)
        .map_err(|e| rustler::Error::Term(Box::new(format!("tokenize_failed: {}", e))))?;

    let n_tokens = tokens.len() as i32;

    ctx.clear_kv_cache();

    // Create batch with all tokens having logits enabled (for embedding extraction)
    let batch_size = tokens.len().max(512);
    let mut batch = LlamaBatch::new(batch_size, 1);

    for (i, token) in tokens.iter().enumerate() {
        // Enable logits for ALL tokens to get embeddings for each
        batch.add(*token, i as i32, &[0], true)
            .map_err(|e| rustler::Error::Term(Box::new(format!("batch_add: {}", e))))?;
    }

    // Decode
    ctx.decode(&mut batch)
        .map_err(|e| rustler::Error::Term(Box::new(format!("decode_failed: {}", e))))?;

    // Extract embeddings for each token
    let n_embd = resource.n_embd as usize;
    let mut all_embeddings = Vec::with_capacity(n_tokens as usize * n_embd);

    for i in 0..n_tokens {
        match ctx.embeddings_ith(i) {
            Ok(emb) => {
                all_embeddings.extend_from_slice(emb);
            }
            Err(_) => {
                // Fallback: use zeros if embedding extraction fails for this token
                all_embeddings.extend(std::iter::repeat(0.0f32).take(n_embd));
            }
        }
    }

    Ok((all_embeddings, n_tokens))
}

/// Get logits (output probabilities) for the last token
/// Useful for soft-label distillation
/// Returns: Vec<f32> of shape [n_vocab]
#[rustler::nif(schedule = "DirtyCpu")]
fn llm_get_logits(
    resource: ResourceArc<LlmResource>,
    prompt: String,
    ctx_size: u32,
) -> Result<Vec<f32>, rustler::Error> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size));

    let mut ctx = resource.model.new_context(&resource.backend, ctx_params)
        .map_err(|e| rustler::Error::Term(Box::new(format!("context_failed: {}", e))))?;

    let tokens = resource.model.str_to_token(&prompt, AddBos::Always)
        .map_err(|e| rustler::Error::Term(Box::new(format!("tokenize_failed: {}", e))))?;

    ctx.clear_kv_cache();

    let batch_size = tokens.len().max(512);
    let mut batch = LlamaBatch::new(batch_size, 1);

    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(*token, i as i32, &[0], is_last)
            .map_err(|e| rustler::Error::Term(Box::new(format!("batch_add: {}", e))))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| rustler::Error::Term(Box::new(format!("decode_failed: {}", e))))?;

    // Get logits for the last token
    let logits = ctx.get_logits();
    Ok(logits.to_vec())
}

/// Legacy predict function - now returns real embeddings from last token
#[rustler::nif(schedule = "DirtyCpu")]
fn llm_predict(resource: ResourceArc<LlmResource>, prompt: String) -> Result<(String, Vec<f32>), rustler::Error> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_embeddings(true);

    let mut ctx = resource.model.new_context(&resource.backend, ctx_params)
        .map_err(|e| rustler::Error::Term(Box::new(format!("context_failed: {}", e))))?;

    let tokens = resource.model.str_to_token(&prompt, AddBos::Always)
        .map_err(|e| rustler::Error::Term(Box::new(format!("tokenize_failed: {}", e))))?;

    ctx.clear_kv_cache();

    let batch_size = 512;
    let mut batch = LlamaBatch::new(batch_size, 1);

    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(*token, i as i32, &[0], is_last)
            .map_err(|e| rustler::Error::Term(Box::new(format!("batch_add: {}", e))))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| rustler::Error::Term(Box::new(format!("decode_failed: {}", e))))?;

    // Get embedding from last token
    let last_idx = (tokens.len() - 1) as i32;
    let embedding = match ctx.embeddings_ith(last_idx) {
        Ok(emb) => emb.to_vec(),
        Err(_) => vec![0.0; resource.n_embd as usize],  // Fallback
    };

    Ok(("processed".to_string(), embedding))
}

/// Tokenize a string, returning token IDs
#[rustler::nif]
fn llm_tokenize(resource: ResourceArc<LlmResource>, text: String) -> Result<Vec<i32>, rustler::Error> {
    let tokens = resource.model.str_to_token(&text, AddBos::Always)
        .map_err(|e| rustler::Error::Term(Box::new(format!("tokenize_failed: {}", e))))?;

    // Convert LlamaToken to i32
    Ok(tokens.iter().map(|t| t.0).collect())
}

/// Detokenize token IDs back to string
#[rustler::nif]
fn llm_detokenize(resource: ResourceArc<LlmResource>, tokens: Vec<i32>) -> Result<String, rustler::Error> {
    let mut result = String::new();
    for token_id in tokens {
        let token = llama_cpp_2::token::LlamaToken(token_id);
        match resource.model.token_to_str(token, llama_cpp_2::model::Special::Tokenize) {
            Ok(s) => result.push_str(&s),
            Err(_) => result.push_str("<unk>"),
        }
    }
    Ok(result)
}

/// Get system memory status for proprioception
#[rustler::nif]
fn llm_memory_status() -> (u64, u64) {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_memory(MemoryRefreshKind::everything())
    );
    sys.refresh_memory();
    (sys.total_memory(), sys.free_memory())
}

/// Check native capabilities
#[rustler::nif]
fn llm_native_check() -> String {
    #[cfg(target_feature = "avx512f")]
    return "AVX512_ENABLED".to_string();

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    return "AVX2_ENABLED".to_string();

    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    return "SSE_ONLY".to_string();
}

rustler::init!(
    "Elixir.Viva.Llm.Native",
    [
        llm_load_model,
        llm_model_info,
        llm_get_hidden_states,
        llm_get_logits,
        llm_predict,
        llm_tokenize,
        llm_detokenize,
        llm_memory_status,
        llm_native_check
    ],
    load = load
);

fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(LlmResource, env);
    true
}
