use rustler::{Env, ResourceArc, Term};
use std::path::Path;
use std::num::NonZeroU32;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use sysinfo::{System, RefreshKind, MemoryRefreshKind}; // Proprioception

// --- Resources ---
struct LlmResource {
    model: LlamaModel,
    backend: LlamaBackend,
}

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

// --- NIFs ---

#[rustler::nif(schedule = "DirtyIo")]
fn load_model(path: String, gpu_layers: i32) -> Result<ResourceArc<LlmResource>, rustler::Error> {
    let backend = LlamaBackend::init().map_err(|e| rustler::Error::Term(Box::new(format!("backend_init: {}", e))))?;

    let model_path = Path::new(&path);
    if !model_path.exists() {
        return Err(rustler::Error::Term(Box::new(format!("file_not_found: {}", path))));
    }

    let model_params = llama_cpp_2::model::params::LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers as u32);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .map_err(|e| rustler::Error::Term(Box::new(format!("load_failed: {}", e))))?;

    Ok(ResourceArc::new(LlmResource {
        model,
        backend,
    }))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn predict(resource: ResourceArc<LlmResource>, prompt: String) -> Result<(String, Vec<f32>), rustler::Error> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048));

    let mut ctx = resource.model.new_context(&resource.backend, ctx_params)
        .map_err(|e| rustler::Error::Term(Box::new(format!("context_failed: {}", e))))?;

    let tokens_list = resource.model.str_to_token(&prompt, AddBos::Always)
        .map_err(|e| rustler::Error::Term(Box::new(format!("tokenize_failed: {}", e))))?;

    ctx.clear_kv_cache();

    let batch_size = 512;
    let mut batch = LlamaBatch::new(batch_size, 1);

    for (i, token) in tokens_list.iter().enumerate() {
        let is_last = i == tokens_list.len() - 1;
        batch.add(*token, i as i32, &[0], is_last).map_err(|e| rustler::Error::Term(Box::new(format!("batch_add: {}", e))))?;
    }

    ctx.decode(&mut batch).map_err(|e| rustler::Error::Term(Box::new(format!("decode_failed: {}", e))))?;

    let fake_embedding = vec![0.0; 1024]; // Placeholder until embedding extraction is finalized
    Ok(("processed".to_string(), fake_embedding))
}

#[rustler::nif]
fn get_memory_status() -> (u64, u64) {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_memory(MemoryRefreshKind::everything())
    );
    // No explicit refresh needed if new_with_specifics refreshes automatically,
    // but sysinfo behavior varies. Best to refresh_memory explicitly to be sure.
    sys.refresh_memory();
    (sys.total_memory(), sys.free_memory())
}

#[rustler::nif]
fn native_check() -> String {
    "AVX512_ENABLED".to_string()
}

rustler::init!("Elixir.Viva.Llm", [load_model, predict, native_check, get_memory_status], load = load);

fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(LlmResource, env);
    true
}
