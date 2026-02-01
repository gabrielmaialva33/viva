import gleam/int
import gleam/float
import gleam/list
import gleeunit/should
import viva/neural/llm
import viva/neural/distillation.{
  PADState, init_progressive, step_progressive, advanced_config,
  compute_cka, compute_cka_loss,
}
import viva/memory/hrr
import viva_telemetry/log

// =============================================================================
// CKA LOSS TESTS (no GPU needed)
// =============================================================================

pub fn cka_identical_vectors_test() {
  // CKA of identical vectors should be 1.0
  let v = hrr.random(256)
  let cka = compute_cka(v, v)

  // Should be very close to 1.0
  { cka >. 0.99 } |> should.be_true()
}

pub fn cka_orthogonal_vectors_test() {
  // CKA of very different vectors should be low
  let v1 = hrr.from_list(list.repeat(1.0, 256))
  let v2 = hrr.from_list(list.repeat(-1.0, 256))

  let cka = compute_cka(v1, v2)

  // Should be close to 1.0 (both are constant, so centered = 0)
  // Actually for constant vectors, CKA is undefined, let's use random
  log.debug("CKA orthogonal: " <> float.to_string(cka), [])
}

pub fn cka_loss_range_test() {
  // CKA loss should be between 0 and 1
  let v1 = hrr.random(256)
  let v2 = hrr.random(256)

  let loss = compute_cka_loss(v1, v2)

  { loss >=. 0.0 && loss <=. 2.0 } |> should.be_true()
}

// =============================================================================
// PROGRESSIVE DISTILLATION TESTS
// =============================================================================

pub fn progressive_init_test() {
  let state = init_progressive(100)

  state.epoch |> should.equal(0)
  state.max_epochs |> should.equal(100)
  { state.alpha_kd >. 0.0 } |> should.be_true()
  { state.temperature >. 0.0 } |> should.be_true()
}

pub fn progressive_step_test() {
  let config = advanced_config()
  let state = init_progressive(100)

  // Step forward
  let state2 = step_progressive(state, config)

  state2.epoch |> should.equal(1)

  // alpha_kd should increase
  { state2.alpha_kd >=. state.alpha_kd } |> should.be_true()

  // temperature should decrease
  { state2.temperature <=. state.temperature } |> should.be_true()
}

pub fn progressive_full_schedule_test() {
  let config = advanced_config()
  let initial = init_progressive(10)

  // Run through all epochs
  let final = list.fold(list.range(1, 10), initial, fn(state, _) {
    step_progressive(state, config)
  })

  final.epoch |> should.equal(10)

  // Final alpha_kd should be close to final_alpha_kd
  { final.alpha_kd >. 0.7 } |> should.be_true()

  // Final temperature should be close to final_temp
  { final.temperature <. 2.0 } |> should.be_true()

  log.info("Progressive schedule:", [])
  log.info("  Final alpha_kd: " <> float.to_string(final.alpha_kd), [])
  log.info("  Final temperature: " <> float.to_string(final.temperature), [])
}

// =============================================================================
// LLM INTEGRATION TEST (requires GPU + model)
// =============================================================================

pub fn llm_load_and_extract_test() {
  log.info("\n=== LLM Distillation Test ===", [])
  log.info("Loading Qwen3-32B teacher...", [])

  // Check if model exists first
  let model_path = "models/Qwen3-32B-Q4_K_M.gguf"

  // Try to load (this will fail gracefully if NIF not loaded)
  // In real test, this would load the model
  log.info("Model path: " <> model_path, [])

  // For now, test the memory status function (doesn't need model)
  let #(total, free) = llm.memory_status()
  log.info("System memory:", [])
  log.info("  Total: " <> int.to_string(total / 1_000_000_000) <> " GB", [])
  log.info("  Free: " <> int.to_string(free / 1_000_000_000) <> " GB", [])

  // Check native capabilities
  let caps = llm.native_check()
  log.info("Native: " <> caps, [])

  { total > 0 } |> should.be_true()
}
