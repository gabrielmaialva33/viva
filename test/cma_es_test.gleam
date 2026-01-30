//// CMA-ES Unit Tests
////
//// Tests for Covariance Matrix Adaptation Evolution Strategy implementation.
//// These tests run without GPU to verify Pure Gleam logic.

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{None, Some}
import gleeunit/should
import viva/neural/cma_es

// =============================================================================
// CONFIGURATION TESTS
// =============================================================================

pub fn default_config_test() {
  let config = cma_es.default_config()

  // Check default values
  should.equal(config.initial_sigma, 0.3)
  should.be_none(config.lambda)
  should.be_none(config.mu)
}

pub fn sinuca_config_test() {
  let config = cma_es.sinuca_config()

  // Sinuca should have lambda=50
  should.be_some(config.lambda)
  let assert cma_es.CmaEsConfig(lambda: Some(l), ..) = config
  should.equal(l, 50)
}

pub fn small_config_test() {
  let config = cma_es.small_config()

  // Small config for testing
  should.equal(config.initial_sigma, 0.5)
  should.be_some(config.lambda)
}

// =============================================================================
// QD-CMA-ES CONFIG TESTS
// =============================================================================

pub fn default_qd_config_test() {
  let config = cma_es.default_qd_config()

  should.equal(config.max_stagnation, 10)
  should.equal(config.grid_size, 5)
  should.equal(config.behavior_dims, 2)
  should.equal(config.restart_sigma, 0.3)
}

// =============================================================================
// BEHAVIOR EXTRACTION TESTS
// =============================================================================

pub fn extract_behavior_empty_test() {
  let behavior = cma_es.extract_behavior([], 2)

  // Empty outputs should give zero behavior
  should.equal(list.length(behavior), 2)
  should.equal(list.first(behavior), Ok(0.0))
}

pub fn extract_behavior_single_test() {
  let outputs = [[0.5, 0.3, 0.7]]
  let behavior = cma_es.extract_behavior(outputs, 2)

  // Should take first 2 values
  should.equal(list.length(behavior), 2)
  should.equal(list.first(behavior), Ok(0.5))
}

pub fn extract_behavior_multiple_test() {
  let outputs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
  let behavior = cma_es.extract_behavior(outputs, 4)

  // Should flatten and take first 4
  should.equal(list.length(behavior), 4)
}

pub fn weights_to_behavior_test() {
  let weights = [0.5, -0.3, 0.8, 0.1, 0.0]
  let behavior = cma_es.weights_to_behavior(weights, 3)

  should.equal(list.length(behavior), 3)
  // Values should be clamped to [-1, 1]
}

pub fn weights_to_behavior_padding_test() {
  let weights = [0.5]
  let behavior = cma_es.weights_to_behavior(weights, 3)

  // Should pad with zeros
  should.equal(list.length(behavior), 3)
}

// =============================================================================
// TYPE CONSTRUCTION TESTS
// =============================================================================

pub fn cma_es_diagnostics_test() {
  let diag = cma_es.CmaEsDiagnostics(
    sigma: 0.3,
    condition_number: 1.5,
    normalized_ps_norm: 0.95,
  )

  should.equal(diag.sigma, 0.3)
  should.equal(diag.condition_number, 1.5)
  should.equal(diag.normalized_ps_norm, 0.95)
}

pub fn cma_es_step_result_test() {
  let result = cma_es.CmaEsStepResult(
    population: [[1.0, 2.0], [3.0, 4.0]],
    best_estimate: [2.0, 3.0],
    sigma: 0.25,
  )

  should.equal(list.length(result.population), 2)
  should.equal(result.sigma, 0.25)
}

// =============================================================================
// CELL OPTIMIZER TESTS
// =============================================================================

pub fn cell_optimizer_construction_test() {
  // Just test that the type can be constructed
  // (actual NIF calls would be tested in integration tests)
  let cell_id = #(2, 3)

  should.equal(cell_id.0, 2)
  should.equal(cell_id.1, 3)
}

// =============================================================================
// HYBRID TRAINER CONFIG TESTS
// =============================================================================

pub fn hybrid_trainer_config_test() {
  let qd_config = cma_es.default_qd_config()

  // Verify QD config contains CMA config
  should.equal(qd_config.cma_config.initial_sigma, 0.3)
  should.be_some(qd_config.cma_config.lambda)
}

// =============================================================================
// BENCHMARK STRING TEST
// =============================================================================

pub fn benchmark_function_exists_test() {
  // Just verify the benchmark function is exported
  // Actual call would require NIF to be loaded
  let _ = cma_es.benchmark
  should.be_true(True)
}

// =============================================================================
// INTEGRATION TEST (requires NIF)
// =============================================================================

// These tests would run if NIF is loaded:
// pub fn init_and_sample_test() {
//   let config = cma_es.small_config()
//   let initial = list.repeat(0.0, 10)
//   let state = cma_es.init(initial, config)
//   let pop = cma_es.sample(state, 42)
//   should.be_true(list.length(pop) > 0)
// }
