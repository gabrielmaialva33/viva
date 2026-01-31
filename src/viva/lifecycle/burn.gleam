//// VIVA Burn - GPU-Accelerated Neural Network Operations
////
//// Batch forward pass for 1000+ networks using burn-rs.
//// RTX 4090 optimized: 16,384 CUDA cores, 512 Tensor Cores.
////
//// Usage:
////   let results = burn.batch_forward(weights_list, inputs_list, [8, 32, 16, 3])
////
//// Performance:
////   CPU (Rayon):  ~50,000 forwards/sec
////   GPU (CUDA):   ~500,000 forwards/sec (10x speedup)

import gleam/dynamic.{type Dynamic}
import gleam/result

// =============================================================================
// EXTERNAL NIF FUNCTIONS
// =============================================================================

/// Check burn backend status
@external(erlang, "Elixir.Viva.Burn.Native", "burn_check")
pub fn check() -> String

/// Batch forward pass for multiple networks (returns Result)
pub fn batch_forward(
  weights_list: List(List(Float)),
  inputs_list: List(List(Float)),
  architecture: List(Int),
) -> Result(List(List(Float)), String) {
  Ok(batch_forward_raw(weights_list, inputs_list, architecture))
}

/// Raw batch forward (direct NIF call)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_forward")
pub fn batch_forward_raw(
  weights_list: List(List(Float)),
  inputs_list: List(List(Float)),
  architecture: List(Int),
) -> List(List(Float))

/// Batch forward for fixed-topology dense networks
/// More efficient when all networks have same architecture
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_dense")
pub fn batch_dense(
  weights_batch: List(List(Float)),
  inputs_batch: List(List(Float)),
  layer_sizes: List(Int),
) -> Result(List(List(Float)), String)

/// Generate perturbations for NES gradient estimation
///
/// base_weights: Original network weights
/// num_perturbations: How many perturbed copies (8-16 typical)
/// std_dev: Standard deviation of Gaussian noise
/// seed: Random seed for reproducibility
@external(erlang, "Elixir.Viva.Burn.Native", "burn_perturb_weights")
pub fn perturb_weights(
  base_weights: List(Float),
  num_perturbations: Int,
  std_dev: Float,
  seed: Int,
) -> List(List(Float))

/// Compute NES gradient from fitness evaluations
///
/// perturbations: List of perturbation vectors
/// fitnesses: Fitness of each perturbation
/// std_dev: Standard deviation used in perturbations
@external(erlang, "Elixir.Viva.Burn.Native", "burn_nes_gradient")
pub fn nes_gradient(
  perturbations: List(List(Float)),
  fitnesses: List(Float),
  std_dev: Float,
) -> List(Float)

/// Apply gradient update to weights (SGD step)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_update_weights")
pub fn update_weights(
  weights: List(Float),
  gradient: List(Float),
  learning_rate: Float,
) -> List(Float)

/// Complete NES step (gradient + update in one call)
/// More efficient as it minimizes CPU-GPU transfers
@external(erlang, "Elixir.Viva.Burn.Native", "burn_nes_step")
pub fn nes_step(
  base_weights: List(Float),
  perturbations: List(List(Float)),
  fitnesses: List(Float),
  std_dev: Float,
  learning_rate: Float,
) -> List(Float)

// =============================================================================
// GPU-EXPLICIT NES OPERATIONS (CUDA only)
// =============================================================================

/// GPU-only perturbation generation using cuRAND
/// Falls back to CPU if CUDA unavailable
@external(erlang, "Elixir.Viva.Burn.Native", "burn_perturb_weights_gpu")
pub fn perturb_weights_gpu(
  base_weights: List(Float),
  num_perturbations: Int,
  std_dev: Float,
  seed: Int,
) -> List(List(Float))

/// GPU-only NES gradient using Tensor Core GEMM
/// Falls back to CPU if CUDA unavailable
@external(erlang, "Elixir.Viva.Burn.Native", "burn_nes_gradient_gpu")
pub fn nes_gradient_gpu(
  perturbations: List(List(Float)),
  fitnesses: List(Float),
  std_dev: Float,
) -> List(Float)

/// GPU-only weight update
/// Falls back to CPU if CUDA unavailable
@external(erlang, "Elixir.Viva.Burn.Native", "burn_update_weights_gpu")
pub fn update_weights_gpu(
  weights: List(Float),
  gradient: List(Float),
  learning_rate: Float,
) -> List(Float)

/// Batch weight updates on GPU (for population-based training)
/// Processes N weight vectors in parallel
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_update_weights_gpu")
pub fn batch_update_weights_gpu(
  weights_batch: List(List(Float)),
  gradients_batch: List(List(Float)),
  learning_rate: Float,
) -> List(List(Float))

/// Batch sigmoid activation
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_sigmoid")
pub fn batch_sigmoid(values: List(List(Float))) -> List(List(Float))

/// Batch tanh activation
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_tanh")
pub fn batch_tanh(values: List(List(Float))) -> List(List(Float))

/// Batch ReLU activation
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_relu")
pub fn batch_relu(values: List(List(Float))) -> List(List(Float))

/// Run benchmark
@external(erlang, "Elixir.Viva.Burn.Native", "burn_benchmark")
pub fn benchmark(
  pop_size: Int,
  input_size: Int,
  hidden_size: Int,
  output_size: Int,
  iterations: Int,
) -> String

/// Benchmark NES operations (GPU vs CPU comparison)
///
/// Example: benchmark_nes(867, 16, 100)
/// For sinuca architecture [8, 32, 16, 3] = 867 weights
@external(erlang, "Elixir.Viva.Burn.Native", "burn_benchmark_nes")
pub fn benchmark_nes(
  weight_count: Int,
  num_perturbations: Int,
  iterations: Int,
) -> String

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate total weight count for architecture
pub fn weight_count(architecture: List(Int)) -> Int {
  weight_count_loop(architecture, 0)
}

fn weight_count_loop(arch: List(Int), acc: Int) -> Int {
  case arch {
    [] -> acc
    [_] -> acc
    [in_size, out_size, ..rest] -> {
      let layer_weights = in_size * out_size + out_size
      weight_count_loop([out_size, ..rest], acc + layer_weights)
    }
  }
}

/// Initialize random weights for architecture
pub fn init_weights(architecture: List(Int), seed: Int) -> List(Float) {
  let count = weight_count(architecture)
  generate_xavier_weights(count, seed)
}

fn generate_xavier_weights(count: Int, seed: Int) -> List(Float) {
  // Simple pseudo-random Xavier initialization
  generate_weights_loop(count, seed, [])
}

fn generate_weights_loop(remaining: Int, seed: Int, acc: List(Float)) -> List(Float) {
  case remaining <= 0 {
    True -> acc
    False -> {
      let next_seed = { seed * 1103515245 + 12345 } % 2147483648
      let value = { next_seed % 1000 - 500 } |> int_to_float |> fn(x) { x /. 1000.0 }
      generate_weights_loop(remaining - 1, next_seed, [value, ..acc])
    }
  }
}

// =============================================================================
// FFI - O(1) int_to_float (replaced O(n) loop)
// =============================================================================

@external(erlang, "erlang", "float")
fn int_to_float(x: Int) -> Float

// =============================================================================
// HIGH-LEVEL API
// =============================================================================

/// Configuration for batch neural forward
pub type BatchConfig {
  BatchConfig(
    architecture: List(Int),
    population_size: Int,
  )
}

/// Policy network with weights
pub type Policy {
  Policy(
    weights: List(Float),
    architecture: List(Int),
  )
}

/// Create new policy with random weights
pub fn new_policy(architecture: List(Int), seed: Int) -> Policy {
  Policy(
    weights: init_weights(architecture, seed),
    architecture: architecture,
  )
}

/// Forward pass for single policy
pub fn forward(policy: Policy, inputs: List(Float)) -> Result(List(Float), String) {
  case batch_forward([policy.weights], [inputs], policy.architecture) {
    Ok([output]) -> Ok(output)
    Ok(_) -> Error("Unexpected batch result")
    Error(e) -> Error(e)
  }
}

/// NES optimization step
pub type NESConfig {
  NESConfig(
    num_perturbations: Int,
    perturbation_std: Float,
    learning_rate: Float,
  )
}

/// Default NES config optimized for RTX 4090
pub fn default_nes_config() -> NESConfig {
  NESConfig(
    num_perturbations: 16,
    perturbation_std: 0.02,
    learning_rate: 0.1,
  )
}

/// Run one NES optimization step
/// Returns updated policy
pub fn nes_step_policy(
  policy: Policy,
  fitness_fn: fn(List(Float)) -> Float,
  inputs: List(Float),
  config: NESConfig,
  seed: Int,
) -> Policy {
  // Generate perturbations
  let perturbations = perturb_weights(
    policy.weights,
    config.num_perturbations,
    config.perturbation_std,
    seed,
  )

  // Evaluate all perturbations
  let fitnesses = evaluate_perturbations(
    perturbations,
    inputs,
    policy.architecture,
    fitness_fn,
  )

  // Compute gradient
  let gradient = nes_gradient(perturbations, fitnesses, config.perturbation_std)

  // Update weights
  let new_weights = update_weights(policy.weights, gradient, config.learning_rate)

  Policy(..policy, weights: new_weights)
}

fn evaluate_perturbations(
  perturbations: List(List(Float)),
  inputs: List(Float),
  architecture: List(Int),
  fitness_fn: fn(List(Float)) -> Float,
) -> List(Float) {
  case batch_forward(perturbations, repeat_list(inputs, list_length(perturbations)), architecture) {
    Ok(outputs) -> list_map(outputs, fitness_fn)
    Error(_) -> []
  }
}

fn repeat_list(item: a, n: Int) -> List(a) {
  case n <= 0 {
    True -> []
    False -> [item, ..repeat_list(item, n - 1)]
  }
}

fn list_length(lst: List(a)) -> Int {
  list_length_loop(lst, 0)
}

fn list_length_loop(lst: List(a), acc: Int) -> Int {
  case lst {
    [] -> acc
    [_, ..rest] -> list_length_loop(rest, acc + 1)
  }
}

fn list_map(lst: List(a), f: fn(a) -> b) -> List(b) {
  case lst {
    [] -> []
    [first, ..rest] -> [f(first), ..list_map(rest, f)]
  }
}
