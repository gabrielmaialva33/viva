//// CMA-ES - Covariance Matrix Adaptation Evolution Strategy
////
//// GPU-accelerated implementation for neural network weight optimization.
//// Modernizes QD/MAP-Elites with gradient-like adaptation.
////
//// Key Features:
//// - Covariance matrix adaptation (learns search distribution)
//// - Step-size control (automatic learning rate)
//// - Rank-mu and rank-one updates
//// - Integration with MAP-Elites for QD-CMA-ES hybrid
////
//// Architecture target: [8, 32, 16, 3] = 867 weights
//// Uses burn-rs 0.16 with CUDA backend on RTX 4090.
////
//// Reference: Hansen & Ostermeier (2001) "Completely Derandomized
////            Self-Adaptation in Evolution Strategies"
////
//// Created at GATO-PC, Brazil, 2026.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}

// =============================================================================
// TYPES
// =============================================================================

/// Opaque handle to CMA-ES state (Rust NIF resource)
pub type CmaEsState

/// CMA-ES configuration
pub type CmaEsConfig {
  CmaEsConfig(
    /// Initial step size (0.3 recommended for NN weights)
    initial_sigma: Float,
    /// Population size (None = auto = 4 + 3*ln(n))
    lambda: Option(Int),
    /// Number of selected parents (auto = lambda / 2)
    mu: Option(Int),
    /// Learning rate for covariance (None = auto)
    cc: Option(Float),
    /// Rank-one update rate (None = auto)
    c1: Option(Float),
    /// Rank-mu update rate (None = auto)
    cmu: Option(Float),
    /// Sigma learning rate (None = auto)
    csigma: Option(Float),
    /// Sigma damping (None = auto)
    dsigma: Option(Float),
  )
}

/// CMA-ES diagnostics for monitoring convergence
pub type CmaEsDiagnostics {
  CmaEsDiagnostics(
    /// Current step size
    sigma: Float,
    /// Condition number of covariance matrix (high = ill-conditioned)
    condition_number: Float,
    /// Normalized evolution path norm (should be ~1 when adapted)
    normalized_ps_norm: Float,
  )
}

/// Result of CMA-ES optimization step
pub type CmaEsStepResult {
  CmaEsStepResult(
    /// New population to evaluate
    population: List(List(Float)),
    /// Current best estimate (mean)
    best_estimate: List(Float),
    /// Current step size
    sigma: Float,
  )
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Default CMA-ES configuration for neural network optimization
pub fn default_config() -> CmaEsConfig {
  CmaEsConfig(
    initial_sigma: 0.3,
    lambda: None,  // Auto: 4 + 3*ln(n)
    mu: None,      // Auto: lambda / 2
    cc: None,      // Auto
    c1: None,      // Auto
    cmu: None,     // Auto
    csigma: None,  // Auto
    dsigma: None,  // Auto
  )
}

/// Config optimized for sinuca neural network [8, 32, 16, 3] = 867 weights
pub fn sinuca_config() -> CmaEsConfig {
  CmaEsConfig(
    initial_sigma: 0.3,
    lambda: Some(50),   // Match QD population size
    mu: None,           // Auto: 25
    cc: None,
    c1: None,
    cmu: None,
    csigma: None,
    dsigma: None,
  )
}

/// Config for smaller problems (testing)
pub fn small_config() -> CmaEsConfig {
  CmaEsConfig(
    initial_sigma: 0.5,
    lambda: Some(20),
    mu: None,
    cc: None,
    c1: None,
    cmu: None,
    csigma: None,
    dsigma: None,
  )
}

/// Config for large-scale optimization (1000+ dimensions)
pub fn large_scale_config() -> CmaEsConfig {
  CmaEsConfig(
    initial_sigma: 0.2,
    lambda: Some(100),  // Larger population for high dimensions
    mu: None,
    cc: None,
    c1: None,
    cmu: None,
    csigma: None,
    dsigma: None,
  )
}

// =============================================================================
// EXTERNAL NIF FUNCTIONS
// =============================================================================

/// Initialize CMA-ES optimizer with lambda (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_init")
fn cma_es_init_with_lambda_nif(
  initial_mean: List(Float),
  initial_sigma: Float,
  lambda: Int,
) -> CmaEsState

/// Initialize CMA-ES optimizer with auto lambda (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_init_auto")
fn cma_es_init_auto_nif(
  initial_mean: List(Float),
  initial_sigma: Float,
) -> CmaEsState

/// Sample population from CMA-ES (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_sample")
fn cma_es_sample_nif(state: CmaEsState, seed: Int) -> List(List(Float))

/// Update CMA-ES state (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_update")
fn cma_es_update_nif(
  state: CmaEsState,
  population: List(List(Float)),
  fitnesses: List(Float),
) -> Bool

/// Complete CMA-ES step (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_step")
fn cma_es_step_nif(
  state: CmaEsState,
  population: List(List(Float)),
  fitnesses: List(Float),
  seed: Int,
) -> List(List(Float))

/// Get current mean (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_get_mean")
fn cma_es_get_mean_nif(state: CmaEsState) -> List(Float)

/// Get current sigma (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_get_sigma")
fn cma_es_get_sigma_nif(state: CmaEsState) -> Float

/// Get diagnostics (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_get_diagnostics")
fn cma_es_get_diagnostics_nif(state: CmaEsState) -> #(Float, Float, Float)

/// Benchmark CMA-ES (Rust NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_cma_es_benchmark")
pub fn benchmark(n: Int, lambda: Int, iterations: Int) -> String

// =============================================================================
// PUBLIC API
// =============================================================================

/// Initialize CMA-ES optimizer
///
/// Arguments:
/// - initial_mean: Starting point in search space (e.g., random NN weights)
/// - config: CMA-ES configuration
///
/// Returns: CmaEsState handle for subsequent operations
pub fn init(initial_mean: List(Float), config: CmaEsConfig) -> CmaEsState {
  case config.lambda {
    Some(l) -> cma_es_init_with_lambda_nif(initial_mean, config.initial_sigma, l)
    None -> cma_es_init_auto_nif(initial_mean, config.initial_sigma)
  }
}

/// Initialize with Xavier-initialized weights for neural network
pub fn init_xavier(
  architecture: List(Int),
  seed: Int,
  config: CmaEsConfig,
) -> CmaEsState {
  let weights = init_xavier_weights(architecture, seed)
  init(weights, config)
}

/// Sample new population from current distribution
///
/// Arguments:
/// - state: CMA-ES state handle
/// - seed: Random seed for reproducibility
///
/// Returns: List of candidate solutions to evaluate
pub fn sample(state: CmaEsState, seed: Int) -> List(List(Float)) {
  cma_es_sample_nif(state, seed)
}

/// Update CMA-ES state with fitness evaluations
///
/// Arguments:
/// - state: CMA-ES state handle
/// - population: Candidate solutions from sample()
/// - fitnesses: Fitness values (higher = better) for each solution
///
/// Note: This modifies the state in-place (Rust side)
pub fn update(
  state: CmaEsState,
  population: List(List(Float)),
  fitnesses: List(Float),
) -> Nil {
  let _ = cma_es_update_nif(state, population, fitnesses)
  Nil
}

/// Complete CMA-ES step: update with previous results, then sample new population
///
/// Arguments:
/// - state: CMA-ES state handle
/// - population: Previous population (empty for first call)
/// - fitnesses: Previous fitnesses (empty for first call)
/// - seed: Random seed
///
/// Returns: New population to evaluate
pub fn step(
  state: CmaEsState,
  population: List(List(Float)),
  fitnesses: List(Float),
  seed: Int,
) -> List(List(Float)) {
  cma_es_step_nif(state, population, fitnesses, seed)
}

/// Get current best estimate (mean of distribution)
pub fn get_mean(state: CmaEsState) -> List(Float) {
  cma_es_get_mean_nif(state)
}

/// Get current step size (sigma)
pub fn get_sigma(state: CmaEsState) -> Float {
  cma_es_get_sigma_nif(state)
}

/// Get convergence diagnostics
pub fn get_diagnostics(state: CmaEsState) -> CmaEsDiagnostics {
  let #(sigma, cond, ps_norm) = cma_es_get_diagnostics_nif(state)
  CmaEsDiagnostics(
    sigma: sigma,
    condition_number: cond,
    normalized_ps_norm: ps_norm,
  )
}

/// Check if CMA-ES has converged
///
/// Convergence criteria:
/// - Sigma < threshold (search collapsed)
/// - Condition number > threshold (ill-conditioned)
pub fn is_converged(
  state: CmaEsState,
  sigma_threshold: Float,
  condition_threshold: Float,
) -> Bool {
  let diag = get_diagnostics(state)
  diag.sigma <. sigma_threshold || diag.condition_number >. condition_threshold
}

// =============================================================================
// QD-CMA-ES INTEGRATION
// =============================================================================

/// CMA-ES optimizer for a single MAP-Elites cell
pub type CellOptimizer {
  CellOptimizer(
    cell_id: #(Int, Int),
    state: CmaEsState,
    best_fitness: Float,
    stagnation: Int,
  )
}

/// QD-CMA-ES configuration
pub type QdCmaEsConfig {
  QdCmaEsConfig(
    /// CMA-ES config for each cell
    cma_config: CmaEsConfig,
    /// Maximum stagnation before restart
    max_stagnation: Int,
    /// Grid size for MAP-Elites
    grid_size: Int,
    /// Number of behavior dimensions
    behavior_dims: Int,
    /// Sigma reset value on restart
    restart_sigma: Float,
  )
}

/// Default QD-CMA-ES config
pub fn default_qd_config() -> QdCmaEsConfig {
  QdCmaEsConfig(
    cma_config: sinuca_config(),
    max_stagnation: 10,
    grid_size: 5,
    behavior_dims: 2,
    restart_sigma: 0.3,
  )
}

/// Initialize cell optimizer with elite solution
pub fn init_cell_optimizer(
  cell_id: #(Int, Int),
  elite_weights: List(Float),
  config: CmaEsConfig,
) -> CellOptimizer {
  let state = init(elite_weights, config)
  CellOptimizer(
    cell_id: cell_id,
    state: state,
    best_fitness: 0.0,
    stagnation: 0,
  )
}

/// Update cell optimizer with new evaluation
pub fn update_cell_optimizer(
  optimizer: CellOptimizer,
  population: List(List(Float)),
  fitnesses: List(Float),
  config: QdCmaEsConfig,
) -> CellOptimizer {
  // Find best fitness in this batch
  let batch_best = list.fold(fitnesses, 0.0, float.max)

  // Update CMA-ES state
  update(optimizer.state, population, fitnesses)

  // Check for improvement
  case batch_best >. optimizer.best_fitness {
    True -> {
      // Improvement - reset stagnation
      CellOptimizer(
        ..optimizer,
        best_fitness: batch_best,
        stagnation: 0,
      )
    }
    False -> {
      // No improvement - increment stagnation
      let new_stagnation = optimizer.stagnation + 1

      // Check if restart needed
      case new_stagnation >= config.max_stagnation {
        True -> {
          // Restart CMA-ES from current mean with fresh sigma
          let current_mean = get_mean(optimizer.state)
          let fresh_config = CmaEsConfig(
            ..config.cma_config,
            initial_sigma: config.restart_sigma,
          )
          let new_state = init(current_mean, fresh_config)
          CellOptimizer(
            ..optimizer,
            state: new_state,
            stagnation: 0,
          )
        }
        False -> {
          CellOptimizer(..optimizer, stagnation: new_stagnation)
        }
      }
    }
  }
}

// =============================================================================
// BEHAVIOR EXTRACTION FOR QD
// =============================================================================

/// Extract behavior descriptor from neural network output trajectory
///
/// Arguments:
/// - outputs: List of network outputs over episode
/// - n_dims: Number of behavior dimensions to extract
///
/// Returns: Normalized behavior vector
pub fn extract_behavior(outputs: List(List(Float)), n_dims: Int) -> List(Float) {
  case outputs {
    [] -> list.repeat(0.0, n_dims)
    _ -> {
      // Flatten all outputs
      let flat = list.flatten(outputs)

      // Take first n_dims as behavior (or average if more outputs)
      let len = list.length(flat)
      case len >= n_dims {
        True -> list.take(flat, n_dims)
        False -> {
          // Pad with zeros
          list.append(flat, list.repeat(0.0, n_dims - len))
        }
      }
    }
  }
}

/// Compute behavior from neural network weights (phenotypic characterization)
///
/// Uses first few weight components as rough behavior descriptor.
/// For more accurate behavior, evaluate network on test inputs.
pub fn weights_to_behavior(weights: List(Float), n_dims: Int) -> List(Float) {
  // Simple approach: use first n_dims weights scaled to [-1, 1]
  weights
  |> list.take(n_dims)
  |> list.map(fn(w) { float_clamp(w /. 2.0, -1.0, 1.0) })
  |> fn(taken) {
    let len = list.length(taken)
    case len < n_dims {
      True -> list.append(taken, list.repeat(0.0, n_dims - len))
      False -> taken
    }
  }
}

// =============================================================================
// HYBRID CMA-ES + MAP-ELITES
// =============================================================================

/// Hybrid QD-CMA-ES trainer state
pub type HybridTrainer {
  HybridTrainer(
    /// Global CMA-ES for exploration
    global_cma: CmaEsState,
    /// Cell-specific optimizers for exploitation
    cell_optimizers: List(CellOptimizer),
    /// Current generation
    generation: Int,
    /// Best global fitness
    best_fitness: Float,
    /// Configuration
    config: QdCmaEsConfig,
  )
}

/// Initialize hybrid trainer
pub fn init_hybrid_trainer(
  initial_weights: List(Float),
  config: QdCmaEsConfig,
) -> HybridTrainer {
  let global = init(initial_weights, config.cma_config)

  HybridTrainer(
    global_cma: global,
    cell_optimizers: [],
    generation: 0,
    best_fitness: 0.0,
    config: config,
  )
}

/// Generate population from hybrid system
///
/// Mix of:
/// - Global CMA-ES samples (exploration)
/// - Cell CMA-ES samples (exploitation)
pub fn hybrid_sample(
  trainer: HybridTrainer,
  exploration_ratio: Float,
  seed: Int,
) -> List(List(Float)) {
  let global_pop = sample(trainer.global_cma, seed)
  let global_count = list.length(global_pop)

  // Calculate how many from global vs cells
  let from_global = float.round(
    int.to_float(global_count) *. exploration_ratio
  )

  let global_samples = list.take(global_pop, from_global)

  // Sample from cell optimizers
  let cell_samples = trainer.cell_optimizers
    |> list.index_map(fn(opt, idx) {
      sample(opt.state, seed + idx * 1000)
    })
    |> list.flatten
    |> list.take(global_count - from_global)

  list.append(global_samples, cell_samples)
}

/// Update hybrid trainer with evaluations
pub fn hybrid_update(
  trainer: HybridTrainer,
  population: List(List(Float)),
  fitnesses: List(Float),
  behaviors: List(List(Float)),
  seed: Int,
) -> HybridTrainer {
  // Update global CMA-ES
  update(trainer.global_cma, population, fitnesses)

  // Find best fitness
  let batch_best = list.fold(fitnesses, 0.0, float.max)
  let new_best = float.max(trainer.best_fitness, batch_best)

  // Update cell optimizers based on which cells solutions landed in
  // (simplified - in full implementation, route solutions to cells)

  HybridTrainer(
    ..trainer,
    generation: trainer.generation + 1,
    best_fitness: new_best,
  )
}

// =============================================================================
// UTILITIES
// =============================================================================

fn float_clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False -> case x >. max {
      True -> max
      False -> x
    }
  }
}

/// Initialize Xavier weights for neural network
fn init_xavier_weights(architecture: List(Int), seed: Int) -> List(Float) {
  let weight_count = calculate_weight_count(architecture)
  generate_xavier_weights(weight_count, seed, architecture)
}

fn calculate_weight_count(arch: List(Int)) -> Int {
  case arch {
    [] -> 0
    [_] -> 0
    [in_size, out_size, ..rest] -> {
      let layer_weights = in_size * out_size + out_size
      layer_weights + calculate_weight_count([out_size, ..rest])
    }
  }
}

fn generate_xavier_weights(
  count: Int,
  seed: Int,
  architecture: List(Int),
) -> List(Float) {
  let fan_in = case list.first(architecture) {
    Ok(n) -> n
    Error(_) -> 8
  }
  let scale = float_sqrt(2.0 /. int.to_float(fan_in))

  generate_weights_loop(count, seed, scale, [])
}

fn generate_weights_loop(
  remaining: Int,
  seed: Int,
  scale: Float,
  acc: List(Float),
) -> List(Float) {
  case remaining <= 0 {
    True -> list.reverse(acc)
    False -> {
      let next_seed = { seed * 1103515245 + 12345 } % 2147483648
      let value = { int.to_float(next_seed % 2000 - 1000) /. 1000.0 } *. scale
      generate_weights_loop(remaining - 1, next_seed, scale, [value, ..acc])
    }
  }
}

fn float_sqrt(x: Float) -> Float {
  case x <=. 0.0 {
    True -> 0.0
    False -> do_sqrt(x, x /. 2.0, 0)
  }
}

fn do_sqrt(x: Float, guess: Float, iterations: Int) -> Float {
  case iterations > 20 {
    True -> guess
    False -> {
      let new_guess = { guess +. x /. guess } /. 2.0
      let diff = float_abs(new_guess -. guess)
      case diff <. 0.0001 {
        True -> new_guess
        False -> do_sqrt(x, new_guess, iterations + 1)
      }
    }
  }
}

fn float_abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

// =============================================================================
// LOGGING AND MONITORING
// =============================================================================

/// Log CMA-ES state for debugging
pub fn log_state(state: CmaEsState, label: String) -> Nil {
  let diag = get_diagnostics(state)
  let sigma_str = float_str(diag.sigma)
  let cond_str = float_str(diag.condition_number)
  let ps_str = float_str(diag.normalized_ps_norm)

  io.println(
    label <> " | sigma: " <> sigma_str
    <> " | cond: " <> cond_str
    <> " | ps_norm: " <> ps_str
  )
}

fn float_str(x: Float) -> String {
  let scaled = float.truncate(x *. 1000.0)
  let whole = scaled / 1000
  let frac = int.absolute_value(scaled % 1000)
  let frac_str = case frac < 100 {
    True -> case frac < 10 {
      True -> "00" <> int.to_string(frac)
      False -> "0" <> int.to_string(frac)
    }
    False -> int.to_string(frac)
  }
  int.to_string(whole) <> "." <> frac_str
}

// =============================================================================
// TESTS
// =============================================================================

/// Simple test function
pub fn test_basic() -> Bool {
  // Initialize with small dimension for testing
  let initial = list.repeat(0.0, 10)
  let config = small_config()
  let state = init(initial, config)

  // Sample population
  let pop = sample(state, 42)
  let pop_size = list.length(pop)

  // Create fake fitnesses
  let fitnesses = list.map(pop, fn(x) {
    // Sphere function: sum of squares (minimize, so negate)
    let sum = list.fold(x, 0.0, fn(acc, xi) { acc +. xi *. xi })
    0.0 -. sum
  })

  // Update
  update(state, pop, fitnesses)

  // Check state
  let sigma = get_sigma(state)

  pop_size > 0 && sigma >. 0.0
}
