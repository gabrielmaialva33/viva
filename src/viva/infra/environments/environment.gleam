//// VIVA Environments - Standard RL Environment Interface
////
//// Gym-style interface for benchmarking and comparisons.
//// Compatible with academic standards (OpenAI Gym, Brax).
////
//// Reference:
////   - OpenAI Gym API: https://gymnasium.farama.org/
////   - Brax Environments: https://github.com/google/brax

import gleam/list
import gleam/option.{type Option}

// =============================================================================
// STANDARD ENVIRONMENT INTERFACE
// =============================================================================

/// Environment observation (state)
pub type Observation {
  Observation(
    /// State vector (continuous values)
    values: List(Float),
    /// Dimension of observation space
    dim: Int,
  )
}

/// Environment action
pub type Action {
  /// Discrete action (integer index)
  Discrete(Int)
  /// Continuous action (vector of floats)
  Continuous(List(Float))
}

/// Step result returned by environment
pub type StepResult {
  StepResult(
    /// New observation after action
    observation: Observation,
    /// Reward received
    reward: Float,
    /// Whether episode is done
    done: Bool,
    /// Whether episode was truncated (time limit)
    truncated: Bool,
    /// Additional info (metrics, diagnostics)
    info: EnvInfo,
  )
}

/// Environment metadata and info
pub type EnvInfo {
  EnvInfo(
    /// Current timestep
    timestep: Int,
    /// Total episode return so far
    episode_return: Float,
    /// Custom metrics (key-value pairs)
    metrics: List(#(String, Float)),
  )
}

/// Environment specification
pub type EnvSpec {
  EnvSpec(
    /// Environment name
    name: String,
    /// Observation space dimension
    observation_dim: Int,
    /// Action space dimension (discrete: max_action, continuous: num_dims)
    action_dim: Int,
    /// Is action space discrete?
    discrete_actions: Bool,
    /// Maximum episode length
    max_episode_steps: Int,
    /// Reward range (min, max)
    reward_range: #(Float, Float),
  )
}

// =============================================================================
// ENVIRONMENT TRAIT (Polymorphic Interface)
// =============================================================================

/// Generic environment state holder
pub type Environment(state) {
  Environment(
    /// Internal state
    state: state,
    /// Current observation
    observation: Observation,
    /// Current timestep
    timestep: Int,
    /// Episode return
    episode_return: Float,
    /// Whether episode is done
    done: Bool,
  )
}

/// Environment operations (function pointers for polymorphism)
pub type EnvOps(state) {
  EnvOps(
    /// Reset environment to initial state
    reset: fn(Int) -> Environment(state),
    /// Take a step with given action
    step: fn(Environment(state), Action) -> StepResult,
    /// Get environment specification
    spec: fn() -> EnvSpec,
    /// Clone state for parallel evaluation
    clone: fn(Environment(state)) -> Environment(state),
  )
}

// =============================================================================
// OBSERVATION HELPERS
// =============================================================================

/// Create observation from list of floats
pub fn observation_from_list(values: List(Float)) -> Observation {
  Observation(values: values, dim: list.length(values))
}

/// Get observation as normalized vector (clip to [-1, 1])
pub fn normalize_observation(obs: Observation) -> Observation {
  let normalized = list.map(obs.values, fn(v) {
    case v <. -1.0 {
      True -> -1.0
      False -> case v >. 1.0 {
        True -> 1.0
        False -> v
      }
    }
  })
  Observation(values: normalized, dim: obs.dim)
}

/// Get value at index (safe)
pub fn obs_at(obs: Observation, idx: Int) -> Option(Float) {
  list_at(obs.values, idx)
}

// =============================================================================
// ACTION HELPERS
// =============================================================================

/// Create discrete action
pub fn discrete_action(idx: Int) -> Action {
  Discrete(idx)
}

/// Create continuous action from list
pub fn continuous_action(values: List(Float)) -> Action {
  Continuous(values)
}

/// Get continuous action values (returns empty for discrete)
pub fn action_values(action: Action) -> List(Float) {
  case action {
    Discrete(_) -> []
    Continuous(vals) -> vals
  }
}

/// Get discrete action index (returns 0 for continuous)
pub fn action_index(action: Action) -> Int {
  case action {
    Discrete(idx) -> idx
    Continuous(_) -> 0
  }
}

// =============================================================================
// INFO HELPERS
// =============================================================================

/// Create empty info
pub fn empty_info() -> EnvInfo {
  EnvInfo(timestep: 0, episode_return: 0.0, metrics: [])
}

/// Add metric to info
pub fn add_metric(info: EnvInfo, key: String, value: Float) -> EnvInfo {
  EnvInfo(..info, metrics: [#(key, value), ..info.metrics])
}

/// Get metric from info
pub fn get_metric(info: EnvInfo, key: String) -> Option(Float) {
  list.find_map(info.metrics, fn(kv) {
    let #(k, v) = kv
    case k == key {
      True -> Ok(v)
      False -> Error(Nil)
    }
  })
  |> option.from_result
}

// =============================================================================
// BATCH OPERATIONS (for parallel/GPU evaluation)
// =============================================================================

/// Batch step result
pub type BatchStepResult {
  BatchStepResult(
    observations: List(Observation),
    rewards: List(Float),
    dones: List(Bool),
    infos: List(EnvInfo),
  )
}

/// Convert batch results to individual step results
pub fn unbatch_results(batch: BatchStepResult) -> List(StepResult) {
  let obs = batch.observations
  let rew = batch.rewards
  let done = batch.dones
  let info = batch.infos

  zip4(obs, rew, done, info)
  |> list.map(fn(t) {
    let #(o, r, d, i) = t
    StepResult(
      observation: o,
      reward: r,
      done: d,
      truncated: False,
      info: i,
    )
  })
}

// =============================================================================
// BENCHMARK METRICS
// =============================================================================

/// Standard benchmark metrics (compatible with academic papers)
pub type BenchmarkMetrics {
  BenchmarkMetrics(
    /// Environment name
    env_name: String,
    /// Evaluations per second
    evals_per_sec: Float,
    /// Wall clock time (seconds)
    wall_time: Float,
    /// Steps per second (for physics)
    steps_per_sec: Float,
    /// Sample efficiency (reward per step)
    sample_efficiency: Float,
    /// Final performance (mean episode return)
    final_return: Float,
    /// Standard deviation of returns
    return_std: Float,
    /// Number of evaluations
    num_evals: Int,
    /// Hardware used
    hardware: String,
  )
}

/// Create metrics record
pub fn new_metrics(env_name: String) -> BenchmarkMetrics {
  BenchmarkMetrics(
    env_name: env_name,
    evals_per_sec: 0.0,
    wall_time: 0.0,
    steps_per_sec: 0.0,
    sample_efficiency: 0.0,
    final_return: 0.0,
    return_std: 0.0,
    num_evals: 0,
    hardware: "unknown",
  )
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

fn list_at(lst: List(a), idx: Int) -> Option(a) {
  lst
  |> list.drop(idx)
  |> list.first
  |> option.from_result
}

fn zip4(
  a: List(a),
  b: List(b),
  c: List(c),
  d: List(d),
) -> List(#(a, b, c, d)) {
  case a, b, c, d {
    [a1, ..ar], [b1, ..br], [c1, ..cr], [d1, ..dr] ->
      [#(a1, b1, c1, d1), ..zip4(ar, br, cr, dr)]
    _, _, _, _ -> []
  }
}
