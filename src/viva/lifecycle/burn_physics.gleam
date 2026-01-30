//// VIVA Burn Physics - GPU-Accelerated Batch Billiards Simulation
////
//// Batch physics simulation for sinuca tables using viva_burn.
//// Processes 1000+ simultaneous table simulations.
////
//// Performance target: 100-1000x speedup over sequential Jolt
////
//// Usage:
////   let batch = burn_physics.create_batch(4800)
////   let results = burn_physics.simulate_batch(batch, shots, 200)
////   let fitness = burn_physics.calculate_fitness(batch, results)

import gleam/int
import gleam/list
import gleam/result

// =============================================================================
// TYPES
// =============================================================================

/// Batch state for multiple sinuca tables
pub type BatchState {
  BatchState(
    /// Ball X positions [batch, 8]
    positions_x: List(List(Float)),
    /// Ball Z positions [batch, 8]
    positions_z: List(List(Float)),
    /// Ball X velocities [batch, 8]
    velocities_x: List(List(Float)),
    /// Ball Z velocities [batch, 8]
    velocities_z: List(List(Float)),
    /// Pocketed flags [batch, 8] (0.0 or 1.0)
    pocketed: List(List(Float)),
    /// Batch size
    batch_size: Int,
  )
}

/// Shot parameters
pub type BatchShot {
  BatchShot(
    /// Angle in radians (0 = +X direction)
    angle: Float,
    /// Power 0.0 to 1.0
    power: Float,
    /// English (side spin) -1.0 to 1.0
    english: Float,
    /// Elevation 0.0 to 1.0
    elevation: Float,
  )
}

/// Simulation result for a single table
pub type SimResult {
  SimResult(
    /// Final ball X positions [8]
    positions_x: List(Float),
    /// Final ball Z positions [8]
    positions_z: List(Float),
    /// Final pocketed flags [8]
    pocketed: List(Float),
    /// Number of steps taken
    steps_taken: Int,
  )
}

/// Fitness result with behavior descriptors
pub type FitnessResult {
  FitnessResult(
    /// Fitness score
    fitness: Float,
    /// Hit angle (0-1) for behavior descriptor
    hit_angle: Float,
    /// Scatter ratio (0-1) for behavior descriptor
    scatter_ratio: Float,
  )
}

// =============================================================================
// EXTERNAL NIF FUNCTIONS
// =============================================================================

/// Create initial batch of sinuca tables
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_physics_create_tables")
fn native_create_tables(batch_size: Int) -> #(
  List(List(Float)),
  List(List(Float)),
  List(List(Float)),
  List(List(Float)),
  List(List(Float)),
)

/// Run batch physics simulation (returns Ok tuple directly from NIF)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_physics_simulate")
fn native_simulate(
  positions_x: List(List(Float)),
  positions_z: List(List(Float)),
  velocities_x: List(List(Float)),
  velocities_z: List(List(Float)),
  pocketed: List(List(Float)),
  shots: List(List(Float)),
  max_steps: Int,
) -> #(List(List(Float)), List(List(Float)), List(List(Float)), List(Int))

/// Run batch physics simulation WITH SPIN PHYSICS
///
/// This version uses english and elevation to apply realistic spin:
/// - English (side spin): causes curved ball paths via Magnus effect
/// - Elevation (top/back spin): affects ball behavior after contact
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_physics_simulate_with_spin")
fn native_simulate_with_spin(
  positions_x: List(List(Float)),
  positions_z: List(List(Float)),
  velocities_x: List(List(Float)),
  velocities_z: List(List(Float)),
  pocketed: List(List(Float)),
  shots: List(List(Float)),
  max_steps: Int,
) -> #(List(List(Float)), List(List(Float)), List(List(Float)), List(Int))

/// Calculate fitness for batch results
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_physics_calculate_fitness")
fn native_calculate_fitness(
  batch_size: Int,
  initial_pocketed: List(List(Float)),
  final_pocketed: List(List(Float)),
  final_pos_x: List(List(Float)),
  final_pos_z: List(List(Float)),
  initial_pos_x: List(List(Float)),
  initial_pos_z: List(List(Float)),
  target_ball_idx: Int,
) -> List(#(Float, Float, Float))

/// Benchmark batch physics
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_physics_benchmark")
pub fn benchmark(
  batch_size: Int,
  max_steps: Int,
  iterations: Int,
) -> String

// =============================================================================
// PUBLIC API
// =============================================================================

/// Create a batch of sinuca tables in initial configuration
pub fn create_batch(batch_size: Int) -> BatchState {
  let #(px, pz, vx, vz, pocketed) = native_create_tables(batch_size)
  BatchState(
    positions_x: px,
    positions_z: pz,
    velocities_x: vx,
    velocities_z: vz,
    pocketed: pocketed,
    batch_size: batch_size,
  )
}

/// Run batch simulation with given shots
///
/// Returns final state after simulation
pub fn simulate_batch(
  initial: BatchState,
  shots: List(BatchShot),
  max_steps: Int,
) -> Result(BatchState, String) {
  // Convert shots to list format
  let shots_list = list.map(shots, fn(shot) {
    [shot.angle, shot.power, shot.english, shot.elevation]
  })

  let #(final_px, final_pz, final_pocketed, _steps) = native_simulate(
    initial.positions_x,
    initial.positions_z,
    initial.velocities_x,
    initial.velocities_z,
    initial.pocketed,
    shots_list,
    max_steps,
  )

  Ok(BatchState(
    positions_x: final_px,
    positions_z: final_pz,
    velocities_x: list.map(final_px, fn(_) { list.repeat(0.0, 8) }),
    velocities_z: list.map(final_pz, fn(_) { list.repeat(0.0, 8) }),
    pocketed: final_pocketed,
    batch_size: initial.batch_size,
  ))
}

/// Run batch simulation with SPIN PHYSICS
///
/// This version uses english and elevation parameters to apply realistic spin:
/// - English causes curved ball paths via the Magnus effect
/// - Elevation affects top/back spin for draw and follow shots
///
/// Returns final state after simulation
pub fn simulate_batch_with_spin(
  initial: BatchState,
  shots: List(BatchShot),
  max_steps: Int,
) -> Result(BatchState, String) {
  // Convert shots to list format
  let shots_list = list.map(shots, fn(shot) {
    [shot.angle, shot.power, shot.english, shot.elevation]
  })

  let #(final_px, final_pz, final_pocketed, _steps) = native_simulate_with_spin(
    initial.positions_x,
    initial.positions_z,
    initial.velocities_x,
    initial.velocities_z,
    initial.pocketed,
    shots_list,
    max_steps,
  )

  Ok(BatchState(
    positions_x: final_px,
    positions_z: final_pz,
    velocities_x: list.map(final_px, fn(_) { list.repeat(0.0, 8) }),
    velocities_z: list.map(final_pz, fn(_) { list.repeat(0.0, 8) }),
    pocketed: final_pocketed,
    batch_size: initial.batch_size,
  ))
}

/// Simulate and return full results including steps taken
pub fn simulate_batch_full(
  initial: BatchState,
  shots: List(BatchShot),
  max_steps: Int,
) -> Result(List(SimResult), String) {
  let shots_list = list.map(shots, fn(shot) {
    [shot.angle, shot.power, shot.english, shot.elevation]
  })

  let #(final_px, final_pz, final_pocketed, steps) = native_simulate(
    initial.positions_x,
    initial.positions_z,
    initial.velocities_x,
    initial.velocities_z,
    initial.pocketed,
    shots_list,
    max_steps,
  )

  let results = zip4(final_px, final_pz, final_pocketed, steps)
    |> list.map(fn(tuple) {
      let #(px, pz, pck, s) = tuple
      SimResult(
        positions_x: px,
        positions_z: pz,
        pocketed: pck,
        steps_taken: s,
      )
    })
  Ok(results)
}

/// Calculate fitness for all simulations in batch
pub fn calculate_fitness(
  initial: BatchState,
  final_state: BatchState,
  target_ball_idx: Int,
) -> List(FitnessResult) {
  let results = native_calculate_fitness(
    initial.batch_size,
    initial.pocketed,
    final_state.pocketed,
    final_state.positions_x,
    final_state.positions_z,
    initial.positions_x,
    initial.positions_z,
    target_ball_idx,
  )

  list.map(results, fn(r) {
    let #(fitness, hit_angle, scatter) = r
    FitnessResult(
      fitness: fitness,
      hit_angle: hit_angle,
      scatter_ratio: scatter,
    )
  })
}

/// Complete evaluation: simulate + calculate fitness
///
/// This is the main API for QD training.
pub fn evaluate_batch(
  shots: List(BatchShot),
  target_ball_idx: Int,
  max_steps: Int,
) -> Result(List(FitnessResult), String) {
  let batch_size = list.length(shots)
  let initial = create_batch(batch_size)

  case simulate_batch(initial, shots, max_steps) {
    Ok(final_state) -> {
      let fitness_results = calculate_fitness(initial, final_state, target_ball_idx)
      Ok(fitness_results)
    }
    Error(e) -> Error(e)
  }
}

/// Evaluate batch with neural network outputs
///
/// Converts network outputs [angle_adj, power, english] to shots
/// using pocket_angle as base (curriculum learning)
pub fn evaluate_batch_with_outputs(
  outputs: List(List(Float)),
  pocket_angles: List(Float),
  target_ball_idx: Int,
  max_steps: Int,
) -> Result(List(FitnessResult), String) {
  // Convert outputs to shots
  let shots = list.zip(outputs, pocket_angles)
    |> list.map(fn(pair) {
      let #(out, pocket_angle) = pair
      decode_shot(out, pocket_angle)
    })

  evaluate_batch(shots, target_ball_idx, max_steps)
}

/// Decode neural network output to shot
///
/// Network outputs: [angle_adjustment, power, english]
/// - angle_adjustment: 0-1 -> -45 to +45 degrees from pocket angle
/// - power: 0-1 -> 0.1 to 1.0
/// - english: 0-1 -> -0.8 to 0.8
fn decode_shot(outputs: List(Float), pocket_angle: Float) -> BatchShot {
  case outputs {
    [angle_adj_raw, power_raw, english_raw, ..] -> {
      // Adjustment range: +/- 45 degrees
      let angle_adjustment = { angle_adj_raw *. 2.0 -. 1.0 } *. 0.785398  // pi/4
      BatchShot(
        angle: pocket_angle +. angle_adjustment,
        power: 0.1 +. power_raw *. 0.9,
        english: { english_raw *. 2.0 -. 1.0 } *. 0.8,
        elevation: 0.0,
      )
    }
    _ -> BatchShot(angle: pocket_angle, power: 0.5, english: 0.0, elevation: 0.0)
  }
}

// =============================================================================
// UTILITIES
// =============================================================================

/// Zip 4 lists together
fn zip4(
  a: List(a),
  b: List(b),
  c: List(c),
  d: List(d),
) -> List(#(a, b, c, d)) {
  case a, b, c, d {
    [ah, ..at], [bh, ..bt], [ch, ..ct], [dh, ..dt] ->
      [#(ah, bh, ch, dh), ..zip4(at, bt, ct, dt)]
    _, _, _, _ -> []
  }
}

// =============================================================================
// MULTI-SHOT EPISODE SIMULATION (KEY OPTIMIZATION)
// =============================================================================

/// Result of a complete multi-shot episode
pub type EpisodeResult {
  EpisodeResult(
    /// Total accumulated fitness across all shots
    total_fitness: Float,
    /// Final ball X positions [8]
    final_pos_x: List(Float),
    /// Final ball Z positions [8]
    final_pos_z: List(Float),
    /// Final pocketed flags [8]
    final_pocketed: List(Float),
    /// Number of shots taken
    shots_taken: Int,
    /// Number of balls pocketed
    balls_pocketed: Int,
    /// Average hit angle (behavior descriptor)
    avg_hit_angle: Float,
    /// Average scatter ratio (behavior descriptor)
    avg_scatter_ratio: Float,
  )
}

/// Native NIF for episode simulation
/// Returns nested tuples: ((fitness, shots, balls, hit_angle, scatter), (pos_x, pos_z, pocketed))
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_simulate_episodes")
fn native_simulate_episodes(
  population_weights: List(List(Float)),
  architecture: List(Int),
  shots_per_episode: Int,
  max_steps_per_shot: Int,
) -> List(#(#(Float, Int, Int, Float, Float), #(List(Float), List(Float), List(Float))))

/// Native NIF for episode evaluation (simplified)
@external(erlang, "Elixir.Viva.Burn.Native", "burn_batch_evaluate_episodes")
fn native_evaluate_episodes(
  population_weights: List(List(Float)),
  architecture: List(Int),
  shots_per_episode: Int,
  max_steps_per_shot: Int,
) -> List(#(Float, Float, Float))

/// Simulate complete multi-shot episodes for a population
///
/// This is THE KEY OPTIMIZATION: reduces 4800 NIF calls per generation to 1.
///
/// Pipeline per network:
/// 1. Initialize table state
/// 2. For each shot:
///    a. Encode state to neural network inputs
///    b. Forward pass to get shot parameters
///    c. Simulate physics until settled
///    d. Calculate fitness for shot
/// 3. Return total fitness + behavior descriptors
///
/// Arguments:
/// - population_weights: network weights for all individuals
/// - architecture: [input, hidden1, hidden2, ..., output] layer sizes
/// - shots_per_episode: number of shots per episode
/// - max_steps_per_shot: max physics steps per shot
///
/// Returns: EpisodeResult for each network
pub fn simulate_episodes(
  population_weights: List(List(Float)),
  architecture: List(Int),
  shots_per_episode: Int,
  max_steps_per_shot: Int,
) -> List(EpisodeResult) {
  native_simulate_episodes(
    population_weights,
    architecture,
    shots_per_episode,
    max_steps_per_shot,
  )
  |> list.map(fn(nested_tuple) {
    // Unpack nested tuple structure
    let #(metrics, state) = nested_tuple
    let #(fitness, shots, balls, hit_angle, scatter) = metrics
    let #(pos_x, pos_z, pocketed) = state
    EpisodeResult(
      total_fitness: fitness,
      final_pos_x: pos_x,
      final_pos_z: pos_z,
      final_pocketed: pocketed,
      shots_taken: shots,
      balls_pocketed: balls,
      avg_hit_angle: hit_angle,
      avg_scatter_ratio: scatter,
    )
  })
}

/// Evaluate complete episodes (simplified API for QD training)
///
/// Returns only what's needed for MAP-Elites archive:
/// - Total fitness
/// - Hit angle (behavior descriptor)
/// - Scatter ratio (behavior descriptor)
///
/// This is the most efficient single-call API for QD training.
pub fn evaluate_episodes(
  population_weights: List(List(Float)),
  architecture: List(Int),
  shots_per_episode: Int,
  max_steps_per_shot: Int,
) -> List(FitnessResult) {
  native_evaluate_episodes(
    population_weights,
    architecture,
    shots_per_episode,
    max_steps_per_shot,
  )
  |> list.map(fn(tuple) {
    let #(fitness, hit_angle, scatter) = tuple
    FitnessResult(
      fitness: fitness,
      hit_angle: hit_angle,
      scatter_ratio: scatter,
    )
  })
}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Number of balls on table (cue + 7 colored)
pub const num_balls: Int = 8

/// Ball indices
pub const cue_ball_idx: Int = 0
pub const red_ball_idx: Int = 1
pub const yellow_ball_idx: Int = 2
pub const green_ball_idx: Int = 3
pub const brown_ball_idx: Int = 4
pub const blue_ball_idx: Int = 5
pub const pink_ball_idx: Int = 6
pub const black_ball_idx: Int = 7
