//// VIVA Sinuca QD v10 - Multi-shot Episode Simulation
////
//// Key improvement over v9: Multi-shot episode batch simulation
//// Instead of 4800 NIF calls per generation (1 per shot per network),
//// we now do 1 NIF call that runs the entire evaluation pipeline in Rust:
//// - Neural forward pass
//// - Physics simulation
//// - Fitness calculation
//// - Multi-shot sequencing
////
//// Performance target:
//// - v9: 4800 NIF calls x overhead = slow
//// - v10: 1 NIF call per generation = ~100x speedup (no NIF overhead)
////
//// Pipeline (all in single Rust NIF call):
//// 1. For each network in population:
////    a. Initialize table state
////    b. For each shot (1..shots_per_episode):
////       i.   Encode state to neural inputs
////       ii.  Forward pass (dense network)
////       iii. Decode outputs to shot
////       iv.  Simulate physics until settled
////       v.   Calculate shot fitness
////    c. Sum fitness, average behavior descriptors
//// 2. Return all results to Gleam
////
//// Created at GATO-PC, Brazil, 2026.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/lifecycle/burn
import viva/lifecycle/burn_physics.{type BatchShot, type FitnessResult, type EpisodeResult, BatchShot, FitnessResult, EpisodeResult}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub type QDConfig {
  QDConfig(
    // Grid
    grid_size: Int,
    // Network architecture
    architecture: List(Int),
    // NES parameters
    num_perturbations: Int,
    perturbation_std: Float,
    learning_rate: Float,
    // Training
    population_size: Int,
    elites_per_generation: Int,
    gradient_phase_ratio: Float,
    max_drift: Float,
    // Simulation
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    // Logging
    log_interval: Int,
  )
}

pub fn default_config() -> QDConfig {
  QDConfig(
    grid_size: 10,
    architecture: [8, 32, 16, 3],  // 867 weights
    num_perturbations: 16,
    perturbation_std: 0.03,
    learning_rate: 0.1,
    population_size: 100,
    elites_per_generation: 10,
    gradient_phase_ratio: 0.3,  // 30% NES, 70% exploration
    max_drift: 0.35,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
  )
}

pub fn fast_config() -> QDConfig {
  QDConfig(
    ..default_config(),
    grid_size: 5,
    population_size: 50,
    num_perturbations: 8,
    elites_per_generation: 5,
  )
}

// =============================================================================
// BEHAVIOR DESCRIPTOR
// =============================================================================

pub type Behavior {
  Behavior(
    hit_angle: Float,      // 0-1: angle of cue ball hit
    scatter_ratio: Float,  // 0-1: how much balls scattered
  )
}

fn behavior_to_cell(b: Behavior, grid_size: Int) -> #(Int, Int) {
  let x = float.clamp(b.hit_angle, 0.0, 0.999)
  let y = float.clamp(b.scatter_ratio, 0.0, 0.999)
  let cell_x = float.truncate(x *. int.to_float(grid_size))
  let cell_y = float.truncate(y *. int.to_float(grid_size))
  #(cell_x, cell_y)
}

fn behavior_distance(b1: Behavior, b2: Behavior) -> Float {
  let dx = b1.hit_angle -. b2.hit_angle
  let dy = b1.scatter_ratio -. b2.scatter_ratio
  float_sqrt(dx *. dx +. dy *. dy)
}

// =============================================================================
// ELITE ARCHIVE
// =============================================================================

pub type Elite {
  Elite(
    weights: List(Float),
    fitness: Float,
    behavior: Behavior,
    generation: Int,
  )
}

pub type Archive {
  Archive(
    cells: Dict(#(Int, Int), Elite),
    grid_size: Int,
  )
}

fn new_archive(grid_size: Int) -> Archive {
  Archive(cells: dict.new(), grid_size: grid_size)
}

fn try_add_elite(
  archive: Archive,
  weights: List(Float),
  fitness: Float,
  behavior: Behavior,
  generation: Int,
) -> #(Archive, Bool) {
  let cell = behavior_to_cell(behavior, archive.grid_size)

  case dict.get(archive.cells, cell) {
    Ok(existing) -> {
      case fitness >. existing.fitness {
        True -> {
          let elite = Elite(weights:, fitness:, behavior:, generation:)
          let cells = dict.insert(archive.cells, cell, elite)
          #(Archive(..archive, cells:), True)
        }
        False -> #(archive, False)
      }
    }
    Error(_) -> {
      let elite = Elite(weights:, fitness:, behavior:, generation:)
      let cells = dict.insert(archive.cells, cell, elite)
      #(Archive(..archive, cells:), True)
    }
  }
}

fn get_elites(archive: Archive) -> List(Elite) {
  dict.values(archive.cells)
}

fn coverage(archive: Archive) -> Float {
  let filled = int.to_float(dict.size(archive.cells))
  let total = int.to_float(archive.grid_size * archive.grid_size)
  filled /. total *. 100.0
}

fn qd_score(archive: Archive) -> Float {
  dict.values(archive.cells)
  |> list.fold(0.0, fn(acc, e) { acc +. e.fitness })
}

fn best_fitness(archive: Archive) -> Float {
  dict.values(archive.cells)
  |> list.fold(0.0, fn(acc, e) { float.max(acc, e.fitness) })
}

// =============================================================================
// BATCH EVALUATION (Multi-shot Episode Simulation)
// =============================================================================

/// Evaluate a batch of weight vectors using multi-shot episode simulation
///
/// This is THE KEY OPTIMIZATION: Instead of 4800 NIF calls per generation,
/// we do 1 NIF call that runs:
/// - Neural forward passes
/// - Physics simulations
/// - Fitness calculations
/// - Multi-shot sequencing
///
/// All in Rust, with Rayon parallelism.
fn batch_evaluate(
  weights_batch: List(List(Float)),
  config: QDConfig,
  _seed: Int,
) -> List(#(Float, Behavior)) {
  // Single NIF call for entire population evaluation!
  let results = burn_physics.evaluate_episodes(
    weights_batch,
    config.architecture,
    config.shots_per_episode,
    config.max_steps_per_shot,
  )

  // Convert FitnessResult to (fitness, behavior) pairs
  list.map(results, fn(r) {
    let behavior = Behavior(
      hit_angle: r.hit_angle,
      scatter_ratio: r.scatter_ratio,
    )
    #(r.fitness, behavior)
  })
}

/// Legacy batch evaluate for single-shot scenarios (kept for compatibility)
fn batch_evaluate_single_shot(
  weights_batch: List(List(Float)),
  config: QDConfig,
  seed: Int,
) -> List(#(Float, Behavior)) {
  let batch_size = list.length(weights_batch)

  // Generate fixed inputs for initial state (all tables start the same)
  let inputs = list.repeat(initial_inputs(), batch_size)

  // Generate pocket angles (same for all - target is always Red initially)
  // For Brazilian sinuca, cue ball starts at -table_length/4, Red at +table_length/4
  // So angle to red is approximately 0 radians (+X direction)
  let pocket_angles = list.repeat(0.0, batch_size)

  // 1. BATCH NEURAL FORWARD (GPU)
  let outputs = case burn.batch_forward(weights_batch, inputs, config.architecture) {
    Ok(out) -> out
    Error(_) -> list.repeat([0.5, 0.5, 0.5], batch_size)
  }

  // 2. DECODE OUTPUTS TO SHOTS
  let shots = list.zip(outputs, pocket_angles)
    |> list.map(fn(pair) {
      let #(out, pocket_angle) = pair
      decode_shot(out, pocket_angle)
    })

  // 3. BATCH PHYSICS SIMULATION (Rayon parallel)
  let initial_state = burn_physics.create_batch(batch_size)

  case burn_physics.simulate_batch(initial_state, shots, config.max_steps_per_shot) {
    Ok(final_state) -> {
      // 4. BATCH FITNESS CALCULATION
      let fitness_results = burn_physics.calculate_fitness(
        initial_state,
        final_state,
        burn_physics.red_ball_idx,  // Target is Red in initial break
      )

      // Convert to (fitness, behavior) pairs
      list.map(fitness_results, fn(r) {
        let behavior = Behavior(
          hit_angle: r.hit_angle,
          scatter_ratio: r.scatter_ratio,
        )
        #(r.fitness, behavior)
      })
    }
    Error(_) -> {
      // Fallback: return default fitness
      list.repeat(#(0.0, Behavior(hit_angle: 0.5, scatter_ratio: 0.5)), batch_size)
    }
  }
}

/// Initial neural network inputs
///
/// Normalized state of a fresh sinuca table.
fn initial_inputs() -> List(Float) {
  // 8 inputs matching sinuca_trainer.gleam format
  let half_l = 1.0  // Normalized
  let half_w = 0.5

  // Cue ball at -table_length/4
  let cue_x = -0.5 /. half_l
  let cue_z = 0.0 /. half_w

  // Target (Red) at +table_length/4
  let target_x = 0.5 /. half_l
  let target_z = 0.0 /. half_w

  // Pocket angle (to nearest pocket from red)
  // Closest pocket to red is probably top-right or bottom-right
  let pocket_angle = 0.0 /. 3.14159  // Normalized

  // Distance to pocket
  let pocket_dist = 0.5  // Normalized

  // Target value (Red = 1/7)
  let target_value = 1.0 /. 7.0 *. 2.0 -. 1.0

  // Balls remaining
  let balls_left = 8.0 /. 8.0 *. 2.0 -. 1.0

  [
    cue_x,
    cue_z,
    target_x,
    target_z,
    pocket_angle,
    pocket_dist,
    target_value,
    balls_left,
  ]
}

/// Decode neural network output to shot
fn decode_shot(outputs: List(Float), pocket_angle: Float) -> BatchShot {
  case outputs {
    [angle_adj_raw, power_raw, english_raw, ..] -> {
      // Adjustment range: +/- 45 degrees (pi/4)
      let angle_adjustment = { angle_adj_raw *. 2.0 -. 1.0 } *. 0.785398
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
// NES OPTIMIZATION (with batch physics)
// =============================================================================

/// Run NES gradient step on an elite using batch physics
fn nes_step(
  elite: Elite,
  config: QDConfig,
  seed: Int,
) -> Elite {
  // Generate perturbations using burn (fast Rust RNG)
  let perturbations = burn.perturb_weights(
    elite.weights,
    config.num_perturbations,
    config.perturbation_std,
    seed,
  )

  // Evaluate all perturbations IN BATCH
  let results = batch_evaluate(perturbations, config, seed + 10000)

  // Extract fitnesses
  let fitnesses = list.map(results, fn(r) { r.0 })

  // Check drift constraint
  let valid_results = list.zip(perturbations, results)
    |> list.filter(fn(pair) {
      let #(_, #(_, behavior)) = pair
      behavior_distance(elite.behavior, behavior) <. config.max_drift
    })

  case valid_results {
    [] -> elite  // No valid perturbations, keep original
    _ -> {
      let valid_perturbations = list.map(valid_results, fn(p) { p.0 })
      let valid_fitnesses = list.map(valid_results, fn(p) { { p.1 }.0 })

      // Compute gradient using burn (fast vectorized)
      let gradient = burn.nes_gradient(
        valid_perturbations,
        valid_fitnesses,
        config.perturbation_std,
      )

      // Update weights
      let new_weights = burn.update_weights(
        elite.weights,
        gradient,
        config.learning_rate,
      )

      // Evaluate updated weights
      case batch_evaluate([new_weights], config, seed + 20000) {
        [#(new_fitness, new_behavior)] -> {
          case new_fitness >. elite.fitness && behavior_distance(elite.behavior, new_behavior) <. config.max_drift {
            True -> Elite(
              weights: new_weights,
              fitness: new_fitness,
              behavior: new_behavior,
              generation: elite.generation,
            )
            False -> elite
          }
        }
        _ -> elite
      }
    }
  }
}

// =============================================================================
// EXPLORATION PHASE
// =============================================================================

fn exploration_phase(
  archive: Archive,
  config: QDConfig,
  generation: Int,
  seed: Int,
) -> Archive {
  // Generate random weights
  let new_weights = generate_random_population(
    config.population_size,
    burn.weight_count(config.architecture),
    seed,
  )

  // Batch evaluate all (FAST with batch physics!)
  let results = batch_evaluate(new_weights, config, seed + 50000)

  // Add to archive
  list.zip(new_weights, results)
  |> list.fold(archive, fn(arch, pair) {
    let #(weights, #(fitness, behavior)) = pair
    let #(new_arch, _) = try_add_elite(arch, weights, fitness, behavior, generation)
    new_arch
  })
}

fn generate_random_population(
  count: Int,
  weight_count: Int,
  seed: Int,
) -> List(List(Float)) {
  list.range(0, count - 1)
  |> list.map(fn(i) {
    generate_random_weights(weight_count, seed + i * 1000)
  })
}

fn generate_random_weights(count: Int, seed: Int) -> List(Float) {
  generate_weights_loop(count, seed, [])
}

fn generate_weights_loop(remaining: Int, seed: Int, acc: List(Float)) -> List(Float) {
  case remaining <= 0 {
    True -> list.reverse(acc)
    False -> {
      let next = { seed * 1103515245 + 12345 } % 2147483648
      let value = int.to_float(next % 1000 - 500) /. 1000.0
      generate_weights_loop(remaining - 1, next, [value, ..acc])
    }
  }
}

// =============================================================================
// MAIN TRAINING LOOP
// =============================================================================

pub fn train(generations: Int, config: QDConfig) -> #(Archive, Float, Float, Float) {
  io.println("=== VIVA Sinuca QD v10 (Multi-shot Episodes) ===")
  io.println("Backend: " <> burn.check())
  io.println("Grid: " <> int.to_string(config.grid_size) <> "x" <> int.to_string(config.grid_size))
  io.println("Architecture: " <> format_architecture(config.architecture))
  io.println("Population: " <> int.to_string(config.population_size))
  io.println("Shots/Episode: " <> int.to_string(config.shots_per_episode))
  io.println("NES: " <> int.to_string(config.num_perturbations) <> " perturbations")
  io.println("")
  io.println("KEY: Multi-shot episode batch (1 NIF call per generation!)")
  io.println("     Neural + Physics + Fitness all in single Rust call")
  io.println("")

  // Initialize archive with exploration
  let archive = new_archive(config.grid_size)
  let archive = exploration_phase(archive, config, 0, 42)

  // Training loop
  train_loop(archive, generations, 0, config, 42)
}

fn train_loop(
  archive: Archive,
  remaining: Int,
  generation: Int,
  config: QDConfig,
  seed: Int,
) -> #(Archive, Float, Float, Float) {
  case remaining <= 0 {
    True -> {
      let best = best_fitness(archive)
      let cov = coverage(archive)
      let qd = qd_score(archive)

      io.println("")
      io.println("=== Training Complete ===")
      io.println("Best: " <> float_to_string(best))
      io.println("Coverage: " <> float_to_string(cov) <> "%")
      io.println("QD-Score: " <> float_to_string(qd))

      #(archive, best, cov, qd)
    }
    False -> {
      // Log progress
      case generation % config.log_interval == 0 {
        True -> {
          let best = best_fitness(archive)
          let cov = coverage(archive)
          let qd = qd_score(archive)
          let phase = case pseudo_random(seed + generation) <. config.gradient_phase_ratio {
            True -> "[NES]"
            False -> "[EXPLORE]"
          }
          io.println(
            "Gen " <> int.to_string(generation)
            <> " | Best: " <> float_to_string(best)
            <> " | Cov: " <> float_to_string(cov) <> "%"
            <> " | QD: " <> float_to_string(qd)
            <> " " <> phase
          )
        }
        False -> Nil
      }

      // Choose phase
      let archive = case pseudo_random(seed + generation) <. config.gradient_phase_ratio {
        True -> {
          // NES gradient phase
          let elites = get_elites(archive)
            |> list.take(config.elites_per_generation)

          list.index_fold(elites, archive, fn(arch, elite, idx) {
            let improved = nes_step(elite, config, seed + generation * 1000 + idx * 100)
            let #(new_arch, _) = try_add_elite(
              arch,
              improved.weights,
              improved.fitness,
              improved.behavior,
              generation,
            )
            new_arch
          })
        }
        False -> {
          // Exploration phase
          exploration_phase(archive, config, generation, seed + generation * 5000)
        }
      }

      train_loop(archive, remaining - 1, generation + 1, config, seed)
    }
  }
}

// =============================================================================
// BENCHMARK
// =============================================================================

/// Benchmark batch physics vs sequential
pub fn benchmark_comparison(batch_size: Int, max_steps: Int, iterations: Int) -> String {
  io.println("Running batch physics benchmark...")
  let result = burn_physics.benchmark(batch_size, max_steps, iterations)
  io.println(result)
  result
}

// =============================================================================
// UTILITIES
// =============================================================================

fn pseudo_random(seed: Int) -> Float {
  let next = { seed * 1103515245 + 12345 } % 2147483648
  int.to_float(int.absolute_value(next)) /. 2147483648.0
}

fn float_sqrt(x: Float) -> Float {
  case x <=. 0.0 {
    True -> 0.0
    False -> newton_sqrt(x, x /. 2.0, 10)
  }
}

fn newton_sqrt(x: Float, guess: Float, iterations: Int) -> Float {
  case iterations <= 0 {
    True -> guess
    False -> {
      let better = { guess +. x /. guess } /. 2.0
      newton_sqrt(x, better, iterations - 1)
    }
  }
}

fn float_to_string(x: Float) -> String {
  let scaled = float.truncate(x *. 10.0)
  let whole = scaled / 10
  let frac = int.absolute_value(scaled % 10)
  int.to_string(whole) <> "." <> int.to_string(frac)
}

fn format_architecture(arch: List(Int)) -> String {
  "[" <> format_arch_inner(arch) <> "]"
}

fn format_arch_inner(arch: List(Int)) -> String {
  case arch {
    [] -> ""
    [x] -> int.to_string(x)
    [x, ..rest] -> int.to_string(x) <> ", " <> format_arch_inner(rest)
  }
}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  // First run benchmark to show speedup
  io.println("=== Multi-shot Episode Benchmark ===")
  let _ = benchmark_comparison(1000, 200, 10)
  io.println("")

  // Then run training (300 generations for full evolution)
  let config = default_config()
  let #(_archive, best, cov, qd) = train(300, config)

  io.println("")
  io.println("Final Results:")
  io.println("  Best Fitness: " <> float_to_string(best))
  io.println("  Coverage: " <> float_to_string(cov) <> "%")
  io.println("  QD-Score: " <> float_to_string(qd))
}

/// Benchmark multi-shot episode simulation
pub fn benchmark_episodes(pop_size: Int, shots_per_episode: Int, max_steps: Int, iterations: Int) -> String {
  // Generate random weights for population
  let architecture = [8, 32, 16, 3]
  let weight_count = 8 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3  // 867

  let weights_batch = generate_random_population(pop_size, weight_count, 42)

  // Warmup
  let _ = burn_physics.evaluate_episodes(
    weights_batch,
    architecture,
    shots_per_episode,
    max_steps,
  )

  // This would need timing - for now just return info
  "Multi-shot Episode Benchmark:\n"
  <> "  Population: " <> int.to_string(pop_size) <> "\n"
  <> "  Shots/Episode: " <> int.to_string(shots_per_episode) <> "\n"
  <> "  Max Steps: " <> int.to_string(max_steps) <> "\n"
  <> "  NIF Calls: 1 (was " <> int.to_string(pop_size * shots_per_episode) <> " before)\n"
  <> "  Speedup: ~" <> int.to_string(pop_size * shots_per_episode) <> "x NIF overhead reduction"
}
