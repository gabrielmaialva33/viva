//// VIVA Sinuca QD v11 - Qwen3-235B Improvements
////
//// Improvements based on AI analysis:
//// 1. Novelty bonus for empty cells (accelerate coverage)
//// 2. Random restarts to escape local optima (improve fitness)
//// 3. Larger population (200 vs 100)
//// 4. Adaptive mutation scaling
////
//// Created at GATO-PC, Brazil, 2026.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva/lifecycle/burn
import viva/lifecycle/burn_physics

// =============================================================================
// CONFIGURATION - Improved based on Qwen3 suggestions
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
    // Training - INCREASED population
    population_size: Int,
    elites_per_generation: Int,
    gradient_phase_ratio: Float,
    max_drift: Float,
    // Simulation
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    // v11 NEW: Novelty and restart params
    novelty_bonus: Float,        // Bonus for filling empty cells
    restart_interval: Int,       // Generations between random restarts
    restart_ratio: Float,        // Fraction of population to restart
    mutation_decay: Float,       // Decay rate for mutation std
    // Logging
    log_interval: Int,
  )
}

pub fn default_config() -> QDConfig {
  QDConfig(
    grid_size: 10,
    architecture: [8, 32, 16, 3],
    num_perturbations: 16,
    perturbation_std: 0.05,      // Increased from 0.03
    learning_rate: 0.1,
    population_size: 200,        // DOUBLED from 100
    elites_per_generation: 20,   // Doubled
    gradient_phase_ratio: 0.25,  // More exploration
    max_drift: 0.4,              // Slightly increased
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    // v11 NEW params
    novelty_bonus: 5.0,          // +5 fitness for new cells
    restart_interval: 50,        // Restart every 50 gens
    restart_ratio: 0.1,          // Restart 10% of population
    mutation_decay: 0.995,       // Slow decay
    log_interval: 5,
  )
}

// =============================================================================
// BEHAVIOR DESCRIPTOR - Same as v10
// =============================================================================

pub type Behavior {
  Behavior(
    hit_angle: Float,
    scatter_ratio: Float,
  )
}

fn behavior_to_cell(b: Behavior, grid_size: Int) -> #(Int, Int) {
  let x = float.clamp(b.hit_angle, 0.0, 0.999)
  let y = float.clamp(b.scatter_ratio, 0.0, 0.999)
  let cell_x = float.truncate(x *. int.to_float(grid_size))
  let cell_y = float.truncate(y *. int.to_float(grid_size))
  #(cell_x, cell_y)
}

// =============================================================================
// ELITE ARCHIVE with Novelty Tracking
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
    total_discoveries: Int,  // v11: Track total discoveries
  )
}

fn new_archive(grid_size: Int) -> Archive {
  Archive(cells: dict.new(), grid_size: grid_size, total_discoveries: 0)
}

/// Try to add elite with novelty bonus
fn try_add_elite(
  archive: Archive,
  weights: List(Float),
  raw_fitness: Float,
  behavior: Behavior,
  generation: Int,
  novelty_bonus: Float,
) -> #(Archive, Bool, Float) {
  let cell = behavior_to_cell(behavior, archive.grid_size)

  case dict.get(archive.cells, cell) {
    Ok(existing) -> {
      // Cell exists - no novelty bonus, just compare fitness
      case raw_fitness >. existing.fitness {
        True -> {
          let elite = Elite(weights:, fitness: raw_fitness, behavior:, generation:)
          let cells = dict.insert(archive.cells, cell, elite)
          #(Archive(..archive, cells:), True, raw_fitness)
        }
        False -> #(archive, False, raw_fitness)
      }
    }
    Error(_) -> {
      // NEW CELL! Apply novelty bonus
      let boosted_fitness = raw_fitness +. novelty_bonus
      let elite = Elite(weights:, fitness: raw_fitness, behavior:, generation:)
      let cells = dict.insert(archive.cells, cell, elite)
      let new_archive = Archive(
        ..archive,
        cells: cells,
        total_discoveries: archive.total_discoveries + 1,
      )
      #(new_archive, True, boosted_fitness)
    }
  }
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

fn get_elites(archive: Archive) -> List(Elite) {
  dict.values(archive.cells)
}

// =============================================================================
// RANDOM RESTART - Escape local optima
// =============================================================================

fn should_restart(gen: Int, interval: Int) -> Bool {
  gen > 0 && gen % interval == 0
}

fn random_restart_population(
  population: List(List(Float)),
  ratio: Float,
  weight_count: Int,
  seed: Int,
) -> List(List(Float)) {
  let restart_count = float.truncate(int.to_float(list.length(population)) *. ratio)
  let keep_count = list.length(population) - restart_count

  let kept = list.take(population, keep_count)
  let restarted = generate_random_population(restart_count, weight_count, seed)

  list.append(kept, restarted)
}

fn generate_random_population(count: Int, weight_count: Int, seed: Int) -> List(List(Float)) {
  list.range(0, count - 1)
  |> list.map(fn(i) { random_weights(weight_count, seed + i * 1000) })
}

fn random_weights(count: Int, seed: Int) -> List(Float) {
  list.range(0, count - 1)
  |> list.map(fn(i) {
    let x = int.to_float({ seed + i * 7919 } % 10000) /. 10000.0
    { x -. 0.5 } *. 2.0  // Range [-1, 1]
  })
}

// =============================================================================
// BATCH EVALUATION
// =============================================================================

fn batch_evaluate(
  weights_batch: List(List(Float)),
  config: QDConfig,
  _seed: Int,
) -> List(#(Float, Behavior)) {
  let results = burn_physics.evaluate_episodes(
    weights_batch,
    config.architecture,
    config.shots_per_episode,
    config.max_steps_per_shot,
  )

  list.map(results, fn(r) {
    let behavior = Behavior(
      hit_angle: r.hit_angle,
      scatter_ratio: r.scatter_ratio,
    )
    #(r.fitness, behavior)
  })
}

// =============================================================================
// MUTATION with Adaptive Scaling
// =============================================================================

fn mutate_weights(
  weights: List(Float),
  std: Float,
  seed: Int,
) -> List(Float) {
  list.index_map(weights, fn(w, i) {
    let noise = gaussian_noise(seed + i, std)
    float.clamp(w +. noise, -2.0, 2.0)
  })
}

fn gaussian_noise(seed: Int, std: Float) -> Float {
  // Box-Muller approximation
  let u1 = int.to_float({ seed * 1103515245 + 12345 } % 2147483647) /. 2147483647.0
  let u2 = int.to_float({ seed * 1103515245 * 2 + 12345 } % 2147483647) /. 2147483647.0
  let u1_safe = float.max(u1, 0.0001)
  let z = float_sqrt(-2.0 *. float_ln(u1_safe)) *. float_cos(2.0 *. 3.14159 *. u2)
  z *. std
}

// =============================================================================
// TRAINING LOOP with v11 Improvements
// =============================================================================

pub type TrainState {
  TrainState(
    archive: Archive,
    generation: Int,
    config: QDConfig,
    current_std: Float,  // Adaptive mutation std
  )
}

pub fn train(generations: Int) -> TrainState {
  train_with_config(generations, default_config())
}

pub fn train_with_config(generations: Int, config: QDConfig) -> TrainState {
  let archive = new_archive(config.grid_size)
  let weight_count = calculate_weight_count(config.architecture)

  // Initial population - LARGER
  let initial_pop = generate_random_population(
    config.population_size,
    weight_count,
    42,
  )

  let state = TrainState(
    archive: archive,
    generation: 0,
    config: config,
    current_std: config.perturbation_std,
  )

  train_loop(state, initial_pop, generations, weight_count)
}

fn train_loop(
  state: TrainState,
  population: List(List(Float)),
  remaining: Int,
  weight_count: Int,
) -> TrainState {
  case remaining <= 0 {
    True -> state
    False -> {
      let gen = state.generation
      let config = state.config

      // v11: Check for random restart
      let pop = case should_restart(gen, config.restart_interval) {
        True -> {
          io.println("[v11] Random restart at gen " <> int.to_string(gen))
          random_restart_population(population, config.restart_ratio, weight_count, gen)
        }
        False -> population
      }

      // Evaluate population
      let results = batch_evaluate(pop, config, gen)

      // Update archive with novelty bonus
      let #(new_archive, _added) = list.fold(
        list.zip(pop, results),
        #(state.archive, 0),
        fn(acc, pair) {
          let #(archive, count) = acc
          let #(weights, #(fitness, behavior)) = pair
          let #(arch2, added, _boosted) = try_add_elite(
            archive, weights, fitness, behavior, gen, config.novelty_bonus,
          )
          case added {
            True -> #(arch2, count + 1)
            False -> #(arch2, count)
          }
        },
      )

      // Log progress
      case gen % config.log_interval == 0 {
        True -> {
          let phase = case gen % 20 < int.to_float(20) *. config.gradient_phase_ratio |> float.truncate {
            True -> "[NES]"
            False -> "[EXPLORE]"
          }
          let best = best_fitness(new_archive)
          let cov = coverage(new_archive)
          let qd = qd_score(new_archive)
          io.println(
            "Gen " <> int.to_string(gen)
            <> " | Best: " <> float_format(best, 1)
            <> " | Cov: " <> float_format(cov, 1) <> "%"
            <> " | QD: " <> float_format(qd, 1)
            <> " " <> phase
          )
        }
        False -> Nil
      }

      // Generate next population
      let elites = get_elites(new_archive)
      let next_pop = generate_next_population(
        elites,
        config.population_size,
        state.current_std,
        gen,
        weight_count,
      )

      // v11: Decay mutation std
      let new_std = state.current_std *. config.mutation_decay

      let new_state = TrainState(
        ..state,
        archive: new_archive,
        generation: gen + 1,
        current_std: new_std,
      )

      train_loop(new_state, next_pop, remaining - 1, weight_count)
    }
  }
}

fn generate_next_population(
  elites: List(Elite),
  target_size: Int,
  std: Float,
  gen: Int,
  weight_count: Int,
) -> List(List(Float)) {
  case list.length(elites) {
    0 -> generate_random_population(target_size, weight_count, gen)
    _ -> {
      // Elite weights as base
      let elite_weights = list.map(elites, fn(e) { e.weights })
      let elite_count = list.length(elite_weights)

      // Generate mutations from elites
      list.range(0, target_size - 1)
      |> list.map(fn(i) {
        let base_idx = i % elite_count
        let base = case list.drop(elite_weights, base_idx) |> list.first {
          Ok(w) -> w
          Error(_) -> random_weights(weight_count, gen + i)
        }
        mutate_weights(base, std, gen * 1000 + i)
      })
    }
  }
}

fn calculate_weight_count(architecture: List(Int)) -> Int {
  case architecture {
    [] -> 0
    [_] -> 0
    [a, b, ..rest] -> {
      let layer_weights = a * b + b  // weights + biases
      layer_weights + calculate_weight_count([b, ..rest])
    }
  }
}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("=== VIVA Sinuca QD v11 ===")
  io.println("Improvements: Novelty bonus, Random restarts, Larger population")
  io.println("")

  let state = train(300)

  io.println("")
  io.println("=== Training Complete ===")
  io.println("Best: " <> float.to_string(best_fitness(state.archive)))
  io.println("Coverage: " <> float.to_string(coverage(state.archive)) <> "%")
  io.println("QD-Score: " <> float.to_string(qd_score(state.archive)))
  io.println("Total Discoveries: " <> int.to_string(state.archive.total_discoveries))
  io.println("")
  io.println("Final Results:")
  io.println("  Best Fitness: " <> float.to_string(best_fitness(state.archive)))
  io.println("  Coverage: " <> float.to_string(coverage(state.archive)) <> "%")
  io.println("  QD-Score: " <> float.to_string(qd_score(state.archive)))
}

// =============================================================================
// FFI
// =============================================================================

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "log")
fn float_ln(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

fn float_format(x: Float, _decimals: Int) -> String {
  let truncated = float.truncate(x *. 10.0)
  let integer = truncated / 10
  let decimal = int.absolute_value(truncated % 10)
  int.to_string(integer) <> "." <> int.to_string(decimal)
}
