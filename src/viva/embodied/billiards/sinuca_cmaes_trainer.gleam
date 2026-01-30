//// VIVA Sinuca - QD-CMA-ES Hybrid Trainer
////
//// State-of-the-art Quality-Diversity training combining:
//// - CMA-ES for adaptive search distribution
//// - MAP-Elites for behavioral diversity
//// - GPU acceleration via burn-rs NIFs
////
//// Key innovations:
//// - CMA-ES replaces random mutations (gradient-like adaptation)
//// - Per-cell CMA-ES optimizers for exploitation
//// - Global CMA-ES for exploration
//// - Automatic step-size adaptation
////
//// Architecture: [8, 32, 16, 3] = 867 weights
//// GPU: RTX 4090 with CUDA backend
////
//// Created at GATO-PC, Brazil, 2026.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/dict.{type Dict}
import viva/embodied/billiards/sinuca.{type Shot, type Table, Shot}
import viva/embodied/billiards/sinuca_fitness as fitness
import viva/lifecycle/burn
import viva/soul/glands
import viva/lifecycle/jolt.{Vec3}
import viva/neural/cma_es.{
  type CmaEsState, type CmaEsConfig, type CmaEsDiagnostics,
  type QdCmaEsConfig, type CellOptimizer,
}
import viva/neural/holomap.{type HoloMapConfig, type HoloMapStats, type MapElitesGrid, type Elite}
import viva/neural/novelty.{type Behavior}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// QD-CMA-ES trainer configuration
pub type TrainerConfig {
  TrainerConfig(
    /// Network architecture [input, hidden1, hidden2, output]
    architecture: List(Int),
    /// Maximum physics steps per shot
    max_steps_per_shot: Int,
    /// Shots per evaluation episode
    shots_per_episode: Int,
    /// Logging interval (generations)
    log_interval: Int,
    /// Exploration ratio (0.0-1.0, higher = more global CMA-ES)
    exploration_ratio: Float,
    /// Initial exploration ratio
    initial_exploration_ratio: Float,
    /// Final exploration ratio (after annealing)
    final_exploration_ratio: Float,
    /// Annealing generations
    annealing_gens: Int,
    /// CMA-ES step size for exploitation (per-cell)
    exploitation_sigma: Float,
    /// CMA-ES step size for exploration (global)
    exploration_sigma: Float,
    /// MAP-Elites grid size
    grid_size: Int,
    /// HRR dimension for genome encoding
    hrr_dim: Int,
    /// Population size per generation
    population_size: Int,
    /// Maximum stagnation before restart
    max_stagnation: Int,
  )
}

/// Default trainer config optimized for sinuca
pub fn default_config() -> TrainerConfig {
  TrainerConfig(
    architecture: [8, 32, 16, 3],  // 867 weights
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    exploration_ratio: 0.7,
    initial_exploration_ratio: 0.8,
    final_exploration_ratio: 0.3,
    annealing_gens: 50,
    exploitation_sigma: 0.1,
    exploration_sigma: 0.3,
    grid_size: 5,
    hrr_dim: 4096,
    population_size: 50,
    max_stagnation: 10,
  )
}

/// Fast config for testing
pub fn fast_config() -> TrainerConfig {
  TrainerConfig(
    ..default_config(),
    population_size: 20,
    shots_per_episode: 2,
    max_steps_per_shot: 100,
    annealing_gens: 20,
  )
}

// =============================================================================
// TRAINER STATE
// =============================================================================

/// Main trainer state
pub type TrainerState {
  TrainerState(
    /// Global CMA-ES for exploration
    global_cma: CmaEsState,
    /// Per-cell CMA-ES optimizers
    cell_optimizers: Dict(#(Int, Int), CellOptimizer),
    /// MAP-Elites grid
    grid: MapElitesGrid,
    /// Current generation
    generation: Int,
    /// Best fitness ever seen
    best_fitness: Float,
    /// Training config
    config: TrainerConfig,
    /// Random seed
    seed: Int,
  )
}

/// Training result
pub type TrainingResult {
  TrainingResult(
    /// Final MAP-Elites grid
    grid: MapElitesGrid,
    /// Final statistics
    stats: HoloMapStats,
    /// Best weights found
    best_weights: List(Float),
    /// Total generations trained
    generations: Int,
  )
}

// =============================================================================
// INITIALIZATION
// =============================================================================

/// Initialize trainer with random weights
pub fn init(config: TrainerConfig, seed: Int) -> TrainerState {
  // Calculate weight count
  let weight_count = burn.weight_count(config.architecture)

  // Initialize random weights for global CMA-ES
  let initial_weights = init_xavier_weights(weight_count, seed)

  // Create global CMA-ES
  let cma_config = cma_es.CmaEsConfig(
    initial_sigma: config.exploration_sigma,
    lambda: Some(config.population_size),
    mu: None,
    cc: None,
    c1: None,
    cmu: None,
    csigma: None,
    dsigma: None,
  )
  let global = cma_es.init(initial_weights, cma_config)

  // Initialize MAP-Elites grid
  let holomap_config = holomap.HoloMapConfig(
    grid_size: config.grid_size,
    behavior_dims: 2,
    hrr_dim: config.hrr_dim,
    initial_novelty_weight: 0.7,
    final_novelty_weight: 0.3,
    decay_midpoint: 20,
    batch_size: config.population_size,
    tournament_size: 4,
  )
  let grid = holomap.new_grid(holomap_config)

  TrainerState(
    global_cma: global,
    cell_optimizers: dict.new(),
    grid: grid,
    generation: 0,
    best_fitness: 0.0,
    config: config,
    seed: seed,
  )
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

/// Train for specified number of generations
pub fn train(
  generations: Int,
  config: TrainerConfig,
  seed: Int,
) -> TrainingResult {
  io.println("=== VIVA Sinuca QD-CMA-ES Hybrid ===")
  io.println("GPU Status: " <> glands.check())
  io.println("Architecture: " <> int_list_to_string(config.architecture))
  io.println("Population: " <> int.to_string(config.population_size))
  io.println("Grid: " <> int.to_string(config.grid_size) <> "x"
    <> int.to_string(config.grid_size))
  io.println("")

  let state = init(config, seed)

  // Initial population
  let initial_pop = cma_es.sample(state.global_cma, seed)

  // Train
  let final_state = train_loop(state, initial_pop, [], generations, seed)

  // Compute final stats
  let holomap_config = holomap.HoloMapConfig(
    grid_size: config.grid_size,
    behavior_dims: 2,
    hrr_dim: config.hrr_dim,
    initial_novelty_weight: 0.3,
    final_novelty_weight: 0.3,
    decay_midpoint: 20,
    batch_size: config.population_size,
    tournament_size: 4,
  )
  let stats = holomap.compute_stats(
    final_state.grid,
    final_state.generation,
    holomap_config,
  )

  // Get best weights
  let best_weights = cma_es.get_mean(final_state.global_cma)

  io.println("")
  io.println("=== Training Complete ===")
  io.println("Best Fitness: " <> float_str(stats.best_fitness))
  io.println("Coverage: " <> float_str(stats.coverage) <> "%")
  io.println("QD-Score: " <> float_str(stats.qd_score))

  TrainingResult(
    grid: final_state.grid,
    stats: stats,
    best_weights: best_weights,
    generations: final_state.generation,
  )
}

/// Main training loop
fn train_loop(
  state: TrainerState,
  population: List(List(Float)),
  prev_fitnesses: List(Float),
  remaining: Int,
  seed: Int,
) -> TrainerState {
  case remaining <= 0 {
    True -> state
    False -> {
      // Compute annealed exploration ratio
      let exploration_ratio = annealed_exploration_ratio(
        state.generation,
        state.config,
      )

      // Evaluate population
      let evaluated = evaluate_population(population, state.config)

      // Extract fitnesses and behaviors
      let fitnesses = list.map(evaluated, fn(e) { e.0 })
      let behaviors = list.map(evaluated, fn(e) { e.1 })

      // Update global CMA-ES
      cma_es.update(state.global_cma, population, fitnesses)

      // Update MAP-Elites grid
      let new_grid = update_grid(
        state.grid,
        population,
        fitnesses,
        behaviors,
        state.generation,
        state.config,
      )

      // Update cell optimizers
      let new_cell_opts = update_cell_optimizers(
        state.cell_optimizers,
        new_grid,
        population,
        fitnesses,
        behaviors,
        state.config,
      )

      // Track best
      let batch_best = list.fold(fitnesses, 0.0, float.max)
      let new_best = float.max(state.best_fitness, batch_best)

      // Log progress
      case state.generation % state.config.log_interval == 0 {
        True -> {
          let holomap_config = holomap.HoloMapConfig(
            grid_size: state.config.grid_size,
            behavior_dims: 2,
            hrr_dim: state.config.hrr_dim,
            initial_novelty_weight: 0.3,
            final_novelty_weight: 0.3,
            decay_midpoint: 20,
            batch_size: state.config.population_size,
            tournament_size: 4,
          )
          let stats = holomap.compute_stats(
            new_grid,
            state.generation,
            holomap_config,
          )
          let diag = cma_es.get_diagnostics(state.global_cma)

          io.println(
            "Gen " <> int.to_string(state.generation)
            <> " | Best: " <> float_str(batch_best)
            <> " | Cov: " <> float_str(stats.coverage) <> "%"
            <> " | QD: " <> float_str(stats.qd_score)
            <> " | sigma: " <> float_str(diag.sigma)
            <> " | expl: " <> float_str(exploration_ratio)
          )
        }
        False -> Nil
      }

      // Generate next population
      let next_pop = generate_next_population(
        state.global_cma,
        new_cell_opts,
        exploration_ratio,
        seed + state.generation * 1000,
        state.config,
      )

      // Update state
      let new_state = TrainerState(
        ..state,
        cell_optimizers: new_cell_opts,
        grid: new_grid,
        generation: state.generation + 1,
        best_fitness: new_best,
      )

      train_loop(new_state, next_pop, fitnesses, remaining - 1, seed)
    }
  }
}

// =============================================================================
// EVALUATION
// =============================================================================

/// Evaluate population - returns (fitness, behavior) pairs
fn evaluate_population(
  population: List(List(Float)),
  config: TrainerConfig,
) -> List(#(Float, Behavior)) {
  list.map(population, fn(weights) {
    evaluate_single(weights, config)
  })
}

/// Evaluate single set of weights
fn evaluate_single(
  weights: List(Float),
  config: TrainerConfig,
) -> #(Float, Behavior) {
  let table = sinuca.new()
  let episode = fitness.new_episode()

  let #(total_fitness, behavior_features, _, _, _, _) =
    list.fold(
      list.range(1, config.shots_per_episode),
      #(0.0, [], 0, 0, table, episode),
      fn(acc, _shot_num) {
        let #(fitness_sum, features, foul_count, max_combo, current, ep) = acc

        case sinuca.balls_on_table(current) <= 1 {
          True -> acc
          False -> {
            let inputs = encode_inputs(current)
            let outputs = forward_weights(weights, inputs, config.architecture)
            let shot = decode_outputs(outputs)

            let #(shot_fitness, next, new_ep) =
              fitness.quick_evaluate_full(
                current,
                shot,
                config.max_steps_per_shot,
                ep,
                fitness.default_config(),
              )

            let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(next) {
              Some(Vec3(x, _, z)) -> #(x, z)
              None -> #(0.0, 0.0)
            }

            let new_features = list.append(features, [
              shot.angle /. 6.28,
              shot.power,
              cue_x /. 1.5,
              cue_z /. 0.75,
            ])

            let new_fouls = case sinuca.is_scratch(next) {
              True -> foul_count + 1
              False -> foul_count
            }

            let new_max_combo = int.max(max_combo, new_ep.consecutive_pockets)

            let table_next = case sinuca.is_scratch(next) {
              True -> sinuca.reset_cue_ball(next)
              False -> next
            }

            #(
              fitness_sum +. shot_fitness,
              new_features,
              new_fouls,
              new_max_combo,
              table_next,
              new_ep,
            )
          }
        }
      },
    )

  let behavior = novelty.behavior_from_features(behavior_features)
  #(total_fitness, behavior)
}

// =============================================================================
// GRID UPDATE
// =============================================================================

/// Update MAP-Elites grid with new evaluations
fn update_grid(
  grid: MapElitesGrid,
  population: List(List(Float)),
  fitnesses: List(Float),
  behaviors: List(Behavior),
  generation: Int,
  config: TrainerConfig,
) -> MapElitesGrid {
  // Zip population, fitnesses, behaviors
  let entries = zip3(population, fitnesses, behaviors)

  list.fold(entries, grid, fn(g, entry) {
    let #(weights, fit, behavior) = entry
    let hrr = weights_to_hrr(weights, config.hrr_dim, generation)
    let genome_id = generation * 1000 + list.length(dict.keys(g.cells))
    let #(new_grid, _added) = holomap.try_add_elite(
      g,
      genome_id,
      behavior,
      hrr,
      fit,
      generation,
    )
    new_grid
  })
}

/// Update cell-specific CMA-ES optimizers
fn update_cell_optimizers(
  cell_opts: Dict(#(Int, Int), CellOptimizer),
  grid: MapElitesGrid,
  population: List(List(Float)),
  fitnesses: List(Float),
  behaviors: List(Behavior),
  config: TrainerConfig,
) -> Dict(#(Int, Int), CellOptimizer) {
  // Get all elites
  let elites = holomap.get_elites(grid)

  // For each elite, ensure we have a cell optimizer
  list.fold(elites, cell_opts, fn(opts, elite) {
    let cell = holomap.behavior_to_cell(elite.behavior, grid)

    case dict.get(opts, cell) {
      Ok(existing_opt) -> {
        // Update existing optimizer with any solutions that landed in this cell
        // (simplified - just track best fitness)
        case elite.fitness >. existing_opt.best_fitness {
          True -> {
            // Re-initialize CMA-ES from new elite
            let cma_config = cma_es.CmaEsConfig(
              initial_sigma: config.exploitation_sigma,
              lambda: Some(10),  // Smaller population for cell optimization
              mu: None,
              cc: None,
              c1: None,
              cmu: None,
              csigma: None,
              dsigma: None,
            )
            let new_cma = cma_es.init(elite.hrr_vector, cma_config)
            let new_opt = cma_es.CellOptimizer(
              cell_id: cell,
              state: new_cma,
              best_fitness: elite.fitness,
              stagnation: 0,
            )
            dict.insert(opts, cell, new_opt)
          }
          False -> {
            // Increment stagnation
            let updated = cma_es.CellOptimizer(
              ..existing_opt,
              stagnation: existing_opt.stagnation + 1,
            )
            dict.insert(opts, cell, updated)
          }
        }
      }
      Error(_) -> {
        // Create new cell optimizer
        let cma_config = cma_es.CmaEsConfig(
          initial_sigma: config.exploitation_sigma,
          lambda: Some(10),
          mu: None,
          cc: None,
          c1: None,
          cmu: None,
          csigma: None,
          dsigma: None,
        )
        let new_cma = cma_es.init(elite.hrr_vector, cma_config)
        let new_opt = cma_es.CellOptimizer(
          cell_id: cell,
          state: new_cma,
          best_fitness: elite.fitness,
          stagnation: 0,
        )
        dict.insert(opts, cell, new_opt)
      }
    }
  })
}

// =============================================================================
// POPULATION GENERATION
// =============================================================================

/// Generate next population from hybrid system
fn generate_next_population(
  global_cma: CmaEsState,
  cell_opts: Dict(#(Int, Int), CellOptimizer),
  exploration_ratio: Float,
  seed: Int,
  config: TrainerConfig,
) -> List(List(Float)) {
  let total_pop = config.population_size

  // Calculate split
  let from_global = float.round(
    int.to_float(total_pop) *. exploration_ratio
  )
  let from_cells = total_pop - from_global

  // Sample from global CMA-ES
  let global_samples = cma_es.sample(global_cma, seed)
    |> list.take(from_global)

  // Sample from cell optimizers
  let cell_list = dict.values(cell_opts)
  let cell_samples = case list.length(cell_list) {
    0 -> []
    num_cells -> {
      let per_cell = from_cells / num_cells + 1
      cell_list
        |> list.index_map(fn(opt, idx) {
          cma_es.sample(opt.state, seed + idx * 100)
            |> list.take(per_cell)
        })
        |> list.flatten
        |> list.take(from_cells)
    }
  }

  list.append(global_samples, cell_samples)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn annealed_exploration_ratio(generation: Int, config: TrainerConfig) -> Float {
  let progress = int.to_float(int.min(generation, config.annealing_gens))
  let total = int.to_float(config.annealing_gens)
  let range = config.initial_exploration_ratio -. config.final_exploration_ratio

  config.initial_exploration_ratio -. { progress /. total } *. range
}

fn encode_inputs(table: Table) -> List(Float) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0

  let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(table) {
    Some(Vec3(x, _y, z)) -> #(x, z)
    None -> #(0.0, 0.0)
  }

  let target = table.target_ball
  let #(target_x, target_z) = case sinuca.get_ball_position(table, target) {
    Some(Vec3(x, _y, z)) -> #(x, z)
    None -> #(0.0, 0.0)
  }

  let #(pocket_angle, pocket_dist) = fitness.best_pocket_angle(table)
  let target_value = int.to_float(sinuca.point_value(target)) /. 7.0
  let balls_left = int.to_float(sinuca.balls_on_table(table)) /. 8.0

  [
    cue_x /. half_l,
    cue_z /. half_w,
    target_x /. half_l,
    target_z /. half_w,
    pocket_angle /. 3.14159,
    float_clamp(pocket_dist /. 3.0, 0.0, 1.0) *. 2.0 -. 1.0,
    target_value *. 2.0 -. 1.0,
    balls_left *. 2.0 -. 1.0,
  ]
}

fn decode_outputs(outputs: List(Float)) -> Shot {
  case outputs {
    [angle_raw, power_raw, english_raw, ..] -> {
      Shot(
        angle: angle_raw *. 2.0 *. 3.14159,
        power: 0.1 +. power_raw *. 0.9,
        english: { english_raw *. 2.0 -. 1.0 } *. 0.8,
        elevation: 0.0,
      )
    }
    _ -> Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)
  }
}

/// Simple forward pass using burn NIF
fn forward_weights(
  weights: List(Float),
  inputs: List(Float),
  architecture: List(Int),
) -> List(Float) {
  case burn.batch_forward([weights], [inputs], architecture) {
    Ok([output]) -> output
    Ok(_) -> list.repeat(0.5, 3)
    Error(_) -> list.repeat(0.5, 3)
  }
}

fn weights_to_hrr(weights: List(Float), dim: Int, seed: Int) -> List(Float) {
  let len = list.length(weights)
  case len >= dim {
    True -> list.take(weights, dim)
    False -> {
      let padding = list.range(0, dim - len - 1)
        |> list.map(fn(i) { pseudo_random(seed + i * 17) *. 0.01 })
      list.append(weights, padding)
    }
  }
}

fn init_xavier_weights(count: Int, seed: Int) -> List(Float) {
  let scale = float_sqrt(2.0 /. 8.0)  // Assuming input size 8
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

fn float_clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False -> case x >. max {
      True -> max
      False -> x
    }
  }
}

fn pseudo_random(seed: Int) -> Float {
  let a = 1103515245
  let c = 12345
  let m = 2147483648
  let next = { a * seed + c } % m
  int.to_float(int.absolute_value(next)) /. int.to_float(m)
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

fn float_str(x: Float) -> String {
  let scaled = float.truncate(x *. 100.0)
  let whole = scaled / 100
  let frac = int.absolute_value(scaled % 100)
  let frac_str = case frac < 10 {
    True -> "0" <> int.to_string(frac)
    False -> int.to_string(frac)
  }
  int.to_string(whole) <> "." <> frac_str
}

fn int_list_to_string(lst: List(Int)) -> String {
  "[" <> list.map(lst, int.to_string) |> string_join(", ") <> "]"
}

fn string_join(parts: List(String), sep: String) -> String {
  case parts {
    [] -> ""
    [single] -> single
    [first, ..rest] -> first <> sep <> string_join(rest, sep)
  }
}

fn zip3(a: List(a), b: List(b), c: List(c)) -> List(#(a, b, c)) {
  case a, b, c {
    [a_head, ..a_rest], [b_head, ..b_rest], [c_head, ..c_rest] -> {
      [#(a_head, b_head, c_head), ..zip3(a_rest, b_rest, c_rest)]
    }
    _, _, _ -> []
  }
}

// =============================================================================
// ENTRY POINT
// =============================================================================

/// Main training entry point
pub fn main() {
  let config = default_config()
  let result = train(100, config, 42)

  io.println("")
  io.println("Training complete!")
  io.println("Final best: " <> float_str(result.stats.best_fitness))
}
