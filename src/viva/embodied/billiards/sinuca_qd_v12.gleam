//// VIVA Sinuca QD v12 - Multi-Model AI Improvements
////
//// Based on analysis from Qwen3-235B + Gemini:
//// 1. Adaptive mutation (increases on stagnation, not fixed decay)
//// 2. Frontier targeting (prioritize elites near empty cells)
//// 3. Hill climbing every 10 gens on top elites
//// 4. Larger population (300) + higher novelty bonus (15.0)
////
//// Target: Best=85+, Coverage=90%+
////
//// Created at GATO-PC, Brazil, 2026.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/set.{type Set}
import viva/lifecycle/burn_physics

// =============================================================================
// CONFIGURATION v12
// =============================================================================

pub type QDConfig {
  QDConfig(
    grid_size: Int,
    architecture: List(Int),
    num_perturbations: Int,
    perturbation_std: Float,
    learning_rate: Float,
    population_size: Int,
    elites_per_generation: Int,
    gradient_phase_ratio: Float,
    max_drift: Float,
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    // v12: Improved params
    novelty_bonus: Float,
    restart_interval: Int,
    restart_ratio: Float,
    // v12 NEW: Adaptive mutation
    stagnation_threshold: Int,
    mutation_heat_factor: Float,
    max_mutation_std: Float,
    // v12 NEW: Hill climbing
    hc_interval: Int,
    hc_steps: Int,
    hc_top_k: Int,
    // v12 NEW: Frontier targeting
    frontier_ratio: Float,
    log_interval: Int,
  )
}

pub fn default_config() -> QDConfig {
  QDConfig(
    grid_size: 10,
    architecture: [8, 32, 16, 3],
    num_perturbations: 16,
    perturbation_std: 0.03,         // Start lower
    learning_rate: 0.1,
    population_size: 300,           // Increased from 200
    elites_per_generation: 30,
    gradient_phase_ratio: 0.2,
    max_drift: 0.4,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    // v12 params
    novelty_bonus: 15.0,            // Tripled from 5.0
    restart_interval: 100,          // Less frequent (adaptive handles it)
    restart_ratio: 0.05,            // Smaller restart
    // Adaptive mutation
    stagnation_threshold: 10,       // Gens before heating up
    mutation_heat_factor: 1.3,      // 30% increase per stagnation check
    max_mutation_std: 0.3,          // Cap
    // Hill climbing
    hc_interval: 10,                // Every 10 gens
    hc_steps: 5,                    // 5 micro-steps
    hc_top_k: 5,                    // Top 5 elites
    // Frontier
    frontier_ratio: 0.2,            // 20% from frontier
    log_interval: 5,
  )
}

// =============================================================================
// BEHAVIOR & ARCHIVE (same as v11)
// =============================================================================

pub type Behavior {
  Behavior(hit_angle: Float, scatter_ratio: Float)
}

fn behavior_to_cell(b: Behavior, grid_size: Int) -> #(Int, Int) {
  let x = float.clamp(b.hit_angle, 0.0, 0.999)
  let y = float.clamp(b.scatter_ratio, 0.0, 0.999)
  #(float.truncate(x *. int.to_float(grid_size)),
    float.truncate(y *. int.to_float(grid_size)))
}

pub type Elite {
  Elite(weights: List(Float), fitness: Float, behavior: Behavior, generation: Int)
}

pub type Archive {
  Archive(
    cells: Dict(#(Int, Int), Elite),
    grid_size: Int,
    total_discoveries: Int,
  )
}

fn new_archive(grid_size: Int) -> Archive {
  Archive(cells: dict.new(), grid_size: grid_size, total_discoveries: 0)
}

fn try_add_elite(
  archive: Archive,
  weights: List(Float),
  fitness: Float,
  behavior: Behavior,
  generation: Int,
  novelty_bonus: Float,
) -> #(Archive, Bool, Bool) {
  let cell = behavior_to_cell(behavior, archive.grid_size)
  case dict.get(archive.cells, cell) {
    Ok(existing) -> {
      case fitness >. existing.fitness {
        True -> {
          let elite = Elite(weights:, fitness:, behavior:, generation:)
          #(Archive(..archive, cells: dict.insert(archive.cells, cell, elite)), True, False)
        }
        False -> #(archive, False, False)
      }
    }
    Error(_) -> {
      // New cell discovered!
      let elite = Elite(weights:, fitness: fitness +. novelty_bonus, behavior:, generation:)
      #(Archive(
        ..archive,
        cells: dict.insert(archive.cells, cell, elite),
        total_discoveries: archive.total_discoveries + 1,
      ), True, True)
    }
  }
}

fn coverage(archive: Archive) -> Float {
  int.to_float(dict.size(archive.cells)) /. int.to_float(archive.grid_size * archive.grid_size) *. 100.0
}

fn qd_score(archive: Archive) -> Float {
  dict.values(archive.cells) |> list.fold(0.0, fn(acc, e) { acc +. e.fitness })
}

fn best_fitness(archive: Archive) -> Float {
  dict.values(archive.cells) |> list.fold(0.0, fn(acc, e) { float.max(acc, e.fitness) })
}

fn get_elites(archive: Archive) -> List(Elite) {
  dict.values(archive.cells)
}

// =============================================================================
// v12 NEW: FRONTIER TARGETING
// =============================================================================

fn get_frontier_elites(archive: Archive) -> List(Elite) {
  let filled_cells = dict.keys(archive.cells) |> set.from_list
  let grid = archive.grid_size

  dict.to_list(archive.cells)
  |> list.filter(fn(pair) {
    let #(#(x, y), _elite) = pair
    has_empty_neighbor(x, y, grid, filled_cells)
  })
  |> list.map(fn(pair) { pair.1 })
}

fn has_empty_neighbor(x: Int, y: Int, grid: Int, filled: Set(#(Int, Int))) -> Bool {
  let neighbors = [
    #(x - 1, y), #(x + 1, y), #(x, y - 1), #(x, y + 1),
    #(x - 1, y - 1), #(x + 1, y + 1), #(x - 1, y + 1), #(x + 1, y - 1),
  ]
  list.any(neighbors, fn(n) {
    let #(nx, ny) = n
    nx >= 0 && nx < grid && ny >= 0 && ny < grid && !set.contains(filled, n)
  })
}

// =============================================================================
// v12 NEW: ADAPTIVE MUTATION
// =============================================================================

fn update_adaptive_mutation(
  current_std: Float,
  stagnation: Int,
  current_best: Float,
  last_best: Float,
  config: QDConfig,
) -> #(Float, Int) {
  case current_best >. last_best +. 0.1 {
    True -> {
      // Improvement! Reset
      #(config.perturbation_std, 0)
    }
    False -> {
      let new_stag = stagnation + 1
      case new_stag > config.stagnation_threshold {
        True -> {
          // Heat up!
          let heated = float.min(current_std *. config.mutation_heat_factor, config.max_mutation_std)
          #(heated, new_stag)
        }
        False -> #(current_std, new_stag)
      }
    }
  }
}

// =============================================================================
// v12 NEW: HILL CLIMBING
// =============================================================================

fn hill_climb_elite(
  elite: Elite,
  steps: Int,
  config: QDConfig,
  seed: Int,
) -> Elite {
  hill_climb_loop(elite, steps, config, seed)
}

fn hill_climb_loop(elite: Elite, remaining: Int, config: QDConfig, seed: Int) -> Elite {
  case remaining <= 0 {
    True -> elite
    False -> {
      // Try small perturbation
      let candidate_weights = mutate_weights(elite.weights, 0.01, seed + remaining)
      let results = batch_evaluate([candidate_weights], config, seed)
      case list.first(results) {
        Ok(#(fitness, behavior)) -> {
          case fitness >. elite.fitness {
            True -> {
              let new_elite = Elite(..elite, weights: candidate_weights, fitness: fitness, behavior: behavior)
              hill_climb_loop(new_elite, remaining - 1, config, seed + 100)
            }
            False -> hill_climb_loop(elite, remaining - 1, config, seed + 100)
          }
        }
        Error(_) -> elite
      }
    }
  }
}

// =============================================================================
// EVALUATION & MUTATION
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
    #(r.fitness, Behavior(hit_angle: r.hit_angle, scatter_ratio: r.scatter_ratio))
  })
}

fn mutate_weights(weights: List(Float), std: Float, seed: Int) -> List(Float) {
  list.index_map(weights, fn(w, i) {
    float.clamp(w +. gaussian_noise(seed + i, std), -2.0, 2.0)
  })
}

fn gaussian_noise(seed: Int, std: Float) -> Float {
  let u1 = int.to_float({ seed * 1103515245 + 12345 } % 2147483647) /. 2147483647.0
  let u2 = int.to_float({ seed * 1103515245 * 2 + 12345 } % 2147483647) /. 2147483647.0
  let z = float_sqrt(-2.0 *. float_ln(float.max(u1, 0.0001))) *. float_cos(6.28318 *. u2)
  z *. std
}

fn random_weights(count: Int, seed: Int) -> List(Float) {
  list.range(0, count - 1)
  |> list.map(fn(i) { { int.to_float({ seed + i * 7919 } % 10000) /. 10000.0 -. 0.5 } *. 2.0 })
}

fn generate_random_population(count: Int, weight_count: Int, seed: Int) -> List(List(Float)) {
  list.range(0, count - 1) |> list.map(fn(i) { random_weights(weight_count, seed + i * 1000) })
}

// =============================================================================
// TRAINING STATE
// =============================================================================

pub type TrainState {
  TrainState(
    archive: Archive,
    generation: Int,
    config: QDConfig,
    current_std: Float,
    stagnation_counter: Int,
    last_best_fitness: Float,
  )
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

pub fn train(generations: Int) -> TrainState {
  let config = default_config()
  let archive = new_archive(config.grid_size)
  let weight_count = calc_weights(config.architecture)
  let pop = generate_random_population(config.population_size, weight_count, 42)

  let state = TrainState(
    archive: archive,
    generation: 0,
    config: config,
    current_std: config.perturbation_std,
    stagnation_counter: 0,
    last_best_fitness: 0.0,
  )

  train_loop(state, pop, generations, weight_count)
}

fn train_loop(state: TrainState, pop: List(List(Float)), remaining: Int, wc: Int) -> TrainState {
  case remaining <= 0 {
    True -> state
    False -> {
      let gen = state.generation
      let config = state.config

      // Evaluate
      let results = batch_evaluate(pop, config, gen)

      // Update archive
      let #(new_archive, new_cells) = list.fold(
        list.zip(pop, results),
        #(state.archive, 0),
        fn(acc, pair) {
          let #(arch, nc) = acc
          let #(w, #(f, b)) = pair
          let #(a2, _, is_new) = try_add_elite(arch, w, f, b, gen, config.novelty_bonus)
          #(a2, nc + case is_new { True -> 1 False -> 0 })
        },
      )

      // v12: Adaptive mutation
      let current_best = best_fitness(new_archive)
      let #(new_std, new_stag) = update_adaptive_mutation(
        state.current_std, state.stagnation_counter,
        current_best, state.last_best_fitness, config,
      )

      // v12: Hill climbing every hc_interval
      let final_archive = case gen > 0 && gen % config.hc_interval == 0 {
        True -> {
          let top_elites = get_elites(new_archive)
            |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })
            |> list.take(config.hc_top_k)
          list.index_fold(top_elites, new_archive, fn(arch, elite, i) {
            let improved = hill_climb_elite(elite, config.hc_steps, config, gen * 1000 + i)
            let cell = behavior_to_cell(improved.behavior, arch.grid_size)
            Archive(..arch, cells: dict.insert(arch.cells, cell, improved))
          })
        }
        False -> new_archive
      }

      // Log
      case gen % config.log_interval == 0 {
        True -> {
          let phase = case new_stag > config.stagnation_threshold { True -> "[HEAT]" False -> "[NORM]" }
          let hc = case gen > 0 && gen % config.hc_interval == 0 { True -> "+HC" False -> "" }
          io.println(
            "Gen " <> int.to_string(gen)
            <> " | Best: " <> fmt(best_fitness(final_archive))
            <> " | Cov: " <> fmt(coverage(final_archive)) <> "%"
            <> " | QD: " <> fmt(qd_score(final_archive))
            <> " | σ=" <> fmt(new_std)
            <> " " <> phase <> hc
            <> case new_cells > 0 { True -> " +" <> int.to_string(new_cells) <> "cells" False -> "" }
          )
        }
        False -> Nil
      }

      // Generate next population with frontier targeting
      let elites = get_elites(final_archive)
      let frontier = get_frontier_elites(final_archive)
      let next_pop = generate_next_pop_v12(elites, frontier, config, new_std, gen, wc)

      // Random restart check
      let final_pop = case gen > 0 && gen % config.restart_interval == 0 {
        True -> {
          io.println("[v12] Scheduled restart")
          let keep = list.take(next_pop, config.population_size - float.truncate(int.to_float(config.population_size) *. config.restart_ratio))
          let fresh = generate_random_population(float.truncate(int.to_float(config.population_size) *. config.restart_ratio), wc, gen * 999)
          list.append(keep, fresh)
        }
        False -> next_pop
      }

      let new_state = TrainState(
        ..state,
        archive: final_archive,
        generation: gen + 1,
        current_std: new_std,
        stagnation_counter: new_stag,
        last_best_fitness: current_best,
      )

      train_loop(new_state, final_pop, remaining - 1, wc)
    }
  }
}

fn generate_next_pop_v12(
  elites: List(Elite),
  frontier: List(Elite),
  config: QDConfig,
  std: Float,
  gen: Int,
  wc: Int,
) -> List(List(Float)) {
  let target = config.population_size
  let frontier_count = float.truncate(int.to_float(target) *. config.frontier_ratio)
  let normal_count = target - frontier_count

  // From frontier elites (for coverage)
  let frontier_pop = case list.length(frontier) {
    0 -> generate_random_population(frontier_count, wc, gen)
    fc -> {
      let fw = list.map(frontier, fn(e) { e.weights })
      list.range(0, frontier_count - 1)
      |> list.map(fn(i) {
        let base = case list.drop(fw, i % fc) |> list.first {
          Ok(w) -> w
          Error(_) -> random_weights(wc, gen + i)
        }
        mutate_weights(base, std *. 1.5, gen * 2000 + i)  // Higher mutation for frontier
      })
    }
  }

  // From all elites
  let normal_pop = case list.length(elites) {
    0 -> generate_random_population(normal_count, wc, gen)
    ec -> {
      let ew = list.map(elites, fn(e) { e.weights })
      list.range(0, normal_count - 1)
      |> list.map(fn(i) {
        let base = case list.drop(ew, i % ec) |> list.first {
          Ok(w) -> w
          Error(_) -> random_weights(wc, gen + i)
        }
        mutate_weights(base, std, gen * 3000 + i)
      })
    }
  }

  list.append(frontier_pop, normal_pop)
}

fn calc_weights(arch: List(Int)) -> Int {
  case arch {
    [] | [_] -> 0
    [a, b, ..rest] -> a * b + b + calc_weights([b, ..rest])
  }
}

fn fmt(x: Float) -> String {
  let t = float.truncate(x *. 10.0)
  int.to_string(t / 10) <> "." <> int.to_string(int.absolute_value(t % 10))
}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("=== VIVA Sinuca QD v12 ===")
  io.println("Multi-AI Improvements: Adaptive mutation + Frontier targeting + Hill climbing")
  io.println("")

  let state = train(300)

  io.println("")
  io.println("=== Training Complete ===")
  io.println("Best: " <> float.to_string(best_fitness(state.archive)))
  io.println("Coverage: " <> float.to_string(coverage(state.archive)) <> "%")
  io.println("QD-Score: " <> float.to_string(qd_score(state.archive)))
  io.println("Discoveries: " <> int.to_string(state.archive.total_discoveries))
}

// FFI
@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "log")
fn float_ln(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float
