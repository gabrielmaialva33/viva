//// VIVA Sinuca QD - Quality-Diversity Training
////
//// Consolidated QD-MAP-Elites implementation with all optimizations:
//// - Adaptive mutation (heats up on stagnation)
//// - Hill climbing on top elites
//// - Frontier targeting for coverage
//// - TOR (Targeted Objective Relaxation) for empty cells
//// - Empty cell seeding
////
//// Results: Best=86.2, Coverage=86%, QD-Score=3941
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
// CONFIGURATION
// =============================================================================

pub type QDConfig {
  QDConfig(
    grid_size: Int,
    architecture: List(Int),
    population_size: Int,
    perturbation_std: Float,
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    // Novelty & Coverage
    novelty_bonus: Float,
    empty_cell_bonus: Float,
    tor_threshold: Int,
    frontier_ratio: Float,
    frontier_mutation_mult: Float,
    empty_seed_ratio: Float,
    // Adaptive mutation
    stagnation_threshold: Int,
    mutation_heat_factor: Float,
    max_mutation_std: Float,
    // Hill climbing
    hc_interval: Int,
    hc_steps: Int,
    hc_top_k: Int,
    // Restart
    restart_interval: Int,
    restart_ratio: Float,
    log_interval: Int,
  )
}

pub fn default_config() -> QDConfig {
  QDConfig(
    grid_size: 10,
    architecture: [8, 32, 16, 3],
    population_size: 300,
    perturbation_std: 0.03,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    // Coverage
    novelty_bonus: 15.0,
    empty_cell_bonus: 10.0,
    tor_threshold: 30,
    frontier_ratio: 0.25,
    frontier_mutation_mult: 2.0,
    empty_seed_ratio: 0.1,
    // Adaptive
    stagnation_threshold: 10,
    mutation_heat_factor: 1.3,
    max_mutation_std: 0.3,
    // Hill climbing
    hc_interval: 10,
    hc_steps: 5,
    hc_top_k: 5,
    // Restart
    restart_interval: 100,
    restart_ratio: 0.05,
    log_interval: 5,
  )
}

pub fn fast_config() -> QDConfig {
  QDConfig(..default_config(), population_size: 100, log_interval: 10)
}

// =============================================================================
// TYPES
// =============================================================================

pub type Behavior {
  Behavior(hit_angle: Float, scatter_ratio: Float)
}

pub type Elite {
  Elite(weights: List(Float), fitness: Float, behavior: Behavior, generation: Int)
}

pub type Archive {
  Archive(
    cells: Dict(#(Int, Int), Elite),
    cell_ages: Dict(#(Int, Int), Int),
    grid_size: Int,
    discoveries: Int,
  )
}

pub type TrainState {
  TrainState(
    archive: Archive,
    generation: Int,
    config: QDConfig,
    current_std: Float,
    stagnation: Int,
    last_best: Float,
  )
}

pub type TrainResult {
  TrainResult(
    best_fitness: Float,
    coverage: Float,
    qd_score: Float,
    generations: Int,
    discoveries: Int,
  )
}

// =============================================================================
// ARCHIVE
// =============================================================================

fn new_archive(grid_size: Int) -> Archive {
  let ages = list.flat_map(list.range(0, grid_size - 1), fn(x) {
    list.map(list.range(0, grid_size - 1), fn(y) { #(#(x, y), 0) })
  }) |> dict.from_list
  Archive(cells: dict.new(), cell_ages: ages, grid_size: grid_size, discoveries: 0)
}

fn behavior_to_cell(b: Behavior, grid: Int) -> #(Int, Int) {
  #(float.truncate(float.clamp(b.hit_angle, 0.0, 0.999) *. int.to_float(grid)),
    float.truncate(float.clamp(b.scatter_ratio, 0.0, 0.999) *. int.to_float(grid)))
}

fn try_add_elite(
  archive: Archive,
  weights: List(Float),
  fitness: Float,
  behavior: Behavior,
  gen: Int,
  config: QDConfig,
) -> #(Archive, Bool, Bool) {
  let cell = behavior_to_cell(behavior, archive.grid_size)
  let cell_age = dict.get(archive.cell_ages, cell) |> result_unwrap(0)
  let empty_time = gen - cell_age

  case dict.get(archive.cells, cell) {
    Ok(existing) -> {
      case fitness >. existing.fitness {
        True -> {
          let e = Elite(weights:, fitness:, behavior:, generation: gen)
          #(Archive(..archive, cells: dict.insert(archive.cells, cell, e)), True, False)
        }
        False -> #(archive, False, False)
      }
    }
    Error(_) -> {
      let bonus = config.novelty_bonus +. case empty_time > config.tor_threshold {
        True -> config.empty_cell_bonus
        False -> 0.0
      }
      let e = Elite(weights:, fitness: fitness +. bonus, behavior:, generation: gen)
      #(Archive(
        ..archive,
        cells: dict.insert(archive.cells, cell, e),
        cell_ages: dict.insert(archive.cell_ages, cell, gen),
        discoveries: archive.discoveries + 1,
      ), True, True)
    }
  }
}

fn result_unwrap(r: Result(a, b), default: a) -> a {
  case r { Ok(v) -> v Error(_) -> default }
}

pub fn coverage(archive: Archive) -> Float {
  int.to_float(dict.size(archive.cells)) /. int.to_float(archive.grid_size * archive.grid_size) *. 100.0
}

pub fn qd_score(archive: Archive) -> Float {
  dict.values(archive.cells) |> list.fold(0.0, fn(acc, e) { acc +. e.fitness })
}

pub fn best_fitness(archive: Archive) -> Float {
  dict.values(archive.cells) |> list.fold(0.0, fn(acc, e) { float.max(acc, e.fitness) })
}

fn get_elites(archive: Archive) -> List(Elite) {
  dict.values(archive.cells)
}

fn get_empty_cells(archive: Archive) -> List(#(Int, Int)) {
  let filled = dict.keys(archive.cells) |> set.from_list
  list.flat_map(list.range(0, archive.grid_size - 1), fn(x) {
    list.filter_map(list.range(0, archive.grid_size - 1), fn(y) {
      case set.contains(filled, #(x, y)) { True -> Error(Nil) False -> Ok(#(x, y)) }
    })
  })
}

fn get_frontier_elites(archive: Archive) -> List(Elite) {
  let filled = dict.keys(archive.cells) |> set.from_list
  dict.to_list(archive.cells)
  |> list.filter(fn(p) {
    let #(#(x, y), _) = p
    has_empty_neighbor(x, y, archive.grid_size, filled)
  })
  |> list.map(fn(p) { p.1 })
}

fn has_empty_neighbor(x: Int, y: Int, grid: Int, filled: Set(#(Int, Int))) -> Bool {
  [#(x-1,y), #(x+1,y), #(x,y-1), #(x,y+1), #(x-1,y-1), #(x+1,y+1), #(x-1,y+1), #(x+1,y-1)]
  |> list.any(fn(n) {
    let #(nx, ny) = n
    nx >= 0 && nx < grid && ny >= 0 && ny < grid && !set.contains(filled, n)
  })
}

// =============================================================================
// ADAPTIVE MUTATION
// =============================================================================

fn update_mutation(std: Float, stag: Int, best: Float, last: Float, cfg: QDConfig) -> #(Float, Int) {
  case best >. last +. 0.1 {
    True -> #(cfg.perturbation_std, 0)
    False -> {
      let new_stag = stag + 1
      case new_stag > cfg.stagnation_threshold {
        True -> #(float.min(std *. cfg.mutation_heat_factor, cfg.max_mutation_std), new_stag)
        False -> #(std, new_stag)
      }
    }
  }
}

// =============================================================================
// HILL CLIMBING
// =============================================================================

fn hill_climb(elite: Elite, steps: Int, cfg: QDConfig, seed: Int) -> Elite {
  case steps <= 0 {
    True -> elite
    False -> {
      let cand = mutate_weights(elite.weights, 0.01, seed + steps)
      let results = batch_evaluate([cand], cfg, seed)
      case list.first(results) {
        Ok(#(f, b)) if f >. elite.fitness ->
          hill_climb(Elite(..elite, weights: cand, fitness: f, behavior: b), steps - 1, cfg, seed + 100)
        _ -> hill_climb(elite, steps - 1, cfg, seed + 100)
      }
    }
  }
}

// =============================================================================
// EVALUATION & MUTATION
// =============================================================================

fn batch_evaluate(weights: List(List(Float)), cfg: QDConfig, _seed: Int) -> List(#(Float, Behavior)) {
  burn_physics.evaluate_episodes(weights, cfg.architecture, cfg.shots_per_episode, cfg.max_steps_per_shot)
  |> list.map(fn(r) { #(r.fitness, Behavior(hit_angle: r.hit_angle, scatter_ratio: r.scatter_ratio)) })
}

fn mutate_weights(weights: List(Float), std: Float, seed: Int) -> List(Float) {
  list.index_map(weights, fn(w, i) { float.clamp(w +. gaussian(seed + i, std), -2.0, 2.0) })
}

fn gaussian(seed: Int, std: Float) -> Float {
  let u1 = int.to_float({ seed * 1103515245 + 12345 } % 2147483647) /. 2147483647.0
  let u2 = int.to_float({ seed * 1103515245 * 2 + 12345 } % 2147483647) /. 2147483647.0
  float_sqrt(-2.0 *. float_ln(float.max(u1, 0.0001))) *. float_cos(6.28318 *. u2) *. std
}

fn random_weights(count: Int, seed: Int) -> List(Float) {
  list.range(0, count - 1)
  |> list.map(fn(i) { { int.to_float({ seed + i * 7919 } % 10000) /. 10000.0 -. 0.5 } *. 2.0 })
}

fn biased_weights(cell: #(Int, Int), grid: Int, count: Int, seed: Int) -> List(Float) {
  let #(tx, ty) = cell
  list.range(0, count - 1) |> list.map(fn(i) {
    case i {
      0 -> { int.to_float(tx) /. int.to_float(grid) +. 0.05 } *. 2.0 -. 1.0 +. small_noise(seed + i)
      1 -> { int.to_float(ty) /. int.to_float(grid) +. 0.05 } *. 2.0 -. 1.0 +. small_noise(seed + i + 100)
      _ -> { int.to_float({ seed + i * 7919 } % 10000) /. 10000.0 -. 0.5 } *. 2.0
    }
  })
}

fn small_noise(seed: Int) -> Float {
  { int.to_float({ seed * 1103515245 + 12345 } % 10000) /. 10000.0 -. 0.5 } *. 0.2
}

fn calc_weights(arch: List(Int)) -> Int {
  case arch { [] | [_] -> 0  [a, b, ..rest] -> a * b + b + calc_weights([b, ..rest]) }
}

// =============================================================================
// TRAINING
// =============================================================================

pub fn train(generations: Int) -> TrainResult {
  train_with_config(generations, default_config())
}

pub fn train_with_config(generations: Int, config: QDConfig) -> TrainResult {
  let wc = calc_weights(config.architecture)
  let pop = list.range(0, config.population_size - 1) |> list.map(fn(i) { random_weights(wc, 42 + i * 1000) })
  let state = TrainState(
    archive: new_archive(config.grid_size),
    generation: 0, config: config,
    current_std: config.perturbation_std, stagnation: 0, last_best: 0.0,
  )
  let final = train_loop(state, pop, generations, wc)
  TrainResult(
    best_fitness: best_fitness(final.archive),
    coverage: coverage(final.archive),
    qd_score: qd_score(final.archive),
    generations: final.generation,
    discoveries: final.archive.discoveries,
  )
}

fn train_loop(state: TrainState, pop: List(List(Float)), remaining: Int, wc: Int) -> TrainState {
  case remaining <= 0 {
    True -> state
    False -> {
      let gen = state.generation
      let cfg = state.config

      // Evaluate & update archive
      let results = batch_evaluate(pop, cfg, gen)
      let #(arch1, new_cells) = list.fold(list.zip(pop, results), #(state.archive, 0), fn(acc, p) {
        let #(a, nc) = acc
        let #(w, #(f, b)) = p
        let #(a2, _, is_new) = try_add_elite(a, w, f, b, gen, cfg)
        #(a2, nc + case is_new { True -> 1 False -> 0 })
      })

      // Adaptive mutation
      let best = best_fitness(arch1)
      let #(new_std, new_stag) = update_mutation(state.current_std, state.stagnation, best, state.last_best, cfg)

      // Hill climbing
      let arch2 = case gen > 0 && gen % cfg.hc_interval == 0 {
        True -> {
          get_elites(arch1)
          |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })
          |> list.take(cfg.hc_top_k)
          |> list.index_fold(arch1, fn(a, e, i) {
            let improved = hill_climb(e, cfg.hc_steps, cfg, gen * 1000 + i)
            Archive(..a, cells: dict.insert(a.cells, behavior_to_cell(improved.behavior, a.grid_size), improved))
          })
        }
        False -> arch1
      }

      // Log
      case gen % cfg.log_interval == 0 {
        True -> {
          let heat = case new_stag > cfg.stagnation_threshold { True -> "[HEAT]" False -> "" }
          let hc = case gen > 0 && gen % cfg.hc_interval == 0 { True -> "+HC" False -> "" }
          let cells = case new_cells > 0 { True -> " +" <> int.to_string(new_cells) False -> "" }
          io.println("Gen " <> int.to_string(gen) <> " | Best: " <> fmt(best_fitness(arch2))
            <> " | Cov: " <> fmt(coverage(arch2)) <> "%" <> " | QD: " <> fmt(qd_score(arch2))
            <> " " <> heat <> hc <> cells)
        }
        False -> Nil
      }

      // Generate next population
      let next = generate_population(arch2, cfg, new_std, gen, wc)

      // Random restart
      let final_pop = case gen > 0 && gen % cfg.restart_interval == 0 {
        True -> {
          io.println("[RESTART]")
          let keep = float.truncate(int.to_float(cfg.population_size) *. { 1.0 -. cfg.restart_ratio })
          let fresh = cfg.population_size - keep
          list.append(list.take(next, keep), list.range(0, fresh - 1) |> list.map(fn(i) { random_weights(wc, gen * 999 + i) }))
        }
        False -> next
      }

      train_loop(TrainState(..state, archive: arch2, generation: gen + 1, current_std: new_std, stagnation: new_stag, last_best: best), final_pop, remaining - 1, wc)
    }
  }
}

fn generate_population(archive: Archive, cfg: QDConfig, std: Float, gen: Int, wc: Int) -> List(List(Float)) {
  let target = cfg.population_size
  let empty = get_empty_cells(archive)
  let frontier = get_frontier_elites(archive)
  let elites = get_elites(archive)

  let empty_n = float.truncate(int.to_float(target) *. cfg.empty_seed_ratio)
  let front_n = float.truncate(int.to_float(target) *. cfg.frontier_ratio)
  let normal_n = target - empty_n - front_n

  let empty_pop = case list.length(empty) {
    0 -> []
    ec -> list.range(0, empty_n - 1) |> list.map(fn(i) {
      let cell = list.drop(empty, i % ec) |> list.first |> result_unwrap(#(5, 5))
      biased_weights(cell, cfg.grid_size, wc, gen * 5000 + i)
    })
  }

  let front_pop = case list.length(frontier) {
    0 -> list.range(0, front_n - 1) |> list.map(fn(i) { random_weights(wc, gen * 6000 + i) })
    fc -> {
      let fw = list.map(frontier, fn(e) { e.weights })
      list.range(0, front_n - 1) |> list.map(fn(i) {
        let base = list.drop(fw, i % fc) |> list.first |> result_unwrap(random_weights(wc, gen + i))
        mutate_weights(base, std *. cfg.frontier_mutation_mult, gen * 7000 + i)
      })
    }
  }

  let normal_pop = case list.length(elites) {
    0 -> list.range(0, normal_n - 1) |> list.map(fn(i) { random_weights(wc, gen * 8000 + i) })
    ec -> {
      let ew = list.map(elites, fn(e) { e.weights })
      list.range(0, normal_n - 1) |> list.map(fn(i) {
        let base = list.drop(ew, i % ec) |> list.first |> result_unwrap(random_weights(wc, gen + i))
        mutate_weights(base, std, gen * 9000 + i)
      })
    }
  }

  list.flatten([empty_pop, front_pop, normal_pop])
}

fn fmt(x: Float) -> String {
  let t = float.truncate(x *. 10.0)
  int.to_string(t / 10) <> "." <> int.to_string(int.absolute_value(t % 10))
}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("=== VIVA Sinuca QD Training ===")
  io.println("")
  let result = train(300)
  io.println("")
  io.println("=== Complete ===")
  io.println("Best Fitness: " <> float.to_string(result.best_fitness))
  io.println("Coverage: " <> float.to_string(result.coverage) <> "%")
  io.println("QD-Score: " <> float.to_string(result.qd_score))
}

// FFI
@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "log")
fn float_ln(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float
