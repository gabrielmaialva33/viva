//// VIVA Sinuca QD v13 - Coverage Focus 90%+
////
//// Techniques from AI analysis to maximize coverage:
//// 1. Targeted Objective Relaxation (TOR): Accept any valid individual in empty cells
//// 2. Empty Cell Seeding: Direct sampling toward empty cell coordinates
//// 3. Aggressive Frontier Mutation: 2x mutation for frontier offspring
//// 4. Cell Age Tracking: Prioritize cells empty for many generations
////
//// Target: Coverage 90%+ (currently stuck at 84%)
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
// CONFIGURATION v13 - Coverage focused
// =============================================================================

pub type QDConfig {
  QDConfig(
    grid_size: Int,
    architecture: List(Int),
    population_size: Int,
    perturbation_std: Float,
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    // v13: Coverage params
    novelty_bonus: Float,
    empty_cell_bonus: Float,        // Extra bonus for long-empty cells
    tor_threshold: Int,             // Generations before TOR activates
    frontier_ratio: Float,
    frontier_mutation_mult: Float,  // 2x mutation for frontier
    empty_seed_ratio: Float,        // % of pop targeting empty cells directly
    log_interval: Int,
  )
}

pub fn default_config() -> QDConfig {
  QDConfig(
    grid_size: 10,
    architecture: [8, 32, 16, 3],
    population_size: 400,           // Even larger for coverage
    perturbation_std: 0.05,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    // v13 coverage params
    novelty_bonus: 20.0,            // Higher novelty bonus
    empty_cell_bonus: 10.0,         // Extra for cells empty > tor_threshold
    tor_threshold: 30,              // Activate TOR after 30 gens empty
    frontier_ratio: 0.3,            // 30% from frontier (was 20%)
    frontier_mutation_mult: 2.0,    // 2x mutation for frontier
    empty_seed_ratio: 0.1,          // 10% directly seeded toward empty cells
    log_interval: 5,
  )
}

// =============================================================================
// BEHAVIOR
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

// =============================================================================
// ARCHIVE with Cell Age Tracking
// =============================================================================

pub type Elite {
  Elite(weights: List(Float), fitness: Float, behavior: Behavior, generation: Int)
}

pub type Archive {
  Archive(
    cells: Dict(#(Int, Int), Elite),
    cell_ages: Dict(#(Int, Int), Int),  // v13: Track how long each cell has been empty
    grid_size: Int,
    total_discoveries: Int,
  )
}

fn new_archive(grid_size: Int) -> Archive {
  // Initialize all cells as "empty since gen 0"
  let all_cells = list.flat_map(list.range(0, grid_size - 1), fn(x) {
    list.map(list.range(0, grid_size - 1), fn(y) { #(#(x, y), 0) })
  })
  Archive(
    cells: dict.new(),
    cell_ages: dict.from_list(all_cells),
    grid_size: grid_size,
    total_discoveries: 0,
  )
}

/// v13: TOR - Accept with relaxed fitness if cell empty for too long
fn try_add_elite_tor(
  archive: Archive,
  weights: List(Float),
  fitness: Float,
  behavior: Behavior,
  generation: Int,
  config: QDConfig,
) -> #(Archive, Bool, Bool) {
  let cell = behavior_to_cell(behavior, archive.grid_size)
  let cell_age = dict.get(archive.cell_ages, cell) |> result_or(0)
  let empty_time = generation - cell_age

  case dict.get(archive.cells, cell) {
    Ok(existing) -> {
      // Cell filled - normal comparison
      case fitness >. existing.fitness {
        True -> {
          let elite = Elite(weights:, fitness:, behavior:, generation:)
          let new_cells = dict.insert(archive.cells, cell, elite)
          #(Archive(..archive, cells: new_cells), True, False)
        }
        False -> #(archive, False, False)
      }
    }
    Error(_) -> {
      // Empty cell - apply bonuses!
      let bonus = config.novelty_bonus +. case empty_time > config.tor_threshold {
        True -> config.empty_cell_bonus  // Extra bonus for long-empty
        False -> 0.0
      }
      let elite = Elite(weights:, fitness: fitness +. bonus, behavior:, generation:)
      let new_cells = dict.insert(archive.cells, cell, elite)
      let new_ages = dict.insert(archive.cell_ages, cell, generation)
      #(Archive(
        ..archive,
        cells: new_cells,
        cell_ages: new_ages,
        total_discoveries: archive.total_discoveries + 1,
      ), True, True)
    }
  }
}

fn result_or(r: Result(a, b), default: a) -> a {
  case r {
    Ok(v) -> v
    Error(_) -> default
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
// v13: EMPTY CELL TARGETING
// =============================================================================

/// Get list of empty cells
fn get_empty_cells(archive: Archive) -> List(#(Int, Int)) {
  let filled = dict.keys(archive.cells) |> set.from_list
  list.flat_map(list.range(0, archive.grid_size - 1), fn(x) {
    list.filter_map(list.range(0, archive.grid_size - 1), fn(y) {
      case set.contains(filled, #(x, y)) {
        True -> Error(Nil)
        False -> Ok(#(x, y))
      }
    })
  })
}

/// Get elites adjacent to empty cells (frontier)
fn get_frontier_elites(archive: Archive) -> List(Elite) {
  let filled = dict.keys(archive.cells) |> set.from_list
  dict.to_list(archive.cells)
  |> list.filter(fn(pair) {
    let #(#(x, y), _) = pair
    has_empty_neighbor(x, y, archive.grid_size, filled)
  })
  |> list.map(fn(pair) { pair.1 })
}

fn has_empty_neighbor(x: Int, y: Int, grid: Int, filled: Set(#(Int, Int))) -> Bool {
  let neighbors = [#(x-1,y), #(x+1,y), #(x,y-1), #(x,y+1), #(x-1,y-1), #(x+1,y+1), #(x-1,y+1), #(x+1,y-1)]
  list.any(neighbors, fn(n) {
    let #(nx, ny) = n
    nx >= 0 && nx < grid && ny >= 0 && ny < grid && !set.contains(filled, n)
  })
}

/// v13: Generate weights biased toward a specific cell coordinate
fn generate_biased_weights(
  target_cell: #(Int, Int),
  grid_size: Int,
  weight_count: Int,
  seed: Int,
) -> List(Float) {
  // The first few weights influence behavior descriptors
  // Bias them toward target cell coordinates
  let #(tx, ty) = target_cell
  let target_angle = int.to_float(tx) /. int.to_float(grid_size) +. 0.05  // Center of cell
  let target_scatter = int.to_float(ty) /. int.to_float(grid_size) +. 0.05

  list.range(0, weight_count - 1)
  |> list.map(fn(i) {
    case i {
      // Bias first weights toward target behavior
      0 -> target_angle *. 2.0 -. 1.0 +. small_noise(seed + i)
      1 -> target_scatter *. 2.0 -. 1.0 +. small_noise(seed + i + 100)
      _ -> {
        // Random for rest
        let x = int.to_float({ seed + i * 7919 } % 10000) /. 10000.0
        { x -. 0.5 } *. 2.0
      }
    }
  })
}

fn small_noise(seed: Int) -> Float {
  let x = int.to_float({ seed * 1103515245 + 12345 } % 10000) /. 10000.0
  { x -. 0.5 } *. 0.2  // Small noise ±0.1
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

// =============================================================================
// TRAINING
// =============================================================================

pub type TrainState {
  TrainState(archive: Archive, generation: Int, config: QDConfig)
}

pub fn train(generations: Int) -> TrainState {
  let config = default_config()
  let archive = new_archive(config.grid_size)
  let wc = calc_weights(config.architecture)

  // Initial population
  let pop = list.range(0, config.population_size - 1)
    |> list.map(fn(i) { random_weights(wc, 42 + i * 1000) })

  train_loop(TrainState(archive:, generation: 0, config:), pop, generations, wc)
}

fn train_loop(state: TrainState, pop: List(List(Float)), remaining: Int, wc: Int) -> TrainState {
  case remaining <= 0 {
    True -> state
    False -> {
      let gen = state.generation
      let config = state.config

      // Evaluate
      let results = batch_evaluate(pop, config, gen)

      // Update archive with TOR
      let #(new_archive, new_cells) = list.fold(
        list.zip(pop, results),
        #(state.archive, 0),
        fn(acc, pair) {
          let #(arch, nc) = acc
          let #(w, #(f, b)) = pair
          let #(a2, _, is_new) = try_add_elite_tor(arch, w, f, b, gen, config)
          #(a2, nc + case is_new { True -> 1 False -> 0 })
        },
      )

      // Log
      let cov = coverage(new_archive)
      case gen % config.log_interval == 0 {
        True -> {
          let empty_count = 100 - float.truncate(cov)
          io.println(
            "Gen " <> int.to_string(gen)
            <> " | Best: " <> fmt(best_fitness(new_archive))
            <> " | Cov: " <> fmt(cov) <> "%"
            <> " | Empty: " <> int.to_string(empty_count)
            <> " | QD: " <> fmt(qd_score(new_archive))
            <> case new_cells > 0 { True -> " +" <> int.to_string(new_cells) False -> "" }
          )
        }
        False -> Nil
      }

      // Generate next population with v13 techniques
      let next_pop = generate_next_pop_v13(new_archive, config, gen, wc)

      train_loop(TrainState(..state, archive: new_archive, generation: gen + 1), next_pop, remaining - 1, wc)
    }
  }
}

fn generate_next_pop_v13(archive: Archive, config: QDConfig, gen: Int, wc: Int) -> List(List(Float)) {
  let target = config.population_size
  let empty_cells = get_empty_cells(archive)
  let frontier = get_frontier_elites(archive)
  let all_elites = get_elites(archive)

  // Split population into 3 groups:
  // 1. Empty cell seeding (10%) - directly target empty cells
  // 2. Frontier (30%) - mutate from frontier elites
  // 3. Normal (60%) - mutate from all elites

  let empty_count = float.truncate(int.to_float(target) *. config.empty_seed_ratio)
  let frontier_count = float.truncate(int.to_float(target) *. config.frontier_ratio)
  let normal_count = target - empty_count - frontier_count

  // 1. Empty cell seeding
  let empty_pop = case list.length(empty_cells) {
    0 -> []
    ec -> {
      list.range(0, empty_count - 1)
      |> list.map(fn(i) {
        let cell_idx = i % ec
        let target_cell = case list.drop(empty_cells, cell_idx) |> list.first {
          Ok(c) -> c
          Error(_) -> #(5, 5)
        }
        generate_biased_weights(target_cell, config.grid_size, wc, gen * 5000 + i)
      })
    }
  }

  // 2. Frontier population (high mutation)
  let frontier_pop = case list.length(frontier) {
    0 -> list.range(0, frontier_count - 1) |> list.map(fn(i) { random_weights(wc, gen * 6000 + i) })
    fc -> {
      let fw = list.map(frontier, fn(e) { e.weights })
      list.range(0, frontier_count - 1)
      |> list.map(fn(i) {
        let base = case list.drop(fw, i % fc) |> list.first {
          Ok(w) -> w
          Error(_) -> random_weights(wc, gen + i)
        }
        mutate_weights(base, config.perturbation_std *. config.frontier_mutation_mult, gen * 7000 + i)
      })
    }
  }

  // 3. Normal population
  let normal_pop = case list.length(all_elites) {
    0 -> list.range(0, normal_count - 1) |> list.map(fn(i) { random_weights(wc, gen * 8000 + i) })
    ec -> {
      let ew = list.map(all_elites, fn(e) { e.weights })
      list.range(0, normal_count - 1)
      |> list.map(fn(i) {
        let base = case list.drop(ew, i % ec) |> list.first {
          Ok(w) -> w
          Error(_) -> random_weights(wc, gen + i)
        }
        mutate_weights(base, config.perturbation_std, gen * 9000 + i)
      })
    }
  }

  list.flatten([empty_pop, frontier_pop, normal_pop])
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
  io.println("=== VIVA Sinuca QD v13 - Coverage Focus ===")
  io.println("Target: 90%+ coverage")
  io.println("Techniques: TOR + Empty Cell Seeding + High Frontier Mutation")
  io.println("")

  let state = train(400)  // More generations for coverage

  io.println("")
  io.println("=== Training Complete ===")
  io.println("Best: " <> float.to_string(best_fitness(state.archive)))
  io.println("Coverage: " <> float.to_string(coverage(state.archive)) <> "%")
  io.println("QD-Score: " <> float.to_string(qd_score(state.archive)))
}

// FFI
@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "log")
fn float_ln(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float
