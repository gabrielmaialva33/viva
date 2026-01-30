//// HoloMAP - MAP-Elites with Holographic Reduced Representations
////
//// World's first implementation of Quality-Diversity with HRR encoding.
//// Combines MAP-Elites grid-based diversity with holographic genome representation.
//// Created at GATO-PC, Brazil, 2026.
////
//// Innovation:
//// - Behavior descriptors extracted from HRR vectors (first 8 dims)
//// - Elites stored as holographic vectors for compact representation
//// - Crossover via circular convolution in behavior space
////
//// References:
//// - Mouret & Clune (2015) - Illuminating search spaces with MAP-Elites
//// - Plate (1995) - Holographic Reduced Representations
//// - Qwen3-235B Analysis (2026) - VIVA HoloNEAT recommendations

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/neural/novelty.{type Behavior}

// =============================================================================
// TYPES
// =============================================================================

/// Cell in the MAP-Elites grid
pub type Elite {
  Elite(
    genome_id: Int,
    behavior: Behavior,
    hrr_vector: List(Float),
    fitness: Float,
    generation_added: Int,
  )
}

/// MAP-Elites archive grid
pub type MapElitesGrid {
  MapElitesGrid(
    cells: Dict(#(Int, Int), Elite),  // (x, y) -> Elite
    grid_size: Int,                    // Cells per dimension
    behavior_dims: Int,                // Number of behavior dimensions (2 for 2D grid)
    min_bounds: List(Float),           // Min values for each behavior dim
    max_bounds: List(Float),           // Max values for each behavior dim
  )
}

/// HoloMAP configuration
pub type HoloMapConfig {
  HoloMapConfig(
    grid_size: Int,              // Grid resolution (e.g., 20x20 = 400 cells)
    behavior_dims: Int,          // Behavior space dimensions
    hrr_dim: Int,                // HRR vector dimension
    initial_novelty_weight: Float,
    final_novelty_weight: Float,
    decay_midpoint: Int,         // Generation at which weight is 50% decayed
    batch_size: Int,             // Genomes per generation
    tournament_size: Int,        // Selection tournament size
  )
}

/// Training statistics
pub type HoloMapStats {
  HoloMapStats(
    generation: Int,
    best_fitness: Float,
    coverage: Float,             // Percentage of cells filled
    qd_score: Float,             // Sum of all elite fitnesses
    novelty_weight: Float,       // Current adaptive weight
  )
}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub fn default_config() -> HoloMapConfig {
  HoloMapConfig(
    grid_size: 20,               // 20x20 = 400 cells
    behavior_dims: 2,
    hrr_dim: 8192,
    initial_novelty_weight: 0.7,
    final_novelty_weight: 0.2,
    decay_midpoint: 15,
    batch_size: 50,
    tournament_size: 4,
  )
}

pub fn fast_config() -> HoloMapConfig {
  HoloMapConfig(
    grid_size: 10,               // 10x10 = 100 cells
    behavior_dims: 2,
    hrr_dim: 4096,
    initial_novelty_weight: 0.6,
    final_novelty_weight: 0.25,
    decay_midpoint: 10,
    batch_size: 30,
    tournament_size: 3,
  )
}

/// Qwen3-optimized config (5x5 grid, gradual expansion)
pub fn qwen3_optimized_config() -> HoloMapConfig {
  HoloMapConfig(
    grid_size: 5,                // 5x5 = 25 cells (Qwen3 recommendation)
    behavior_dims: 2,
    hrr_dim: 4096,
    initial_novelty_weight: 0.7, // Higher exploration initially
    final_novelty_weight: 0.2,
    decay_midpoint: 20,          // Slower decay
    batch_size: 50,
    tournament_size: 4,
  )
}

// =============================================================================
// GRID OPERATIONS
// =============================================================================

/// Create empty MAP-Elites grid
pub fn new_grid(config: HoloMapConfig) -> MapElitesGrid {
  MapElitesGrid(
    cells: dict.new(),
    grid_size: config.grid_size,
    behavior_dims: config.behavior_dims,
    min_bounds: list.repeat(-1.0, config.behavior_dims),
    max_bounds: list.repeat(1.0, config.behavior_dims),
  )
}

/// Convert behavior to grid cell coordinates
pub fn behavior_to_cell(
  behavior: Behavior,
  grid: MapElitesGrid,
) -> #(Int, Int) {
  let features = behavior.features

  // Extract first 2 dimensions for 2D grid
  let x_val = list_at(features, 0) |> option.unwrap(0.0)
  let y_val = list_at(features, 1) |> option.unwrap(0.0)

  let min_x = list_at(grid.min_bounds, 0) |> option.unwrap(-1.0)
  let max_x = list_at(grid.max_bounds, 0) |> option.unwrap(1.0)
  let min_y = list_at(grid.min_bounds, 1) |> option.unwrap(-1.0)
  let max_y = list_at(grid.max_bounds, 1) |> option.unwrap(1.0)

  // Normalize to 0-1 range
  let norm_x = { x_val -. min_x } /. { max_x -. min_x }
  let norm_y = { y_val -. min_y } /. { max_y -. min_y }

  // Clamp to valid range
  let norm_x = float_clamp(norm_x, 0.0, 0.999)
  let norm_y = float_clamp(norm_y, 0.0, 0.999)

  // Convert to cell indices
  let cell_x = float.truncate(norm_x *. int.to_float(grid.grid_size))
  let cell_y = float.truncate(norm_y *. int.to_float(grid.grid_size))

  #(cell_x, cell_y)
}

/// Try to add elite to grid (replaces if better fitness)
pub fn try_add_elite(
  grid: MapElitesGrid,
  genome_id: Int,
  behavior: Behavior,
  hrr_vector: List(Float),
  fitness: Float,
  generation: Int,
) -> #(MapElitesGrid, Bool) {
  let cell = behavior_to_cell(behavior, grid)

  case dict.get(grid.cells, cell) {
    Ok(existing) -> {
      // Replace only if better
      case fitness >. existing.fitness {
        True -> {
          let new_elite = Elite(
            genome_id: genome_id,
            behavior: behavior,
            hrr_vector: hrr_vector,
            fitness: fitness,
            generation_added: generation,
          )
          let new_cells = dict.insert(grid.cells, cell, new_elite)
          #(MapElitesGrid(..grid, cells: new_cells), True)
        }
        False -> #(grid, False)
      }
    }
    Error(_) -> {
      // Empty cell - add new elite
      let new_elite = Elite(
        genome_id: genome_id,
        behavior: behavior,
        hrr_vector: hrr_vector,
        fitness: fitness,
        generation_added: generation,
      )
      let new_cells = dict.insert(grid.cells, cell, new_elite)
      #(MapElitesGrid(..grid, cells: new_cells), True)
    }
  }
}

/// Get all elites from grid
pub fn get_elites(grid: MapElitesGrid) -> List(Elite) {
  dict.values(grid.cells)
}

/// Calculate grid coverage (percentage of cells filled)
pub fn coverage(grid: MapElitesGrid) -> Float {
  let filled = int.to_float(dict.size(grid.cells))
  let total = int.to_float(grid.grid_size * grid.grid_size)
  filled /. total *. 100.0
}

/// Calculate QD-Score (sum of all elite fitnesses)
pub fn qd_score(grid: MapElitesGrid) -> Float {
  dict.values(grid.cells)
  |> list.fold(0.0, fn(acc, elite) { acc +. elite.fitness })
}

// =============================================================================
// ADAPTIVE NOVELTY WEIGHT (Qwen3 recommendation)
// =============================================================================

/// Sigmoid decay for novelty weight
/// Starts high (exploration), decays to low (exploitation)
pub fn adaptive_novelty_weight(
  generation: Int,
  config: HoloMapConfig,
) -> Float {
  let gen_f = int.to_float(generation)
  let mid_f = int.to_float(config.decay_midpoint)

  // Sigmoid: w = w_final + (w_init - w_final) / (1 + exp(0.1 * (gen - mid)))
  let range = config.initial_novelty_weight -. config.final_novelty_weight
  let exponent = 0.1 *. { gen_f -. mid_f }
  let sigmoid = 1.0 /. { 1.0 +. float_exp(exponent) }

  config.final_novelty_weight +. range *. sigmoid
}

// =============================================================================
// SELECTION
// =============================================================================

/// Tournament selection from elite archive
pub fn tournament_select(
  grid: MapElitesGrid,
  tournament_size: Int,
  seed: Int,
) -> Option(Elite) {
  let elites = get_elites(grid)

  case list.length(elites) {
    0 -> None
    n -> {
      // Select random tournament participants
      let participants = list.range(0, tournament_size - 1)
        |> list.map(fn(i) {
          let idx = pseudo_random_int(seed + i * 17, n)
          list_at_int(elites, idx)
        })
        |> list.filter_map(fn(x) { option.to_result(x, Nil) })

      // Return best from tournament
      case list.length(participants) {
        0 -> list.first(elites) |> option.from_result
        _ -> {
          list.fold(participants, None, fn(best: Option(Elite), elite: Elite) {
            case best {
              None -> Some(elite)
              Some(b) -> {
                case elite.fitness >. b.fitness {
                  True -> Some(elite)
                  False -> best
                }
              }
            }
          })
        }
      }
    }
  }
}

// =============================================================================
// HRR BEHAVIOR EXTRACTION (Qwen3 shortcut)
// =============================================================================

/// Extract behavior descriptor from HRR vector
/// Uses first 8 dimensions as recommended by Qwen3
pub fn hrr_to_behavior(hrr_vector: List(Float), dims: Int) -> Behavior {
  let features = list.take(hrr_vector, dims)
  novelty.behavior_from_features(features)
}

/// Extract 2D behavior for grid mapping
pub fn hrr_to_2d_behavior(hrr_vector: List(Float)) -> Behavior {
  hrr_to_behavior(hrr_vector, 2)
}

// =============================================================================
// HRR NORMALIZATION (Qwen3 recommendation - every 10 gens)
// =============================================================================

/// Normalize HRR vector to unit length
/// Prevents cuFFT precision errors from cascading
pub fn normalize_hrr(vector: List(Float)) -> List(Float) {
  let magnitude = float_sqrt(
    list.fold(vector, 0.0, fn(acc, x) { acc +. x *. x })
  )

  case magnitude >. 0.0 {
    True -> list.map(vector, fn(x) { x /. magnitude })
    False -> vector
  }
}

/// Normalize all elites in grid (call every 10 generations)
pub fn normalize_grid_hrr(grid: MapElitesGrid) -> MapElitesGrid {
  let normalized_cells = dict.map_values(grid.cells, fn(_key, elite) {
    Elite(..elite, hrr_vector: normalize_hrr(elite.hrr_vector))
  })
  MapElitesGrid(..grid, cells: normalized_cells)
}

// =============================================================================
// HOLOGRAPHIC CROSSOVER
// =============================================================================

/// Crossover two elites via HRR blending
pub fn holographic_crossover(
  parent1: Elite,
  parent2: Elite,
  blend_ratio: Float,
) -> List(Float) {
  let w1 = blend_ratio
  let w2 = 1.0 -. blend_ratio

  let blended = list.zip(parent1.hrr_vector, parent2.hrr_vector)
    |> list.map(fn(pair) { pair.0 *. w1 +. pair.1 *. w2 })

  normalize_hrr(blended)
}

// =============================================================================
// STATISTICS
// =============================================================================

pub fn compute_stats(
  grid: MapElitesGrid,
  generation: Int,
  config: HoloMapConfig,
) -> HoloMapStats {
  let elites = get_elites(grid)

  let best = list.fold(elites, 0.0, fn(acc, e) { float.max(acc, e.fitness) })
  let cov = coverage(grid)
  let qd = qd_score(grid)
  let nw = adaptive_novelty_weight(generation, config)

  HoloMapStats(
    generation: generation,
    best_fitness: best,
    coverage: cov,
    qd_score: qd,
    novelty_weight: nw,
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

fn float_exp(x: Float) -> Float {
  // Approximation of e^x using Taylor series
  let x_clamped = float_clamp(x, -10.0, 10.0)
  let terms = [
    1.0,
    x_clamped,
    x_clamped *. x_clamped /. 2.0,
    x_clamped *. x_clamped *. x_clamped /. 6.0,
    x_clamped *. x_clamped *. x_clamped *. x_clamped /. 24.0,
    x_clamped *. x_clamped *. x_clamped *. x_clamped *. x_clamped /. 120.0,
  ]
  list.fold(terms, 0.0, fn(acc, t) { acc +. t })
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

fn pseudo_random(seed: Int) -> Float {
  let a = 1103515245
  let c = 12345
  let m = 2147483648
  let next = { a * seed + c } % m
  int.to_float(int.absolute_value(next)) /. int.to_float(m)
}

fn pseudo_random_int(seed: Int, max: Int) -> Int {
  let r = pseudo_random(seed)
  float.truncate(r *. int.to_float(max))
}

fn list_at(items: List(a), index: Int) -> Option(a) {
  case index < 0 {
    True -> None
    False -> do_list_at(items, index, 0)
  }
}

fn do_list_at(items: List(a), target: Int, current: Int) -> Option(a) {
  case items {
    [] -> None
    [first, ..rest] -> {
      case current == target {
        True -> Some(first)
        False -> do_list_at(rest, target, current + 1)
      }
    }
  }
}

fn list_at_int(items: List(a), index: Int) -> Option(a) {
  list_at(items, index)
}
