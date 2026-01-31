//// Novelty Search for NEAT
////
//// Implements behavioral diversity objective to escape local optima.
//// Based on Lehman & Stanley (2011) "Abandoning Objectives"
////
//// Key insight: reward behavioral novelty, not just fitness.
//// This encourages exploration and helps escape fitness plateaus.

import gleam/float
import gleam/int
import gleam/list
import viva/neural/math_ffi

// =============================================================================
// TYPES
// =============================================================================

/// Behavior vector - characterizes what a genome DOES, not its structure
pub type Behavior {
  Behavior(features: List(Float))
}

/// Archive of novel behaviors discovered during evolution
pub type NoveltyArchive {
  NoveltyArchive(
    behaviors: List(Behavior),
    max_size: Int,
    add_threshold: Float,
  )
}

/// Novelty search configuration
pub type NoveltyConfig {
  NoveltyConfig(
    k_nearest: Int,           // Number of neighbors for novelty calc
    archive_threshold: Float, // Min novelty to add to archive
    archive_max_size: Int,    // Max archive size
    novelty_weight: Float,    // Weight of novelty vs fitness (0-1)
  )
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Default novelty search config
pub fn default_config() -> NoveltyConfig {
  NoveltyConfig(
    k_nearest: 15,
    archive_threshold: 0.3,
    archive_max_size: 500,
    novelty_weight: 0.4,  // 40% novelty, 60% fitness
  )
}

/// Exploration-focused config (higher novelty weight)
pub fn exploration_config() -> NoveltyConfig {
  NoveltyConfig(
    k_nearest: 20,
    archive_threshold: 0.2,
    archive_max_size: 1000,
    novelty_weight: 0.6,  // 60% novelty, 40% fitness
  )
}

/// Exploitation-focused config (lower novelty weight)
pub fn exploitation_config() -> NoveltyConfig {
  NoveltyConfig(
    k_nearest: 10,
    archive_threshold: 0.5,
    archive_max_size: 200,
    novelty_weight: 0.2,  // 20% novelty, 80% fitness
  )
}

// =============================================================================
// ARCHIVE OPERATIONS
// =============================================================================

/// Create empty novelty archive
pub fn new_archive(config: NoveltyConfig) -> NoveltyArchive {
  NoveltyArchive(
    behaviors: [],
    max_size: config.archive_max_size,
    add_threshold: config.archive_threshold,
  )
}

/// Add behavior to archive if novel enough
pub fn maybe_add_to_archive(
  archive: NoveltyArchive,
  behavior: Behavior,
  novelty_score: Float,
) -> NoveltyArchive {
  case novelty_score >. archive.add_threshold {
    True -> {
      let new_behaviors = [behavior, ..archive.behaviors]
      // Trim if over max size (keep most recent)
      let trimmed = list.take(new_behaviors, archive.max_size)
      NoveltyArchive(..archive, behaviors: trimmed)
    }
    False -> archive
  }
}

// =============================================================================
// NOVELTY CALCULATION
// =============================================================================

/// Calculate novelty score for a behavior
/// Novelty = average distance to k-nearest neighbors in archive + population
pub fn calculate_novelty(
  behavior: Behavior,
  population_behaviors: List(Behavior),
  archive: NoveltyArchive,
  config: NoveltyConfig,
) -> Float {
  // Combine archive and current population
  let all_behaviors = list.append(archive.behaviors, population_behaviors)

  case list.is_empty(all_behaviors) {
    True -> 1.0  // Max novelty if no comparison points
    False -> {
      // Calculate distances to all other behaviors
      let distances =
        all_behaviors
        |> list.filter(fn(b) { b != behavior })
        |> list.map(fn(other) { behavior_distance(behavior, other) })
        |> list.sort(float.compare)

      // Average of k-nearest distances
      let k = int.min(config.k_nearest, list.length(distances))
      let nearest = list.take(distances, k)

      case list.length(nearest) {
        0 -> 1.0
        n -> list.fold(nearest, 0.0, fn(acc, d) { acc +. d }) /. int.to_float(n)
      }
    }
  }
}

/// Euclidean distance between two behavior vectors
fn behavior_distance(a: Behavior, b: Behavior) -> Float {
  let squared_diffs =
    list.zip(a.features, b.features)
    |> list.map(fn(pair) {
      let #(x, y) = pair
      let diff = x -. y
      diff *. diff
    })

  let sum = list.fold(squared_diffs, 0.0, fn(acc, x) { acc +. x })
  float_sqrt(sum)
}

// =============================================================================
// COMBINED FITNESS
// =============================================================================

/// Combine objective fitness with novelty score
pub fn combined_fitness(
  objective_fitness: Float,
  novelty_score: Float,
  config: NoveltyConfig,
) -> Float {
  let w = config.novelty_weight
  // Normalize novelty: clamp to 0-2 range, then scale to fitness range
  // Typical novelty values are 0-5, fitness is 0-100
  let clamped_novelty = float_min(novelty_score, 2.0)
  let normalized_novelty = clamped_novelty *. 30.0  // Max 60 from novelty

  { 1.0 -. w } *. objective_fitness +. w *. normalized_novelty
}

fn float_min(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}

/// Batch calculate combined fitness for population
pub fn batch_combined_fitness(
  fitness_scores: List(Float),
  behaviors: List(Behavior),
  archive: NoveltyArchive,
  config: NoveltyConfig,
) -> #(List(Float), NoveltyArchive) {
  // Calculate novelty for each behavior
  let novelty_scores =
    list.map(behaviors, fn(b) {
      calculate_novelty(b, behaviors, archive, config)
    })

  // Combine fitness and novelty
  let combined =
    list.zip(fitness_scores, novelty_scores)
    |> list.map(fn(pair) {
      let #(fit, nov) = pair
      combined_fitness(fit, nov, config)
    })

  // Update archive with novel behaviors
  let final_archive =
    list.zip(behaviors, novelty_scores)
    |> list.fold(archive, fn(arch, pair) {
      let #(behavior, novelty) = pair
      maybe_add_to_archive(arch, behavior, novelty)
    })

  #(combined, final_archive)
}

// =============================================================================
// BEHAVIOR EXTRACTION (Generic interface)
// =============================================================================

/// Create behavior from raw feature list
pub fn behavior_from_features(features: List(Float)) -> Behavior {
  Behavior(features: features)
}

/// Get behavior dimension
pub fn behavior_dimension(behavior: Behavior) -> Int {
  list.length(behavior.features)
}

// =============================================================================
// UTILITIES
// =============================================================================

// Use centralized FFI (O(1) vs O(n) Newton-Raphson)
fn float_sqrt(x: Float) -> Float {
  math_ffi.safe_sqrt(x)
}
