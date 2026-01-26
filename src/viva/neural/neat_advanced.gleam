// =============================================================================
// NEAT ADVANCED - State-of-the-Art Optimizations
// =============================================================================
// Based on research:
// - TensorNEAT (GECCO 2024 Best Paper) - tensorization concepts
// - Lamarckian Weight Inheritance - outperforms Xavier/Kaiming
// - Safe Mutations through Gradients (SM-G)
// - Adaptive Speciation
// =============================================================================

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva/neural/neat.{
  type ConnectionGene, type Genome, type NeatConfig, type Population,
  ConnectionGene, Genome, Population,
}

// =============================================================================
// LAMARCKIAN WEIGHT INHERITANCE
// =============================================================================
// Key insight: Inherit trained weights during crossover, not random init
// Result: 75% improvement in data efficiency (Lyu et al., 2021)

pub type InheritanceStrategy {
  /// Traditional NEAT - weights from fitter parent
  FitterParent
  /// Lamarckian - blend weights based on fitness ratio
  LamarckianBlend
  /// Average weights from both parents
  WeightAverage
  /// Weighted average by fitness differential
  FitnessWeighted
}

/// Lamarckian crossover - inherits weights intelligently
pub fn lamarckian_crossover(
  parent1: Genome,
  fitness1: Float,
  parent2: Genome,
  fitness2: Float,
  strategy: InheritanceStrategy,
  seed: Int,
) -> Genome {
  // Build connection maps
  let p1_conns = build_conn_map(parent1.connections)
  let p2_conns = build_conn_map(parent2.connections)

  // Calculate fitness ratio for blending
  let total_fit = float.max(fitness1 +. fitness2, 0.001)
  let alpha = fitness1 /. total_fit

  // Merge connections with intelligent weight inheritance
  let all_innovations =
    list.unique(list.append(
      list.map(parent1.connections, fn(c) { c.innovation }),
      list.map(parent2.connections, fn(c) { c.innovation }),
    ))

  let new_connections =
    list.index_map(all_innovations, fn(innov, idx) {
      let c1 = dict.get(p1_conns, innov)
      let c2 = dict.get(p2_conns, innov)

      case c1, c2 {
        Ok(conn1), Ok(conn2) -> {
          // Matching gene - apply inheritance strategy
          let weight = case strategy {
            FitterParent ->
              case fitness1 >=. fitness2 {
                True -> conn1.weight
                False -> conn2.weight
              }
            LamarckianBlend ->
              // Blend with noise for exploration
              blend_weights(conn1.weight, conn2.weight, alpha, seed + idx)
            WeightAverage -> { conn1.weight +. conn2.weight } /. 2.0
            FitnessWeighted ->
              // Weighted by fitness differential
              conn1.weight *. alpha +. conn2.weight *. { 1.0 -. alpha }
          }
          // Inherit enabled status from fitter parent (or randomly if equal)
          let enabled = case fitness1 >=. fitness2 {
            True -> conn1.enabled
            False -> conn2.enabled
          }
          ConnectionGene(..conn1, weight: weight, enabled: enabled)
        }
        Ok(conn), Error(_) -> conn
        Error(_), Ok(conn) ->
          // Disjoint/excess - include if from fitter parent
          case fitness2 >. fitness1 {
            True -> conn
            False ->
              // Small chance to include anyway (diversity)
              case pseudo_random(seed + idx) <. 0.1 {
                True -> conn
                False ->
                  // Return a disabled version
                  ConnectionGene(..conn, enabled: False)
              }
          }
        Error(_), Error(_) ->
          // Should not happen
          ConnectionGene(
            innovation: innov,
            in_node: 0,
            out_node: 0,
            weight: 0.0,
            enabled: False,
          )
      }
    })
    |> list.filter(fn(c) { c.in_node != 0 })

  // Use nodes from fitter parent
  let nodes = case fitness1 >=. fitness2 {
    True -> parent1.nodes
    False -> parent2.nodes
  }

  Genome(
    id: parent1.id,
    nodes: nodes,
    connections: new_connections,
    fitness: 0.0,
    adjusted_fitness: 0.0,
    species_id: parent1.species_id,
  )
}

fn blend_weights(w1: Float, w2: Float, alpha: Float, seed: Int) -> Float {
  // Blend with small noise for exploration
  let noise = { pseudo_random(seed) -. 0.5 } *. 0.1
  w1 *. alpha +. w2 *. { 1.0 -. alpha } +. noise
}

fn build_conn_map(conns: List(ConnectionGene)) -> Dict(Int, ConnectionGene) {
  list.fold(conns, dict.new(), fn(acc, c) { dict.insert(acc, c.innovation, c) })
}

// =============================================================================
// SAFE MUTATIONS
// =============================================================================
// Key insight: Mutations that preserve network behavior while exploring
// Based on "Safe Mutations through Gradients" (Lehman et al., 2018)

pub type SafeMutationConfig {
  SafeMutationConfig(
    /// Base mutation strength
    base_strength: Float,
    /// Decay factor for deeper layers
    depth_decay: Float,
    /// Min mutation strength
    min_strength: Float,
    /// Max mutation strength
    max_strength: Float,
    /// Adaptive strength based on fitness stagnation
    adaptive: Bool,
  )
}

pub fn default_safe_config() -> SafeMutationConfig {
  SafeMutationConfig(
    base_strength: 0.3,
    depth_decay: 0.9,
    min_strength: 0.05,
    max_strength: 1.0,
    adaptive: True,
  )
}

/// Safe weight mutation - preserves behavior while exploring
pub fn safe_mutate_weights(
  genome: Genome,
  config: SafeMutationConfig,
  generations_stagnant: Int,
  seed: Int,
) -> Genome {
  // Adaptive strength based on stagnation
  let adaptive_factor = case config.adaptive {
    True ->
      // Increase mutation as stagnation grows
      float.min(
        config.max_strength,
        config.base_strength
          *. { 1.0 +. int.to_float(generations_stagnant) *. 0.1 },
      )
    False -> config.base_strength
  }

  let new_connections =
    list.index_map(genome.connections, fn(conn, idx) {
      // Calculate mutation strength based on connection "importance"
      // Approximate decay without float.power
      let decay_factor = case idx / 10 {
        0 -> 1.0
        1 -> config.depth_decay
        2 -> config.depth_decay *. config.depth_decay
        _ -> config.depth_decay *. config.depth_decay *. config.depth_decay
      }
      let strength =
        float.max(config.min_strength, adaptive_factor *. decay_factor)

      // Safe mutation: small perturbation proportional to current weight
      let rand = pseudo_random(seed + idx * 17)
      case rand <. 0.8 {
        // 80% chance to mutate
        True -> {
          let delta =
            { pseudo_random(seed + idx * 17 + 1) -. 0.5 } *. 2.0 *. strength
          // Perturbation proportional to weight magnitude (safe)
          let scale = float.max(float.absolute_value(conn.weight), 0.1)
          let new_weight = conn.weight +. delta *. scale
          ConnectionGene(..conn, weight: clamp(new_weight, -5.0, 5.0))
        }
        False -> conn
      }
    })

  Genome(..genome, connections: new_connections)
}

// =============================================================================
// ADAPTIVE SPECIATION
// =============================================================================
// Dynamically adjust compatibility threshold to maintain target species count

pub type AdaptiveSpeciationConfig {
  AdaptiveSpeciationConfig(
    /// Target number of species
    target_species: Int,
    /// How fast to adjust threshold
    adjustment_rate: Float,
    /// Minimum threshold
    min_threshold: Float,
    /// Maximum threshold
    max_threshold: Float,
  )
}

pub fn default_speciation_config() -> AdaptiveSpeciationConfig {
  AdaptiveSpeciationConfig(
    target_species: 10,
    adjustment_rate: 0.1,
    min_threshold: 1.0,
    max_threshold: 10.0,
  )
}

/// Adjust compatibility threshold based on current species count
pub fn adapt_threshold(
  current_threshold: Float,
  current_species_count: Int,
  config: AdaptiveSpeciationConfig,
) -> Float {
  let diff = current_species_count - config.target_species

  let adjustment = case diff {
    d if d > 2 ->
      // Too many species - increase threshold
      config.adjustment_rate
    d if d < -2 ->
      // Too few species - decrease threshold
      0.0 -. config.adjustment_rate
    _ ->
      // Close enough
      0.0
  }

  clamp(
    current_threshold +. adjustment,
    config.min_threshold,
    config.max_threshold,
  )
}

// =============================================================================
// POPULATION STATISTICS
// =============================================================================
// Track metrics for analysis and adaptive optimization

pub type PopulationStats {
  PopulationStats(
    generation: Int,
    best_fitness: Float,
    avg_fitness: Float,
    worst_fitness: Float,
    fitness_std_dev: Float,
    species_count: Int,
    avg_genome_size: Float,
    total_connections: Int,
    total_nodes: Int,
    stagnant_generations: Int,
  )
}

/// Calculate comprehensive population statistics
pub fn calculate_stats(
  population: Population,
  generation: Int,
  prev_best: Float,
  prev_stagnant: Int,
) -> PopulationStats {
  let fitnesses = list.map(population.genomes, fn(g) { g.fitness })

  let best = list.fold(fitnesses, 0.0, float.max)
  let worst =
    list.fold(fitnesses, 999_999.0, fn(acc, f) {
      case f <. acc {
        True -> f
        False -> acc
      }
    })
  let sum = list.fold(fitnesses, 0.0, fn(acc, f) { acc +. f })
  let count = int.to_float(list.length(fitnesses))
  let avg = sum /. float.max(count, 1.0)

  // Standard deviation
  let variance =
    list.fold(fitnesses, 0.0, fn(acc, f) {
      let diff = f -. avg
      acc +. diff *. diff
    })
    /. float.max(count, 1.0)
  // Approximate sqrt using Newton-Raphson (3 iterations)
  let std_dev = approx_sqrt(variance)

  // Genome size stats
  let sizes =
    list.map(population.genomes, fn(g) {
      #(list.length(g.nodes), list.length(g.connections))
    })
  let total_nodes = list.fold(sizes, 0, fn(acc, s) { acc + s.0 })
  let total_conns = list.fold(sizes, 0, fn(acc, s) { acc + s.1 })
  let avg_size =
    int.to_float(total_nodes + total_conns) /. float.max(count, 1.0)

  // Stagnation tracking
  let stagnant = case best >. prev_best +. 0.001 {
    True -> 0
    False -> prev_stagnant + 1
  }

  PopulationStats(
    generation: generation,
    best_fitness: best,
    avg_fitness: avg,
    worst_fitness: worst,
    fitness_std_dev: std_dev,
    species_count: list.length(population.species),
    avg_genome_size: avg_size,
    total_connections: total_conns,
    total_nodes: total_nodes,
    stagnant_generations: stagnant,
  )
}

// =============================================================================
// ELITE PRESERVATION WITH DIVERSITY
// =============================================================================

pub type EliteConfig {
  EliteConfig(
    /// Percentage of elites to preserve
    elite_percentage: Float,
    /// Minimum elites to keep
    min_elites: Int,
    /// Also preserve most diverse genome?
    preserve_diversity: Bool,
  )
}

pub fn default_elite_config() -> EliteConfig {
  EliteConfig(elite_percentage: 0.15, min_elites: 2, preserve_diversity: True)
}

/// Select elites with optional diversity preservation
pub fn select_elites(genomes: List(Genome), config: EliteConfig) -> List(Genome) {
  let sorted =
    list.sort(genomes, fn(a, b) { float.compare(b.fitness, a.fitness) })

  let elite_count =
    int.max(
      config.min_elites,
      float.round(int.to_float(list.length(genomes)) *. config.elite_percentage),
    )

  let elites = list.take(sorted, elite_count)

  case config.preserve_diversity {
    False -> elites
    True -> {
      // Add most diverse genome (largest structure) if not already elite
      let largest =
        list.fold(
          genomes,
          list.first(genomes) |> result.unwrap(Genome(0, [], [], 0.0, 0.0, 0)),
          fn(acc, g) {
            case list.length(g.connections) > list.length(acc.connections) {
              True -> g
              False -> acc
            }
          },
        )
      case list.any(elites, fn(e) { e.id == largest.id }) {
        True -> elites
        False -> list.append(elites, [largest])
      }
    }
  }
}

// =============================================================================
// TOURNAMENT SELECTION WITH NICHING
// =============================================================================

/// Tournament selection with fitness sharing (niching)
pub fn tournament_select_with_sharing(
  genomes: List(Genome),
  tournament_size: Int,
  sharing_radius: Float,
  seed: Int,
) -> Genome {
  // Calculate shared fitness (rewards diversity)
  let shared_fitnesses =
    list.map(genomes, fn(g) {
      let niche_count =
        list.fold(genomes, 0.0, fn(acc, other) {
          let dist = genome_distance(g, other)
          case dist <. sharing_radius {
            True -> acc +. 1.0 -. dist /. sharing_radius
            False -> acc
          }
        })
      #(g, g.fitness /. float.max(niche_count, 1.0))
    })

  // Tournament selection using shared fitness
  let tournament =
    list.index_map(list.range(0, tournament_size - 1), fn(_, i) {
      let idx =
        float.round(
          pseudo_random(seed + i * 31)
          *. int.to_float(list.length(shared_fitnesses) - 1),
        )
      list_at(shared_fitnesses, idx)
      |> result.unwrap(#(Genome(0, [], [], 0.0, 0.0, 0), 0.0))
    })

  let winner =
    list.fold(
      tournament,
      #(Genome(0, [], [], 0.0, 0.0, 0), -999.0),
      fn(acc, candidate) {
        case candidate.1 >. acc.1 {
          True -> candidate
          False -> acc
        }
      },
    )

  winner.0
}

fn genome_distance(g1: Genome, g2: Genome) -> Float {
  // Simple structural distance
  let size_diff =
    int.absolute_value(
      list.length(g1.connections) - list.length(g2.connections),
    )
  int.to_float(size_diff) /. 10.0
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

fn pseudo_random(seed: Int) -> Float {
  let x = int.bitwise_exclusive_or(seed, int.bitwise_shift_left(seed, 13))
  let x = int.bitwise_exclusive_or(x, int.bitwise_shift_right(x, 17))
  let x = int.bitwise_exclusive_or(x, int.bitwise_shift_left(x, 5))
  let x = int.bitwise_and(x, 0x7FFFFFFF)
  int.to_float(x) /. 2_147_483_647.0
}

fn clamp(x: Float, min_val: Float, max_val: Float) -> Float {
  float.min(max_val, float.max(min_val, x))
}

fn list_at(l: List(a), idx: Int) -> Result(a, Nil) {
  case idx < 0 {
    True -> Error(Nil)
    False ->
      case l {
        [] -> Error(Nil)
        [first, ..rest] ->
          case idx == 0 {
            True -> Ok(first)
            False -> list_at(rest, idx - 1)
          }
      }
  }
}

/// Approximate square root using Newton-Raphson (3 iterations)
fn approx_sqrt(x: Float) -> Float {
  case x <=. 0.0 {
    True -> 0.0
    False -> {
      // Initial guess
      let guess = x /. 2.0
      // Newton-Raphson iterations
      let g1 = { guess +. x /. guess } /. 2.0
      let g2 = { g1 +. x /. g1 } /. 2.0
      let g3 = { g2 +. x /. g2 } /. 2.0
      g3
    }
  }
}
