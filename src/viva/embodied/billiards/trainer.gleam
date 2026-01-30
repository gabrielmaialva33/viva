//// VIVA Billiards - NEAT Trainer with Jolt 3D Physics
////
//// Neuroevolution training loop for pool AI.
//// Network learns: angle, power, english from ball positions.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva/embodied/billiards/fitness
import viva/embodied/billiards/table.{type Shot, type Table, Shot}
import viva/lifecycle/jolt.{type Vec3, Vec3}
import viva/neural/neat.{
  type FitnessResult, type Genome, type NeatConfig, type Population,
  FitnessResult, Genome, NeatConfig,
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Training configuration
pub type TrainerConfig {
  TrainerConfig(
    /// NEAT population size
    population_size: Int,
    /// Max simulation steps per shot
    max_steps_per_shot: Int,
    /// Shots per evaluation episode
    shots_per_episode: Int,
    /// Print progress every N generations
    log_interval: Int,
    /// Fitness config
    fitness_config: fitness.FitnessConfig,
  )
}

/// Default trainer config
pub fn default_config() -> TrainerConfig {
  TrainerConfig(
    population_size: 150,
    max_steps_per_shot: 600,
    shots_per_episode: 5,
    log_interval: 10,
    fitness_config: fitness.default_config(),
  )
}

/// Fast training config (smaller pop, fewer shots)
pub fn fast_config() -> TrainerConfig {
  TrainerConfig(
    population_size: 50,
    max_steps_per_shot: 300,
    shots_per_episode: 3,
    log_interval: 5,
    fitness_config: fitness.aggressive_config(),
  )
}

/// Thorough training config
pub fn thorough_config() -> TrainerConfig {
  TrainerConfig(
    population_size: 300,
    max_steps_per_shot: 900,
    shots_per_episode: 10,
    log_interval: 20,
    fitness_config: fitness.default_config(),
  )
}

/// Training statistics
pub type TrainingStats {
  TrainingStats(
    generation: Int,
    best_fitness: Float,
    avg_fitness: Float,
    best_genome_id: Int,
    total_balls_pocketed: Int,
    total_scratches: Int,
    species_count: Int,
  )
}

// =============================================================================
// NEAT CONFIG FOR BILLIARDS
// =============================================================================

/// Create NEAT config for billiards
/// Inputs: 6 (cue ball x,z + nearest target x,z + angle to pocket, distance)
/// Outputs: 3 (angle, power, english)
pub fn billiards_neat_config(population_size: Int) -> NeatConfig {
  NeatConfig(
    population_size: population_size,
    num_inputs: 6,
    num_outputs: 3,
    weight_mutation_rate: 0.8,
    weight_perturb_rate: 0.9,
    add_node_rate: 0.03,
    add_connection_rate: 0.05,
    disable_rate: 0.01,
    excess_coefficient: 1.0,
    disjoint_coefficient: 1.0,
    weight_coefficient: 0.4,
    compatibility_threshold: 3.0,
    survival_threshold: 0.2,
    max_stagnation: 15,
    elitism: 2,
  )
}

// =============================================================================
// INPUT ENCODING
// =============================================================================

/// Encode table state as neural network inputs
/// Returns 6 floats normalized to [-1, 1]
pub fn encode_inputs(t: Table) -> List(Float) {
  let half_length = table.table_length /. 2.0
  let half_width = table.table_width /. 2.0

  // Get cue ball position
  let #(cue_x, cue_z) = case table.get_cue_ball_position(t) {
    Ok(Vec3(x, _y, z)) -> #(x, z)
    Error(_) -> #(0.0, 0.0)
  }

  // Find nearest non-cue ball
  let positions = table.get_all_positions(t)
  let #(target_x, target_z) = find_nearest_ball(cue_x, cue_z, positions)

  // Find best pocket angle and distance
  let #(pocket_angle, pocket_dist) =
    calculate_best_pocket(cue_x, cue_z, target_x, target_z)

  // Normalize all inputs to [-1, 1]
  [
    cue_x /. half_length,
    cue_z /. half_width,
    target_x /. half_length,
    target_z /. half_width,
    pocket_angle /. 3.14159,
    float_clamp(pocket_dist /. 3.0, -1.0, 1.0),
  ]
}

/// Find nearest ball to cue ball
fn find_nearest_ball(
  cue_x: Float,
  cue_z: Float,
  positions: List(#(table.BallType, Vec3)),
) -> #(Float, Float) {
  let result =
    list.fold(positions, #(999.0, 0.0, 0.0), fn(acc, pos) {
      let #(best_dist, _best_x, _best_z) = acc
      let #(ball_type, Vec3(x, _y, z)) = pos
      case ball_type {
        table.CueBall -> acc
        _ -> {
          let dx = x -. cue_x
          let dz = z -. cue_z
          let dist = float_sqrt(dx *. dx +. dz *. dz)
          case dist <. best_dist {
            True -> #(dist, x, z)
            False -> acc
          }
        }
      }
    })

  let #(_dist, x, z) = result
  #(x, z)
}

/// Calculate angle and distance to best pocket through target ball
fn calculate_best_pocket(
  cue_x: Float,
  cue_z: Float,
  target_x: Float,
  target_z: Float,
) -> #(Float, Float) {
  let half_length = table.table_length /. 2.0
  let half_width = table.table_width /. 2.0

  // Corner pockets
  let neg_half_length = 0.0 -. half_length
  let neg_half_width = 0.0 -. half_width
  let pockets = [
    #(neg_half_length, half_width),
    #(half_length, half_width),
    #(neg_half_length, neg_half_width),
    #(half_length, neg_half_width),
    #(0.0, half_width),
    #(0.0, neg_half_width),
  ]

  // Find pocket with best angle (most direct line through target)
  let result =
    list.fold(pockets, #(999.0, 0.0, 999.0), fn(acc, pocket) {
      let #(best_angle_diff, _best_angle, _best_dist) = acc
      let #(px, pz) = pocket

      // Angle from cue to target
      let cue_to_target_angle = float_atan2(target_z -. cue_z, target_x -. cue_x)

      // Angle from target to pocket
      let target_to_pocket_angle = float_atan2(pz -. target_z, px -. target_x)

      // Difference (smaller = more direct)
      let angle_diff = float_abs(cue_to_target_angle -. target_to_pocket_angle)

      // Distance to pocket
      let dx = px -. target_x
      let dz = pz -. target_z
      let dist = float_sqrt(dx *. dx +. dz *. dz)

      case angle_diff <. best_angle_diff {
        True -> #(angle_diff, cue_to_target_angle, dist)
        False -> acc
      }
    })

  let #(_diff, angle, dist) = result
  #(angle, dist)
}

// =============================================================================
// OUTPUT DECODING
// =============================================================================

/// Decode neural network outputs to shot parameters
/// Outputs are assumed to be in [0, 1] range (sigmoid)
pub fn decode_outputs(outputs: List(Float)) -> Shot {
  case outputs {
    [angle_raw, power_raw, english_raw] -> {
      // Angle: full 360 degrees
      let angle = angle_raw *. 2.0 *. 3.14159

      // Power: 0.1 to 1.0 (avoid zero-power shots)
      let power = 0.1 +. power_raw *. 0.9

      // English: -1 to 1
      let english = english_raw *. 2.0 -. 1.0

      Shot(angle: angle, power: power, english: english, elevation: 0.0)
    }
    [angle_raw, power_raw] -> {
      Shot(
        angle: angle_raw *. 2.0 *. 3.14159,
        power: 0.1 +. power_raw *. 0.9,
        english: 0.0,
        elevation: 0.0,
      )
    }
    _ -> {
      // Default shot
      Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)
    }
  }
}

// =============================================================================
// EVALUATION
// =============================================================================

/// Evaluate a single genome over multiple shots
pub fn evaluate_genome(
  genome: Genome,
  config: TrainerConfig,
) -> #(Float, Int, Int) {
  // Create fresh table
  let t = table.new()

  // Play multiple shots
  let #(total_fitness, balls_pocketed, scratches, _final_table) =
    list.fold(
      list.range(1, config.shots_per_episode),
      #(0.0, 0, 0, t),
      fn(acc, _shot_num) {
        let #(fitness_sum, pocketed, scratch_count, current_table) = acc

        // Skip if table is cleared
        case table.balls_on_table(current_table) <= 1 {
          True -> acc
          False -> {
            // Encode state
            let inputs = encode_inputs(current_table)

            // Forward pass
            let outputs = neat.forward(genome, inputs)

            // Decode to shot
            let shot = decode_outputs(outputs)

            // Execute and evaluate
            let #(shot_fitness, next_table) =
              fitness.quick_evaluate(
                current_table,
                shot,
                config.max_steps_per_shot,
              )

            // Track stats
            let new_scratches = case table.is_scratch(next_table) {
              True -> scratch_count + 1
              False -> scratch_count
            }

            let pocketed_now =
              table.balls_on_table(current_table)
              - table.balls_on_table(next_table)

            // Handle scratch - reset cue ball
            let table_for_next = case table.is_scratch(next_table) {
              True -> table.reset_cue_ball(next_table)
              False -> next_table
            }

            #(
              fitness_sum +. shot_fitness,
              pocketed + pocketed_now,
              new_scratches,
              table_for_next,
            )
          }
        }
      },
    )

  #(total_fitness, balls_pocketed, scratches)
}

/// Evaluate entire population
pub fn evaluate_population(
  population: Population,
  config: TrainerConfig,
) -> #(List(FitnessResult), TrainingStats) {
  let results_with_stats =
    list.map(population.genomes, fn(genome) {
      let #(fitness, pocketed, scratches) = evaluate_genome(genome, config)
      #(FitnessResult(genome_id: genome.id, fitness: fitness), pocketed, scratches)
    })

  // Extract results
  let results = list.map(results_with_stats, fn(r) { r.0 })

  // Calculate stats
  let total_pocketed =
    list.fold(results_with_stats, 0, fn(acc, r) { acc + r.1 })
  let total_scratches =
    list.fold(results_with_stats, 0, fn(acc, r) { acc + r.2 })

  let fitnesses = list.map(results, fn(r) { r.fitness })
  let best_fitness =
    list.fold(fitnesses, 0.0, fn(acc, f) { float.max(acc, f) })
  let avg_fitness =
    list.fold(fitnesses, 0.0, fn(acc, f) { acc +. f })
    /. int.to_float(list.length(fitnesses))

  let best_id =
    list.fold(results, 0, fn(acc, r) {
      case r.fitness == best_fitness {
        True -> r.genome_id
        False -> acc
      }
    })

  let stats =
    TrainingStats(
      generation: population.generation,
      best_fitness: best_fitness,
      avg_fitness: avg_fitness,
      best_genome_id: best_id,
      total_balls_pocketed: total_pocketed,
      total_scratches: total_scratches,
      species_count: list.length(population.species),
    )

  #(results, stats)
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

/// Train for N generations
pub fn train(generations: Int, config: TrainerConfig) -> #(Population, Genome) {
  let neat_config = billiards_neat_config(config.population_size)
  let initial_pop = neat.create_population(neat_config, 42)

  io.println("=== VIVA Billiards 3D Training ===")
  io.println(
    "Population: " <> int.to_string(config.population_size)
    <> " | Shots/episode: " <> int.to_string(config.shots_per_episode),
  )
  io.println("")

  let #(final_pop, best_genome) =
    train_loop(initial_pop, generations, config, neat_config, 42)

  io.println("")
  io.println("=== Training Complete ===")

  #(final_pop, best_genome)
}

fn train_loop(
  population: Population,
  remaining: Int,
  config: TrainerConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(Population, Genome) {
  case remaining <= 0 {
    True -> {
      // Return best genome
      let best =
        list.fold(population.genomes, list.first(population.genomes), fn(acc, g) {
          case acc {
            Ok(current) ->
              case g.fitness >. current.fitness {
                True -> Ok(g)
                False -> acc
              }
            Error(_) -> Ok(g)
          }
        })
      case best {
        Ok(b) -> #(population, b)
        Error(_) ->
          #(population, Genome(
            id: 0,
            nodes: [],
            connections: [],
            fitness: 0.0,
            adjusted_fitness: 0.0,
            species_id: 0,
          ))
      }
    }
    False -> {
      // Evaluate population
      let #(results, stats) = evaluate_population(population, config)

      // Log progress
      case population.generation % config.log_interval == 0 {
        True -> log_stats(stats)
        False -> Nil
      }

      // Evolve
      let next_pop = neat.evolve(population, results, neat_config, seed + population.generation)

      train_loop(next_pop, remaining - 1, config, neat_config, seed)
    }
  }
}

fn log_stats(stats: TrainingStats) -> Nil {
  io.println(
    "Gen " <> int.to_string(stats.generation)
    <> " | Best: " <> float.to_string(stats.best_fitness)
    <> " | Avg: " <> float.to_string(stats.avg_fitness)
    <> " | Pocketed: " <> int.to_string(stats.total_balls_pocketed)
    <> " | Scratches: " <> int.to_string(stats.total_scratches)
    <> " | Species: " <> int.to_string(stats.species_count),
  )
}

// =============================================================================
// DEMO / WATCH MODE
// =============================================================================

/// Play a single game with a trained genome and return shot-by-shot results
pub fn play_game(genome: Genome, max_shots: Int) -> List(#(Shot, Int, Bool)) {
  let t = table.new()
  play_game_loop(genome, t, max_shots, [])
}

fn play_game_loop(
  genome: Genome,
  t: Table,
  remaining: Int,
  results: List(#(Shot, Int, Bool)),
) -> List(#(Shot, Int, Bool)) {
  case remaining <= 0 || table.balls_on_table(t) <= 1 {
    True -> list.reverse(results)
    False -> {
      let balls_before = table.balls_on_table(t)
      let inputs = encode_inputs(t)
      let outputs = neat.forward(genome, inputs)
      let shot = decode_outputs(outputs)

      let #(_fitness, next_table) = fitness.quick_evaluate(t, shot, 600)

      let balls_after = table.balls_on_table(next_table)
      let pocketed = balls_before - balls_after
      let scratched = table.is_scratch(next_table)

      let t_next = case scratched {
        True -> table.reset_cue_ball(next_table)
        False -> next_table
      }

      play_game_loop(genome, t_next, remaining - 1, [
        #(shot, pocketed, scratched),
        ..results
      ])
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn float_clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False ->
      case x >. max {
        True -> max
        False -> x
      }
  }
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "atan2")
fn float_atan2(y: Float, x: Float) -> Float

@external(erlang, "erlang", "abs")
fn float_abs(x: Float) -> Float
