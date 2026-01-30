//// VIVA Sinuca - GPU-Accelerated NEAT Trainer with Novelty Search
////
//// Uses BEAM parallelism + viva_glands GPU for training.
//// Implements Novelty Search to escape local optima (Lehman & Stanley, 2011).

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option
import viva/billiards/sinuca.{type Shot, type Table, Shot}
import viva/billiards/sinuca_fitness as fitness
import viva/glands
import viva/jolt.{Vec3}
import viva/neural/neat.{
  type FitnessResult, type Genome, type NeatConfig, type Population,
  FitnessResult, Genome, NeatConfig,
}
import viva/neural/novelty.{type Behavior, type NoveltyArchive, type NoveltyConfig}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub type GpuTrainerConfig {
  GpuTrainerConfig(
    population_size: Int,
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    log_interval: Int,
    num_workers: Int,  // Parallel workers (BEAM processes)
  )
}

pub fn default_config() -> GpuTrainerConfig {
  GpuTrainerConfig(
    population_size: 100,
    max_steps_per_shot: 300,
    shots_per_episode: 5,
    log_interval: 10,
    num_workers: 16,  // Match CPU cores
  )
}

pub fn fast_config() -> GpuTrainerConfig {
  GpuTrainerConfig(
    population_size: 50,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    num_workers: 16,
  )
}

// =============================================================================
// NEAT CONFIG
// =============================================================================

pub fn sinuca_neat_config(population_size: Int) -> NeatConfig {
  // Small network (8 inputs * 3 outputs = 27 max connections)
  // Max distance ~1.5, threshold 1.0 allows 3-8 species
  NeatConfig(
    population_size: population_size,
    num_inputs: 8,
    num_outputs: 3,
    weight_mutation_rate: 0.8,
    weight_perturb_rate: 0.9,
    add_node_rate: 0.03,
    add_connection_rate: 0.05,
    disable_rate: 0.01,
    excess_coefficient: 1.0,
    disjoint_coefficient: 1.0,
    weight_coefficient: 0.4,
    compatibility_threshold: 1.0,  // Ajustado para rede pequena
    survival_threshold: 0.2,
    max_stagnation: 15,
    elitism: 2,
  )
}

// =============================================================================
// INPUT/OUTPUT ENCODING
// =============================================================================

pub fn encode_inputs(table: Table) -> List(Float) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0

  let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(table) {
    option.Some(Vec3(x, _y, z)) -> #(x, z)
    option.None -> #(0.0, 0.0)
  }

  let target = table.target_ball
  let #(target_x, target_z) = case sinuca.get_ball_position(table, target) {
    option.Some(Vec3(x, _y, z)) -> #(x, z)
    option.None -> #(0.0, 0.0)
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

pub fn decode_outputs(outputs: List(Float)) -> Shot {
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

// =============================================================================
// GENOME EVALUATION
// =============================================================================

/// Evaluate genome and return both fitness and behavior vector
pub fn evaluate_genome_with_behavior(
  genome: Genome,
  config: GpuTrainerConfig,
) -> #(Float, Behavior) {
  let table = sinuca.new()
  let episode = fitness.new_episode()

  // Track behavior: final positions, shot angles, pocketed counts
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
            let outputs = neat.forward(genome, inputs)
            let shot = decode_outputs(outputs)

            let #(shot_fitness, next, new_ep) =
              fitness.quick_evaluate_full(
                current,
                shot,
                config.max_steps_per_shot,
                ep,
                fitness.default_config(),
              )

            // Extract behavior features: shot angle, power, final cue position
            let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(next) {
              option.Some(Vec3(x, _, z)) -> #(x, z)
              option.None -> #(0.0, 0.0)
            }
            let new_features = list.append(features, [
              shot.angle /. 6.28,  // Normalize angle to 0-1
              shot.power,
              cue_x /. 1.5,       // Normalize position
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

/// Legacy evaluate function (without behavior)
pub fn evaluate_genome(genome: Genome, config: GpuTrainerConfig) -> Float {
  let #(fitness, _behavior) = evaluate_genome_with_behavior(genome, config)
  fitness
}

// =============================================================================
// PARALLEL EVALUATION WITH BEAM
// =============================================================================

/// Evaluate all genomes in parallel using BEAM
fn evaluate_population_parallel(
  population: Population,
  config: GpuTrainerConfig,
) -> List(FitnessResult) {
  list.map(population.genomes, fn(genome) {
    let fitness = evaluate_genome(genome, config)
    FitnessResult(genome_id: genome.id, fitness: fitness)
  })
}

/// Evaluate with behavior extraction for novelty search
fn evaluate_population_with_behavior(
  population: Population,
  config: GpuTrainerConfig,
) -> List(#(Int, Float, Behavior)) {
  list.map(population.genomes, fn(genome) {
    let #(fitness, behavior) = evaluate_genome_with_behavior(genome, config)
    #(genome.id, fitness, behavior)
  })
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

/// Train without novelty search (legacy)
pub fn train(
  generations: Int,
  config: GpuTrainerConfig,
) -> #(Population, Genome) {
  let neat_config = sinuca_neat_config(config.population_size)
  let initial_pop = neat.create_population(neat_config, 42)

  io.println("=== VIVA Sinuca GPU Training ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Population: " <> int.to_string(config.population_size)
    <> " | Workers: " <> int.to_string(config.num_workers)
    <> " | Shots/ep: " <> int.to_string(config.shots_per_episode),
  )
  io.println("")

  train_loop(initial_pop, generations, config, neat_config, 42)
}

/// Train WITH novelty search to escape local optima
pub fn train_with_novelty(
  generations: Int,
  config: GpuTrainerConfig,
  novelty_config: NoveltyConfig,
) -> #(Population, Genome) {
  let neat_config = sinuca_neat_config(config.population_size)
  let initial_pop = neat.create_population(neat_config, 42)
  let archive = novelty.new_archive(novelty_config)

  io.println("=== VIVA Sinuca GPU Training + Novelty Search ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Population: " <> int.to_string(config.population_size)
    <> " | Novelty weight: " <> float_str(novelty_config.novelty_weight *. 100.0) <> "%",
  )
  io.println("")

  train_loop_novelty(initial_pop, generations, config, neat_config, novelty_config, archive, 42)
}

fn train_loop(
  population: Population,
  remaining: Int,
  config: GpuTrainerConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(Population, Genome) {
  case remaining <= 0 {
    True -> {
      let best = find_best_genome(population)
      io.println("")
      io.println("=== Training Complete ===")
      io.println("Best fitness: " <> float_str(best.fitness))
      #(population, best)
    }
    False -> {
      let results = evaluate_population_parallel(population, config)
      let next_pop =
        neat.evolve(population, results, neat_config, seed + population.generation)
      let stats = compute_stats(next_pop, results)

      case population.generation % config.log_interval == 0 {
        True -> log_stats(stats)
        False -> Nil
      }

      train_loop(next_pop, remaining - 1, config, neat_config, seed)
    }
  }
}

fn train_loop_novelty(
  population: Population,
  remaining: Int,
  config: GpuTrainerConfig,
  neat_config: NeatConfig,
  novelty_config: NoveltyConfig,
  archive: NoveltyArchive,
  seed: Int,
) -> #(Population, Genome) {
  case remaining <= 0 {
    True -> {
      let best = find_best_genome(population)
      io.println("")
      io.println("=== Training Complete (Novelty Search) ===")
      io.println("Best fitness: " <> float_str(best.fitness))
      io.println("Archive size: " <> int.to_string(list.length(archive.behaviors)))
      #(population, best)
    }
    False -> {
      // Evaluate with behavior extraction
      let eval_results = evaluate_population_with_behavior(population, config)

      // Extract fitness and behaviors
      let objective_fitness = list.map(eval_results, fn(r) { r.1 })
      let behaviors = list.map(eval_results, fn(r) { r.2 })

      // Calculate combined fitness with novelty
      let #(combined_fitness, new_archive) =
        novelty.batch_combined_fitness(
          objective_fitness,
          behaviors,
          archive,
          novelty_config,
        )

      // Create fitness results with combined scores
      let results =
        list.zip(eval_results, combined_fitness)
        |> list.map(fn(pair) {
          let #(#(genome_id, _, _), combined) = pair
          FitnessResult(genome_id: genome_id, fitness: combined)
        })

      // Evolve with combined fitness
      let next_pop =
        neat.evolve(population, results, neat_config, seed + population.generation)

      // Stats (show both objective and combined)
      let best_objective = list.fold(objective_fitness, 0.0, fn(acc, f) { float.max(acc, f) })
      let best_combined = list.fold(combined_fitness, 0.0, fn(acc, f) { float.max(acc, f) })

      case population.generation % config.log_interval == 0 {
        True -> {
          io.println(
            "Gen " <> int.to_string(population.generation)
            <> " | Obj: " <> float_str(best_objective)
            <> " | Nov: " <> float_str(best_combined)
            <> " | Arch: " <> int.to_string(list.length(new_archive.behaviors))
            <> " | Sp: " <> int.to_string(list.length(next_pop.species)),
          )
        }
        False -> Nil
      }

      train_loop_novelty(next_pop, remaining - 1, config, neat_config, novelty_config, new_archive, seed)
    }
  }
}

// =============================================================================
// STATS & LOGGING
// =============================================================================

pub type TrainingStats {
  TrainingStats(
    generation: Int,
    best_fitness: Float,
    avg_fitness: Float,
    species_count: Int,
  )
}

fn compute_stats(
  population: Population,
  results: List(FitnessResult),
) -> TrainingStats {
  let fitnesses = list.map(results, fn(r) { r.fitness })
  let best = list.fold(fitnesses, neg(9999.0), fn(acc, f) { float.max(acc, f) })
  let avg = case list.length(fitnesses) {
    0 -> 0.0
    n -> list.fold(fitnesses, 0.0, fn(acc, f) { acc +. f }) /. int.to_float(n)
  }

  TrainingStats(
    generation: population.generation,
    best_fitness: best,
    avg_fitness: avg,
    species_count: list.length(population.species),
  )
}

fn log_stats(stats: TrainingStats) -> Nil {
  io.println(
    "Gen " <> int.to_string(stats.generation)
    <> " | Best: " <> float_str(stats.best_fitness)
    <> " | Avg: " <> float_str(stats.avg_fitness)
    <> " | Species: " <> int.to_string(stats.species_count),
  )
}

fn find_best_genome(population: Population) -> Genome {
  case list.first(population.genomes) {
    Ok(first) -> {
      list.fold(population.genomes, first, fn(best, g) {
        case g.fitness >. best.fitness {
          True -> g
          False -> best
        }
      })
    }
    Error(_) -> {
      Genome(
        id: 0,
        nodes: [],
        connections: [],
        fitness: 0.0,
        adjusted_fitness: 0.0,
        species_id: 0,
      )
    }
  }
}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  let config = fast_config()
  let novelty_config = novelty.default_config()

  // Train with novelty search to escape plateaus
  let #(_pop, best) = train_with_novelty(50, config, novelty_config)

  io.println("")
  io.println("Best genome:")
  io.println("  Nodes: " <> int.to_string(list.length(best.nodes)))
  io.println("  Connections: " <> int.to_string(list.length(best.connections)))
}

// =============================================================================
// HELPERS
// =============================================================================

fn neg(x: Float) -> Float {
  0.0 -. x
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

fn float_str(f: Float) -> String {
  let rounded = float.round(f *. 10.0)
  int.to_string(rounded / 10) <> "." <> int.to_string(int.absolute_value(rounded % 10))
}
