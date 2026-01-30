//// VIVA Sinuca - NEAT Trainer (Fixed)
////
//// Simplified trainer with stable fitness values.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option
import viva/embodied/billiards/sinuca.{type Table, type Shot, Shot}
import viva/embodied/billiards/sinuca_fitness as fitness
import viva/lifecycle/jolt.{Vec3}
import viva/neural/neat.{
  type FitnessResult, type Genome, type NeatConfig, type Population,
  FitnessResult, Genome, NeatConfig,
}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub type TrainerConfig {
  TrainerConfig(
    population_size: Int,
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    log_interval: Int,
    fitness_config: fitness.FitnessConfig,
  )
}

pub fn default_config() -> TrainerConfig {
  TrainerConfig(
    population_size: 100,
    max_steps_per_shot: 300,
    shots_per_episode: 5,
    log_interval: 10,
    fitness_config: fitness.default_config(),
  )
}

pub fn fast_config() -> TrainerConfig {
  TrainerConfig(
    population_size: 50,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    fitness_config: fitness.aggressive_config(),
  )
}

pub fn thorough_config() -> TrainerConfig {
  TrainerConfig(
    population_size: 150,
    max_steps_per_shot: 400,
    shots_per_episode: 8,
    log_interval: 10,
    fitness_config: fitness.positional_config(),
  )
}

pub type TrainingStats {
  TrainingStats(
    generation: Int,
    best_fitness: Float,
    avg_fitness: Float,
    best_genome_id: Int,
    total_pocketed: Int,
    total_fouls: Int,
    best_combo: Int,
    species_count: Int,
  )
}

// =============================================================================
// NEAT CONFIG
// =============================================================================

/// 8 inputs, 3 outputs - NEAT config otimizado para especiação
pub fn sinuca_neat_config(population_size: Int, _use_full: Bool) -> NeatConfig {
  NeatConfig(
    population_size: population_size,
    num_inputs: 8,
    num_outputs: 3,  // angle, power, english
    // Mutação mais agressiva para diversidade
    weight_mutation_rate: 0.9,
    weight_perturb_rate: 0.8,  // 20% chance de reset total
    add_node_rate: 0.05,       // Mais estrutura
    add_connection_rate: 0.08, // Mais conexões
    disable_rate: 0.02,
    // Especiação - threshold baixo para criar espécies
    excess_coefficient: 1.0,
    disjoint_coefficient: 1.0,
    weight_coefficient: 0.5,   // Peso importa mais
    compatibility_threshold: 0.5,  // BAIXO = mais espécies
    // Seleção
    survival_threshold: 0.3,   // Top 30% sobrevive
    max_stagnation: 20,        // Mais paciência
    elitism: 3,                // Preserva top 3
  )
}

// =============================================================================
// INPUT ENCODING (simple 8 inputs)
// =============================================================================

pub fn encode_inputs(table: Table, _use_full: Bool) -> List(Float) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0

  // Cue ball position
  let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(table) {
    option.Some(Vec3(x, _y, z)) -> #(x, z)
    option.None -> #(0.0, 0.0)
  }

  // Target ball position
  let target = table.target_ball
  let #(target_x, target_z) = case sinuca.get_ball_position(table, target) {
    option.Some(Vec3(x, _y, z)) -> #(x, z)
    option.None -> #(0.0, 0.0)
  }

  // Best pocket angle and distance
  let #(pocket_angle, pocket_dist) = fitness.best_pocket_angle(table)

  // Target ball value (normalized 0-1)
  let target_value = int.to_float(sinuca.point_value(target)) /. 7.0

  // Game state (how many balls left, normalized)
  let balls_left = int.to_float(sinuca.balls_on_table(table)) /. 8.0

  // Normalize all inputs to [-1, 1]
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

// =============================================================================
// OUTPUT DECODING
// =============================================================================

pub fn decode_outputs(outputs: List(Float)) -> Shot {
  case outputs {
    [angle_raw, power_raw, english_raw, ..] -> {
      let angle = angle_raw *. 2.0 *. 3.14159
      let power = 0.1 +. power_raw *. 0.9
      let english = { english_raw *. 2.0 -. 1.0 } *. 0.8

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
    _ -> Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)
  }
}

// =============================================================================
// EVALUATION
// =============================================================================

pub fn evaluate_genome(
  genome: Genome,
  config: TrainerConfig,
) -> #(Float, Int, Int, Int) {
  let table = sinuca.new()
  let episode = fitness.new_episode()

  let #(total_fitness, pocketed, fouls, best_combo, final_table, final_episode) =
    list.fold(
      list.range(1, config.shots_per_episode),
      #(0.0, 0, 0, 0, table, episode),
      fn(acc, _shot_num) {
        let #(fitness_sum, total_pocketed, foul_count, max_combo, current, ep) = acc

        case sinuca.balls_on_table(current) <= 1 {
          True -> acc
          False -> {
            let inputs = encode_inputs(current, False)
            let outputs = neat.forward(genome, inputs)
            let shot = decode_outputs(outputs)

            let #(shot_fitness, next, new_ep) =
              fitness.quick_evaluate_full(
                current,
                shot,
                config.max_steps_per_shot,
                ep,
                config.fitness_config,
              )

            let new_fouls = case sinuca.is_scratch(next) {
              True -> foul_count + 1
              False -> foul_count
            }

            let newly_pocketed =
              sinuca.balls_on_table(current) - sinuca.balls_on_table(next)

            let new_max_combo = int.max(max_combo, new_ep.consecutive_pockets)

            let table_next = case sinuca.is_scratch(next) {
              True -> sinuca.reset_cue_ball(next)
              False -> next
            }

            #(
              fitness_sum +. shot_fitness,
              total_pocketed + newly_pocketed,
              new_fouls,
              new_max_combo,
              table_next,
              new_ep,
            )
          }
        }
      },
    )

  let remaining = sinuca.balls_on_table(final_table)
  let ep_bonus = fitness.episode_bonus(final_episode, remaining)

  #(total_fitness +. ep_bonus, pocketed, fouls, best_combo)
}

pub fn evaluate_population(
  population: Population,
  config: TrainerConfig,
) -> #(List(FitnessResult), TrainingStats) {
  let results_with_stats =
    list.map(population.genomes, fn(genome) {
      let #(fit, pocketed, fouls, combo) = evaluate_genome(genome, config)
      #(FitnessResult(genome_id: genome.id, fitness: fit), pocketed, fouls, combo)
    })

  let results = list.map(results_with_stats, fn(r) { r.0 })

  let total_pocketed = list.fold(results_with_stats, 0, fn(acc, r) { acc + r.1 })
  let total_fouls = list.fold(results_with_stats, 0, fn(acc, r) { acc + r.2 })
  let best_combo = list.fold(results_with_stats, 0, fn(acc, r) { int.max(acc, r.3) })

  let pop_size = list.length(results_with_stats)
  let fitnesses = list.map(results, fn(r) { r.fitness })
  let best_fitness = list.fold(fitnesses, neg(9999.0), fn(acc, f) { float.max(acc, f) })
  let avg_fitness = case pop_size {
    0 -> 0.0
    n -> list.fold(fitnesses, 0.0, fn(acc, f) { acc +. f }) /. int.to_float(n)
  }

  let best_id = list.fold(results, 0, fn(acc, r) {
    case r.fitness == best_fitness {
      True -> r.genome_id
      False -> acc
    }
  })

  let stats = TrainingStats(
    generation: population.generation,
    best_fitness: best_fitness,
    avg_fitness: avg_fitness,
    best_genome_id: best_id,
    total_pocketed: total_pocketed,
    total_fouls: total_fouls,
    best_combo: best_combo,
    species_count: list.length(population.species),
  )

  #(results, stats)
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

pub fn train(generations: Int, config: TrainerConfig) -> #(Population, Genome) {
  let neat_config = sinuca_neat_config(config.population_size, False)
  let initial_pop = neat.create_population(neat_config, 42)

  io.println("=== VIVA Sinuca 3D Training ===")
  io.println(
    "Population: " <> int.to_string(config.population_size)
    <> " | 8 inputs | 3 outputs | " <> int.to_string(config.shots_per_episode) <> " shots/ep",
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
      let best = find_best_genome(population)
      #(population, best)
    }
    False -> {
      let #(results, stats) = evaluate_population(population, config)

      case population.generation % config.log_interval == 0 {
        True -> log_stats(stats)
        False -> Nil
      }

      let next_pop = neat.evolve(population, results, neat_config, seed + population.generation)

      train_loop(next_pop, remaining - 1, config, neat_config, seed)
    }
  }
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

fn log_stats(stats: TrainingStats) -> Nil {
  io.println(
    "Gen " <> int.to_string(stats.generation)
    <> " | Best: " <> float_to_str(stats.best_fitness)
    <> " | Avg: " <> float_to_str(stats.avg_fitness)
    <> " | Pocketed: " <> int.to_string(stats.total_pocketed)
    <> " | Fouls: " <> int.to_string(stats.total_fouls)
    <> " | Combo: " <> int.to_string(stats.best_combo)
    <> " | Species: " <> int.to_string(stats.species_count),
  )
}

fn float_to_str(f: Float) -> String {
  let rounded = float.round(f *. 10.0)
  int.to_string(rounded / 10) <> "." <> int.to_string(int.absolute_value(rounded % 10))
}

// =============================================================================
// DEMO
// =============================================================================

pub fn play_game(
  genome: Genome,
  max_shots: Int,
  _use_full: Bool,
) -> List(#(Shot, Int, Bool, Int)) {
  let table = sinuca.new()
  let episode = fitness.new_episode()
  play_loop(genome, table, max_shots, episode, [])
}

fn play_loop(
  genome: Genome,
  table: Table,
  remaining: Int,
  episode: fitness.EpisodeState,
  results: List(#(Shot, Int, Bool, Int)),
) -> List(#(Shot, Int, Bool, Int)) {
  case remaining <= 0 || sinuca.balls_on_table(table) <= 1 {
    True -> list.reverse(results)
    False -> {
      let balls_before = sinuca.balls_on_table(table)
      let inputs = encode_inputs(table, False)
      let outputs = neat.forward(genome, inputs)
      let shot = decode_outputs(outputs)

      let #(_fit, next, new_ep) = fitness.quick_evaluate_full(
        table, shot, 600, episode, fitness.default_config()
      )

      let balls_after = sinuca.balls_on_table(next)
      let pocketed = balls_before - balls_after
      let foul = sinuca.is_scratch(next)
      let combo = new_ep.consecutive_pockets

      let table_next = case foul {
        True -> sinuca.reset_cue_ball(next)
        False -> next
      }

      play_loop(genome, table_next, remaining - 1, new_ep, [
        #(shot, pocketed, foul, combo),
        ..results
      ])
    }
  }
}

pub fn display_game_results(results: List(#(Shot, Int, Bool, Int))) -> Nil {
  io.println("\n=== Game Results ===")
  list.index_map(results, fn(result, i) {
    let #(shot, pocketed, foul, combo) = result
    let foul_str = case foul { True -> " [FOUL]"  False -> "" }
    let combo_str = case combo >= 2 {
      True -> " [COMBO x" <> int.to_string(combo) <> "]"
      False -> ""
    }
    io.println(
      "Shot " <> int.to_string(i + 1)
      <> ": angle=" <> float_to_str(shot.angle)
      <> " power=" <> float_to_str(shot.power)
      <> " -> Pocketed: " <> int.to_string(pocketed)
      <> foul_str <> combo_str,
    )
  })
  Nil
}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  long_training()
}

pub fn long_training() -> Nil {
  io.println("=== VIVA Sinuca Training ===")
  let config = default_config()
  let #(_pop, best) = train(100, config)

  io.println("")
  io.println("Best fitness: " <> float_to_str(best.fitness))
  io.println("Nodes: " <> int.to_string(list.length(best.nodes)))
  io.println("Connections: " <> int.to_string(list.length(best.connections)))

  io.println("")
  let game = play_game(best, 15, False)
  display_game_results(game)
}

pub fn benchmark(generations: Int) -> Nil {
  io.println("=== Benchmark ===")
  let config = fast_config()
  let #(_pop, best) = train(generations, config)

  io.println("")
  io.println("Best: " <> float_to_str(best.fitness))
  let game = play_game(best, 10, False)
  display_game_results(game)
}

pub fn quick_test() -> Nil {
  let config = TrainerConfig(
    population_size: 30,
    max_steps_per_shot: 150,
    shots_per_episode: 3,
    log_interval: 5,
    fitness_config: fitness.default_config(),
  )
  let #(_pop, best) = train(20, config)
  io.println("Best: " <> float_to_str(best.fitness))
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
