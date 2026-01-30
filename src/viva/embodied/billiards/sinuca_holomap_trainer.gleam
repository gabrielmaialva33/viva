//// VIVA Sinuca - HoloMAP Trainer v2
////
//// Hybrid Quality-Diversity: MAP-Elites grid + NEAT evolution.
//// Grid tracks elites for diversity, NEAT handles actual genome evolution.
//// Created at GATO-PC, Brazil, 2026.
////
//// Key insight: Use MAP-Elites for SELECTION, not genome reconstruction.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/embodied/billiards/sinuca.{type Shot, type Table, Shot}
import viva/embodied/billiards/sinuca_fitness as fitness
import viva/soul/glands
import viva/lifecycle/jolt.{Vec3}
import viva/neural/neat.{type Genome, type NeatConfig, Genome, NeatConfig}
import viva/neural/holomap.{type HoloMapConfig, type HoloMapStats, type MapElitesGrid}
import viva/neural/novelty.{type Behavior}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub type TrainerConfig {
  TrainerConfig(
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    log_interval: Int,
    elite_selection_ratio: Float,
  )
}

pub fn default_trainer_config() -> TrainerConfig {
  TrainerConfig(
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    elite_selection_ratio: 0.5,
  )
}

fn sinuca_neat_config() -> NeatConfig {
  NeatConfig(
    population_size: 50,
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
    compatibility_threshold: 1.0,
    survival_threshold: 0.2,
    max_stagnation: 15,
    elitism: 2,
  )
}

// =============================================================================
// GENOME EVALUATION
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

/// Evaluate genome and extract behavior for MAP-Elites
fn evaluate_genome_with_behavior(
  genome: Genome,
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

            // Extract behavior features
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
// GENOME TO HRR ENCODING (simplified - uses genome weights)
// =============================================================================

fn genome_to_hrr(genome: Genome, dim: Int, seed: Int) -> List(Float) {
  let weight_features = list.map(genome.connections, fn(c) { c.weight })
  let node_count = int.to_float(list.length(genome.nodes)) /. 20.0
  let conn_count = int.to_float(list.length(genome.connections)) /. 50.0

  let features = list.flatten([
    weight_features,
    [node_count, conn_count, int.to_float(seed % 100) /. 100.0],
  ])

  pad_to_dim(features, dim, seed)
}

fn pad_to_dim(features: List(Float), dim: Int, seed: Int) -> List(Float) {
  let len = list.length(features)
  case len >= dim {
    True -> list.take(features, dim)
    False -> {
      let padding = list.range(0, dim - len - 1)
        |> list.map(fn(i) { pseudo_random(seed + i * 17) *. 0.01 })
      list.append(features, padding)
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

fn float_str(x: Float) -> String {
  let scaled = float.truncate(x *. 10.0)
  let whole = scaled / 10
  let frac = int.absolute_value(scaled % 10)
  int.to_string(whole) <> "." <> int.to_string(frac)
}

// =============================================================================
// MAIN TRAINING
// =============================================================================

pub fn main() {
  let trainer_config = default_trainer_config()
  let holomap_config = holomap.fast_config()

  let #(_grid, stats) = train(50, holomap_config, trainer_config)

  io.println("")
  io.println("Final Stats:")
  io.println("  Best Fitness: " <> float_str(stats.best_fitness))
  io.println("  Coverage: " <> float_str(stats.coverage) <> "%")
  io.println("  QD-Score: " <> float_str(stats.qd_score))
}

pub fn train(
  generations: Int,
  holomap_config: HoloMapConfig,
  trainer_config: TrainerConfig,
) -> #(MapElitesGrid, HoloMapStats) {
  let neat_config = sinuca_neat_config()

  io.println("=== VIVA Sinuca HoloMAP v2 Training ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Grid: " <> int.to_string(holomap_config.grid_size) <> "x"
    <> int.to_string(holomap_config.grid_size)
    <> " | Pop: " <> int.to_string(neat_config.population_size)
    <> " | Elite ratio: " <> float_str(trainer_config.elite_selection_ratio *. 100.0) <> "%",
  )
  io.println("")

  // Initialize
  let grid = holomap.new_grid(holomap_config)
  let population = neat.create_population(neat_config, 42)

  // Evaluate and seed grid with initial population
  let #(grid_seeded, pop_evaluated) = evaluate_and_update_grid(
    grid, population.genomes, trainer_config, holomap_config, 0
  )

  // Main evolution loop
  train_loop(
    grid_seeded,
    pop_evaluated,
    population,
    generations,
    0,
    holomap_config,
    trainer_config,
    neat_config,
    42,
  )
}

/// Evaluate genomes and update MAP-Elites grid
fn evaluate_and_update_grid(
  grid: MapElitesGrid,
  genomes: List(Genome),
  trainer_config: TrainerConfig,
  holomap_config: HoloMapConfig,
  generation: Int,
) -> #(MapElitesGrid, List(#(Genome, Float, Behavior))) {
  let evaluated = list.map(genomes, fn(genome) {
    let #(fitness, behavior) = evaluate_genome_with_behavior(genome, trainer_config)
    #(genome, fitness, behavior)
  })

  let new_grid = list.fold(evaluated, grid, fn(g, item) {
    let #(genome, fitness, behavior) = item
    let hrr = genome_to_hrr(genome, holomap_config.hrr_dim, genome.id * 1000)
    let #(updated_grid, _) = holomap.try_add_elite(g, genome.id, behavior, hrr, fitness, generation)
    updated_grid
  })

  #(new_grid, evaluated)
}

/// Main evolution loop
fn train_loop(
  grid: MapElitesGrid,
  current_pop: List(#(Genome, Float, Behavior)),
  population: neat.Population,
  remaining: Int,
  generation: Int,
  holomap_config: HoloMapConfig,
  trainer_config: TrainerConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(MapElitesGrid, HoloMapStats) {
  case remaining <= 0 {
    True -> {
      let stats = holomap.compute_stats(grid, generation, holomap_config)
      io.println("")
      io.println("=== HoloMAP v2 Training Complete ===")
      io.println("Best fitness: " <> float_str(stats.best_fitness))
      io.println("Coverage: " <> float_str(stats.coverage) <> "%")
      io.println("QD-Score: " <> float_str(stats.qd_score))
      #(grid, stats)
    }
    False -> {
      // Log progress
      case generation % trainer_config.log_interval == 0 {
        True -> {
          let stats = holomap.compute_stats(grid, generation, holomap_config)
          io.println(
            "Gen " <> int.to_string(generation)
            <> " | Best: " <> float_str(stats.best_fitness)
            <> " | Cov: " <> float_str(stats.coverage) <> "%"
            <> " | QD: " <> float_str(stats.qd_score)
            <> " | Nov: " <> float_str(stats.novelty_weight *. 100.0) <> "%"
          )
        }
        False -> Nil
      }

      // Generate offspring using HYBRID selection (grid elites + current pop)
      let #(offspring, new_population) = generate_offspring_hybrid(
        grid,
        current_pop,
        population,
        neat_config.population_size,
        trainer_config,
        neat_config,
        seed + generation * 1000,
      )

      // Evaluate offspring and update grid
      let #(new_grid, new_pop) = evaluate_and_update_grid(
        grid, offspring, trainer_config, holomap_config, generation + 1
      )

      // Continue
      train_loop(
        new_grid,
        new_pop,
        new_population,
        remaining - 1,
        generation + 1,
        holomap_config,
        trainer_config,
        neat_config,
        seed,
      )
    }
  }
}

/// Generate offspring using hybrid selection
fn generate_offspring_hybrid(
  grid: MapElitesGrid,
  current_pop: List(#(Genome, Float, Behavior)),
  population: neat.Population,
  count: Int,
  trainer_config: TrainerConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  let elite_count = float.truncate(int.to_float(count) *. trainer_config.elite_selection_ratio)
  let pop_count = count - elite_count

  // Get elites from grid
  let elites = holomap.get_elites(grid)

  // Get genomes from current pop sorted by fitness
  let pop_sorted = current_pop
    |> list.sort(fn(a, b) {
      let #(_, fit_a, _) = a
      let #(_, fit_b, _) = b
      float.compare(fit_b, fit_a)
    })
    |> list.map(fn(item) { item.0 })

  // Generate offspring from elites
  let #(elite_offspring, pop1) = case elites != [] {
    True -> {
      generate_from_elites(elites, pop_sorted, elite_count, population, neat_config, seed)
    }
    False -> #([], population)
  }

  // Generate offspring from current population
  let #(pop_offspring, pop2) = case pop_sorted != [] {
    True -> {
      generate_from_population(pop_sorted, pop_count, pop1, neat_config, seed + 5000)
    }
    False -> #([], pop1)
  }

  #(list.append(elite_offspring, pop_offspring), pop2)
}

fn generate_from_elites(
  elites: List(holomap.Elite),
  pop_sorted: List(Genome),
  count: Int,
  population: neat.Population,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  list.fold(list.range(1, count), #([], population), fn(acc, i) {
    let #(genomes, pop) = acc
    let idx = pseudo_random_int(seed + i * 13, list.length(elites))
    case list_at(elites, idx) {
      Some(elite) -> {
        case find_genome_by_id(pop_sorted, elite.genome_id) {
          Some(genome) -> {
            let #(mutated, new_pop) = apply_neat_mutations(genome, pop, neat_config, seed + i * 100)
            #([mutated, ..genomes], new_pop)
          }
          None -> {
            case list.first(pop_sorted) {
              Ok(genome) -> {
                let #(mutated, new_pop) = apply_neat_mutations(genome, pop, neat_config, seed + i * 100)
                #([mutated, ..genomes], new_pop)
              }
              Error(_) -> acc
            }
          }
        }
      }
      None -> acc
    }
  })
}

fn generate_from_population(
  pop_sorted: List(Genome),
  count: Int,
  population: neat.Population,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  list.fold(list.range(1, count), #([], population), fn(acc, i) {
    let #(genomes, pop) = acc
    let winner = tournament_select_from_list(pop_sorted, 3, seed + i * 29)
    case winner {
      Some(genome) -> {
        let #(mutated, new_pop) = apply_neat_mutations(genome, pop, neat_config, seed + i * 200)
        #([mutated, ..genomes], new_pop)
      }
      None -> acc
    }
  })
}

fn find_genome_by_id(genomes: List(Genome), id: Int) -> Option(Genome) {
  list.find(genomes, fn(g) { g.id == id })
  |> option.from_result
}

fn tournament_select_from_list(
  genomes: List(Genome),
  size: Int,
  seed: Int,
) -> Option(Genome) {
  let len = list.length(genomes)
  case len == 0 {
    True -> None
    False -> {
      let participants = list.range(0, size - 1)
        |> list.filter_map(fn(i) {
          let idx = pseudo_random_int(seed + i * 11, len)
          case list_at(genomes, idx) {
            Some(g) -> Ok(g)
            None -> Error(Nil)
          }
        })
      list.first(participants) |> option.from_result
    }
  }
}

fn apply_neat_mutations(
  genome: Genome,
  population: neat.Population,
  config: NeatConfig,
  seed: Int,
) -> #(Genome, neat.Population) {
  // Apply weight mutation
  let g1 = neat.mutate_weights(genome, config, seed)

  // Maybe add node
  let #(g2, pop1) = case pseudo_random(seed + 100) <. config.add_node_rate {
    True -> neat.mutate_add_node(g1, population, seed + 200)
    False -> #(g1, population)
  }

  // Maybe add connection
  let #(g3, pop2) = case pseudo_random(seed + 300) <. config.add_connection_rate {
    True -> neat.mutate_add_connection(g2, pop1, seed + 400)
    False -> #(g2, pop1)
  }

  #(Genome(..g3, id: seed % 100000), pop2)
}

fn pseudo_random_int(seed: Int, max: Int) -> Int {
  case max <= 0 {
    True -> 0
    False -> {
      let r = pseudo_random(seed)
      float.truncate(r *. int.to_float(max))
    }
  }
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
