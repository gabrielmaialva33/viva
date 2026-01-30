//// VIVA Sinuca - Hybrid Novelty + MAP-Elites Trainer
////
//// Implementation based on Qwen3-235B recommendations:
//// - 70% Novelty Search for exploration
//// - 30% MAP-Elites for quality tracking
//// - 5x5 grid initially, dynamic expansion
//// - 20% elite selection ratio
////
//// Created at GATO-PC, Brazil, 2026.

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
import viva/neural/novelty.{type Behavior, type NoveltyArchive, type NoveltyConfig}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub type HybridConfig {
  HybridConfig(
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    log_interval: Int,
    novelty_ratio: Float,     // 70% from novelty archive
    elite_ratio: Float,       // 20% from MAP-Elites
    population_ratio: Float,  // 10% from top performers
  )
}

pub fn default_hybrid_config() -> HybridConfig {
  HybridConfig(
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    novelty_ratio: 0.70,      // Qwen3 recommendation
    elite_ratio: 0.20,        // Qwen3 recommendation for <30% coverage
    population_ratio: 0.10,
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

fn novelty_config() -> NoveltyConfig {
  novelty.exploration_config()  // Higher novelty weight
}

// =============================================================================
// INPUT/OUTPUT ENCODING
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

// =============================================================================
// GENOME EVALUATION
// =============================================================================

fn evaluate_genome_full(
  genome: Genome,
  config: HybridConfig,
) -> #(Float, Behavior, List(Float)) {
  let table = sinuca.new()
  let episode = fitness.new_episode()

  let #(total_fitness, behavior_features, entropy_score, _, _, _) =
    list.fold(
      list.range(1, config.shots_per_episode),
      #(0.0, [], 0.0, 0, table, episode),
      fn(acc, _shot_num) {
        let #(fitness_sum, features, entropy, foul_count, current, ep) = acc

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

            // Enhanced behavior features (Qwen3 recommendation)
            let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(next) {
              Some(Vec3(x, _, z)) -> #(x, z)
              None -> #(0.0, 0.0)
            }

            // Ball distribution entropy (simplified)
            let balls = sinuca.balls_on_table(next)
            let distribution_score = int.to_float(balls) /. 16.0

            let new_features = list.append(features, [
              shot.angle /. 6.28,
              shot.power,
              cue_x /. 1.5,
              cue_z /. 0.75,
              distribution_score,
            ])

            let new_fouls = case sinuca.is_scratch(next) {
              True -> foul_count + 1
              False -> foul_count
            }

            let table_next = case sinuca.is_scratch(next) {
              True -> sinuca.reset_cue_ball(next)
              False -> next
            }

            #(
              fitness_sum +. shot_fitness,
              new_features,
              entropy +. distribution_score,
              new_fouls,
              table_next,
              new_ep,
            )
          }
        }
      },
    )

  let behavior = novelty.behavior_from_features(behavior_features)
  let hrr_features = list.append(behavior_features, [entropy_score])
  #(total_fitness, behavior, hrr_features)
}

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
  let hybrid_config = default_hybrid_config()
  let holomap_config = holomap.qwen3_optimized_config()

  let #(_grid, stats, _archive) = train(50, holomap_config, hybrid_config)

  io.println("")
  io.println("Final Stats:")
  io.println("  Best Fitness: " <> float_str(stats.best_fitness))
  io.println("  Coverage: " <> float_str(stats.coverage) <> "%")
  io.println("  QD-Score: " <> float_str(stats.qd_score))
}

pub fn train(
  generations: Int,
  holomap_config: HoloMapConfig,
  hybrid_config: HybridConfig,
) -> #(MapElitesGrid, HoloMapStats, NoveltyArchive) {
  let neat_config = sinuca_neat_config()
  let nov_config = novelty_config()

  io.println("=== VIVA Sinuca Hybrid (Novelty + MAP-Elites) ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Grid: " <> int.to_string(holomap_config.grid_size) <> "x"
    <> int.to_string(holomap_config.grid_size)
    <> " | Novelty: " <> float_str(hybrid_config.novelty_ratio *. 100.0) <> "%"
    <> " | Elite: " <> float_str(hybrid_config.elite_ratio *. 100.0) <> "%",
  )
  io.println("")

  // Initialize
  let grid = holomap.new_grid(holomap_config)
  let archive = novelty.new_archive(nov_config)
  let population = neat.create_population(neat_config, 42)

  // Evaluate initial population
  let #(grid_init, archive_init, pop_evaluated) = evaluate_population(
    grid, archive, population.genomes, hybrid_config, holomap_config, nov_config, 0
  )

  // Main loop
  train_loop(
    grid_init,
    archive_init,
    pop_evaluated,
    population,
    generations,
    0,
    holomap_config,
    hybrid_config,
    neat_config,
    nov_config,
    42,
  )
}

fn evaluate_population(
  grid: MapElitesGrid,
  archive: NoveltyArchive,
  genomes: List(Genome),
  hybrid_config: HybridConfig,
  holomap_config: HoloMapConfig,
  nov_config: NoveltyConfig,
  generation: Int,
) -> #(MapElitesGrid, NoveltyArchive, List(#(Genome, Float, Behavior))) {
  // Evaluate all genomes
  let evaluated = list.map(genomes, fn(genome) {
    let #(fitness, behavior, _hrr_features) = evaluate_genome_full(genome, hybrid_config)
    #(genome, fitness, behavior)
  })

  // Extract behaviors for novelty calculation
  let behaviors = list.map(evaluated, fn(item) { item.2 })

  // Calculate novelty scores and update archive
  let #(combined_fitness, new_archive) = novelty.batch_combined_fitness(
    list.map(evaluated, fn(item) { item.1 }),
    behaviors,
    archive,
    nov_config,
  )

  // Update grid with elites
  let new_grid = list.fold(
    list.zip(evaluated, combined_fitness),
    grid,
    fn(g, pair) {
      let #(#(genome, _obj_fit, behavior), comb_fit) = pair
      let hrr = genome_to_hrr(genome, holomap_config.hrr_dim, genome.id * 1000)
      let #(updated_grid, _) = holomap.try_add_elite(g, genome.id, behavior, hrr, comb_fit, generation)
      updated_grid
    }
  )

  // Return evaluated with combined fitness
  let result = list.zip(evaluated, combined_fitness)
    |> list.map(fn(pair) {
      let #(#(genome, _, behavior), comb_fit) = pair
      #(genome, comb_fit, behavior)
    })

  #(new_grid, new_archive, result)
}

fn train_loop(
  grid: MapElitesGrid,
  archive: NoveltyArchive,
  current_pop: List(#(Genome, Float, Behavior)),
  population: neat.Population,
  remaining: Int,
  generation: Int,
  holomap_config: HoloMapConfig,
  hybrid_config: HybridConfig,
  neat_config: NeatConfig,
  nov_config: NoveltyConfig,
  seed: Int,
) -> #(MapElitesGrid, HoloMapStats, NoveltyArchive) {
  case remaining <= 0 {
    True -> {
      let stats = holomap.compute_stats(grid, generation, holomap_config)
      io.println("")
      io.println("=== Hybrid Training Complete ===")
      io.println("Best fitness: " <> float_str(stats.best_fitness))
      io.println("Coverage: " <> float_str(stats.coverage) <> "%")
      io.println("Archive size: " <> int.to_string(list.length(archive.behaviors)))
      #(grid, stats, archive)
    }
    False -> {
      // Log progress
      case generation % hybrid_config.log_interval == 0 {
        True -> {
          let stats = holomap.compute_stats(grid, generation, holomap_config)
          let best_obj = list.fold(current_pop, 0.0, fn(acc, item) {
            float.max(acc, item.1)
          })
          io.println(
            "Gen " <> int.to_string(generation)
            <> " | Obj: " <> float_str(best_obj)
            <> " | Cov: " <> float_str(stats.coverage) <> "%"
            <> " | QD: " <> float_str(stats.qd_score)
            <> " | Arch: " <> int.to_string(list.length(archive.behaviors))
          )
        }
        False -> Nil
      }

      // Generate offspring with hybrid selection
      let #(offspring, new_pop) = generate_hybrid_offspring(
        grid,
        archive,
        current_pop,
        population,
        neat_config.population_size,
        hybrid_config,
        neat_config,
        seed + generation * 1000,
      )

      // Evaluate offspring
      let #(new_grid, new_archive, evaluated) = evaluate_population(
        grid, archive, offspring, hybrid_config, holomap_config, nov_config, generation + 1
      )

      // Continue
      train_loop(
        new_grid,
        new_archive,
        evaluated,
        new_pop,
        remaining - 1,
        generation + 1,
        holomap_config,
        hybrid_config,
        neat_config,
        nov_config,
        seed,
      )
    }
  }
}

/// Hybrid selection: 70% novelty, 20% elites, 10% population
fn generate_hybrid_offspring(
  grid: MapElitesGrid,
  archive: NoveltyArchive,
  current_pop: List(#(Genome, Float, Behavior)),
  population: neat.Population,
  count: Int,
  hybrid_config: HybridConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  let novelty_count = float.truncate(int.to_float(count) *. hybrid_config.novelty_ratio)
  let elite_count = float.truncate(int.to_float(count) *. hybrid_config.elite_ratio)
  let pop_count = count - novelty_count - elite_count

  // Get genomes sorted by fitness
  let pop_sorted = current_pop
    |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
    |> list.map(fn(item) { item.0 })

  // Generate from novelty archive (random selection from diverse behaviors)
  let #(novelty_offspring, pop1) = generate_from_novelty(
    archive, pop_sorted, novelty_count, population, neat_config, seed
  )

  // Generate from MAP-Elites grid
  let #(elite_offspring, pop2) = generate_from_elites(
    grid, pop_sorted, elite_count, pop1, neat_config, seed + 3000
  )

  // Generate from top population performers
  let #(pop_offspring, pop3) = generate_from_top_performers(
    pop_sorted, pop_count, pop2, neat_config, seed + 6000
  )

  #(list.flatten([novelty_offspring, elite_offspring, pop_offspring]), pop3)
}

fn generate_from_novelty(
  archive: NoveltyArchive,
  pop_sorted: List(Genome),
  count: Int,
  population: neat.Population,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  case archive.behaviors != [] && pop_sorted != [] {
    True -> {
      list.fold(list.range(1, count), #([], population), fn(acc, i) {
        let #(genomes, pop) = acc
        // Select random genome from top performers
        let idx = pseudo_random_int(seed + i * 17, int.min(10, list.length(pop_sorted)))
        case list_at(pop_sorted, idx) {
          Some(genome) -> {
            let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 100)
            #([mutated, ..genomes], new_pop)
          }
          None -> acc
        }
      })
    }
    False -> {
      // Fallback to random from population
      list.fold(list.range(1, count), #([], population), fn(acc, i) {
        let #(genomes, pop) = acc
        case list.first(pop_sorted) {
          Ok(genome) -> {
            let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 100)
            #([mutated, ..genomes], new_pop)
          }
          Error(_) -> acc
        }
      })
    }
  }
}

fn generate_from_elites(
  grid: MapElitesGrid,
  pop_sorted: List(Genome),
  count: Int,
  population: neat.Population,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  let elites = holomap.get_elites(grid)
  case elites != [] && pop_sorted != [] {
    True -> {
      list.fold(list.range(1, count), #([], population), fn(acc, i) {
        let #(genomes, pop) = acc
        let idx = pseudo_random_int(seed + i * 13, list.length(elites))
        case list_at(elites, idx) {
          Some(elite) -> {
            case find_genome_by_id(pop_sorted, elite.genome_id) {
              Some(genome) -> {
                let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 100)
                #([mutated, ..genomes], new_pop)
              }
              None -> {
                case list.first(pop_sorted) {
                  Ok(genome) -> {
                    let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 100)
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
    False -> #([], population)
  }
}

fn generate_from_top_performers(
  pop_sorted: List(Genome),
  count: Int,
  population: neat.Population,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  case pop_sorted != [] {
    True -> {
      list.fold(list.range(1, count), #([], population), fn(acc, i) {
        let #(genomes, pop) = acc
        // Tournament selection from top 5
        let idx = pseudo_random_int(seed + i * 29, int.min(5, list.length(pop_sorted)))
        case list_at(pop_sorted, idx) {
          Some(genome) -> {
            let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 200)
            #([mutated, ..genomes], new_pop)
          }
          None -> acc
        }
      })
    }
    False -> #([], population)
  }
}

fn find_genome_by_id(genomes: List(Genome), id: Int) -> Option(Genome) {
  list.find(genomes, fn(g) { g.id == id })
  |> option.from_result
}

fn apply_mutations(
  genome: Genome,
  population: neat.Population,
  config: NeatConfig,
  seed: Int,
) -> #(Genome, neat.Population) {
  let g1 = neat.mutate_weights(genome, config, seed)

  let #(g2, pop1) = case pseudo_random(seed + 100) <. config.add_node_rate {
    True -> neat.mutate_add_node(g1, population, seed + 200)
    False -> #(g1, population)
  }

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
