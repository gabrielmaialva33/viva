//// VIVA Sinuca - QD Trainer v3 (Decoupled Selection)
////
//// Implementation based on Qwen3-235B Round 2 recommendations:
//// - Decoupled selection/fitness (critical fix)
//// - Cell-wise elitism (store best fitness per cell)
//// - Linear weight annealing (w=0.7 → w=0.3)
//// - Archive-based parent selection from diverse cells
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
import viva/neural/novelty.{type Behavior}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub type QDConfig {
  QDConfig(
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    log_interval: Int,
    initial_novelty_weight: Float,  // 0.7
    final_novelty_weight: Float,    // 0.3
    annealing_gens: Int,            // 50
  )
}

pub fn default_qd_config() -> QDConfig {
  QDConfig(
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    initial_novelty_weight: 0.7,
    final_novelty_weight: 0.3,
    annealing_gens: 50,
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
// LINEAR WEIGHT ANNEALING (Qwen3 recommendation)
// =============================================================================

fn annealed_novelty_weight(generation: Int, config: QDConfig) -> Float {
  let progress = int.to_float(int.min(generation, config.annealing_gens))
  let total = int.to_float(config.annealing_gens)
  let range = config.initial_novelty_weight -. config.final_novelty_weight

  // Linear annealing: w = w_init - (progress/total) * range
  config.initial_novelty_weight -. { progress /. total } *. range
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
// GENOME EVALUATION (DECOUPLED - fitness only, no combined metric)
// =============================================================================

fn evaluate_genome(
  genome: Genome,
  config: QDConfig,
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
  // Return RAW fitness, not combined (Qwen3 critical fix)
  #(total_fitness, behavior)
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
  let qd_config = default_qd_config()
  let holomap_config = holomap.qwen3_optimized_config()

  let #(_grid, stats) = train(50, holomap_config, qd_config)

  io.println("")
  io.println("Final Stats:")
  io.println("  Best Fitness: " <> float_str(stats.best_fitness))
  io.println("  Coverage: " <> float_str(stats.coverage) <> "%")
  io.println("  QD-Score: " <> float_str(stats.qd_score))
}

pub fn train(
  generations: Int,
  holomap_config: HoloMapConfig,
  qd_config: QDConfig,
) -> #(MapElitesGrid, HoloMapStats) {
  let neat_config = sinuca_neat_config()

  io.println("=== VIVA Sinuca QD v3 (Decoupled Selection) ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Grid: " <> int.to_string(holomap_config.grid_size) <> "x"
    <> int.to_string(holomap_config.grid_size)
    <> " | Annealing: w=" <> float_str(qd_config.initial_novelty_weight)
    <> " -> " <> float_str(qd_config.final_novelty_weight),
  )
  io.println("")

  // Initialize
  let grid = holomap.new_grid(holomap_config)
  let population = neat.create_population(neat_config, 42)

  // Evaluate initial population and seed grid
  let #(grid_seeded, pop_evaluated) = evaluate_and_update_grid(
    grid, population.genomes, qd_config, holomap_config, 0
  )

  // Main loop
  train_loop(
    grid_seeded,
    pop_evaluated,
    population,
    generations,
    0,
    holomap_config,
    qd_config,
    neat_config,
    42,
  )
}

/// Evaluate genomes and update grid (cell-wise elitism)
fn evaluate_and_update_grid(
  grid: MapElitesGrid,
  genomes: List(Genome),
  qd_config: QDConfig,
  holomap_config: HoloMapConfig,
  generation: Int,
) -> #(MapElitesGrid, List(#(Genome, Float, Behavior))) {
  // Evaluate all genomes (RAW fitness, not combined)
  let evaluated = list.map(genomes, fn(genome) {
    let #(raw_fitness, behavior) = evaluate_genome(genome, qd_config)
    #(genome, raw_fitness, behavior)
  })

  // Update grid with cell-wise elitism
  // Only add if: cell empty OR new fitness > existing fitness
  let new_grid = list.fold(evaluated, grid, fn(g, item) {
    let #(genome, raw_fitness, behavior) = item
    let hrr = genome_to_hrr(genome, holomap_config.hrr_dim, genome.id * 1000)
    // try_add_elite implements cell-wise elitism
    let #(updated_grid, _) = holomap.try_add_elite(g, genome.id, behavior, hrr, raw_fitness, generation)
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
  qd_config: QDConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(MapElitesGrid, HoloMapStats) {
  case remaining <= 0 {
    True -> {
      let stats = holomap.compute_stats(grid, generation, holomap_config)
      io.println("")
      io.println("=== QD v3 Training Complete ===")
      io.println("Best fitness: " <> float_str(stats.best_fitness))
      io.println("Coverage: " <> float_str(stats.coverage) <> "%")
      io.println("QD-Score: " <> float_str(stats.qd_score))
      #(grid, stats)
    }
    False -> {
      let novelty_w = annealed_novelty_weight(generation, qd_config)

      // Log progress
      case generation % qd_config.log_interval == 0 {
        True -> {
          let stats = holomap.compute_stats(grid, generation, holomap_config)
          let best_raw = list.fold(current_pop, 0.0, fn(acc, item) {
            float.max(acc, item.1)
          })
          io.println(
            "Gen " <> int.to_string(generation)
            <> " | Best: " <> float_str(best_raw)
            <> " | Cov: " <> float_str(stats.coverage) <> "%"
            <> " | QD: " <> float_str(stats.qd_score)
            <> " | w: " <> float_str(novelty_w)
          )
        }
        False -> Nil
      }

      // Generate offspring with archive-based selection (diverse cells)
      let #(offspring, new_pop) = generate_offspring_decoupled(
        grid,
        current_pop,
        population,
        neat_config.population_size,
        novelty_w,
        neat_config,
        seed + generation * 1000,
      )

      // Evaluate offspring
      let #(new_grid, evaluated) = evaluate_and_update_grid(
        grid, offspring, qd_config, holomap_config, generation + 1
      )

      // Continue
      train_loop(
        new_grid,
        evaluated,
        new_pop,
        remaining - 1,
        generation + 1,
        holomap_config,
        qd_config,
        neat_config,
        seed,
      )
    }
  }
}

/// Decoupled selection: Select parents from diverse cells (based on novelty_w)
/// Higher novelty_w = more diversity-focused selection
/// Lower novelty_w = more fitness-focused selection
fn generate_offspring_decoupled(
  grid: MapElitesGrid,
  current_pop: List(#(Genome, Float, Behavior)),
  population: neat.Population,
  count: Int,
  novelty_w: Float,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  let elites = holomap.get_elites(grid)

  // Sort population by fitness for fallback
  let pop_sorted = current_pop
    |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
    |> list.map(fn(item) { item.0 })

  // Mix of diverse (random cell) and elite (best fitness) selection
  // novelty_w controls the ratio
  let diverse_count = float.truncate(int.to_float(count) *. novelty_w)
  let elite_count = count - diverse_count

  // Generate from random cells (diversity)
  let #(diverse_offspring, pop1) = generate_from_random_cells(
    elites, pop_sorted, diverse_count, population, neat_config, seed
  )

  // Generate from best fitness elites (exploitation)
  let #(elite_offspring, pop2) = generate_from_best_elites(
    elites, pop_sorted, elite_count, pop1, neat_config, seed + 5000
  )

  #(list.append(diverse_offspring, elite_offspring), pop2)
}

/// Select parents from random cells (promotes diversity)
fn generate_from_random_cells(
  elites: List(holomap.Elite),
  pop_sorted: List(Genome),
  count: Int,
  population: neat.Population,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  case elites != [] && pop_sorted != [] {
    True -> {
      list.fold(list.range(1, count), #([], population), fn(acc, i) {
        let #(genomes, pop) = acc
        // Random cell selection (diversity)
        let idx = pseudo_random_int(seed + i * 17, list.length(elites))
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

/// Select parents from best fitness elites (exploitation)
fn generate_from_best_elites(
  elites: List(holomap.Elite),
  pop_sorted: List(Genome),
  count: Int,
  population: neat.Population,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  // Sort elites by fitness (descending)
  let sorted_elites = elites
    |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })

  case sorted_elites != [] && pop_sorted != [] {
    True -> {
      list.fold(list.range(1, count), #([], population), fn(acc, i) {
        let #(genomes, pop) = acc
        // Tournament from top elites
        let top_n = int.min(5, list.length(sorted_elites))
        let idx = pseudo_random_int(seed + i * 23, top_n)
        case list_at(sorted_elites, idx) {
          Some(elite) -> {
            case find_genome_by_id(pop_sorted, elite.genome_id) {
              Some(genome) -> {
                let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 200)
                #([mutated, ..genomes], new_pop)
              }
              None -> {
                case list.first(pop_sorted) {
                  Ok(genome) -> {
                    let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 200)
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
