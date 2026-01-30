//// VIVA Sinuca - QD Trainer v4 (Fixed Selection)
////
//// Key fix: Store actual genomes in elites, not just IDs
//// This ensures we can always retrieve the elite genome for mutation
////
//// Created at GATO-PC, Brazil, 2026.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/billiards/sinuca.{type Shot, type Table, Shot}
import viva/billiards/sinuca_fitness as fitness
import viva/glands
import viva/jolt.{Vec3}
import viva/neural/neat.{type Genome, type NeatConfig, Genome, NeatConfig}
import viva/neural/novelty.{type Behavior}

// =============================================================================
// ELITE ARCHIVE (stores actual genomes)
// =============================================================================

pub type EliteCell {
  EliteCell(
    genome: Genome,
    fitness: Float,
    behavior: Behavior,
    generation: Int,
  )
}

pub type EliteGrid {
  EliteGrid(
    cells: Dict(#(Int, Int), EliteCell),
    grid_size: Int,
  )
}

fn new_grid(size: Int) -> EliteGrid {
  EliteGrid(
    cells: dict.new(),
    grid_size: size,
  )
}

fn behavior_to_cell(behavior: Behavior, grid_size: Int) -> #(Int, Int) {
  let features = behavior.features
  let x_val = list_at(features, 0) |> option.unwrap(0.0)
  let y_val = list_at(features, 1) |> option.unwrap(0.0)

  let norm_x = float_clamp({ x_val +. 1.0 } /. 2.0, 0.0, 0.999)
  let norm_y = float_clamp({ y_val +. 1.0 } /. 2.0, 0.0, 0.999)

  let cell_x = float.truncate(norm_x *. int.to_float(grid_size))
  let cell_y = float.truncate(norm_y *. int.to_float(grid_size))

  #(cell_x, cell_y)
}

fn try_add_elite(
  grid: EliteGrid,
  genome: Genome,
  fitness: Float,
  behavior: Behavior,
  generation: Int,
) -> #(EliteGrid, Bool) {
  let cell = behavior_to_cell(behavior, grid.grid_size)

  case dict.get(grid.cells, cell) {
    Ok(existing) -> {
      case fitness >. existing.fitness {
        True -> {
          let new_elite = EliteCell(
            genome: genome,
            fitness: fitness,
            behavior: behavior,
            generation: generation,
          )
          let new_cells = dict.insert(grid.cells, cell, new_elite)
          #(EliteGrid(..grid, cells: new_cells), True)
        }
        False -> #(grid, False)
      }
    }
    Error(_) -> {
      let new_elite = EliteCell(
        genome: genome,
        fitness: fitness,
        behavior: behavior,
        generation: generation,
      )
      let new_cells = dict.insert(grid.cells, cell, new_elite)
      #(EliteGrid(..grid, cells: new_cells), True)
    }
  }
}

fn get_elites(grid: EliteGrid) -> List(EliteCell) {
  dict.values(grid.cells)
}

fn coverage(grid: EliteGrid) -> Float {
  let filled = int.to_float(dict.size(grid.cells))
  let total = int.to_float(grid.grid_size * grid.grid_size)
  filled /. total *. 100.0
}

fn qd_score(grid: EliteGrid) -> Float {
  dict.values(grid.cells)
  |> list.fold(0.0, fn(acc, elite) { acc +. elite.fitness })
}

fn best_fitness(grid: EliteGrid) -> Float {
  dict.values(grid.cells)
  |> list.fold(0.0, fn(acc, elite) { float.max(acc, elite.fitness) })
}

// =============================================================================
// CONFIGURATION
// =============================================================================

pub type QDConfig {
  QDConfig(
    grid_size: Int,
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    log_interval: Int,
    initial_novelty_weight: Float,
    final_novelty_weight: Float,
    annealing_gens: Int,
  )
}

pub fn default_qd_config() -> QDConfig {
  QDConfig(
    grid_size: 5,
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

fn annealed_novelty_weight(generation: Int, config: QDConfig) -> Float {
  let progress = int.to_float(int.min(generation, config.annealing_gens))
  let total = int.to_float(config.annealing_gens)
  let range = config.initial_novelty_weight -. config.final_novelty_weight
  config.initial_novelty_weight -. { progress /. total } *. range
}

// =============================================================================
// INPUT/OUTPUT
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
  #(total_fitness, behavior)
}

fn pseudo_random(seed: Int) -> Float {
  let a = 1103515245
  let c = 12345
  let m = 2147483648
  let next = { a * seed + c } % m
  int.to_float(int.absolute_value(next)) /. int.to_float(m)
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

fn float_str(x: Float) -> String {
  let scaled = float.truncate(x *. 10.0)
  let whole = scaled / 10
  let frac = int.absolute_value(scaled % 10)
  int.to_string(whole) <> "." <> int.to_string(frac)
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

// =============================================================================
// MAIN TRAINING
// =============================================================================

pub fn main() {
  let qd_config = default_qd_config()

  let #(_grid, best, cov, qd) = train(50, qd_config)

  io.println("")
  io.println("Final Stats:")
  io.println("  Best Fitness: " <> float_str(best))
  io.println("  Coverage: " <> float_str(cov) <> "%")
  io.println("  QD-Score: " <> float_str(qd))
}

pub fn train(
  generations: Int,
  qd_config: QDConfig,
) -> #(EliteGrid, Float, Float, Float) {
  let neat_config = sinuca_neat_config()

  io.println("=== VIVA Sinuca QD v4 (Fixed Selection) ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Grid: " <> int.to_string(qd_config.grid_size) <> "x"
    <> int.to_string(qd_config.grid_size)
    <> " | Annealing: w=" <> float_str(qd_config.initial_novelty_weight)
    <> " -> " <> float_str(qd_config.final_novelty_weight),
  )
  io.println("")

  // Initialize
  let grid = new_grid(qd_config.grid_size)
  let population = neat.create_population(neat_config, 42)

  // Evaluate initial population
  let #(grid_seeded, pop_evaluated) = evaluate_population(
    grid, population.genomes, qd_config, 0
  )

  // Main loop
  train_loop(
    grid_seeded,
    pop_evaluated,
    population,
    generations,
    0,
    qd_config,
    neat_config,
    42,
  )
}

fn evaluate_population(
  grid: EliteGrid,
  genomes: List(Genome),
  qd_config: QDConfig,
  generation: Int,
) -> #(EliteGrid, List(#(Genome, Float, Behavior))) {
  let evaluated = list.map(genomes, fn(genome) {
    let #(fit, behavior) = evaluate_genome(genome, qd_config)
    #(genome, fit, behavior)
  })

  let new_grid = list.fold(evaluated, grid, fn(g, item) {
    let #(genome, fit, behavior) = item
    let #(updated, _) = try_add_elite(g, genome, fit, behavior, generation)
    updated
  })

  #(new_grid, evaluated)
}

fn train_loop(
  grid: EliteGrid,
  current_pop: List(#(Genome, Float, Behavior)),
  population: neat.Population,
  remaining: Int,
  generation: Int,
  qd_config: QDConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(EliteGrid, Float, Float, Float) {
  case remaining <= 0 {
    True -> {
      let best = best_fitness(grid)
      let cov = coverage(grid)
      let qd = qd_score(grid)

      io.println("")
      io.println("=== QD v4 Training Complete ===")
      io.println("Best fitness: " <> float_str(best))
      io.println("Coverage: " <> float_str(cov) <> "%")
      io.println("QD-Score: " <> float_str(qd))
      #(grid, best, cov, qd)
    }
    False -> {
      let novelty_w = annealed_novelty_weight(generation, qd_config)

      // Log progress
      case generation % qd_config.log_interval == 0 {
        True -> {
          let best = best_fitness(grid)
          let cov = coverage(grid)
          let qd = qd_score(grid)
          io.println(
            "Gen " <> int.to_string(generation)
            <> " | Best: " <> float_str(best)
            <> " | Cov: " <> float_str(cov) <> "%"
            <> " | QD: " <> float_str(qd)
            <> " | w: " <> float_str(novelty_w)
          )
        }
        False -> Nil
      }

      // Generate offspring using ACTUAL elite genomes
      let #(offspring, new_pop) = generate_offspring(
        grid,
        population,
        neat_config.population_size,
        novelty_w,
        neat_config,
        seed + generation * 1000,
      )

      // Evaluate offspring
      let #(new_grid, evaluated) = evaluate_population(
        grid, offspring, qd_config, generation + 1
      )

      // Continue
      train_loop(
        new_grid,
        evaluated,
        new_pop,
        remaining - 1,
        generation + 1,
        qd_config,
        neat_config,
        seed,
      )
    }
  }
}

/// Generate offspring using actual elite genomes (key fix)
fn generate_offspring(
  grid: EliteGrid,
  population: neat.Population,
  count: Int,
  novelty_w: Float,
  neat_config: NeatConfig,
  seed: Int,
) -> #(List(Genome), neat.Population) {
  let elites = get_elites(grid)

  case elites != [] {
    True -> {
      // Mix diverse (random cell) and elite (best fitness) selection
      let diverse_count = float.truncate(int.to_float(count) *. novelty_w)
      let elite_count = count - diverse_count

      // Sort elites by fitness for elite selection
      let sorted_elites = elites
        |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })

      // Generate from random cells (diversity)
      let #(diverse_offspring, pop1) = list.fold(
        list.range(1, diverse_count),
        #([], population),
        fn(acc, i) {
          let #(genomes, pop) = acc
          let idx = pseudo_random_int(seed + i * 17, list.length(elites))
          case list_at(elites, idx) {
            Some(elite) -> {
              let #(mutated, new_pop) = apply_mutations(elite.genome, pop, neat_config, seed + i * 100)
              #([mutated, ..genomes], new_pop)
            }
            None -> acc
          }
        }
      )

      // Generate from best elites (exploitation)
      let #(elite_offspring, pop2) = list.fold(
        list.range(1, elite_count),
        #([], pop1),
        fn(acc, i) {
          let #(genomes, pop) = acc
          let top_n = int.min(5, list.length(sorted_elites))
          let idx = pseudo_random_int(seed + i * 23, top_n)
          case list_at(sorted_elites, idx) {
            Some(elite) -> {
              let #(mutated, new_pop) = apply_mutations(elite.genome, pop, neat_config, seed + i * 200)
              #([mutated, ..genomes], new_pop)
            }
            None -> acc
          }
        }
      )

      #(list.append(diverse_offspring, elite_offspring), pop2)
    }
    False -> {
      // Fallback: create random genomes
      let initial_pop = neat.create_population(neat_config, seed)
      #(initial_pop.genomes, initial_pop)
    }
  }
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
