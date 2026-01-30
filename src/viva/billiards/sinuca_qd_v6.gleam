//// VIVA Sinuca - QD v6 (QD-NEAT Hybrid)
////
//// Combines MAP-Elites diversity with NEAT exploitation phases
//// Based on Gemini/Qwen3 analysis: "Hibridizar QD com Exploracao NEAT"
////
//// Every N generations, do micro-evolution on each elite cell using
//// pure fitness selection (no BD) to close the 16% gap to NEAT.
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

// =============================================================================
// IMPACT-BASED BEHAVIOR (from v5)
// =============================================================================

pub type ImpactBehavior {
  ImpactBehavior(
    hit_angle: Float,
    scatter_ratio: Float,
    power_used: Float,
    combo_potential: Float,
  )
}

fn behavior_to_cell(behavior: ImpactBehavior, grid_size: Int) -> #(Int, Int) {
  let x_val = float_clamp(behavior.hit_angle, 0.0, 0.999)
  let y_val = float_clamp(behavior.scatter_ratio, 0.0, 0.999)
  let cell_x = float.truncate(x_val *. int.to_float(grid_size))
  let cell_y = float.truncate(y_val *. int.to_float(grid_size))
  #(cell_x, cell_y)
}

// =============================================================================
// ELITE ARCHIVE
// =============================================================================

pub type EliteCell {
  EliteCell(
    genome: Genome,
    fitness: Float,
    behavior: ImpactBehavior,
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
  EliteGrid(cells: dict.new(), grid_size: size)
}

fn try_add_elite(
  grid: EliteGrid,
  genome: Genome,
  fitness: Float,
  behavior: ImpactBehavior,
  generation: Int,
) -> #(EliteGrid, Bool) {
  let cell = behavior_to_cell(behavior, grid.grid_size)

  case dict.get(grid.cells, cell) {
    Ok(existing) -> {
      case fitness >. existing.fitness {
        True -> {
          let new_elite = EliteCell(genome:, fitness:, behavior:, generation:)
          let new_cells = dict.insert(grid.cells, cell, new_elite)
          #(EliteGrid(..grid, cells: new_cells), True)
        }
        False -> #(grid, False)
      }
    }
    Error(_) -> {
      let new_elite = EliteCell(genome:, fitness:, behavior:, generation:)
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
// CONFIGURATION (QD-NEAT Hybrid)
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
    fitness_boost_threshold: Float,
    // QD-NEAT Hybrid parameters
    exploitation_frequency: Int,    // Do NEAT phase every N generations
    exploitation_micro_gens: Int,   // Number of micro-generations per elite
    exploitation_pop_size: Int,     // Sub-population size for exploitation
  )
}

pub fn default_qd_config() -> QDConfig {
  QDConfig(
    grid_size: 10,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    initial_novelty_weight: 0.6,
    final_novelty_weight: 0.2,
    annealing_gens: 100,
    fitness_boost_threshold: 50.0,
    // Hybrid settings from Qwen3 recommendations
    exploitation_frequency: 10,   // Every 10 generations
    exploitation_micro_gens: 5,   // 5 micro-generations
    exploitation_pop_size: 10,    // Small sub-pop per elite
  )
}

fn sinuca_neat_config() -> NeatConfig {
  NeatConfig(
    population_size: 100,  // Increased from 50
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

fn annealed_novelty_weight(generation: Int, config: QDConfig, best: Float) -> Float {
  let progress = int.to_float(int.min(generation, config.annealing_gens))
  let total = int.to_float(config.annealing_gens)
  let range = config.initial_novelty_weight -. config.final_novelty_weight
  let base = config.initial_novelty_weight -. { progress /. total } *. range

  case best >. config.fitness_boost_threshold {
    True -> float.max(base -. 0.1, config.final_novelty_weight)
    False -> base
  }
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
    False -> float.truncate(pseudo_random(seed) *. int.to_float(max))
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
    [first, ..rest] ->
      case current == target {
        True -> Some(first)
        False -> do_list_at(rest, target, current + 1)
      }
  }
}

// =============================================================================
// INPUT/OUTPUT
// =============================================================================

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
// EVALUATION
// =============================================================================

fn evaluate_genome(genome: Genome, config: QDConfig) -> #(Float, ImpactBehavior) {
  let table = sinuca.new()
  let episode = fitness.new_episode()

  let #(total_fitness, hit_angles, scatter_counts, powers, _, _, _) =
    list.fold(
      list.range(1, config.shots_per_episode),
      #(0.0, [], [], [], 0, table, episode),
      fn(acc, _shot_num) {
        let #(fitness_sum, angles, scatters, pows, foul_count, current, ep) = acc

        case sinuca.balls_on_table(current) <= 1 {
          True -> acc
          False -> {
            let inputs = encode_inputs(current)
            let outputs = neat.forward(genome, inputs)
            let shot = decode_outputs(outputs)

            let balls_before = get_all_ball_positions(current)

            let #(shot_fitness, next, new_ep) =
              fitness.quick_evaluate_full(
                current,
                shot,
                config.max_steps_per_shot,
                ep,
                fitness.default_config(),
              )

            let balls_after = get_all_ball_positions(next)
            let scatter_ratio = calculate_scatter_ratio(balls_before, balls_after)

            let normalized_angle = { shot.angle +. 3.14159 } /. { 2.0 *. 3.14159 }
            let clamped_angle = float_clamp(normalized_angle, 0.0, 1.0)

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
              [clamped_angle, ..angles],
              [scatter_ratio, ..scatters],
              [shot.power, ..pows],
              new_fouls,
              table_next,
              new_ep,
            )
          }
        }
      },
    )

  let avg_angle = safe_average(hit_angles, 0.5)
  let avg_scatter = safe_average(scatter_counts, 0.0)
  let avg_power = safe_average(powers, 0.5)
  let combo_potential = float_clamp(total_fitness /. 100.0, 0.0, 1.0)

  let behavior = ImpactBehavior(
    hit_angle: avg_angle,
    scatter_ratio: avg_scatter,
    power_used: avg_power,
    combo_potential: combo_potential,
  )

  #(total_fitness, behavior)
}

fn evaluate_fitness_only(genome: Genome, config: QDConfig) -> Float {
  let #(fitness, _) = evaluate_genome(genome, config)
  fitness
}

fn get_all_ball_positions(table: Table) -> List(#(Float, Float)) {
  list.filter_map(
    [sinuca.Red, sinuca.Yellow, sinuca.Green, sinuca.Brown, sinuca.Blue, sinuca.Pink, sinuca.Black],
    fn(ball) {
      case sinuca.get_ball_position(table, ball) {
        Some(Vec3(x, _y, z)) -> Ok(#(x, z))
        None -> Error(Nil)
      }
    }
  )
}

fn calculate_scatter_ratio(before: List(#(Float, Float)), after: List(#(Float, Float))) -> Float {
  let pairs = list.zip(before, after)
  let moved = list.filter(pairs, fn(pair) {
    let #(#(bx, bz), #(ax, az)) = pair
    let dx = ax -. bx
    let dz = az -. bz
    dx *. dx +. dz *. dz >. 0.01
  })

  case list.length(before) {
    0 -> 0.0
    n -> int.to_float(list.length(moved)) /. int.to_float(n)
  }
}

fn safe_average(values: List(Float), default: Float) -> Float {
  case list.length(values) {
    0 -> default
    n -> list.fold(values, 0.0, fn(acc, v) { acc +. v }) /. int.to_float(n)
  }
}

// =============================================================================
// QD-NEAT HYBRID: EXPLOITATION PHASE
// =============================================================================

/// Run NEAT micro-evolution on each elite (pure fitness selection)
fn exploitation_phase(
  grid: EliteGrid,
  population: neat.Population,
  config: QDConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(EliteGrid, neat.Population) {
  let elites = get_elites(grid)

  // For each elite, run micro-evolution
  list.fold(
    list.index_map(elites, fn(elite, idx) { #(elite, idx) }),
    #(grid, population),
    fn(acc, item) {
      let #(current_grid, current_pop) = acc
      let #(elite, idx) = item

      // Create sub-population from elite
      let #(improved_genome, new_pop) = micro_evolve_elite(
        elite.genome,
        current_pop,
        config,
        neat_config,
        seed + idx * 1000,
      )

      // Re-evaluate improved genome
      let #(new_fitness, new_behavior) = evaluate_genome(improved_genome, config)

      // Try to update the grid with improved genome
      let cell = behavior_to_cell(elite.behavior, current_grid.grid_size)
      case new_fitness >. elite.fitness {
        True -> {
          let new_elite = EliteCell(
            genome: improved_genome,
            fitness: new_fitness,
            behavior: new_behavior,
            generation: elite.generation,
          )
          let new_cells = dict.insert(current_grid.cells, cell, new_elite)
          #(EliteGrid(..current_grid, cells: new_cells), new_pop)
        }
        False -> #(current_grid, new_pop)
      }
    }
  )
}

/// Micro-evolve a single elite using pure fitness selection (no BD)
fn micro_evolve_elite(
  elite: Genome,
  population: neat.Population,
  config: QDConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(Genome, neat.Population) {
  // Create sub-population of mutants from elite
  let initial_mutants = list.map(
    list.range(1, config.exploitation_pop_size),
    fn(i) {
      let #(mutated, _) = apply_mutations(elite, population, neat_config, seed + i * 100)
      mutated
    }
  )

  // Run micro-generations with pure fitness selection
  let #(final_best, final_pop) = list.fold(
    list.range(1, config.exploitation_micro_gens),
    #(elite, population),
    fn(acc, gen_num) {
      let #(current_best, current_pop) = acc

      // Evaluate all mutants
      let evaluated = [#(current_best, evaluate_fitness_only(current_best, config)),
        ..list.map(initial_mutants, fn(g) { #(g, evaluate_fitness_only(g, config)) })]

      // Find best by pure fitness
      let best = list.fold(evaluated, #(current_best, 0.0), fn(best_so_far, item) {
        let #(_best_g, best_f) = best_so_far
        let #(g, f) = item
        case f >. best_f {
          True -> #(g, f)
          False -> best_so_far
        }
      })

      let #(best_genome, _) = best

      // Create next generation from best
      let #(next_gen, next_pop) = list.fold(
        list.range(1, config.exploitation_pop_size),
        #([], current_pop),
        fn(acc2, i) {
          let #(genomes, pop) = acc2
          let #(mutated, new_pop) = apply_mutations(best_genome, pop, neat_config, seed + gen_num * 1000 + i * 10)
          #([mutated, ..genomes], new_pop)
        }
      )

      let _ = next_gen  // Discard, we only care about best
      #(best_genome, next_pop)
    }
  )

  #(final_best, final_pop)
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

// =============================================================================
// MAIN TRAINING
// =============================================================================

pub fn main() {
  let qd_config = default_qd_config()
  let #(_grid, best, cov, qd) = train(100, qd_config)

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

  io.println("=== VIVA Sinuca QD v6 (QD-NEAT Hybrid) ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Grid: " <> int.to_string(qd_config.grid_size) <> "x"
    <> int.to_string(qd_config.grid_size)
    <> " | Pop: " <> int.to_string(neat_config.population_size)
  )
  io.println(
    "Exploitation: every " <> int.to_string(qd_config.exploitation_frequency)
    <> " gen, " <> int.to_string(qd_config.exploitation_micro_gens) <> " micro-gens"
  )
  io.println("")

  let grid = new_grid(qd_config.grid_size)
  let population = neat.create_population(neat_config, 42)

  let #(grid_seeded, pop_evaluated) = evaluate_population(
    grid, population.genomes, qd_config, 0
  )

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
) -> #(EliteGrid, List(#(Genome, Float, ImpactBehavior))) {
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
  _current_pop: List(#(Genome, Float, ImpactBehavior)),
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
      io.println("=== QD v6 Training Complete ===")
      io.println("Best fitness: " <> float_str(best))
      io.println("Coverage: " <> float_str(cov) <> "%")
      io.println("QD-Score: " <> float_str(qd))
      #(grid, best, cov, qd)
    }
    False -> {
      let best = best_fitness(grid)
      let novelty_w = annealed_novelty_weight(generation, qd_config, best)

      // Log progress
      case generation % qd_config.log_interval == 0 {
        True -> {
          let cov = coverage(grid)
          let qd = qd_score(grid)
          let is_exploit = generation > 0 && generation % qd_config.exploitation_frequency == 0
          let suffix = case is_exploit {
            True -> " [EXPLOIT]"
            False -> ""
          }
          io.println(
            "Gen " <> int.to_string(generation)
            <> " | Best: " <> float_str(best)
            <> " | Cov: " <> float_str(cov) <> "%"
            <> " | QD: " <> float_str(qd)
            <> " | w: " <> float_str(novelty_w)
            <> suffix
          )
        }
        False -> Nil
      }

      // Run exploitation phase if it's time
      let #(grid_after_exploit, pop_after_exploit) = case generation > 0 && generation % qd_config.exploitation_frequency == 0 {
        True -> exploitation_phase(grid, population, qd_config, neat_config, seed + generation * 10000)
        False -> #(grid, population)
      }

      // Generate offspring using elite genomes
      let #(offspring, new_pop) = generate_offspring(
        grid_after_exploit,
        pop_after_exploit,
        neat_config.population_size,
        novelty_w,
        neat_config,
        seed + generation * 1000,
      )

      // Evaluate offspring
      let #(new_grid, evaluated) = evaluate_population(
        grid_after_exploit, offspring, qd_config, generation + 1
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
      let diverse_count = float.truncate(int.to_float(count) *. novelty_w)
      let elite_count = count - diverse_count

      let sorted_elites = elites
        |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })

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
      let initial_pop = neat.create_population(neat_config, seed)
      #(initial_pop.genomes, initial_pop)
    }
  }
}
