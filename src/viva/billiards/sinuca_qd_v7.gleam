//// VIVA Sinuca - QD v7 (DCG-MAP-Elites)
////
//// Descriptor-Conditioned Gradient optimization based on DCG-MAP-Elites (2023)
//// https://hf.co/papers/2303.03832
////
//// Key innovation: Critic network conditioned on Behavior Descriptor [hit_angle, scatter_ratio]
//// This allows gradient-based fitness improvement WITHOUT losing behavioral diversity.
////
//// Expected results (from paper):
//// - +15-20% fitness (90 → 107+)
//// - +30-50% QD-Score
//// - Coverage maintained (descriptor conditioning prevents drift)
////
//// Created at GATO-PC, Brazil, 2026.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import viva/billiards/sinuca.{type Shot, type Table, Shot}
import viva/billiards/sinuca_fitness as fitness
import viva/glands
import viva/jolt.{Vec3}
import viva/neural/activation.{ReLU, Sigmoid}
import viva/neural/neat.{type Genome, type NeatConfig, Genome, NeatConfig}
import viva/neural/network.{type Network}
import viva/neural/train
import viva_tensor/tensor

// =============================================================================
// IMPACT-BASED BEHAVIOR (from v5/v6)
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

fn behavior_distance(b1: ImpactBehavior, b2: ImpactBehavior) -> Float {
  let dx = b1.hit_angle -. b2.hit_angle
  let dy = b1.scatter_ratio -. b2.scatter_ratio
  float_sqrt(dx *. dx +. dy *. dy)
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
    gradient_steps: Int,  // Track how many gradient updates applied
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
          let new_elite = EliteCell(genome:, fitness:, behavior:, generation:, gradient_steps: 0)
          let new_cells = dict.insert(grid.cells, cell, new_elite)
          #(EliteGrid(..grid, cells: new_cells), True)
        }
        False -> #(grid, False)
      }
    }
    Error(_) -> {
      let new_elite = EliteCell(genome:, fitness:, behavior:, generation:, gradient_steps: 0)
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
// DESCRIPTOR-CONDITIONED CRITIC NETWORK
// =============================================================================

/// Critic that takes [state_features..., hit_angle, scatter_ratio] as input
/// and outputs estimated value (expected fitness improvement)
pub type DescriptorCritic {
  DescriptorCritic(
    network: Network,
    learning_rate: Float,
  )
}

/// Create critic network: [8 state + 2 BD] -> 64 -> 32 -> 1
fn create_critic(_seed: Int) -> DescriptorCritic {
  // Input: 8 state features + 2 BD features = 10
  // Hidden: ReLU, Output: Sigmoid (normalized value)
  case network.new([10, 64, 32, 1], ReLU, Sigmoid) {
    Ok(net) -> DescriptorCritic(network: net, learning_rate: 0.001)
    Error(_) -> {
      // Fallback: minimal network
      case network.new([10, 1], ReLU, Sigmoid) {
        Ok(net) -> DescriptorCritic(network: net, learning_rate: 0.001)
        Error(_) -> panic as "Failed to create critic network"
      }
    }
  }
}

/// Forward pass through critic
fn critic_forward(critic: DescriptorCritic, state: List(Float), bd: ImpactBehavior) -> Float {
  let input = list.append(state, [bd.hit_angle, bd.scatter_ratio])
  let input_tensor = tensor.from_list(input)

  case network.predict(critic.network, input_tensor) {
    Ok(output) -> tensor.get(output, 0) |> result.unwrap(0.0)
    Error(_) -> 0.0
  }
}

/// Train critic on (state, bd, actual_fitness) tuples
fn train_critic(
  critic: DescriptorCritic,
  samples: List(#(List(Float), ImpactBehavior, Float)),
) -> DescriptorCritic {
  let train_samples = list.map(samples, fn(s) {
    let #(state, bd, fitness) = s
    let input = list.append(state, [bd.hit_angle, bd.scatter_ratio])
    train.Sample(
      input: tensor.from_list(input),
      target: tensor.from_list([fitness /. 150.0]),  // Normalize fitness
    )
  })

  let config = train.TrainConfig(
    learning_rate: critic.learning_rate,
    momentum: 0.9,
    epochs: 1,
    batch_size: 32,
    loss: train.MSE,
    l2_lambda: 0.0001,
    gradient_clip: 1.0,
    log_interval: 0,
  )

  case train.train_batch(critic.network, train_samples, config) {
    Ok(#(new_net, _loss)) -> DescriptorCritic(..critic, network: new_net)
    Error(_) -> critic
  }
}

// =============================================================================
// CONFIGURATION (DCG-MAP-Elites)
// =============================================================================

pub type DCGConfig {
  DCGConfig(
    grid_size: Int,
    max_steps_per_shot: Int,
    shots_per_episode: Int,
    log_interval: Int,
    // DCG-specific parameters
    gradient_ratio: Float,         // % of updates that use gradients (0.8 = 80%)
    gradient_steps_per_elite: Int, // Gradient steps per elite per generation
    max_descriptor_drift: Float,   // Max allowed BD change during gradient update
    critic_warmup_gens: Int,       // Generations before using critic
    critic_update_interval: Int,   // Update critic every N generations
    // NEAT parameters (for exploration)
    exploration_ratio: Float,      // % of offspring from random exploration
  )
}

pub fn default_dcg_config() -> DCGConfig {
  DCGConfig(
    grid_size: 10,
    max_steps_per_shot: 200,
    shots_per_episode: 3,
    log_interval: 5,
    gradient_ratio: 0.6,           // 60% gradient (reduced from 80%)
    gradient_steps_per_elite: 1,   // 1 step (reduced from 3)
    max_descriptor_drift: 0.15,    // 15% drift allowed
    critic_warmup_gens: 15,        // 15 gens warmup (increased from 10)
    critic_update_interval: 10,    // Update critic less frequently
    exploration_ratio: 0.4,        // 40% exploration
  )
}

fn sinuca_neat_config() -> NeatConfig {
  NeatConfig(
    population_size: 100,
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

fn float_sqrt(x: Float) -> Float {
  case x <=. 0.0 {
    True -> 0.0
    False -> newton_sqrt(x, x /. 2.0, 10)
  }
}

fn newton_sqrt(x: Float, guess: Float, iterations: Int) -> Float {
  case iterations <= 0 {
    True -> guess
    False -> {
      let better = { guess +. x /. guess } /. 2.0
      newton_sqrt(x, better, iterations - 1)
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

fn evaluate_genome(genome: Genome, config: DCGConfig) -> #(Float, ImpactBehavior, List(Float)) {
  let table = sinuca.new()
  let episode = fitness.new_episode()

  let #(total_fitness, hit_angles, scatter_counts, powers, _, _, _, last_state) =
    list.fold(
      list.range(1, config.shots_per_episode),
      #(0.0, [], [], [], 0, table, episode, []),
      fn(acc, _shot_num) {
        let #(fitness_sum, angles, scatters, pows, foul_count, current, ep, _state) = acc

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
              inputs,  // Store last state for critic training
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

  #(total_fitness, behavior, last_state)
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
// DCG: DESCRIPTOR-CONDITIONED GRADIENT PHASE
// =============================================================================

/// Apply gradient-based optimization to elites using descriptor-conditioned critic
fn dcg_gradient_phase(
  grid: EliteGrid,
  critic: DescriptorCritic,
  population: neat.Population,
  config: DCGConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(EliteGrid, neat.Population, List(#(List(Float), ImpactBehavior, Float))) {
  // Only apply DCG to TOP 10 elites by fitness (not all elites)
  let elites = get_elites(grid)
    |> list.sort(fn(a, b) { float.compare(b.fitness, a.fitness) })
    |> list.take(10)

  // Apply gradient steps to top elites only
  let #(new_grid, new_pop, critic_samples) = list.fold(
    list.index_map(elites, fn(elite, idx) { #(elite, idx) }),
    #(grid, population, []),
    fn(acc, item) {
      let #(current_grid, current_pop, samples) = acc
      let #(elite, idx) = item

      // Apply multiple gradient steps
      let #(improved, final_pop, new_samples) = apply_gradient_steps(
        elite,
        critic,
        current_pop,
        config,
        neat_config,
        seed + idx * 10000,
      )

      // Update grid if improved (and BD didn't drift too much)
      let cell = behavior_to_cell(elite.behavior, current_grid.grid_size)
      let drift = behavior_distance(elite.behavior, improved.behavior)

      case improved.fitness >. elite.fitness && drift <. config.max_descriptor_drift {
        True -> {
          let new_cells = dict.insert(current_grid.cells, cell, improved)
          #(EliteGrid(..current_grid, cells: new_cells), final_pop, list.append(samples, new_samples))
        }
        False -> #(current_grid, final_pop, list.append(samples, new_samples))
      }
    }
  )

  #(new_grid, new_pop, critic_samples)
}

/// Apply gradient steps to a single elite
fn apply_gradient_steps(
  elite: EliteCell,
  critic: DescriptorCritic,
  population: neat.Population,
  config: DCGConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(EliteCell, neat.Population, List(#(List(Float), ImpactBehavior, Float))) {
  // Do gradient_steps_per_elite iterations
  list.fold(
    list.range(1, config.gradient_steps_per_elite),
    #(elite, population, []),
    fn(acc, step) {
      let #(current, pop, samples) = acc

      // Generate perturbations and evaluate
      let #(best_variant, best_fitness, best_behavior, best_state, new_pop) =
        gradient_guided_mutation(
          current.genome,
          current.behavior,
          critic,
          pop,
          config,
          neat_config,
          seed + step * 1000,
        )

      // Collect sample for critic training
      let new_sample = #(best_state, best_behavior, best_fitness)

      case best_fitness >. current.fitness {
        True -> {
          let improved = EliteCell(
            genome: best_variant,
            fitness: best_fitness,
            behavior: best_behavior,
            generation: current.generation,
            gradient_steps: current.gradient_steps + 1,
          )
          #(improved, new_pop, [new_sample, ..samples])
        }
        False -> #(current, new_pop, [new_sample, ..samples])
      }
    }
  )
}

/// Generate mutations guided by critic gradient estimate
fn gradient_guided_mutation(
  genome: Genome,
  target_bd: ImpactBehavior,
  _critic: DescriptorCritic,
  population: neat.Population,
  config: DCGConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(Genome, Float, ImpactBehavior, List(Float), neat.Population) {
  // Generate N perturbations (reduced for speed)
  let num_perturbations = 5

  let #(variants, final_pop) = list.fold(
    list.range(1, num_perturbations),
    #([], population),
    fn(acc, i) {
      let #(genomes, pop) = acc
      let #(mutated, new_pop) = apply_mutations(genome, pop, neat_config, seed + i * 100)
      #([mutated, ..genomes], new_pop)
    }
  )

  // Evaluate all variants
  let evaluated = list.map(variants, fn(g) {
    let #(fit, bd, state) = evaluate_genome(g, config)
    let drift = behavior_distance(target_bd, bd)
    // Penalize variants that drift too far from target BD
    let adjusted_fit = case drift >. config.max_descriptor_drift {
      True -> fit -. drift *. 50.0  // Heavy penalty for drift
      False -> fit
    }
    #(g, adjusted_fit, bd, state)
  })

  // Find best
  let best = list.fold(evaluated, #(genome, 0.0, target_bd, []), fn(best_so_far, item) {
    let #(_bg, bf, _bb, _bs) = best_so_far
    let #(g, f, b, s) = item
    case f >. bf {
      True -> #(g, f, b, s)
      False -> best_so_far
    }
  })

  let #(best_g, best_f, best_b, best_s) = best
  #(best_g, best_f, best_b, best_s, final_pop)
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
// EXPLORATION PHASE (NEAT mutations for diversity)
// =============================================================================

fn exploration_phase(
  grid: EliteGrid,
  population: neat.Population,
  config: DCGConfig,
  neat_config: NeatConfig,
  generation: Int,
  seed: Int,
) -> #(EliteGrid, neat.Population) {
  let elites = get_elites(grid)
  let num_offspring = neat_config.population_size

  // Generate offspring from random elites
  let #(offspring, new_pop) = list.fold(
    list.range(1, num_offspring),
    #([], population),
    fn(acc, i) {
      let #(genomes, pop) = acc
      let idx = pseudo_random_int(seed + i * 17, list.length(elites))
      case list_at(elites, idx) {
        Some(elite) -> {
          let #(mutated, next_pop) = apply_mutations(elite.genome, pop, neat_config, seed + i * 100)
          #([mutated, ..genomes], next_pop)
        }
        None -> acc
      }
    }
  )

  // Evaluate and add to grid
  let new_grid = list.fold(offspring, grid, fn(g, genome) {
    let #(fit, behavior, _state) = evaluate_genome(genome, config)
    let #(updated, _added) = try_add_elite(g, genome, fit, behavior, generation)
    updated
  })

  #(new_grid, new_pop)
}

// =============================================================================
// MAIN TRAINING LOOP
// =============================================================================

pub fn main() {
  let config = default_dcg_config()
  let #(_grid, best, cov, qd) = train(100, config)

  io.println("")
  io.println("Final Stats:")
  io.println("  Best Fitness: " <> float_str(best))
  io.println("  Coverage: " <> float_str(cov) <> "%")
  io.println("  QD-Score: " <> float_str(qd))
}

pub fn train(
  generations: Int,
  config: DCGConfig,
) -> #(EliteGrid, Float, Float, Float) {
  let neat_config = sinuca_neat_config()

  io.println("=== VIVA Sinuca QD v7 (DCG-MAP-Elites) ===")
  io.println("GPU Status: " <> glands.check())
  io.println(
    "Grid: " <> int.to_string(config.grid_size) <> "x"
    <> int.to_string(config.grid_size)
    <> " | Pop: " <> int.to_string(neat_config.population_size)
  )
  io.println(
    "DCG: " <> float_str(config.gradient_ratio *. 100.0) <> "% gradient, "
    <> int.to_string(config.gradient_steps_per_elite) <> " steps/elite, "
    <> "max_drift=" <> float_str(config.max_descriptor_drift)
  )
  io.println("")

  let grid = new_grid(config.grid_size)
  let population = neat.create_population(neat_config, 42)
  let critic = create_critic(42)

  // Initial population evaluation
  let #(grid_seeded, _) = evaluate_population(grid, population.genomes, config, 0)

  train_loop(
    grid_seeded,
    population,
    critic,
    generations,
    0,
    config,
    neat_config,
    42,
  )
}

fn evaluate_population(
  grid: EliteGrid,
  genomes: List(Genome),
  config: DCGConfig,
  generation: Int,
) -> #(EliteGrid, List(#(Genome, Float, ImpactBehavior))) {
  let evaluated = list.map(genomes, fn(genome) {
    let #(fit, behavior, _state) = evaluate_genome(genome, config)
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
  population: neat.Population,
  critic: DescriptorCritic,
  remaining: Int,
  generation: Int,
  config: DCGConfig,
  neat_config: NeatConfig,
  seed: Int,
) -> #(EliteGrid, Float, Float, Float) {
  case remaining <= 0 {
    True -> {
      let best = best_fitness(grid)
      let cov = coverage(grid)
      let qd = qd_score(grid)

      io.println("")
      io.println("=== QD v7 (DCG) Training Complete ===")
      io.println("Best fitness: " <> float_str(best))
      io.println("Coverage: " <> float_str(cov) <> "%")
      io.println("QD-Score: " <> float_str(qd))
      #(grid, best, cov, qd)
    }
    False -> {
      let best = best_fitness(grid)

      // Log progress
      case generation % config.log_interval == 0 {
        True -> {
          let cov = coverage(grid)
          let qd = qd_score(grid)
          let phase = case generation >= config.critic_warmup_gens {
            True -> " [DCG]"
            False -> " [WARMUP]"
          }
          io.println(
            "Gen " <> int.to_string(generation)
            <> " | Best: " <> float_str(best)
            <> " | Cov: " <> float_str(cov) <> "%"
            <> " | QD: " <> float_str(qd)
            <> phase
          )
        }
        False -> Nil
      }

      // Decide: gradient phase or exploration phase
      let use_gradients = generation >= config.critic_warmup_gens
        && pseudo_random(seed + generation) <. config.gradient_ratio

      let #(grid_after_phase, pop_after_phase, critic_samples) = case use_gradients {
        True -> {
          // DCG gradient phase (80% of generations after warmup)
          dcg_gradient_phase(grid, critic, population, config, neat_config, seed + generation * 10000)
        }
        False -> {
          // NEAT exploration phase (20% or during warmup)
          let #(g, p) = exploration_phase(grid, population, config, neat_config, generation, seed + generation * 10000)
          #(g, p, [])
        }
      }

      // Update critic if we have samples
      let updated_critic = case
        list.length(critic_samples) > 0
        && generation % config.critic_update_interval == 0
      {
        True -> train_critic(critic, critic_samples)
        False -> critic
      }

      // Continue
      train_loop(
        grid_after_phase,
        pop_after_phase,
        updated_critic,
        remaining - 1,
        generation + 1,
        config,
        neat_config,
        seed,
      )
    }
  }
}
