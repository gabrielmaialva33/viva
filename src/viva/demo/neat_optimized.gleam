// =============================================================================
// NEAT OPTIMIZED - State-of-the-Art Neuroevolution (Pure NEAT)
// =============================================================================
// Implements Lamarckian inheritance, safe mutations, adaptive speciation
// =============================================================================

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string
import viva/neural/neat.{
  type Genome, type NeatConfig, type Population, Genome, Population,
}
import viva/neural/neat_advanced.{
  type EliteConfig, type PopulationStats, type SafeMutationConfig,
  LamarckianBlend,
}

// =============================================================================
// CONFIGURATION
// =============================================================================

const population_size: Int = 60

const max_generations: Int = 25

const target_fitness: Float = 0.80

const samples_per_digit: Int = 20

const test_samples: Int = 80

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println(
    "╔══════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║        VIVA NEAT OPTIMIZED - Pure Neuroevolution v2.0        ║",
  )
  io.println(
    "╠══════════════════════════════════════════════════════════════╣",
  )
  io.println(
    "║  Lamarckian Inheritance • Safe Mutations • Elite Diversity   ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Generate dataset
  io.println("Generating digit dataset...")
  let train_data = generate_dataset(samples_per_digit)
  let test_data = generate_test_set(test_samples)
  io.println(
    "   Train: " <> int.to_string(list.length(train_data)) <> " samples",
  )
  io.println("   Test:  " <> int.to_string(list.length(test_data)) <> " samples")
  io.println("")

  // Configuration
  let config =
    neat.NeatConfig(
      ..neat.default_config(),
      population_size: population_size,
      num_inputs: 64,
      num_outputs: 10,
      weight_mutation_rate: 0.8,
      weight_perturb_rate: 0.9,
      add_node_rate: 0.03,
      add_connection_rate: 0.08,
      compatibility_threshold: 3.0,
    )

  let safe_config = neat_advanced.default_safe_config()
  let elite_config = neat_advanced.default_elite_config()

  // Create initial population
  io.println("Creating optimized population...")
  let population = neat.create_population(config, 42)
  io.println("   Population: " <> int.to_string(population_size))
  io.println("   Architecture: 64 → evolved → 10")
  io.println("")

  // Evolution header
  io.println("Starting optimized evolution...")
  io.println(" Gen │ Best   │ Avg   │ StdDev │ Stagnant │ Nodes │ Conns")
  io.println("─────┼────────┼───────┼────────┼──────────┼───────┼───────")

  // Initial stats
  let initial_stats =
    neat_advanced.PopulationStats(
      generation: 0,
      best_fitness: 0.0,
      avg_fitness: 0.0,
      worst_fitness: 0.0,
      fitness_std_dev: 0.0,
      species_count: 1,
      avg_genome_size: 0.0,
      total_connections: 0,
      total_nodes: 0,
      stagnant_generations: 0,
    )

  // Run evolution
  let #(final_pop, final_stats, best) =
    evolve(
      population,
      config,
      safe_config,
      elite_config,
      train_data,
      test_data,
      0,
      initial_stats,
    )

  // Final results
  io.println("")
  io.println(
    "╔══════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║                     EVOLUTION COMPLETE                       ║",
  )
  io.println(
    "╠══════════════════════════════════════════════════════════════╣",
  )

  let final_acc = evaluate_genome(best, test_data)
  io.println(
    "║  Final Accuracy: "
    <> format_percent(final_acc)
    <> " ("
    <> int.to_string(float.round(final_acc *. int.to_float(test_samples)))
    <> "/"
    <> int.to_string(test_samples)
    <> " correct)        ║",
  )
  io.println(
    "║  Generations:    "
    <> pad_left(int.to_string(final_stats.generation), 3)
    <> "                                      ║",
  )
  io.println(
    "║  Total Nodes:    "
    <> pad_left(int.to_string(final_stats.total_nodes), 5)
    <> "                                  ║",
  )
  io.println(
    "║  Total Conns:    "
    <> pad_left(int.to_string(final_stats.total_connections), 5)
    <> "                                  ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════╝",
  )

  // Analyze best genome
  io.println("")
  io.println("Best Genome Analysis:")
  io.println("   Nodes:       " <> int.to_string(list.length(best.nodes)))
  io.println("   Connections: " <> int.to_string(list.length(best.connections)))
  io.println("   Fitness:     " <> float.to_string(best.fitness))
}

// =============================================================================
// EVOLUTION LOOP
// =============================================================================

fn evolve(
  population: Population,
  config: NeatConfig,
  safe_config: SafeMutationConfig,
  elite_config: EliteConfig,
  train_data: List(#(List(Float), Int)),
  test_data: List(#(List(Float), Int)),
  generation: Int,
  prev_stats: PopulationStats,
) -> #(Population, PopulationStats, Genome) {
  // Evaluate fitness
  let evaluated =
    Population(
      ..population,
      genomes: list.map(population.genomes, fn(g) {
        let fitness = evaluate_genome(g, train_data)
        Genome(..g, fitness: fitness, adjusted_fitness: fitness)
      }),
    )

  // Calculate statistics
  let stats =
    neat_advanced.calculate_stats(
      evaluated,
      generation,
      prev_stats.best_fitness,
      prev_stats.stagnant_generations,
    )

  // Print progress
  io.println(
    " "
    <> pad_left(int.to_string(generation), 3)
    <> " │ "
    <> format_percent(stats.best_fitness)
    <> " │ "
    <> format_percent(stats.avg_fitness)
    <> " │ "
    <> pad_left(float.to_string(float_truncate(stats.fitness_std_dev, 3)), 6)
    <> " │    "
    <> pad_left(int.to_string(stats.stagnant_generations), 2)
    <> "    │ "
    <> pad_left(int.to_string(stats.total_nodes), 5)
    <> " │ "
    <> pad_left(int.to_string(stats.total_connections), 5),
  )

  // Get best genome
  let best =
    list.fold(
      evaluated.genomes,
      case list.first(evaluated.genomes) {
        Ok(g) -> g
        Error(_) -> panic as "Empty population"
      },
      fn(acc, g) {
        case g.fitness >. acc.fitness {
          True -> g
          False -> acc
        }
      },
    )

  // Check termination
  case generation >= max_generations || stats.best_fitness >=. target_fitness {
    True -> #(evaluated, stats, best)
    False -> {
      // Select elites
      let elites = neat_advanced.select_elites(evaluated.genomes, elite_config)

      // Create offspring
      let offspring =
        breed_population(evaluated, config, safe_config, stats.stagnant_generations, generation)

      // Combine elites + offspring
      let next_genomes =
        list.append(elites, offspring)
        |> list.take(population_size)
        |> assign_new_ids(0)

      let next_pop = Population(..evaluated, genomes: next_genomes)

      // Recurse
      evolve(
        next_pop,
        config,
        safe_config,
        elite_config,
        train_data,
        test_data,
        generation + 1,
        stats,
      )
    }
  }
}

fn breed_population(
  population: Population,
  config: NeatConfig,
  safe_config: SafeMutationConfig,
  stagnant: Int,
  gen: Int,
) -> List(Genome) {
  let pop_size = list.length(population.genomes)
  let elite_count = pop_size / 5
  let offspring_count = pop_size - elite_count

  list.index_map(list.range(0, offspring_count - 1), fn(_, i) {
    let seed = gen * 10_000 + i * 100

    // Tournament selection for parents
    let parent1 = tournament_select(population.genomes, 4, seed)
    let parent2 = tournament_select(population.genomes, 4, seed + 50)

    // Lamarckian crossover
    let child =
      neat_advanced.lamarckian_crossover(
        parent1,
        parent1.fitness,
        parent2,
        parent2.fitness,
        LamarckianBlend,
        seed + 1,
      )

    // Safe mutation on weights
    let mutated =
      neat_advanced.safe_mutate_weights(child, safe_config, stagnant, seed + 2)

    // Structural mutations
    let #(struct_mutated, _) = case pseudo_random(seed + 3) <. config.add_node_rate {
      True -> neat.mutate_add_node(mutated, population, seed + 4)
      False -> #(mutated, population)
    }

    case pseudo_random(seed + 5) <. config.add_connection_rate {
      True -> neat.mutate_add_connection(struct_mutated, population, seed + 6).0
      False -> struct_mutated
    }
  })
}

fn tournament_select(genomes: List(Genome), tournament_size: Int, seed: Int) -> Genome {
  let pop_size = list.length(genomes)
  let candidates =
    list.index_map(list.range(0, tournament_size - 1), fn(_, i) {
      let idx = float.round(pseudo_random(seed + i * 7) *. int.to_float(pop_size - 1))
      list_at(genomes, idx)
      |> result.unwrap(case list.first(genomes) {
        Ok(g) -> g
        Error(_) -> panic as "Empty population"
      })
    })

  list.fold(
    candidates,
    case list.first(candidates) {
      Ok(g) -> g
      Error(_) -> panic as "Empty tournament"
    },
    fn(best, candidate) {
      case candidate.fitness >. best.fitness {
        True -> candidate
        False -> best
      }
    },
  )
}

fn assign_new_ids(genomes: List(Genome), start_id: Int) -> List(Genome) {
  list.index_map(genomes, fn(g, i) { Genome(..g, id: start_id + i) })
}

// =============================================================================
// FITNESS EVALUATION
// =============================================================================

fn evaluate_genome(genome: Genome, data: List(#(List(Float), Int))) -> Float {
  let correct =
    list.fold(data, 0, fn(acc, sample) {
      let #(input, label) = sample
      let output = neat.forward(genome, input)
      let predicted = argmax(output)
      case predicted == label {
        True -> acc + 1
        False -> acc
      }
    })
  int.to_float(correct) /. int.to_float(list.length(data))
}

fn argmax(values: List(Float)) -> Int {
  let indexed = list.index_map(values, fn(v, i) { #(i, v) })
  let best =
    list.fold(indexed, #(0, -999_999.0), fn(acc, item) {
      case item.1 >. acc.1 {
        True -> item
        False -> acc
      }
    })
  best.0
}

// =============================================================================
// DATA GENERATION
// =============================================================================

fn generate_dataset(samples_per_digit: Int) -> List(#(List(Float), Int)) {
  list.flatten(
    list.map(list.range(0, 9), fn(digit) {
      list.map(list.range(0, samples_per_digit - 1), fn(variation) {
        let seed = digit * 1000 + variation
        #(generate_digit_image(digit, seed), digit)
      })
    }),
  )
}

fn generate_test_set(total: Int) -> List(#(List(Float), Int)) {
  list.map(list.range(0, total - 1), fn(i) {
    let digit = i % 10
    let seed = 99_999 + i
    #(generate_digit_image(digit, seed), digit)
  })
}

fn generate_digit_image(digit: Int, seed: Int) -> List(Float) {
  let base = get_digit_template(digit)
  list.index_map(base, fn(v, i) {
    let n = { pseudo_random(seed + i * 3) -. 0.5 } *. 0.2
    clamp(v +. n, 0.0, 1.0)
  })
}

fn get_digit_template(digit: Int) -> List(Float) {
  case digit {
    0 ->
      [
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
      ]
    1 ->
      [
        0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
      ]
    2 ->
      [
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
      ]
    3 ->
      [
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
      ]
    4 ->
      [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
      ]
    5 ->
      [
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
      ]
    6 ->
      [
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
      ]
    7 ->
      [
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ]
    8 ->
      [
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
      ]
    9 ->
      [
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      ]
    _ -> list.repeat(0.0, 64)
  }
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

fn float_truncate(x: Float, decimals: Int) -> Float {
  let mult = int.to_float(pow10(decimals))
  int.to_float(float.round(x *. mult)) /. mult
}

fn pow10(n: Int) -> Int {
  case n {
    0 -> 1
    1 -> 10
    2 -> 100
    3 -> 1000
    _ -> 10_000
  }
}

fn format_percent(x: Float) -> String {
  let pct = float_truncate(x *. 100.0, 1)
  pad_left(float.to_string(pct), 5) <> "%"
}

fn pad_left(s: String, width: Int) -> String {
  let len = string.length(s)
  case len >= width {
    True -> s
    False -> string.repeat(" ", width - len) <> s
  }
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
