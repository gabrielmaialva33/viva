//// MNIST NEAT Hybrid Demo - Evolving Neural Architecture for Digit Recognition
////
//// Demonstrates NEAT Hybrid evolving architectures with Conv/Attention modules
//// for handwritten digit classification. Evolution discovers optimal architecture.
////
//// Key Features:
//// - NEAT evolves topology (nodes + connections)
//// - Hybrid modules (Conv, Attention, Pool) can be evolved
//// - Fitness = classification accuracy on test set
//// - Shows architecture of best genome each generation
////
//// Usage: gleam run -m viva/demo/mnist_neat

import gleam/dict
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import viva/neural/neat
import viva/neural/neat_hybrid.{
  type HybridConfig, type HybridGenome, type HybridPopulation, AttentionBlock,
  ConvBlock, Dense, HybridPopulation, PoolBlock,
}
import viva/neural/tensor.{type Tensor}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Image size (8x8 = 64 pixels)
const image_size: Int = 64

/// Number of classes (0-9)
const num_classes: Int = 10

/// Population size
const population_size: Int = 50

/// Number of generations
const max_generations: Int = 30

/// Target fitness to stop early
const target_fitness: Float = 0.85

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║       VIVA MNIST NEAT Hybrid - Evolving Architectures        ║")
  io.println("╠══════════════════════════════════════════════════════════════╣")
  io.println("║  8x8 digits • Pop: 50 • Gen: 30 • Conv/Attention Evolution   ║")
  io.println("╚══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Load data
  io.println("📦 Loading digit samples...")
  let #(train_data, test_data) = load_data()
  io.println(
    "   Train: " <> int.to_string(list.length(train_data)) <> " samples",
  )
  io.println("   Test:  " <> int.to_string(list.length(test_data)) <> " samples")
  io.println("")

  // Create NEAT config for image classification
  io.println("🧬 Creating NEAT Hybrid population...")
  let neat_config =
    neat.NeatConfig(
      ..neat.default_config(),
      population_size: population_size,
      num_inputs: image_size,
      num_outputs: num_classes,
      add_node_rate: 0.1,
      add_connection_rate: 0.15,
      weight_mutation_rate: 0.9,
      compatibility_threshold: 2.0,
    )

  let hybrid_config =
    neat_hybrid.HybridConfig(
      base_config: neat_config,
      add_conv_rate: 0.05,
      add_attention_rate: 0.03,
      add_pool_rate: 0.02,
      module_param_mutation_rate: 0.2,
      max_modules: 3,
      input_is_2d: True,
      input_height: 8,
      input_width: 8,
      input_channels: 1,
    )

  // Create initial population
  let base_pop = neat.create_population(neat_config, 42)
  let population = neat_hybrid.from_population(base_pop)
  io.println("   Population size: " <> int.to_string(population_size))
  io.println("   Inputs: " <> int.to_string(image_size) <> " (8x8 image)")
  io.println("   Outputs: " <> int.to_string(num_classes) <> " (digits 0-9)")
  io.println("")

  // Evolution loop
  io.println("🧬 Starting evolution...")
  io.println("   Gen │ Best Fit │ Avg Fit │ Species │ Modules │ Architecture")
  io.println("───────┼──────────┼─────────┼─────────┼─────────┼─────────────────")

  let #(final_pop, best_genome) =
    evolve(population, hybrid_config, train_data, test_data, 0)

  // Final results
  io.println("")
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║                      EVOLUTION COMPLETE                      ║")
  io.println("╠══════════════════════════════════════════════════════════════╣")

  let final_fitness =
    evaluate_genome(best_genome, test_data, hybrid_config)
  io.println(
    "║  Best Fitness: "
    <> string.pad_end(format_percent(final_fitness), 8, " ")
    <> "                                 ║",
  )
  io.println(
    "║  Generation:   "
    <> string.pad_end(int.to_string(final_pop.generation), 8, " ")
    <> "                                 ║",
  )
  io.println(
    "║  Modules:      "
    <> string.pad_end(int.to_string(list.length(best_genome.modules)), 8, " ")
    <> "                                 ║",
  )
  io.println("╠══════════════════════════════════════════════════════════════╣")
  io.println("║  Best Architecture:                                          ║")
  describe_architecture(best_genome)
  io.println("╚══════════════════════════════════════════════════════════════╝")

  // Show some predictions
  io.println("")
  io.println("🎯 Sample Predictions:")
  show_predictions(best_genome, test_data, hybrid_config)
}

// =============================================================================
// EVOLUTION
// =============================================================================

fn evolve(
  pop: HybridPopulation,
  config: HybridConfig,
  train_data: List(#(Tensor, Int)),
  test_data: List(#(Tensor, Int)),
  gen: Int,
) -> #(HybridPopulation, HybridGenome) {
  // Evaluate population
  let evaluated =
    neat_hybrid.evaluate_population(pop, fn(genome) {
      evaluate_genome(genome, train_data, config)
    })

  // Find best genome
  let best =
    list.fold(evaluated.genomes, list.first(evaluated.genomes), fn(acc, g) {
      case acc {
        Ok(best) ->
          case g.base.fitness >. best.base.fitness {
            True -> Ok(g)
            False -> acc
          }
        Error(_) -> Ok(g)
      }
    })

  let best_genome = case best {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }

  // Calculate stats
  let avg_fitness =
    list.fold(evaluated.genomes, 0.0, fn(acc, g) { acc +. g.base.fitness })
    /. int_to_float(list.length(evaluated.genomes))

  let num_modules =
    list.fold(evaluated.genomes, 0, fn(acc, g) {
      acc + list.length(g.modules)
    })
    / list.length(evaluated.genomes)

  // Count species (simplified - count unique module patterns)
  let species_count = count_species(evaluated.genomes)

  // Print progress
  print_generation(
    gen,
    best_genome.base.fitness,
    avg_fitness,
    species_count,
    num_modules,
    best_genome,
  )

  // Check termination conditions
  case gen >= max_generations || best_genome.base.fitness >=. target_fitness {
    True -> #(evaluated, best_genome)
    False -> {
      // Create next generation
      let next_pop = next_generation(evaluated, config, gen)
      evolve(next_pop, config, train_data, test_data, gen + 1)
    }
  }
}

fn next_generation(
  pop: HybridPopulation,
  config: HybridConfig,
  gen: Int,
) -> HybridPopulation {
  // Sort by fitness
  let sorted =
    list.sort(pop.genomes, fn(a, b) {
      float.compare(b.base.fitness, a.base.fitness)
    })

  // Keep top 10% as elites
  let elite_count = list.length(sorted) / 10
  let elites = list.take(sorted, int.max(2, elite_count))

  // Generate children through crossover and mutation
  let children =
    list.range(0, population_size - list.length(elites) - 1)
    |> list.map(fn(i) {
      // Tournament selection
      let parent1 = tournament_select(sorted, gen * 1000 + i)
      let parent2 = tournament_select(sorted, gen * 1000 + i + 500)

      // Crossover
      let child = neat_hybrid.crossover(parent1, parent2, gen * 1000 + i)

      // Mutation
      neat_hybrid.mutate(child, config, gen * 1000 + i + 1000)
    })

  HybridPopulation(
    ..pop,
    genomes: list.append(elites, children),
    generation: pop.generation + 1,
  )
}

fn tournament_select(genomes: List(HybridGenome), seed: Int) -> HybridGenome {
  let tournament_size = 3
  let candidates =
    list.range(0, tournament_size - 1)
    |> list.filter_map(fn(i) {
      let idx = pseudo_random(seed + i) % list.length(genomes)
      list.drop(genomes, idx) |> list.first()
    })

  // Return best from tournament
  case
    list.reduce(candidates, fn(best, curr) {
      case curr.base.fitness >. best.base.fitness {
        True -> curr
        False -> best
      }
    })
  {
    Ok(winner) -> winner
    Error(_) ->
      case list.first(genomes) {
        Ok(g) -> g
        Error(_) -> panic as "No genomes for selection"
      }
  }
}

fn count_species(genomes: List(HybridGenome)) -> Int {
  // Simplified species counting based on module types
  let patterns =
    list.map(genomes, fn(g) {
      list.map(g.modules, fn(m) {
        case m.module_type {
          Dense -> 0
          ConvBlock(_) -> 1
          AttentionBlock(_) -> 2
          PoolBlock(_) -> 3
          _ -> 4
        }
      })
    })

  list.unique(patterns) |> list.length()
}

// =============================================================================
// EVALUATION
// =============================================================================

fn evaluate_genome(
  genome: HybridGenome,
  data: List(#(Tensor, Int)),
  config: HybridConfig,
) -> Float {
  let results =
    list.map(data, fn(sample) {
      let #(input, label) = sample
      let output = neat_hybrid.forward(genome, input, config)

      // Get predicted class (argmax of output)
      let predicted = argmax(output)

      case predicted == label {
        True -> 1.0
        False -> 0.0
      }
    })

  // Return accuracy
  let correct = list.fold(results, 0.0, fn(a, b) { a +. b })
  correct /. int_to_float(list.length(data))
}

fn argmax(values: List(Float)) -> Int {
  values
  |> list.index_map(fn(val, idx) { #(idx, val) })
  |> list.fold(#(0, -1_000_000.0), fn(best, curr) {
    case curr.1 >. best.1 {
      True -> curr
      False -> best
    }
  })
  |> fn(pair) { pair.0 }
}

// =============================================================================
// DATA LOADING
// =============================================================================

fn load_data() -> #(List(#(Tensor, Int)), List(#(Tensor, Int))) {
  let all_samples = generate_samples()

  // Shuffle deterministically
  let shuffled = shuffle_samples(all_samples, 42)

  // Split 80/20
  let split_idx = list.length(shuffled) * 8 / 10
  list.split(shuffled, split_idx)
}

fn shuffle_samples(
  samples: List(#(Tensor, Int)),
  seed: Int,
) -> List(#(Tensor, Int)) {
  samples
  |> list.index_map(fn(s, i) {
    let key = pseudo_random(seed + i * 1000)
    #(key, s)
  })
  |> list.sort(fn(a, b) { int.compare(a.0, b.0) })
  |> list.map(fn(pair) { pair.1 })
}

fn generate_samples() -> List(#(Tensor, Int)) {
  // Generate 15 variations of each digit
  list.flatten([
    generate_digit_variations(0, digit_0_pattern()),
    generate_digit_variations(1, digit_1_pattern()),
    generate_digit_variations(2, digit_2_pattern()),
    generate_digit_variations(3, digit_3_pattern()),
    generate_digit_variations(4, digit_4_pattern()),
    generate_digit_variations(5, digit_5_pattern()),
    generate_digit_variations(6, digit_6_pattern()),
    generate_digit_variations(7, digit_7_pattern()),
    generate_digit_variations(8, digit_8_pattern()),
    generate_digit_variations(9, digit_9_pattern()),
  ])
}

fn generate_digit_variations(
  label: Int,
  pattern: List(Float),
) -> List(#(Tensor, Int)) {
  list.range(0, 14)
  |> list.map(fn(i) {
    let augmented = augment_pattern(pattern, i)
    #(tensor.from_list(augmented), label)
  })
}

fn augment_pattern(pattern: List(Float), seed: Int) -> List(Float) {
  // Add small noise
  list.index_map(pattern, fn(p, i) {
    let noise = { pseudo_random_float(seed * 100 + i) -. 0.5 } *. 0.2
    float.clamp(p +. noise, 0.0, 1.0)
  })
}

// =============================================================================
// OUTPUT
// =============================================================================

fn print_generation(
  gen: Int,
  best_fit: Float,
  avg_fit: Float,
  species: Int,
  modules: Int,
  best: HybridGenome,
) {
  let arch = describe_architecture_short(best)

  io.println(
    "   "
    <> string.pad_start(int.to_string(gen), 3, " ")
    <> " │ "
    <> string.pad_start(format_percent(best_fit), 7, " ")
    <> " │ "
    <> string.pad_start(format_percent(avg_fit), 6, " ")
    <> " │    "
    <> string.pad_start(int.to_string(species), 2, " ")
    <> "   │    "
    <> string.pad_start(int.to_string(modules), 2, " ")
    <> "   │ "
    <> arch,
  )
}

fn describe_architecture_short(genome: HybridGenome) -> String {
  let hidden_count =
    list.filter(genome.base.nodes, fn(n) { n.node_type == neat.Hidden })
    |> list.length()

  let conn_count =
    list.filter(genome.base.connections, fn(c) { c.enabled })
    |> list.length()

  let modules =
    genome.modules
    |> list.filter(fn(m) { m.enabled })
    |> list.map(fn(m) {
      case m.module_type {
        Dense -> "D"
        ConvBlock(_) -> "C"
        AttentionBlock(_) -> "A"
        PoolBlock(_) -> "P"
        _ -> "?"
      }
    })
    |> string.join("")

  let module_str = case modules {
    "" -> ""
    _ -> "[" <> modules <> "]"
  }

  int.to_string(hidden_count)
  <> "h/"
  <> int.to_string(conn_count)
  <> "c"
  <> module_str
}

fn describe_architecture(genome: HybridGenome) {
  let hidden_count =
    list.filter(genome.base.nodes, fn(n) { n.node_type == neat.Hidden })
    |> list.length()

  let conn_count =
    list.filter(genome.base.connections, fn(c) { c.enabled })
    |> list.length()

  io.println(
    "║    Hidden nodes: "
    <> string.pad_end(int.to_string(hidden_count), 4, " ")
    <> "                                   ║",
  )
  io.println(
    "║    Connections:  "
    <> string.pad_end(int.to_string(conn_count), 4, " ")
    <> "                                   ║",
  )

  case genome.modules {
    [] ->
      io.println(
        "║    Modules:      None (pure NEAT)                            ║",
      )
    modules -> {
      io.println("║    Modules:                                                  ║")
      list.each(modules, fn(m) {
        let desc = case m.module_type {
          Dense -> "Dense"
          ConvBlock(c) ->
            "Conv2D("
            <> int.to_string(c.out_channels)
            <> "ch, "
            <> int.to_string(c.kernel_size)
            <> "x"
            <> int.to_string(c.kernel_size)
            <> ")"
          AttentionBlock(a) ->
            "Attention(d="
            <> int.to_string(a.d_model)
            <> ", h="
            <> int.to_string(a.num_heads)
            <> ")"
          PoolBlock(p) ->
            case p.pool_type {
              neat_hybrid.MaxPool -> "MaxPool"
              neat_hybrid.AvgPool -> "AvgPool"
            }
            <> "("
            <> int.to_string(p.kernel_size)
            <> "x"
            <> int.to_string(p.kernel_size)
            <> ")"
          _ -> "Unknown"
        }
        io.println(
          "║      - "
          <> string.pad_end(desc, 50, " ")
          <> "  ║",
        )
      })
    }
  }
}

fn show_predictions(
  genome: HybridGenome,
  data: List(#(Tensor, Int)),
  config: HybridConfig,
) {
  data
  |> list.take(8)
  |> list.each(fn(sample) {
    let #(input, label) = sample
    let output = neat_hybrid.forward(genome, input, config)
    let predicted = argmax(output)
    let confidence = case list.drop(output, predicted) |> list.first() {
      Ok(c) -> c
      Error(_) -> 0.0
    }

    let status = case predicted == label {
      True -> "✓"
      False -> "✗"
    }

    io.println(
      "   "
      <> status
      <> " Actual: "
      <> int.to_string(label)
      <> " → Predicted: "
      <> int.to_string(predicted)
      <> " ("
      <> format_percent(confidence)
      <> ")",
    )
  })
}

// =============================================================================
// DIGIT PATTERNS (8x8)
// =============================================================================

fn digit_0_pattern() -> List(Float) {
  [
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0,
  ]
}

fn digit_1_pattern() -> List(Float) {
  [
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 0.0,
  ]
}

fn digit_2_pattern() -> List(Float) {
  [
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 0.0,
  ]
}

fn digit_3_pattern() -> List(Float) {
  [
    0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0,
  ]
}

fn digit_4_pattern() -> List(Float) {
  [
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0,
  ]
}

fn digit_5_pattern() -> List(Float) {
  [
    0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0,
  ]
}

fn digit_6_pattern() -> List(Float) {
  [
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0,
  ]
}

fn digit_7_pattern() -> List(Float) {
  [
    0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0,
  ]
}

fn digit_8_pattern() -> List(Float) {
  [
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0,
  ]
}

fn digit_9_pattern() -> List(Float) {
  [
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0,
  ]
}

// =============================================================================
// HELPERS
// =============================================================================

fn pseudo_random(seed: Int) -> Int {
  let x = seed * 1_103_515_245 + 12_345
  { x / 65_536 } % 32_768
}

fn pseudo_random_float(seed: Int) -> Float {
  int_to_float(pseudo_random(seed) % 1000) /. 1000.0
}

fn int_to_float(n: Int) -> Float {
  case n >= 0 {
    True -> positive_int_to_float(n, 0.0)
    False -> 0.0 -. positive_int_to_float(0 - n, 0.0)
  }
}

fn positive_int_to_float(n: Int, acc: Float) -> Float {
  case n {
    0 -> acc
    _ -> positive_int_to_float(n - 1, acc +. 1.0)
  }
}

fn format_percent(f: Float) -> String {
  let pct = f *. 100.0
  let whole = float_to_int(pct)
  let frac = float_to_int({ pct -. int_to_float(whole) } *. 10.0)
  int.to_string(whole) <> "." <> int.to_string(frac) <> "%"
}

fn float_to_int(f: Float) -> Int {
  case f <. 0.0 {
    True -> 0 - float_to_int_positive(0.0 -. f)
    False -> float_to_int_positive(f)
  }
}

fn float_to_int_positive(f: Float) -> Int {
  float_to_int_loop(f, 0)
}

fn float_to_int_loop(f: Float, acc: Int) -> Int {
  case f <. 1.0 {
    True -> acc
    False -> float_to_int_loop(f -. 1.0, acc + 1)
  }
}
