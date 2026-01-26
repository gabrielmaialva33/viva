//// MNIST NEAT Hybrid COMPETITIVE - Optimized for Thesis Defense
////
//// Versão otimizada com:
//// - Dataset expandido (50 variações × 10 dígitos = 500 amostras)
//// - Métricas discretas (X/Y corretos)
//// - Configuração de elite preservation
//// - Taxas de mutação otimizadas
////
//// Usage: gleam run -m viva/demo/mnist_neat_competitive

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
// OPTIMIZED CONSTANTS
// =============================================================================

/// Image size (8x8 = 64 pixels)
const image_size: Int = 64

/// Number of classes (0-9)
const num_classes: Int = 10

/// Population size (larger for diversity)
const population_size: Int = 100

/// Number of generations
const max_generations: Int = 50

/// Target fitness to stop early
const target_fitness: Float = 0.9

/// Variations per digit (50 = 500 total samples)
const variations_per_digit: Int = 50

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║    VIVA NEAT Hybrid COMPETITIVE - Thesis Defense Version     ║")
  io.println("╠══════════════════════════════════════════════════════════════╣")
  io.println("║  8x8 digits • Pop: 100 • Gen: 50 • Dataset: 500 samples      ║")
  io.println("╚══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Load expanded data
  io.println("Loading expanded digit samples...")
  let #(train_data, test_data) = load_data()
  let train_size = list.length(train_data)
  let test_size = list.length(test_data)
  io.println("   Train: " <> int.to_string(train_size) <> " samples")
  io.println("   Test:  " <> int.to_string(test_size) <> " samples")
  io.println("")

  // Optimized NEAT config
  io.println("Creating optimized NEAT Hybrid population...")
  let neat_config =
    neat.NeatConfig(
      ..neat.default_config(),
      population_size: population_size,
      num_inputs: image_size,
      num_outputs: num_classes,
      // Optimized mutation rates
      add_node_rate: 0.05,
      add_connection_rate: 0.1,
      weight_mutation_rate: 0.85,
      weight_perturb_rate: 0.9,
      compatibility_threshold: 3.0,
    )

  let hybrid_config =
    neat_hybrid.HybridConfig(
      base_config: neat_config,
      // Higher conv rate for image data
      add_conv_rate: 0.08,
      add_attention_rate: 0.02,
      add_pool_rate: 0.04,
      module_param_mutation_rate: 0.15,
      max_modules: 4,
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
  io.println("Starting evolution...")
  io.println("   Gen │ Best Acc │ Correct/Total │ Avg Fit │ Species │ Modules")
  io.println("───────┼──────────┼───────────────┼─────────┼─────────┼─────────")

  let #(final_pop, best_genome) =
    evolve(population, hybrid_config, train_data, test_data, 0, test_size)

  // Final results
  io.println("")
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║                      EVOLUTION COMPLETE                      ║")
  io.println("╠══════════════════════════════════════════════════════════════╣")

  let #(final_correct, final_total) =
    evaluate_genome_discrete(best_genome, test_data, hybrid_config)
  let final_accuracy =
    int_to_float(final_correct) /. int_to_float(final_total) *. 100.0

  io.println(
    "║  Test Accuracy: "
    <> int.to_string(final_correct)
    <> "/"
    <> int.to_string(final_total)
    <> " = "
    <> format_float(final_accuracy, 1)
    <> "%"
    <> string.repeat(" ", 20)
    <> "║",
  )
  io.println(
    "║  Generation:    "
    <> string.pad_end(int.to_string(final_pop.generation), 40, " ")
    <> "║",
  )
  io.println(
    "║  Modules:       "
    <> string.pad_end(int.to_string(list.length(best_genome.modules)), 40, " ")
    <> "║",
  )
  io.println("╠══════════════════════════════════════════════════════════════╣")
  io.println("║  Best Architecture:                                          ║")
  describe_architecture(best_genome)
  io.println("╚══════════════════════════════════════════════════════════════╝")

  // Show predictions
  io.println("")
  io.println("Sample Predictions:")
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
  test_size: Int,
) -> #(HybridPopulation, HybridGenome) {
  // Evaluate on training data
  let evaluated =
    neat_hybrid.evaluate_population(pop, fn(genome) {
      let #(correct, total) =
        evaluate_genome_discrete(genome, train_data, config)
      int_to_float(correct) /. int_to_float(total)
    })

  // Find best genome and evaluate on test
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

  // Test set evaluation (discrete)
  let #(test_correct, test_total) =
    evaluate_genome_discrete(best_genome, test_data, config)
  let test_accuracy = int_to_float(test_correct) /. int_to_float(test_total)

  // Calculate stats
  let avg_fitness =
    list.fold(evaluated.genomes, 0.0, fn(acc, g) { acc +. g.base.fitness })
    /. int_to_float(list.length(evaluated.genomes))

  let num_modules =
    list.fold(evaluated.genomes, 0, fn(acc, g) { acc + list.length(g.modules) })
    / list.length(evaluated.genomes)

  let species_count = count_species(evaluated.genomes)

  // Print progress with discrete metrics
  print_generation(
    gen,
    test_accuracy,
    test_correct,
    test_total,
    avg_fitness,
    species_count,
    num_modules,
    best_genome,
  )

  // Check termination
  case gen >= max_generations || test_accuracy >=. target_fitness {
    True -> #(evaluated, best_genome)
    False -> {
      let next_pop = next_generation(evaluated, config, gen)
      evolve(next_pop, config, train_data, test_data, gen + 1, test_size)
    }
  }
}

fn next_generation(
  pop: HybridPopulation,
  config: HybridConfig,
  gen: Int,
) -> HybridPopulation {
  let sorted =
    list.sort(pop.genomes, fn(a, b) {
      float.compare(b.base.fitness, a.base.fitness)
    })

  // Elite preservation: top 15%
  let elite_count = list.length(sorted) * 15 / 100
  let elites = list.take(sorted, int.max(3, elite_count))

  // Generate children
  let children =
    list.range(0, population_size - list.length(elites) - 1)
    |> list.map(fn(i) {
      let parent1 = tournament_select(sorted, gen * 1000 + i)
      let parent2 = tournament_select(sorted, gen * 1000 + i + 500)
      let child = neat_hybrid.crossover(parent1, parent2, gen * 1000 + i)
      neat_hybrid.mutate(child, config, gen * 1000 + i + 1000)
    })

  HybridPopulation(
    ..pop,
    genomes: list.append(elites, children),
    generation: pop.generation + 1,
  )
}

fn tournament_select(genomes: List(HybridGenome), seed: Int) -> HybridGenome {
  let tournament_size = 5
  // Larger tournament for more selective pressure
  let candidates =
    list.range(0, tournament_size - 1)
    |> list.filter_map(fn(i) {
      let idx = pseudo_random(seed + i) % list.length(genomes)
      list.drop(genomes, idx) |> list.first()
    })

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
// EVALUATION (DISCRETE)
// =============================================================================

fn evaluate_genome_discrete(
  genome: HybridGenome,
  data: List(#(Tensor, Int)),
  config: HybridConfig,
) -> #(Int, Int) {
  let correct =
    list.fold(data, 0, fn(acc, sample) {
      let #(input, label) = sample
      let output = neat_hybrid.forward(genome, input, config)
      let predicted = argmax(output)
      case predicted == label {
        True -> acc + 1
        False -> acc
      }
    })

  #(correct, list.length(data))
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
// DATA LOADING (EXPANDED)
// =============================================================================

fn load_data() -> #(List(#(Tensor, Int)), List(#(Tensor, Int))) {
  let all_samples = generate_samples()
  let shuffled = shuffle_samples(all_samples, 42)
  // 80/20 split
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
  list.range(0, variations_per_digit - 1)
  |> list.map(fn(i) {
    let augmented = augment_pattern(pattern, i, label)
    #(tensor.from_list(augmented), label)
  })
}

fn augment_pattern(pattern: List(Float), seed: Int, label: Int) -> List(Float) {
  // More sophisticated augmentation
  list.index_map(pattern, fn(p, i) {
    // Noise based on seed
    let noise_factor = case seed % 5 {
      0 -> 0.1
      // Low noise
      1 -> 0.15
      // Medium-low
      2 -> 0.2
      // Medium
      3 -> 0.25
      // Medium-high
      _ -> 0.3
      // High noise
    }
    let noise =
      { pseudo_random_float(seed * 100 + i + label * 1000) -. 0.5 }
      *. noise_factor
    float.clamp(p +. noise, 0.0, 1.0)
  })
}

// =============================================================================
// OUTPUT
// =============================================================================

fn print_generation(
  gen: Int,
  test_acc: Float,
  correct: Int,
  total: Int,
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
    <> string.pad_start(format_float(test_acc *. 100.0, 1) <> "%", 7, " ")
    <> " │ "
    <> string.pad_start(
      int.to_string(correct) <> "/" <> int.to_string(total),
      13,
      " ",
    )
    <> " │ "
    <> string.pad_start(format_float(avg_fit *. 100.0, 1) <> "%", 6, " ")
    <> " │    "
    <> string.pad_start(int.to_string(species), 2, " ")
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
    <> string.pad_end(int.to_string(hidden_count), 40, " ")
    <> "║",
  )
  io.println(
    "║    Connections:  "
    <> string.pad_end(int.to_string(conn_count), 40, " ")
    <> "║",
  )

  case genome.modules {
    [] ->
      io.println(
        "║    Modules:      None (pure NEAT)                            ║",
      )
    modules -> {
      io.println(
        "║    Modules:                                                  ║",
      )
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
        io.println("║      - " <> string.pad_end(desc, 50, " ") <> "  ║")
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
  |> list.take(10)
  |> list.each(fn(sample) {
    let #(input, label) = sample
    let output = neat_hybrid.forward(genome, input, config)
    let predicted = argmax(output)
    let confidence = case list.drop(output, predicted) |> list.first() {
      Ok(c) -> c
      Error(_) -> 0.0
    }

    let status = case predicted == label {
      True -> "OK"
      False -> "XX"
    }

    io.println(
      "   "
      <> status
      <> " Actual: "
      <> int.to_string(label)
      <> " -> Predicted: "
      <> int.to_string(predicted)
      <> " (conf: "
      <> format_float(confidence *. 100.0, 1)
      <> "%)",
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

fn format_float(f: Float, decimals: Int) -> String {
  let factor = case decimals {
    0 -> 1.0
    1 -> 10.0
    2 -> 100.0
    _ -> 1000.0
  }
  let scaled = f *. factor
  let whole = float_to_int(scaled)
  let integer_part = whole / float_to_int(factor)
  let decimal_part = whole % float_to_int(factor)

  case decimals {
    0 -> int.to_string(integer_part)
    _ ->
      int.to_string(integer_part)
      <> "."
      <> string.pad_start(int.to_string(decimal_part), decimals, "0")
  }
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
