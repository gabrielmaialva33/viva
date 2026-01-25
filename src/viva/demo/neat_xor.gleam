//// NEAT XOR Demo - Evolução de rede neural para resolver XOR
////
//// O problema XOR é o teste clássico para NEAT:
//// - Não é linearmente separável (precisa de hidden nodes)
//// - NEAT deve descobrir a topologia automaticamente
////
//// Usage: gleam run -m viva/demo/neat_xor

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{None, Some}
import gleam/string
import viva/neural/neat.{
  type FitnessResult, type Genome, type Population, FitnessResult,
}

// =============================================================================
// XOR DATASET
// =============================================================================

/// Casos de teste XOR
const xor_inputs: List(List(Float)) = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

const xor_outputs: List(Float) = [0.0, 1.0, 1.0, 0.0]

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║           NEAT XOR Demo - Neuroevolução em Gleam             ║")
  io.println("╠══════════════════════════════════════════════════════════════╣")
  io.println("║  Evolui topologia + pesos para resolver XOR                  ║")
  io.println("╚══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Configuração
  let config = neat.xor_config()
  io.println("📋 Configuração:")
  io.println("   População: " <> int.to_string(config.population_size))
  io.println("   Inputs: " <> int.to_string(config.num_inputs))
  io.println("   Outputs: " <> int.to_string(config.num_outputs))
  io.println("")

  // Cria população inicial
  io.println("🧬 Criando população inicial...")
  let population = neat.create_population(config, 42)
  print_stats(population)
  io.println("")

  // Evolui por várias gerações
  io.println("🔄 Evoluindo...")
  io.println("")

  let #(final_pop, solved) = evolve_loop(population, config, 200, 3.85)

  io.println("")

  case solved {
    True -> {
      io.println("✅ XOR RESOLVIDO!")
    }
    False -> {
      io.println("⚠️ Não convergiu em 100 gerações")
    }
  }

  // Mostra melhor genoma
  io.println("")
  io.println("🏆 Melhor genoma:")
  case neat.get_best(final_pop) {
    Some(best) -> {
      io.println("   " <> neat.genome_to_string(best))
      io.println("   Fitness: " <> format_float(best.fitness, 4))
      io.println("")
      io.println("📊 Testando XOR:")
      test_xor(best)
    }
    None -> io.println("   Nenhum genoma encontrado")
  }

  io.println("")
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║                    DEMO COMPLETE                             ║")
  io.println("╚══════════════════════════════════════════════════════════════╝")
}

// =============================================================================
// EVOLUTION LOOP
// =============================================================================

fn evolve_loop(
  population: Population,
  config: neat.NeatConfig,
  max_generations: Int,
  target_fitness: Float,
) -> #(Population, Bool) {
  case population.generation >= max_generations {
    True -> #(population, False)
    False -> {
      // Avalia fitness de todos os genomas
      let fitness_results = evaluate_population(population)

      // Verifica se resolveu
      let best_fitness =
        list.fold(fitness_results, 0.0, fn(acc, r) {
          float.max(acc, r.fitness)
        })

      // Print progresso a cada 10 gerações
      case population.generation % 10 == 0 || best_fitness >=. target_fitness {
        True -> {
          let stats = neat.get_stats(population)
          io.println(
            "   Gen "
            <> string.pad_start(int.to_string(stats.generation), 3, " ")
            <> " │ Best: "
            <> format_float(best_fitness, 3)
            <> " │ Avg: "
            <> format_float(stats.avg_fitness, 3)
            <> " │ Species: "
            <> int.to_string(stats.num_species)
            <> " │ Nodes: "
            <> format_float(stats.avg_nodes, 1),
          )
        }
        False -> Nil
      }

      case best_fitness >=. target_fitness {
        True -> {
          // Atualiza fitness e retorna
          let final_pop = update_pop_fitness(population, fitness_results)
          #(final_pop, True)
        }
        False -> {
          // Evolui para próxima geração
          let next_pop =
            neat.evolve(
              population,
              fitness_results,
              config,
              42 + population.generation * 1000,
            )
          evolve_loop(next_pop, config, max_generations, target_fitness)
        }
      }
    }
  }
}

fn update_pop_fitness(
  population: Population,
  results: List(FitnessResult),
) -> Population {
  let fitness_map =
    list.fold(results, [], fn(acc, r) { [#(r.genome_id, r.fitness), ..acc] })

  let updated_genomes =
    list.map(population.genomes, fn(g) {
      let fitness =
        list.find(fitness_map, fn(pair) { pair.0 == g.id })
        |> result_map(fn(pair) { pair.1 })
        |> result_unwrap(0.0)
      neat.Genome(..g, fitness: fitness)
    })

  neat.Population(..population, genomes: updated_genomes)
}

// =============================================================================
// FITNESS EVALUATION
// =============================================================================

fn evaluate_population(population: Population) -> List(FitnessResult) {
  list.map(population.genomes, fn(genome) {
    let fitness = evaluate_genome(genome)
    FitnessResult(genome_id: genome.id, fitness: fitness)
  })
}

fn evaluate_genome(genome: Genome) -> Float {
  // Testa todos os casos XOR
  let errors =
    list.zip(xor_inputs, xor_outputs)
    |> list.map(fn(pair) {
      let #(inputs, expected) = pair
      let outputs = neat.forward(genome, inputs)
      let actual = case list.first(outputs) {
        Ok(v) -> v
        Error(_) -> 0.0
      }
      let error = expected -. actual
      error *. error
    })

  let total_error = list.fold(errors, 0.0, fn(acc, e) { acc +. e })

  // Fitness = 4 - sum(squared_errors)
  // Max fitness = 4.0 (quando todos os erros são 0)
  4.0 -. total_error
}

// =============================================================================
// TESTING
// =============================================================================

fn test_xor(genome: Genome) {
  list.zip(xor_inputs, xor_outputs)
  |> list.each(fn(pair) {
    let #(inputs, expected) = pair
    let outputs = neat.forward(genome, inputs)
    let actual = case list.first(outputs) {
      Ok(v) -> v
      Error(_) -> 0.0
    }
    let rounded = case actual >. 0.5 {
      True -> 1.0
      False -> 0.0
    }
    let status = case rounded == expected {
      True -> "✓"
      False -> "✗"
    }
    let in0 = format_float(list_at(inputs, 0), 0)
    let in1 = format_float(list_at(inputs, 1), 0)
    io.println(
      "   "
      <> status
      <> " "
      <> in0
      <> " XOR "
      <> in1
      <> " = "
      <> format_float(actual, 3)
      <> " (expected: "
      <> format_float(expected, 0)
      <> ")",
    )
  })
}

fn print_stats(population: Population) {
  let stats = neat.get_stats(population)
  io.println("   Geração: " <> int.to_string(stats.generation))
  io.println("   Genomas: " <> int.to_string(list.length(population.genomes)))
  io.println("   Espécies: " <> int.to_string(stats.num_species))
  io.println("   Avg nodes: " <> format_float(stats.avg_nodes, 1))
  io.println("   Avg connections: " <> format_float(stats.avg_connections, 1))
}

// =============================================================================
// UTILITIES
// =============================================================================

fn format_float(f: Float, decimals: Int) -> String {
  let multiplier = power_of_10(decimals)
  let rounded = int.to_float(float.round(f *. multiplier)) /. multiplier
  let str = float.to_string(rounded)
  case string.contains(str, ".") {
    True -> str
    False -> str <> ".0"
  }
}

fn power_of_10(n: Int) -> Float {
  case n {
    0 -> 1.0
    1 -> 10.0
    2 -> 100.0
    3 -> 1000.0
    4 -> 10000.0
    _ -> 10000.0
  }
}

fn list_at(lst: List(Float), index: Int) -> Float {
  lst
  |> list.drop(index)
  |> list.first
  |> result_unwrap(0.0)
}

fn result_unwrap(result: Result(a, e), default: a) -> a {
  case result {
    Ok(value) -> value
    Error(_) -> default
  }
}

fn result_map(result: Result(a, e), f: fn(a) -> b) -> Result(b, e) {
  case result {
    Ok(value) -> Ok(f(value))
    Error(e) -> Error(e)
  }
}
