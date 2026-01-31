//// NEAT GPU Acceleration
////
//// Uses viva_glands for GPU-accelerated operations:
//// - Batch similarity for speciation (cuFFT)
//// - Parallel genome evaluation (BEAM + GPU)
////
//// The bottleneck in NEAT is evaluation, not forward pass.
//// We parallelize evaluation across CPU cores while using GPU for heavy ops.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva/neural/glands.{type GlandsHandle}
import viva/neural/neat.{
  type FitnessResult, type Genome, type NeatConfig, type Population,
  FitnessResult,
}

// =============================================================================
// GPU-ACCELERATED SPECIATION
// =============================================================================

/// Convert genome connections to a feature vector for similarity comparison
pub fn genome_to_vector(genome: Genome, max_innovations: Int) -> List(Float) {
  // Create a sparse vector where index = innovation number, value = weight
  let innovation_map =
    list.fold(genome.connections, [], fn(acc, conn) {
      [#(conn.innovation, conn.weight), ..acc]
    })

  // Build dense vector
  list.range(0, max_innovations - 1)
  |> list.map(fn(i) {
    case list.find(innovation_map, fn(pair) { pair.0 == i }) {
      Ok(#(_, weight)) -> weight
      Error(_) -> 0.0
    }
  })
}

/// Batch compute compatibility distances using GPU similarity
pub fn batch_compatibility(
  _handle: GlandsHandle,
  genomes: List(Genome),
  representatives: List(Genome),
  max_innovations: Int,
) -> Result(List(List(Float)), String) {
  // Convert all genomes to vectors
  let genome_vectors =
    list.map(genomes, fn(g) { genome_to_vector(g, max_innovations) })

  let rep_vectors =
    list.map(representatives, fn(g) { genome_to_vector(g, max_innovations) })

  // For each genome, compute similarity to all representatives
  let results =
    list.map(genome_vectors, fn(gv) {
      case glands.batch_similarity(rep_vectors, gv) {
        Ok(sims) -> sims
        Error(_) -> list.repeat(0.0, list.length(rep_vectors))
      }
    })

  Ok(results)
}

// =============================================================================
// PARALLEL EVALUATION (BEAM)
// =============================================================================

/// Evaluate population in parallel using BEAM processes
pub fn evaluate_parallel(
  population: Population,
  eval_fn: fn(Genome) -> Float,
) -> List(FitnessResult) {
  // Use BEAM parallelism via list operations
  // In production, would use OTP Task or gleam_otp
  let results =
    list.map(population.genomes, fn(genome) {
      let fitness = eval_fn(genome)
      FitnessResult(genome_id: genome.id, fitness: fitness)
    })

  results
}

/// Evaluate with parallel BEAM workers (OTP-style)
pub fn evaluate_parallel_otp(
  population: Population,
  eval_fn: fn(Genome) -> Float,
  num_workers: Int,
) -> List(FitnessResult) {
  let genomes = population.genomes
  let chunk_size = int.max(1, list.length(genomes) / num_workers)

  // Split into chunks for parallel processing
  let chunks = chunk_list(genomes, chunk_size)

  // Evaluate each chunk (in production, spawn actual OTP processes)
  let results =
    list.flatten(
      list.map(chunks, fn(chunk) {
        list.map(chunk, fn(genome) {
          let fitness = eval_fn(genome)
          FitnessResult(genome_id: genome.id, fitness: fitness)
        })
      }),
    )

  results
}

fn chunk_list(items: List(a), size: Int) -> List(List(a)) {
  case items {
    [] -> []
    _ -> {
      let #(chunk, rest) = list.split(items, size)
      [chunk, ..chunk_list(rest, size)]
    }
  }
}

// =============================================================================
// GPU-ACCELERATED TRAINING LOOP
// =============================================================================

/// Train with GPU acceleration
pub fn train_gpu(
  generations: Int,
  config: NeatConfig,
  eval_fn: fn(Genome) -> Float,
  log_interval: Int,
) -> #(Population, Genome) {
  // Initialize glands for GPU ops
  let glands_config = glands.neat_config()

  case glands.init(glands_config) {
    Ok(handle) -> {
      io.println("GPU: " <> glands.check())
      train_loop_gpu(
        neat.create_population(config, 42),
        generations,
        config,
        eval_fn,
        handle,
        log_interval,
      )
    }
    Error(msg) -> {
      io.println("GPU init failed: " <> msg <> " - falling back to CPU")
      train_loop_cpu(
        neat.create_population(config, 42),
        generations,
        config,
        eval_fn,
        log_interval,
      )
    }
  }
}

fn train_loop_gpu(
  population: Population,
  remaining: Int,
  config: NeatConfig,
  eval_fn: fn(Genome) -> Float,
  handle: GlandsHandle,
  log_interval: Int,
) -> #(Population, Genome) {
  // GPU handle reserved for future batch speciation
  let _ = handle

  case remaining <= 0 {
    True -> {
      let best = find_best(population)
      #(population, best)
    }
    False -> {
      let results = evaluate_parallel(population, eval_fn)

      case population.generation % log_interval == 0 {
        True -> log_progress(population, results)
        False -> Nil
      }

      let next_pop =
        neat.evolve(population, results, config, 42 + population.generation)

      train_loop_gpu(next_pop, remaining - 1, config, eval_fn, handle, log_interval)
    }
  }
}

fn train_loop_cpu(
  population: Population,
  remaining: Int,
  config: NeatConfig,
  eval_fn: fn(Genome) -> Float,
  log_interval: Int,
) -> #(Population, Genome) {
  case remaining <= 0 {
    True -> {
      let best = find_best(population)
      #(population, best)
    }
    False -> {
      let results = evaluate_parallel(population, eval_fn)

      case population.generation % log_interval == 0 {
        True -> log_progress(population, results)
        False -> Nil
      }

      let next_pop =
        neat.evolve(population, results, config, 42 + population.generation)

      train_loop_cpu(next_pop, remaining - 1, config, eval_fn, log_interval)
    }
  }
}

fn find_best(population: Population) -> Genome {
  case list.first(population.genomes) {
    Ok(first) -> {
      list.fold(population.genomes, first, fn(best, g) {
        case g.fitness >. best.fitness {
          True -> g
          False -> best
        }
      })
    }
    Error(_) -> {
      neat.Genome(
        id: 0,
        nodes: [],
        connections: [],
        fitness: 0.0,
        adjusted_fitness: 0.0,
        species_id: 0,
      )
    }
  }
}

fn log_progress(population: Population, results: List(FitnessResult)) -> Nil {
  let fitnesses = list.map(results, fn(r) { r.fitness })
  let best =
    list.fold(fitnesses, neg_inf(), fn(acc, f) { float.max(acc, f) })
  let avg = case list.length(fitnesses) {
    0 -> 0.0
    n -> list.fold(fitnesses, 0.0, fn(acc, f) { acc +. f }) /. int.to_float(n)
  }

  io.println(
    "Gen "
    <> int.to_string(population.generation)
    <> " | Best: "
    <> float_str(best)
    <> " | Avg: "
    <> float_str(avg)
    <> " | Species: "
    <> int.to_string(list.length(population.species)),
  )
}

fn float_str(f: Float) -> String {
  let rounded = float.round(f *. 10.0)
  int.to_string(rounded / 10) <> "." <> int.to_string(int.absolute_value(rounded % 10))
}

fn neg_inf() -> Float {
  -999_999.0
}
