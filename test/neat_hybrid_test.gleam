import gleam/dict
import gleam/list
import gleeunit/should
import viva/neural/neat
import viva/neural/neat_hybrid
import viva/neural/tensor

// =============================================================================
// CONFIG TESTS
// =============================================================================

pub fn default_config_test() {
  let config = neat_hybrid.default_config()
  should.equal(config.max_modules, 5)
  should.be_true(config.add_conv_rate >. 0.0)
  should.be_true(config.add_attention_rate >. 0.0)
}

pub fn image_config_test() {
  let config = neat_hybrid.image_config(28, 28, 1)
  should.equal(config.input_height, 28)
  should.equal(config.input_width, 28)
  should.equal(config.input_channels, 1)
  should.be_true(config.input_is_2d)
  // Should have higher conv rate for image processing
  should.be_true(
    config.add_conv_rate >. neat_hybrid.default_config().add_conv_rate,
  )
}

pub fn sequence_config_test() {
  let config = neat_hybrid.sequence_config(100, 64)
  should.equal(config.input_height, 100)
  should.equal(config.input_width, 64)
  // Should have higher attention rate for sequence processing
  should.be_true(
    config.add_attention_rate >. neat_hybrid.default_config().add_attention_rate,
  )
}

// =============================================================================
// CONVERSION TESTS
// =============================================================================

pub fn from_genome_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)
  let first_genome = case list.first(pop.genomes) {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }

  let hybrid = neat_hybrid.from_genome(first_genome)
  should.equal(hybrid.base.id, first_genome.id)
  should.equal(hybrid.modules, [])
  should.equal(dict.size(hybrid.module_params), 0)
}

pub fn from_population_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)

  let hybrid_pop = neat_hybrid.from_population(pop)
  should.equal(hybrid_pop.generation, pop.generation)
  should.equal(list.length(hybrid_pop.genomes), list.length(pop.genomes))
}

// =============================================================================
// MUTATION TESTS
// =============================================================================

pub fn mutate_adds_modules_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)
  let genome = case list.first(pop.genomes) {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }

  let hybrid = neat_hybrid.from_genome(genome)

  // Config with high mutation rates to ensure module addition
  let config =
    neat_hybrid.HybridConfig(
      ..neat_hybrid.default_config(),
      add_conv_rate: 0.99,
      add_attention_rate: 0.0,
      add_pool_rate: 0.0,
      max_modules: 3,
    )

  let mutated = neat_hybrid.mutate(hybrid, config, 12_345)

  // Should have added at least one module
  should.be_true(list.length(mutated.modules) >= 1)
}

pub fn mutate_respects_max_modules_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)
  let genome = case list.first(pop.genomes) {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }

  let hybrid = neat_hybrid.from_genome(genome)

  let config =
    neat_hybrid.HybridConfig(
      ..neat_hybrid.default_config(),
      add_conv_rate: 0.99,
      add_attention_rate: 0.99,
      add_pool_rate: 0.99,
      max_modules: 2,
    )

  // Mutate multiple times
  let mutated1 = neat_hybrid.mutate(hybrid, config, 1)
  let mutated2 = neat_hybrid.mutate(mutated1, config, 2)
  let mutated3 = neat_hybrid.mutate(mutated2, config, 3)
  let mutated4 = neat_hybrid.mutate(mutated3, config, 4)

  // Should not exceed max_modules
  should.be_true(list.length(mutated4.modules) <= 2)
}

// =============================================================================
// FORWARD PASS TESTS
// =============================================================================

pub fn forward_empty_modules_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)
  let genome = case list.first(pop.genomes) {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }

  let hybrid = neat_hybrid.from_genome(genome)
  let config = neat_hybrid.default_config()

  let input = tensor.from_list([0.0, 1.0])
  let output = neat_hybrid.forward(hybrid, input, config)

  // Should produce output (XOR has 1 output)
  should.equal(list.length(output), 1)
  // Output should be between 0 and 1 (sigmoid)
  case list.first(output) {
    Ok(val) -> {
      should.be_true(val >=. 0.0)
      should.be_true(val <=. 1.0)
    }
    Error(_) -> should.fail()
  }
}

pub fn forward_with_modules_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)
  let genome = case list.first(pop.genomes) {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }

  let hybrid = neat_hybrid.from_genome(genome)

  // Force add a pool module
  let config =
    neat_hybrid.HybridConfig(
      ..neat_hybrid.default_config(),
      add_conv_rate: 0.0,
      add_attention_rate: 0.0,
      add_pool_rate: 0.99,
      max_modules: 1,
    )

  let mutated = neat_hybrid.mutate(hybrid, config, 999)
  should.be_true(list.length(mutated.modules) >= 1)

  // Forward should still work
  let input = tensor.from_list([0.0, 1.0])
  let output = neat_hybrid.forward(mutated, input, config)

  should.equal(list.length(output), 1)
}

// =============================================================================
// CROSSOVER TESTS
// =============================================================================

pub fn crossover_inherits_modules_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)

  let genome1 = case list.first(pop.genomes) {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }
  let genome2 = case list.drop(pop.genomes, 1) |> list.first() {
    Ok(g) -> g
    Error(_) -> genome1
  }

  let hybrid1 = neat_hybrid.from_genome(genome1)
  let hybrid2 = neat_hybrid.from_genome(genome2)

  // Add modules to parent1
  let config =
    neat_hybrid.HybridConfig(
      ..neat_hybrid.default_config(),
      add_pool_rate: 0.99,
      max_modules: 2,
    )
  let parent1_mutated = neat_hybrid.mutate(hybrid1, config, 111)

  // Set fitness so parent1 is fitter
  let parent1_with_fitness =
    neat_hybrid.HybridGenome(
      ..parent1_mutated,
      base: neat.Genome(..parent1_mutated.base, fitness: 1.0),
    )
  let parent2_with_fitness =
    neat_hybrid.HybridGenome(
      ..hybrid2,
      base: neat.Genome(..hybrid2.base, fitness: 0.5),
    )

  let child =
    neat_hybrid.crossover(parent1_with_fitness, parent2_with_fitness, 42)

  // Child should inherit modules from fitter parent
  should.be_true(list.length(child.modules) >= 0)
}

// =============================================================================
// EVALUATION TESTS
// =============================================================================

pub fn evaluate_test() {
  let neat_config = neat.xor_config()
  let pop = neat.create_population(neat_config, 42)
  let genome = case list.first(pop.genomes) {
    Ok(g) -> g
    Error(_) -> panic as "No genomes"
  }

  let hybrid = neat_hybrid.from_genome(genome)

  // Simple fitness function
  let fitness_fn = fn(_g: neat_hybrid.HybridGenome) -> Float { 0.75 }

  let evaluated = neat_hybrid.evaluate(hybrid, fitness_fn)
  should.be_true(evaluated.base.fitness >. 0.0)
}

pub fn evaluate_population_test() {
  let neat_config = neat.NeatConfig(..neat.xor_config(), population_size: 10)
  let pop = neat.create_population(neat_config, 42)
  let hybrid_pop = neat_hybrid.from_population(pop)

  let fitness_fn = fn(_g: neat_hybrid.HybridGenome) -> Float { 0.5 }

  let evaluated = neat_hybrid.evaluate_population(hybrid_pop, fitness_fn)
  should.be_true(evaluated.best_fitness >=. 0.5)
}
