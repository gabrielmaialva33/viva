//// Tests for Sinuca Trainer - NEAT Self-Play Training

import gleam/float
import gleam/list
import viva_telemetry/log
import gleam/option.{None, Some}
import gleeunit/should
import viva/embodied/billiards/sinuca
import viva/embodied/billiards/sinuca_soul.{P1, P2}
import viva/embodied/billiards/sinuca_trainer
import viva/neural/neat

// =============================================================================
// CONFIGURATION TESTS
// =============================================================================

pub fn default_config_test() {
  let config = sinuca_trainer.default_config()

  should.equal(config.population_size, 100)
  should.equal(config.games_per_eval, 3)
  should.equal(config.num_inputs, 16)
  should.equal(config.num_outputs, 4)
}

pub fn fast_config_test() {
  let config = sinuca_trainer.fast_config()

  should.equal(config.population_size, 30)
  should.equal(config.games_per_eval, 1)
  should.equal(config.max_shots_per_game, 15)
}

// =============================================================================
// CONSCIOUSNESS TESTS
// =============================================================================

pub fn init_consciousness_p1_test() {
  let consciousness = sinuca_trainer.init_consciousness(P1)

  should.equal(consciousness.player, P1)
  // Initial emotion should be slightly positive
  should.be_true(consciousness.emotion.arousal >. 0.0)
  should.equal(consciousness.confidence, 0.5)
  should.equal(consciousness.focus, 0.7)
}

pub fn init_consciousness_p2_test() {
  let consciousness = sinuca_trainer.init_consciousness(P2)

  should.equal(consciousness.player, P2)
  should.equal(consciousness.emotion.pleasure, 0.0)
}

// =============================================================================
// NEURAL NETWORK INTEGRATION TESTS
// =============================================================================

pub fn consciousness_to_inputs_test() {
  let consciousness = sinuca_trainer.init_consciousness(P1)
  let inputs = sinuca_trainer.consciousness_to_inputs(consciousness)

  // Should have exactly 16 inputs
  should.equal(list.length(inputs), 16)

  // All values should be in reasonable range
  list.each(inputs, fn(x) {
    should.be_true(x >=. -2.0)
    should.be_true(x <=. 2.0)
  })
}

pub fn outputs_to_intention_test() {
  let consciousness = sinuca_trainer.init_consciousness(P1)
  // Simulate neural network outputs
  let outputs = [0.3, 0.6, 0.7, 0.4]

  let intention = sinuca_trainer.outputs_to_intention(outputs, consciousness)

  // Should have reasonable gut feeling
  should.be_true(intention.gut_feeling >=. 0.0)
  should.be_true(intention.gut_feeling <=. 1.0)
}

pub fn intention_to_shot_test() {
  // Create consciousness with visual field from table
  let table = sinuca.new()
  let consciousness = sinuca_trainer.init_consciousness(P1)
  let visual_field = sinuca_soul.perceive_table(table, P1)
  let consciousness = sinuca_soul.SinucaConsciousness(..consciousness, visual_field: visual_field)

  let outputs = [0.5, 0.5, 0.5, 0.5]
  let intention = sinuca_trainer.outputs_to_intention(outputs, consciousness)
  let shot = sinuca_trainer.intention_to_shot(intention, consciousness)

  // Shot parameters should be valid
  should.be_true(shot.power >=. 0.0)
  should.be_true(shot.power <=. 1.0)
  should.be_true(shot.english >=. -1.0)
  should.be_true(shot.english <=. 1.0)
}

// =============================================================================
// GAME SIMULATION TESTS
// =============================================================================

pub fn play_game_completes_test() {
  let config = sinuca_trainer.fast_config()
  let neat_config = neat.NeatConfig(
    population_size: 10,
    num_inputs: 16,
    num_outputs: 4,
    weight_mutation_rate: 0.8,
    weight_perturb_rate: 0.9,
    add_node_rate: 0.05,
    add_connection_rate: 0.08,
    disable_rate: 0.01,
    compatibility_threshold: 1.5,
    excess_coefficient: 1.0,
    disjoint_coefficient: 1.0,
    weight_coefficient: 0.4,
    survival_threshold: 0.2,
    elitism: 2,
    max_stagnation: 15,
  )

  // Create minimal genome
  let population = neat.create_population(neat_config, 42)
  let genome = case neat.get_best(population) {
    Some(g) -> g
    None -> panic as "No genome in population"
  }

  let result = sinuca_trainer.play_game(genome, config)

  // Game should have completed some shots
  should.be_true(result.total_shots >= 1)
  should.be_true(result.total_shots <= config.max_shots_per_game)

  // Emotional variance should be calculated
  should.be_true(result.emotional_variance >=. 0.0)
}

// =============================================================================
// FITNESS TESTS
// =============================================================================

pub fn evaluate_genome_returns_positive_fitness_test() {
  let config = sinuca_trainer.TrainerConfig(
    ..sinuca_trainer.fast_config(),
    games_per_eval: 1,
    max_shots_per_game: 5,
  )

  let neat_config = neat.NeatConfig(
    population_size: 5,
    num_inputs: 16,
    num_outputs: 4,
    weight_mutation_rate: 0.8,
    weight_perturb_rate: 0.9,
    add_node_rate: 0.05,
    add_connection_rate: 0.08,
    disable_rate: 0.01,
    compatibility_threshold: 1.5,
    excess_coefficient: 1.0,
    disjoint_coefficient: 1.0,
    weight_coefficient: 0.4,
    survival_threshold: 0.2,
    elitism: 1,
    max_stagnation: 15,
  )

  let population = neat.create_population(neat_config, 123)
  let genome = case neat.get_best(population) {
    Some(g) -> g
    None -> panic as "No genome"
  }

  let fitness = sinuca_trainer.evaluate_genome(genome, config)

  // Fitness should be non-negative (at minimum from shot quality)
  should.be_true(fitness >=. 0.0)
}

// =============================================================================
// TRAINING LOOP TESTS
// =============================================================================

pub fn init_training_test() {
  let config = sinuca_trainer.fast_config()
  let state = sinuca_trainer.init_training(config)

  should.equal(state.generation, 0)
  should.equal(state.total_games, 0)
  should.equal(state.best_fitness, 0.0)
  should.be_true(state.best_genome == None)
  should.equal(list.length(state.population.genomes), config.population_size)
}

pub fn run_generation_advances_test() {
  let config = sinuca_trainer.TrainerConfig(
    ..sinuca_trainer.fast_config(),
    population_size: 10,
    games_per_eval: 1,
    max_shots_per_game: 3,
  )

  let state = sinuca_trainer.init_training(config)
  let new_state = sinuca_trainer.run_generation(state)

  should.equal(new_state.generation, 1)
  should.be_true(new_state.total_games > 0)
}

// =============================================================================
// INTEGRATION TEST
// =============================================================================

pub fn mini_training_session_test() {
  log.info("=== Mini Training Session ===", [])

  let config = sinuca_trainer.TrainerConfig(
    population_size: 10,
    games_per_eval: 1,
    max_shots_per_game: 5,
    max_generations: 3,
    num_inputs: 16,
    num_outputs: 4,
    fitness_balls_pocketed: 10.0,
    fitness_game_won: 50.0,
    fitness_shot_quality: 5.0,
    fitness_emotional_stability: 2.0,
  )

  let state0 = sinuca_trainer.init_training(config)
  log.info("Gen 0: population = " <> int_to_string(list.length(state0.population.genomes)), [])

  let state1 = sinuca_trainer.run_generation(state0)
  log.info("Gen 1: best_fitness = " <> float.to_string(state1.best_fitness), [])

  let state2 = sinuca_trainer.run_generation(state1)
  log.info("Gen 2: best_fitness = " <> float.to_string(state2.best_fitness), [])

  // Training should progress
  should.equal(state2.generation, 2)
  should.be_true(state2.total_games >= 20)  // 10 genomes * 2 generations * 1 game each

  log.info("=== Training Complete ===", [])
}

// =============================================================================
// HELPERS
// =============================================================================

fn int_to_string(n: Int) -> String {
  case n < 0 {
    True -> "-" <> int_to_string(-n)
    False -> do_int_to_string(n, "")
  }
}

fn do_int_to_string(n: Int, acc: String) -> String {
  case n < 10 {
    True -> digit_to_char(n) <> acc
    False -> do_int_to_string(n / 10, digit_to_char(n % 10) <> acc)
  }
}

fn digit_to_char(d: Int) -> String {
  case d {
    0 -> "0"
    1 -> "1"
    2 -> "2"
    3 -> "3"
    4 -> "4"
    5 -> "5"
    6 -> "6"
    7 -> "7"
    8 -> "8"
    _ -> "9"
  }
}
