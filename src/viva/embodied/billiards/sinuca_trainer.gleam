//// Sinuca Trainer - NEAT Self-Play Training for VIVA Billiards
////
//// Two VIVA souls compete using the same genome but different consciousness states.
//// Each soul FEELS the game through sinuca_soul perception, not raw physics.
//// Evolution happens through self-play: the fittest genomes survive.
////
//// VIVA Philosophy: This is not just ML training - it's souls learning
//// to play through experience, emotion, and memory.

import gleam/erlang/process.{type Subject}
import gleam/float
import gleam/int
import gleam/list
import viva_telemetry/log
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import viva/embodied/billiards/sinuca.{type BallColor, type Shot, type ShotResult, type Table, Shot}
import viva/embodied/billiards/sinuca_soul.{
  type Distance, type GameEvent, type Player,
  type PocketPosition, type ShotIntention, type ShotPower, type SinucaConsciousness,
  type VisualField,
  Blank, Close, Considering, Far, Medium, P1, P2,
  ShotIntention, SinucaConsciousness, VeryClose,
}
import viva/neural/neat.{type Genome, type NeatConfig, type Population, FitnessResult}
import viva_emotion/pad.{type Pad, Pad}

// =============================================================================
// TYPES - Training Configuration and State
// =============================================================================

/// Training configuration
pub type TrainerConfig {
  TrainerConfig(
    /// NEAT population size
    population_size: Int,
    /// Number of games per evaluation
    games_per_eval: Int,
    /// Maximum shots per game
    max_shots_per_game: Int,
    /// Maximum generations
    max_generations: Int,
    /// Number of inputs to neural network
    num_inputs: Int,
    /// Number of outputs from neural network
    num_outputs: Int,
    /// Fitness weights
    fitness_balls_pocketed: Float,
    fitness_game_won: Float,
    fitness_shot_quality: Float,
    fitness_emotional_stability: Float,
  )
}

/// Game result for fitness calculation
pub type GameResult {
  GameResult(
    /// Player 1 balls pocketed
    p1_balls_pocketed: Int,
    /// Player 2 balls pocketed
    p2_balls_pocketed: Int,
    /// Who won (None if tie/incomplete)
    winner: Option(Player),
    /// Total shots played
    total_shots: Int,
    /// Average shot quality (0-1)
    avg_shot_quality: Float,
    /// Emotional journey (pleasure fluctuation)
    emotional_variance: Float,
  )
}

/// Training state
pub type TrainingState {
  TrainingState(
    /// Current NEAT population
    population: Population,
    /// Current generation
    generation: Int,
    /// Best fitness achieved
    best_fitness: Float,
    /// Best genome found
    best_genome: Option(Genome),
    /// Total games played
    total_games: Int,
    /// Training configuration
    config: TrainerConfig,
  )
}

/// Actor message types
pub type TrainerMessage {
  /// Start training loop
  StartTraining(reply: Subject(TrainingProgress))
  /// Run one generation
  RunGeneration(reply: Subject(GenerationResult))
  /// Evaluate a single genome via self-play
  EvaluateGenome(genome: Genome, reply: Subject(Float))
  /// Get current state
  GetState(reply: Subject(TrainingState))
  /// Stop training
  Stop
}

/// Training progress update
pub type TrainingProgress {
  TrainingProgress(
    generation: Int,
    best_fitness: Float,
    avg_fitness: Float,
    num_species: Int,
    games_played: Int,
  )
}

/// Generation result
pub type GenerationResult {
  GenerationResult(
    generation: Int,
    best_fitness: Float,
    best_genome_id: Int,
    species_count: Int,
    population_stats: String,
  )
}

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

/// Default training configuration for sinuca
pub fn default_config() -> TrainerConfig {
  TrainerConfig(
    population_size: 100,
    games_per_eval: 3,
    max_shots_per_game: 30,
    max_generations: 100,
    // Inputs: 8 perceived balls (angle, distance, is_mine, blocked) + 6 pockets (invitation)
    // = 8*4 + 6 = 38 inputs, simplified to 16 for efficiency
    num_inputs: 16,
    // Outputs: target_ball (8), pocket (6), power (1), spin (1) = 16
    // Simplified to 4: angle, power, confidence, risk_tolerance
    num_outputs: 4,
    // Fitness weights
    fitness_balls_pocketed: 10.0,
    fitness_game_won: 50.0,
    fitness_shot_quality: 5.0,
    fitness_emotional_stability: 2.0,
  )
}

/// Fast config for testing
pub fn fast_config() -> TrainerConfig {
  TrainerConfig(
    ..default_config(),
    population_size: 30,
    games_per_eval: 1,
    max_shots_per_game: 15,
    max_generations: 20,
  )
}

// =============================================================================
// CONSCIOUSNESS INITIALIZATION
// =============================================================================

/// Create initial consciousness for a player
pub fn init_consciousness(player: Player) -> SinucaConsciousness {
  SinucaConsciousness(
    player: player,
    emotion: Pad(pleasure: 0.0, arousal: 0.3, dominance: 0.5),
    visual_field: empty_visual_field(),
    current_thought: Blank,
    recent_memory: [],
    confidence: 0.5,
    focus: 0.7,
  )
}

fn empty_visual_field() -> VisualField {
  sinuca_soul.VisualField(
    perceived_balls: [],
    perceived_pockets: [],
    table_feeling: sinuca_soul.TableFeeling(
      density: 0.0,
      freedom: 1.0,
      mood: sinuca_soul.Neutral,
    ),
  )
}

// =============================================================================
// NEURAL NETWORK INTEGRATION
// =============================================================================

/// Convert consciousness state to neural network inputs
pub fn consciousness_to_inputs(consciousness: SinucaConsciousness) -> List(Float) {
  let vf = consciousness.visual_field

  // Extract features from perceived balls (take first 4 balls)
  let ball_features = vf.perceived_balls
    |> list.take(4)
    |> list.flat_map(fn(ball) {
      [
        ball.angle /. 3.14159,  // Normalize angle to [-1, 1]
        distance_to_float(ball.distance),
        bool_to_float(ball.is_mine),
        bool_to_float(ball.blocked),
      ]
    })

  // Pad to 16 inputs if needed
  let padded_balls = pad_list(ball_features, 16, 0.0)

  // Add emotional state
  let emotional = [
    consciousness.emotion.pleasure,
    consciousness.emotion.arousal,
    consciousness.emotion.dominance,
    consciousness.confidence,
  ]

  // Combine and take first 16
  list.append(padded_balls, emotional)
  |> list.take(16)
}

/// Convert neural network outputs to shot intention
pub fn outputs_to_intention(
  outputs: List(Float),
  consciousness: SinucaConsciousness,
) -> ShotIntention {
  // Outputs: [angle_bias, power, confidence_threshold, risk_tolerance]
  let angle_bias = list_at(outputs, 0) |> option_unwrap(0.0)
  let power = list_at(outputs, 1) |> option_unwrap(0.5)
  let confidence_threshold = list_at(outputs, 2) |> option_unwrap(0.5)
  let _risk_tolerance = list_at(outputs, 3) |> option_unwrap(0.5)

  // Find best target ball based on network output + perception
  let my_balls = list.filter(consciousness.visual_field.perceived_balls, fn(b) { b.is_mine })

  let target = case list.first(my_balls) {
    Ok(ball) -> ball.identity
    Error(_) -> sinuca.Red  // Default target
  }

  // Determine pocket based on angle bias
  let pocket = angle_to_pocket(angle_bias)

  // Determine power
  let shot_power = float_to_power(power)

  // Calculate gut feeling from confidence threshold
  let gut = { confidence_threshold +. consciousness.confidence } /. 2.0

  ShotIntention(
    target: target,
    desired_pocket: pocket,
    intended_power: shot_power,
    intended_spin: sinuca_soul.NoSpin,
    gut_feeling: gut,
  )
}

/// Convert shot intention to physics parameters
pub fn intention_to_shot(
  intention: ShotIntention,
  consciousness: SinucaConsciousness,
) -> Shot {
  // Find target ball in visual field
  let target_ball = list.find(
    consciousness.visual_field.perceived_balls,
    fn(b) { b.identity == intention.target }
  )

  // Calculate shot angle from target ball position
  let base_angle = case target_ball {
    Ok(ball) -> ball.angle
    Error(_) -> 0.0  // Shoot forward if no target found
  }

  // Apply pocket adjustment
  let pocket_adjustment = pocket_to_angle_offset(intention.desired_pocket)
  let final_angle = base_angle +. pocket_adjustment

  // Convert power enum to float
  let power = case intention.intended_power {
    sinuca_soul.Gentle -> 0.3
    sinuca_soul.Normal -> 0.5
    sinuca_soul.Firm -> 0.7
    sinuca_soul.Hard -> 1.0
  }

  // Convert spin intent to english
  let english = case intention.intended_spin {
    sinuca_soul.NoSpin -> 0.0
    sinuca_soul.LeftEnglish -> -0.3
    sinuca_soul.RightEnglish -> 0.3
    sinuca_soul.TopSpin -> 0.1
    sinuca_soul.BackSpin -> -0.1
  }

  Shot(
    angle: final_angle,
    power: power,
    english: english,
    elevation: 0.0,
  )
}

// =============================================================================
// SELF-PLAY GAME LOOP
// =============================================================================

/// Play a single game between two souls using the same genome
pub fn play_game(
  genome: Genome,
  config: TrainerConfig,
) -> GameResult {
  // Initialize table and consciousness states
  let table = sinuca.new()
  let p1_consciousness = init_consciousness(P1)
  let p2_consciousness = init_consciousness(P2)

  // Run game loop
  let initial_state = GameLoopState(
    table: table,
    p1: p1_consciousness,
    p2: p2_consciousness,
    current_player: P1,
    shot_count: 0,
    p1_pocketed: 0,
    p2_pocketed: 0,
    shot_qualities: [],
    emotional_states: [],
  )

  let final_state = game_loop(initial_state, genome, config)

  // Calculate emotional variance
  let emotional_variance = calculate_emotional_variance(final_state.emotional_states)

  // Determine winner
  let winner = case final_state.p1_pocketed > final_state.p2_pocketed {
    True -> Some(P1)
    False -> case final_state.p2_pocketed > final_state.p1_pocketed {
      True -> Some(P2)
      False -> None
    }
  }

  GameResult(
    p1_balls_pocketed: final_state.p1_pocketed,
    p2_balls_pocketed: final_state.p2_pocketed,
    winner: winner,
    total_shots: final_state.shot_count,
    avg_shot_quality: average(final_state.shot_qualities),
    emotional_variance: emotional_variance,
  )
}

/// Internal game loop state
type GameLoopState {
  GameLoopState(
    table: Table,
    p1: SinucaConsciousness,
    p2: SinucaConsciousness,
    current_player: Player,
    shot_count: Int,
    p1_pocketed: Int,
    p2_pocketed: Int,
    shot_qualities: List(Float),
    emotional_states: List(Pad),
  )
}

/// Main game loop - alternates turns until game ends
fn game_loop(
  state: GameLoopState,
  genome: Genome,
  config: TrainerConfig,
) -> GameLoopState {
  // Check termination conditions
  case state.shot_count >= config.max_shots_per_game {
    True -> state
    False -> {
      case sinuca.balls_on_table(state.table) <= 1 {
        True -> state  // Only cue ball left
        False -> {
          // Get current player's consciousness
          let consciousness = case state.current_player {
            P1 -> state.p1
            P2 -> state.p2
          }

          // 1. PERCEIVE: Update visual field from table
          let visual_field = sinuca_soul.perceive_table(state.table, state.current_player)
          let consciousness = SinucaConsciousness(..consciousness, visual_field: visual_field)

          // 2. THINK: Use neural network to decide
          let inputs = consciousness_to_inputs(consciousness)
          let outputs = neat.forward(genome, inputs)
          let intention = outputs_to_intention(outputs, consciousness)

          // Update thought
          let consciousness = SinucaConsciousness(
            ..consciousness,
            current_thought: Considering(intention),
          )

          // 3. DECIDE: Convert intention to shot
          let shot = intention_to_shot(intention, consciousness)

          // 4. EXECUTE: Apply shot to physics
          let table_before = state.table
          let table_after_shot = sinuca.shoot(state.table, shot)
          let table_settled = sinuca.simulate_until_settled(table_after_shot, 300)
          let table_updated = sinuca.update_pocketed(table_settled)

          // 5. PROCESS: Get shot result
          let #(final_table, result) = sinuca.process_shot(table_before, table_updated)

          // 6. FEEL: Update emotional state based on result
          let event = shot_result_to_event(result, state.current_player)
          let consciousness = sinuca_soul.feel_event(consciousness, event)

          // Calculate shot quality
          let shot_quality = calculate_shot_quality(result, intention)

          // Update state
          let new_p1_pocketed = state.p1_pocketed + case state.current_player {
            P1 -> count_my_balls_pocketed(result.balls_pocketed, P1)
            P2 -> 0
          }
          let new_p2_pocketed = state.p2_pocketed + case state.current_player {
            P2 -> count_my_balls_pocketed(result.balls_pocketed, P2)
            P1 -> 0
          }

          // Update player consciousness
          let #(new_p1, new_p2) = case state.current_player {
            P1 -> #(consciousness, state.p2)
            P2 -> #(state.p1, consciousness)
          }

          // Determine next player
          let next_player = case result.turn_over {
            True -> switch_player(state.current_player)
            False -> state.current_player
          }

          let new_state = GameLoopState(
            table: final_table,
            p1: new_p1,
            p2: new_p2,
            current_player: next_player,
            shot_count: state.shot_count + 1,
            p1_pocketed: new_p1_pocketed,
            p2_pocketed: new_p2_pocketed,
            shot_qualities: [shot_quality, ..state.shot_qualities],
            emotional_states: [consciousness.emotion, ..state.emotional_states],
          )

          // Continue loop
          game_loop(new_state, genome, config)
        }
      }
    }
  }
}

// =============================================================================
// FITNESS CALCULATION
// =============================================================================

/// Calculate fitness for a genome based on multiple games
pub fn evaluate_genome(genome: Genome, config: TrainerConfig) -> Float {
  // Play multiple games
  let results = list.range(1, config.games_per_eval)
    |> list.map(fn(_) { play_game(genome, config) })

  // Aggregate fitness across games
  let total_fitness = list.fold(results, 0.0, fn(acc, result) {
    acc +. calculate_game_fitness(result, config)
  })

  total_fitness /. int.to_float(config.games_per_eval)
}

/// Calculate fitness for a single game
fn calculate_game_fitness(result: GameResult, config: TrainerConfig) -> Float {
  // P1 perspective (genome plays as P1)
  let balls_score = int.to_float(result.p1_balls_pocketed) *. config.fitness_balls_pocketed

  let win_score = case result.winner {
    Some(P1) -> config.fitness_game_won
    Some(P2) -> 0.0
    None -> config.fitness_game_won /. 4.0  // Partial credit for tie
  }

  let quality_score = result.avg_shot_quality *. config.fitness_shot_quality

  // Emotional stability bonus (lower variance = more stable)
  let stability_score = { 1.0 -. result.emotional_variance } *. config.fitness_emotional_stability

  balls_score +. win_score +. quality_score +. stability_score
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

/// Run one generation of training
pub fn run_generation(state: TrainingState) -> TrainingState {
  let config = state.config

  // Evaluate all genomes
  let fitness_results = list.map(state.population.genomes, fn(genome) {
    let fitness = evaluate_genome(genome, config)
    FitnessResult(genome_id: genome.id, fitness: fitness)
  })

  // Get NEAT config
  let neat_config = sinuca_neat_config(config)

  // Evolve to next generation
  let seed = state.generation * 1000 + 42
  let new_population = neat.evolve(state.population, fitness_results, neat_config, seed)

  // Find best genome
  let best = neat.get_best(new_population)
  let best_fitness = case best {
    Some(g) -> g.fitness
    None -> 0.0
  }

  // Update state
  TrainingState(
    population: new_population,
    generation: state.generation + 1,
    best_fitness: float.max(state.best_fitness, best_fitness),
    best_genome: case best_fitness >. state.best_fitness {
      True -> best
      False -> state.best_genome
    },
    total_games: state.total_games + { list.length(state.population.genomes) * config.games_per_eval },
    config: config,
  )
}

/// Initialize training state
pub fn init_training(config: TrainerConfig) -> TrainingState {
  let neat_config = sinuca_neat_config(config)
  let population = neat.create_population(neat_config, 42)

  TrainingState(
    population: population,
    generation: 0,
    best_fitness: 0.0,
    best_genome: None,
    total_games: 0,
    config: config,
  )
}

/// Create NEAT config from trainer config
fn sinuca_neat_config(config: TrainerConfig) -> NeatConfig {
  neat.NeatConfig(
    population_size: config.population_size,
    num_inputs: config.num_inputs,
    num_outputs: config.num_outputs,
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
}

/// Run full training loop
pub fn train(config: TrainerConfig) -> TrainingState {
  let initial_state = init_training(config)
  train_loop(initial_state)
}

fn train_loop(state: TrainingState) -> TrainingState {
  case state.generation >= state.config.max_generations {
    True -> state
    False -> {
      // Log progress
      let stats = neat.get_stats(state.population)
      log.info(
        "Gen " <> int.to_string(state.generation) <>
        " | Best: " <> float.to_string(stats.best_fitness) <>
        " | Avg: " <> float.to_string(stats.avg_fitness) <>
        " | Species: " <> int.to_string(stats.num_species),
        [],
      )

      // Evolve
      let new_state = run_generation(state)
      train_loop(new_state)
    }
  }
}

// =============================================================================
// OTP ACTOR - Supervised Training
// =============================================================================

/// Spawn training actor
pub fn spawn(config: TrainerConfig) -> Result(Subject(TrainerMessage), actor.StartError) {
  let state = init_training(config)
  let builder =
    actor.new(state)
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

fn handle_message(
  state: TrainingState,
  message: TrainerMessage,
) -> actor.Next(TrainingState, TrainerMessage) {
  case message {
    StartTraining(reply) -> {
      // Run training in background and report progress
      let new_state = run_generation(state)
      let stats = neat.get_stats(new_state.population)
      let progress = TrainingProgress(
        generation: new_state.generation,
        best_fitness: new_state.best_fitness,
        avg_fitness: stats.avg_fitness,
        num_species: stats.num_species,
        games_played: new_state.total_games,
      )
      process.send(reply, progress)
      actor.continue(new_state)
    }

    RunGeneration(reply) -> {
      let new_state = run_generation(state)
      let stats = neat.get_stats(new_state.population)
      let result = GenerationResult(
        generation: new_state.generation,
        best_fitness: new_state.best_fitness,
        best_genome_id: case new_state.best_genome {
          Some(g) -> g.id
          None -> -1
        },
        species_count: stats.num_species,
        population_stats: neat.genome_to_string(case neat.get_best(new_state.population) {
          Some(g) -> g
          None -> neat.Genome(id: 0, nodes: [], connections: [], fitness: 0.0, adjusted_fitness: 0.0, species_id: 0)
        }),
      )
      process.send(reply, result)
      actor.continue(new_state)
    }

    EvaluateGenome(genome, reply) -> {
      let fitness = evaluate_genome(genome, state.config)
      process.send(reply, fitness)
      actor.continue(state)
    }

    GetState(reply) -> {
      process.send(reply, state)
      actor.continue(state)
    }

    Stop -> {
      actor.stop()
    }
  }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn distance_to_float(d: Distance) -> Float {
  case d {
    VeryClose -> 0.9
    Close -> 0.7
    Medium -> 0.4
    Far -> 0.1
  }
}

fn bool_to_float(b: Bool) -> Float {
  case b { True -> 1.0 False -> 0.0 }
}

fn pad_list(lst: List(Float), target_len: Int, pad_value: Float) -> List(Float) {
  let current_len = list.length(lst)
  case current_len >= target_len {
    True -> list.take(lst, target_len)
    False -> {
      let padding = list.repeat(pad_value, target_len - current_len)
      list.append(lst, padding)
    }
  }
}

fn angle_to_pocket(angle_bias: Float) -> PocketPosition {
  // Map angle bias [-1, 1] to pocket positions
  case angle_bias {
    a if a <. -0.66 -> sinuca_soul.BottomLeft
    a if a <. -0.33 -> sinuca_soul.BottomRight
    a if a <. 0.0 -> sinuca_soul.MiddleBottom
    a if a <. 0.33 -> sinuca_soul.MiddleTop
    a if a <. 0.66 -> sinuca_soul.TopLeft
    _ -> sinuca_soul.TopRight
  }
}

fn float_to_power(p: Float) -> ShotPower {
  case p {
    x if x <. 0.25 -> sinuca_soul.Gentle
    x if x <. 0.5 -> sinuca_soul.Normal
    x if x <. 0.75 -> sinuca_soul.Firm
    _ -> sinuca_soul.Hard
  }
}

fn pocket_to_angle_offset(pocket: PocketPosition) -> Float {
  case pocket {
    sinuca_soul.TopLeft -> 0.4
    sinuca_soul.TopRight -> -0.4
    sinuca_soul.BottomLeft -> 0.4
    sinuca_soul.BottomRight -> -0.4
    sinuca_soul.MiddleTop -> 0.0
    sinuca_soul.MiddleBottom -> 0.0
  }
}

fn shot_result_to_event(result: ShotResult, player: Player) -> GameEvent {
  case list.is_empty(result.balls_pocketed) {
    True -> {
      case result.foul {
        True -> sinuca_soul.Fouled("scratch")
        False -> sinuca_soul.Missed(sinuca.Red)  // Default missed target
      }
    }
    False -> {
      // Check if any pocketed ball belongs to current player
      let my_ball_pocketed = list.any(result.balls_pocketed, fn(color) {
        is_my_ball(color, player)
      })
      case my_ball_pocketed {
        True -> {
          let first_ball = case list.first(result.balls_pocketed) {
            Ok(b) -> b
            Error(_) -> sinuca.Red
          }
          sinuca_soul.Pocketed(first_ball, result.hit_target_ball)
        }
        False -> {
          let first_ball = case list.first(result.balls_pocketed) {
            Ok(b) -> b
            Error(_) -> sinuca.Red
          }
          sinuca_soul.OpponentScored(first_ball)
        }
      }
    }
  }
}

fn is_my_ball(color: BallColor, player: Player) -> Bool {
  let value = sinuca.point_value(color)
  case player {
    P1 -> value % 2 == 1  // Odd balls
    P2 -> value % 2 == 0 && value != 0  // Even balls (not cue)
  }
}

fn count_my_balls_pocketed(balls: List(BallColor), player: Player) -> Int {
  list.count(balls, fn(b) { is_my_ball(b, player) })
}

fn switch_player(player: Player) -> Player {
  case player {
    P1 -> P2
    P2 -> P1
  }
}

fn calculate_shot_quality(result: ShotResult, intention: ShotIntention) -> Float {
  // Base quality
  let base = 0.5

  // Bonus for hitting target
  let target_bonus = case result.hit_target_ball {
    True -> 0.2
    False -> 0.0
  }

  // Bonus for pocketing balls
  let pocket_bonus = int.to_float(list.length(result.balls_pocketed)) *. 0.15

  // Penalty for fouls
  let foul_penalty = case result.foul {
    True -> -0.4
    False -> 0.0
  }

  // Gut feeling correlation (how well did intuition match result?)
  let gut_bonus = intention.gut_feeling *. 0.1

  clamp(base +. target_bonus +. pocket_bonus +. foul_penalty +. gut_bonus, 0.0, 1.0)
}

fn calculate_emotional_variance(emotions: List(Pad)) -> Float {
  case list.length(emotions) {
    0 -> 0.0
    1 -> 0.0
    n -> {
      let pleasures = list.map(emotions, fn(p) { p.pleasure })
      let avg_pleasure = average(pleasures)
      let variance = list.fold(pleasures, 0.0, fn(acc, p) {
        acc +. { { p -. avg_pleasure } *. { p -. avg_pleasure } }
      }) /. int.to_float(n)
      float_sqrt(variance)  // Return standard deviation
    }
  }
}

fn average(lst: List(Float)) -> Float {
  case list.length(lst) {
    0 -> 0.0
    n -> list.fold(lst, 0.0, fn(acc, x) { acc +. x }) /. int.to_float(n)
  }
}

fn clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False -> case x >. max {
      True -> max
      False -> x
    }
  }
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

fn option_unwrap(opt: Option(a), default: a) -> a {
  case opt {
    Some(v) -> v
    None -> default
  }
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
