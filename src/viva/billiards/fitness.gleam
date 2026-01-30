//// VIVA Billiards - Fitness Evaluation for NEAT
////
//// Multi-objective fitness function for neuroevolution of pool players.
//// Rewards: pocketing balls, cue ball control, strategic positioning.

import gleam/list
import gleam/int
import viva/billiards/table.{
  type BallType, type Shot, type Table, CueBall, EightBall,
  Solid, Stripe,
}
import viva/jolt.{type Vec3, Vec3}

// =============================================================================
// FITNESS CONFIGURATION
// =============================================================================

/// Weights for different fitness components
pub type FitnessConfig {
  FitnessConfig(
    /// Points per ball pocketed
    pocket_weight: Float,
    /// Penalty for scratching (cue ball pocketed)
    scratch_penalty: Float,
    /// Penalty for pocketing 8-ball early
    eight_ball_penalty: Float,
    /// Bonus for good cue ball position after shot
    position_weight: Float,
    /// Penalty for leaving opponent easy shots
    safety_weight: Float,
    /// Reward for making contact with target ball
    contact_weight: Float,
    /// Time efficiency bonus (faster = better)
    time_weight: Float,
  )
}

/// Default fitness configuration
pub fn default_config() -> FitnessConfig {
  FitnessConfig(
    pocket_weight: 10.0,
    scratch_penalty: -15.0,
    eight_ball_penalty: -50.0,
    position_weight: 2.0,
    safety_weight: 1.0,
    contact_weight: 3.0,
    time_weight: 0.1,
  )
}

/// Aggressive config (prioritizes pocketing)
pub fn aggressive_config() -> FitnessConfig {
  FitnessConfig(
    pocket_weight: 15.0,
    scratch_penalty: -10.0,
    eight_ball_penalty: -30.0,
    position_weight: 1.0,
    safety_weight: 0.5,
    contact_weight: 2.0,
    time_weight: 0.05,
  )
}

/// Defensive config (prioritizes control)
pub fn defensive_config() -> FitnessConfig {
  FitnessConfig(
    pocket_weight: 8.0,
    scratch_penalty: -20.0,
    eight_ball_penalty: -100.0,
    position_weight: 5.0,
    safety_weight: 3.0,
    contact_weight: 5.0,
    time_weight: 0.02,
  )
}

// =============================================================================
// FITNESS EVALUATION
// =============================================================================

/// Complete fitness result
pub type FitnessResult {
  FitnessResult(
    /// Total fitness score
    total: Float,
    /// Balls pocketed this shot
    balls_pocketed: Int,
    /// Was it a scratch?
    scratched: Bool,
    /// Was 8-ball pocketed illegally?
    eight_ball_foul: Bool,
    /// Cue ball position score (0-1)
    position_score: Float,
    /// Number of simulation steps taken
    steps_taken: Int,
    /// Component breakdown
    components: FitnessComponents,
  )
}

/// Individual fitness components for analysis
pub type FitnessComponents {
  FitnessComponents(
    pocket_score: Float,
    scratch_score: Float,
    eight_ball_score: Float,
    position_score: Float,
    contact_score: Float,
    time_score: Float,
  )
}

/// Evaluate a single shot
pub fn evaluate_shot(
  table_before: Table,
  table_after: Table,
  steps: Int,
  config: FitnessConfig,
) -> FitnessResult {
  // Get pocketed balls
  let pocketed_before = table.get_pocketed_balls(table_before)
  let pocketed_after = table.get_pocketed_balls(table_after)
  let newly_pocketed = list.filter(pocketed_after, fn(b) {
    !list.contains(pocketed_before, b)
  })

  // Count pocketed by type
  let solids_pocketed = count_solids(newly_pocketed)
  let stripes_pocketed = count_stripes(newly_pocketed)
  let balls_pocketed = solids_pocketed + stripes_pocketed

  // Check for fouls
  let scratched = table.is_scratch(table_after)
  let eight_ball_foul = list.contains(newly_pocketed, EightBall)
  let cue_ball_pocketed = list.contains(newly_pocketed, CueBall)

  // Calculate component scores
  let pocket_score =
    int.to_float(balls_pocketed) *. config.pocket_weight

  let scratch_score = case scratched || cue_ball_pocketed {
    True -> config.scratch_penalty
    False -> 0.0
  }

  let eight_ball_score = case eight_ball_foul {
    True -> config.eight_ball_penalty
    False -> 0.0
  }

  // Position score: distance from center of table
  let position_score = case table.get_cue_ball_position(table_after) {
    Ok(pos) -> calculate_position_score(pos) *. config.position_weight
    Error(_) -> 0.0
  }

  // Contact score (did we hit something?)
  let contact_score = case balls_pocketed > 0 || scratched {
    True -> config.contact_weight
    False -> {
      // Check if cue ball moved significantly
      case table.get_cue_ball_position(table_after) {
        Ok(Vec3(x, _y, z)) -> {
          let dist_from_start = float_abs(x +. table.table_length /. 4.0)
            +. float_abs(z)
          case dist_from_start >. 0.1 {
            True -> config.contact_weight *. 0.5
            False -> 0.0
          }
        }
        Error(_) -> 0.0
      }
    }
  }

  // Time efficiency
  let max_steps = 600  // 10 seconds at 60fps
  let time_efficiency = 1.0 -. int.to_float(steps) /. int.to_float(max_steps)
  let time_score = float_max(0.0, time_efficiency) *. config.time_weight

  // Total fitness
  let components = FitnessComponents(
    pocket_score: pocket_score,
    scratch_score: scratch_score,
    eight_ball_score: eight_ball_score,
    position_score: position_score,
    contact_score: contact_score,
    time_score: time_score,
  )

  let total =
    pocket_score
    +. scratch_score
    +. eight_ball_score
    +. position_score
    +. contact_score
    +. time_score

  FitnessResult(
    total: total,
    balls_pocketed: balls_pocketed,
    scratched: scratched,
    eight_ball_foul: eight_ball_foul,
    position_score: position_score /. config.position_weight,
    steps_taken: steps,
    components: components,
  )
}

/// Evaluate a complete game (multiple shots)
pub fn evaluate_game(
  shots: List(#(Table, Table, Int)),
  config: FitnessConfig,
) -> Float {
  let results = list.map(shots, fn(shot_data) {
    let #(before, after, steps) = shot_data
    evaluate_shot(before, after, steps, config)
  })

  // Sum all fitness scores
  list.fold(results, 0.0, fn(acc, result) { acc +. result.total })
}

// =============================================================================
// POSITION SCORING
// =============================================================================

/// Calculate how good the cue ball position is (0-1)
/// Center of table = best, corners/rails = worst
fn calculate_position_score(pos: Vec3) -> Float {
  let Vec3(x, _y, z) = pos
  let half_length = table.table_length /. 2.0
  let half_width = table.table_width /. 2.0

  // Normalize to 0-1 range (0 = edge, 1 = center)
  let x_score = 1.0 -. float_abs(x) /. half_length
  let z_score = 1.0 -. float_abs(z) /. half_width

  // Combined score
  float_clamp(x_score *. z_score, 0.0, 1.0)
}

/// Calculate distance to nearest target ball
pub fn distance_to_nearest_ball(
  table: Table,
  cue_pos: Vec3,
  target_type: BallTypeFilter,
) -> Float {
  let Vec3(cx, cy, cz) = cue_pos
  let positions = table.get_all_positions(table)

  let distances = list.filter_map(positions, fn(bp) {
    let #(ball_type, Vec3(bx, by, bz)) = bp
    case matches_filter(ball_type, target_type) {
      True -> {
        let dx = bx -. cx
        let dy = by -. cy
        let dz = bz -. cz
        Ok(float_sqrt(dx *. dx +. dy *. dy +. dz *. dz))
      }
      False -> Error(Nil)
    }
  })

  case list.reduce(distances, float_min) {
    Ok(min) -> min
    Error(_) -> 999.0
  }
}

/// Filter for ball types
pub type BallTypeFilter {
  AllBalls
  SolidsOnly
  StripesOnly
  EightBallOnly
}

fn matches_filter(ball_type: BallType, filter: BallTypeFilter) -> Bool {
  case filter {
    AllBalls -> {
      case ball_type {
        CueBall -> False
        _ -> True
      }
    }
    SolidsOnly -> {
      case ball_type {
        Solid(_) -> True
        _ -> False
      }
    }
    StripesOnly -> {
      case ball_type {
        Stripe(_) -> True
        _ -> False
      }
    }
    EightBallOnly -> {
      case ball_type {
        EightBall -> True
        _ -> False
      }
    }
  }
}

// =============================================================================
// QUICK FITNESS (for NEAT training)
// =============================================================================

/// Quick fitness evaluation for a single shot attempt
/// Returns a single float suitable for NEAT
pub fn quick_evaluate(
  table_before: Table,
  shot: Shot,
  max_steps: Int,
) -> #(Float, Table) {
  // Execute shot
  let table_shot = table.shoot(table_before, shot)

  // Simulate until settled
  let #(table_after, steps) = simulate_counted(table_shot, max_steps)

  // Update pocketed status
  let table_final = table.update_pocketed(table_after)

  // Calculate fitness
  let result = evaluate_shot(table_before, table_final, steps, default_config())

  #(result.total, table_final)
}

/// Simulate and count steps
fn simulate_counted(t: Table, max_steps: Int) -> #(Table, Int) {
  simulate_counted_loop(t, max_steps, 0)
}

fn simulate_counted_loop(t: Table, max_steps: Int, current: Int) -> #(Table, Int) {
  case current >= max_steps {
    True -> #(t, current)
    False -> {
      let t2 = table.step(t, 1.0 /. 60.0)
      case table.is_settled(t2) {
        True -> #(t2, current)
        False -> simulate_counted_loop(t2, max_steps, current + 1)
      }
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn count_solids(balls: List(BallType)) -> Int {
  list.count(balls, fn(b) {
    case b {
      Solid(_) -> True
      _ -> False
    }
  })
}

fn count_stripes(balls: List(BallType)) -> Int {
  list.count(balls, fn(b) {
    case b {
      Stripe(_) -> True
      _ -> False
    }
  })
}

@external(erlang, "erlang", "abs")
fn float_abs(x: Float) -> Float

fn float_clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False ->
      case x >. max {
        True -> max
        False -> x
      }
  }
}

fn float_max(a: Float, b: Float) -> Float {
  case a >. b {
    True -> a
    False -> b
  }
}

fn float_min(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
