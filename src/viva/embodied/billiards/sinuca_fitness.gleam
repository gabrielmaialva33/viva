//// VIVA Sinuca - Fitness System (Simplified & Fixed)
////
//// Clean fitness with sane value ranges.
//// All scores are bounded and normalized.

import gleam/float
import gleam/int
import gleam/list
import gleam/option
import viva/embodied/billiards/sinuca.{type Table, type Shot, White}
import viva/lifecycle/jolt.{type Vec3, Vec3}

// =============================================================================
// FITNESS CONFIGURATION
// =============================================================================

/// Fitness weights (all values produce scores in reasonable ranges)
pub type FitnessConfig {
  FitnessConfig(
    target_pocketed: Float,     // +20 for pocketing target
    other_pocketed: Float,      // +10 for pocketing other balls
    scratch_penalty: Float,     // -15 for scratching
    miss_penalty: Float,        // -3 for missing everything
    position_bonus: Float,      // 0-5 for good leave position
    combo_bonus: Float,         // +5 per consecutive pocket
  )
}

/// Default configuration
pub fn default_config() -> FitnessConfig {
  FitnessConfig(
    target_pocketed: 20.0,
    other_pocketed: 10.0,
    scratch_penalty: 15.0,
    miss_penalty: 3.0,
    position_bonus: 5.0,
    combo_bonus: 5.0,
  )
}

/// Aggressive config
pub fn aggressive_config() -> FitnessConfig {
  FitnessConfig(
    target_pocketed: 25.0,
    other_pocketed: 15.0,
    scratch_penalty: 10.0,
    miss_penalty: 2.0,
    position_bonus: 3.0,
    combo_bonus: 8.0,
  )
}

/// Positional config
pub fn positional_config() -> FitnessConfig {
  FitnessConfig(
    target_pocketed: 15.0,
    other_pocketed: 8.0,
    scratch_penalty: 20.0,
    miss_penalty: 5.0,
    position_bonus: 10.0,
    combo_bonus: 3.0,
  )
}

// =============================================================================
// EPISODE TRACKING
// =============================================================================

pub type EpisodeState {
  EpisodeState(
    consecutive_pockets: Int,
    total_pocketed: Int,
    total_fouls: Int,
    shots_taken: Int,
  )
}

pub fn new_episode() -> EpisodeState {
  EpisodeState(
    consecutive_pockets: 0,
    total_pocketed: 0,
    total_fouls: 0,
    shots_taken: 0,
  )
}

// =============================================================================
// SIMPLE EVALUATION
// =============================================================================

/// Evaluate a single shot - returns bounded fitness
pub fn evaluate_shot(
  table_before: Table,
  table_after: Table,
  episode: EpisodeState,
  config: FitnessConfig,
) -> #(Float, EpisodeState) {
  // Count pocketed balls
  let pocketed_before = sinuca.get_pocketed_balls(table_before)
  let pocketed_after = sinuca.get_pocketed_balls(table_after)
  let newly_pocketed = list.filter(pocketed_after, fn(color) {
    !list.contains(pocketed_before, color)
  })

  // Target ball check
  let target = table_before.target_ball
  let target_hit = list.contains(newly_pocketed, target)

  // Other balls (not target, not cue)
  let others = list.filter(newly_pocketed, fn(c) { c != target && c != White })

  // Foul check
  let foul = sinuca.is_scratch(table_after)

  // Calculate score components
  let target_score = case target_hit {
    True -> config.target_pocketed
    False -> 0.0
  }

  let others_score = int.to_float(list.length(others)) *. config.other_pocketed

  let foul_score = case foul {
    True -> neg(config.scratch_penalty)
    False -> 0.0
  }

  // Miss penalty + approach bonus (gives gradient even when missing)
  let #(miss_score, approach_score) = case list.is_empty(newly_pocketed) && !foul {
    True -> {
      // Calculate how close cue ball got to target (approach reward)
      let approach = calculate_approach_bonus(table_before, table_after, target)
      #(neg(config.miss_penalty) +. approach, approach)
    }
    False -> #(0.0, 0.0)
  }
  let _ = approach_score  // Used for logging if needed

  // Position bonus (simple: center = good)
  let position_score = case sinuca.get_cue_ball_position(table_after) {
    option.Some(Vec3(x, _y, z)) -> {
      let half_l = sinuca.table_length /. 2.0
      let half_w = sinuca.table_width /. 2.0
      let center_score = { 1.0 -. float_abs(x) /. half_l }
        *. { 1.0 -. float_abs(z) /. half_w }
      float_clamp(center_score, 0.0, 1.0) *. config.position_bonus
    }
    option.None -> 0.0
  }

  // Combo tracking
  let new_consecutive = case list.is_empty(newly_pocketed) || foul {
    True -> 0
    False -> episode.consecutive_pockets + 1
  }

  let combo_score = case new_consecutive >= 2 {
    True -> int.to_float(new_consecutive - 1) *. config.combo_bonus
    False -> 0.0
  }

  // Total (bounded roughly -20 to +50 per shot)
  let total = target_score
    +. others_score
    +. foul_score
    +. miss_score
    +. position_score
    +. combo_score

  // Update episode
  let new_episode = EpisodeState(
    consecutive_pockets: new_consecutive,
    total_pocketed: episode.total_pocketed + list.length(newly_pocketed),
    total_fouls: episode.total_fouls + case foul { True -> 1  False -> 0 },
    shots_taken: episode.shots_taken + 1,
  )

  #(total, new_episode)
}

// =============================================================================
// QUICK EVALUATION (for NEAT)
// =============================================================================

/// Legacy API - simple evaluation
pub fn quick_evaluate(
  table: Table,
  shot: Shot,
  max_steps: Int,
) -> #(Float, Table) {
  let episode = new_episode()
  let #(fitness, table_out, _ep) = quick_evaluate_full(
    table, shot, max_steps, episode, default_config()
  )
  #(fitness, table_out)
}

/// Full evaluation with episode tracking
pub fn quick_evaluate_full(
  table: Table,
  shot: Shot,
  max_steps: Int,
  episode: EpisodeState,
  config: FitnessConfig,
) -> #(Float, Table, EpisodeState) {
  // Execute shot
  let table_shot = sinuca.shoot(table, shot)

  // Simulate until settled
  let table_final = simulate_until_settled(table_shot, max_steps)

  // Update pocketed status
  let table_updated = sinuca.update_pocketed(table_final)

  // Evaluate
  let #(fitness, new_episode) = evaluate_shot(table, table_updated, episode, config)

  #(fitness, table_updated, new_episode)
}

fn simulate_until_settled(table: Table, max_steps: Int) -> Table {
  simulate_loop(table, max_steps, 0)
}

fn simulate_loop(table: Table, max: Int, current: Int) -> Table {
  case current >= max {
    True -> table
    False -> {
      let table2 = sinuca.step(table, 1.0 /. 60.0)
      case sinuca.is_settled(table2) {
        True -> table2
        False -> simulate_loop(table2, max, current + 1)
      }
    }
  }
}

// =============================================================================
// EPISODE BONUS
// =============================================================================

/// Bonus for completing episode well
pub fn episode_bonus(episode: EpisodeState, balls_remaining: Int) -> Float {
  // Bonus for clearing table
  let clear_bonus = case balls_remaining {
    1 -> 30.0
    2 -> 15.0
    3 -> 5.0
    _ -> 0.0
  }

  // Penalty for fouls
  let foul_penalty = int.to_float(episode.total_fouls) *. neg(5.0)

  clear_bonus +. foul_penalty
}

// =============================================================================
// UTILITIES (kept for compatibility)
// =============================================================================

pub fn distance_to_target(table: Table) -> Float {
  let target = table.target_ball
  case sinuca.get_cue_ball_position(table) {
    option.Some(cue_pos) -> {
      case sinuca.get_ball_position(table, target) {
        option.Some(target_pos) -> {
          let Vec3(cx, _, cz) = cue_pos
          let Vec3(tx, _, tz) = target_pos
          float_sqrt({ tx -. cx } *. { tx -. cx } +. { tz -. cz } *. { tz -. cz })
        }
        option.None -> 999.0
      }
    }
    option.None -> 999.0
  }
}

/// Calculate approach bonus: reward for getting cue ball closer to target
/// This gives gradient signal even when shot misses completely
fn calculate_approach_bonus(
  table_before: Table,
  table_after: Table,
  target: sinuca.BallColor,
) -> Float {
  let dist_before = distance_to_ball(table_before, target)
  let dist_after = distance_to_ball(table_after, target)

  // Reward for getting closer, scaled 0-3 points
  case dist_before >. 0.0 {
    True -> {
      let improvement = { dist_before -. dist_after } /. dist_before
      // Clamp to 0-3 range (reward only for getting closer)
      float_clamp(improvement *. 3.0, 0.0, 3.0)
    }
    False -> 0.0
  }
}

fn distance_to_ball(table: Table, target: sinuca.BallColor) -> Float {
  case sinuca.get_cue_ball_position(table) {
    option.Some(cue_pos) -> {
      case sinuca.get_ball_position(table, target) {
        option.Some(target_pos) -> {
          let Vec3(cx, _, cz) = cue_pos
          let Vec3(tx, _, tz) = target_pos
          float_sqrt({ tx -. cx } *. { tx -. cx } +. { tz -. cz } *. { tz -. cz })
        }
        option.None -> 999.0
      }
    }
    option.None -> 999.0
  }
}

pub fn angle_to_target(table: Table) -> Float {
  let target = table.target_ball
  case sinuca.get_cue_ball_position(table) {
    option.Some(cue_pos) -> {
      case sinuca.get_ball_position(table, target) {
        option.Some(target_pos) -> {
          let Vec3(cx, _, cz) = cue_pos
          let Vec3(tx, _, tz) = target_pos
          float_atan2(tz -. cz, tx -. cx)
        }
        option.None -> 0.0
      }
    }
    option.None -> 0.0
  }
}

pub fn best_pocket_angle(table: Table) -> #(Float, Float) {
  let target = table.target_ball
  case sinuca.get_cue_ball_position(table) {
    option.Some(cue_pos) -> {
      case sinuca.get_ball_position(table, target) {
        option.Some(target_pos) -> find_best_pocket(cue_pos, target_pos)
        option.None -> #(0.0, 999.0)
      }
    }
    option.None -> #(0.0, 999.0)
  }
}

fn find_best_pocket(cue_pos: Vec3, target_pos: Vec3) -> #(Float, Float) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0

  let pockets = [
    #(neg(half_l), half_w),
    #(half_l, half_w),
    #(neg(half_l), neg(half_w)),
    #(half_l, neg(half_w)),
    #(0.0, half_w),
    #(0.0, neg(half_w)),
  ]

  let Vec3(cx, _, cz) = cue_pos
  let Vec3(tx, _, tz) = target_pos

  let result = list.fold(pockets, #(999.0, 0.0, 999.0), fn(acc, pocket) {
    let #(best_diff, _, _) = acc
    let #(px, pz) = pocket

    let cue_to_target = float_atan2(tz -. cz, tx -. cx)
    let target_to_pocket = float_atan2(pz -. tz, px -. tx)
    let diff = float_abs(cue_to_target -. target_to_pocket)

    let dist = float_sqrt({ px -. tx } *. { px -. tx } +. { pz -. tz } *. { pz -. tz })

    case diff <. best_diff {
      True -> #(diff, cue_to_target, dist)
      False -> acc
    }
  })

  let #(_, angle, dist) = result
  #(angle, dist)
}

// =============================================================================
// HELPERS
// =============================================================================

fn neg(x: Float) -> Float {
  0.0 -. x
}

fn float_clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False -> case x >. max {
      True -> max
      False -> x
    }
  }
}

@external(erlang, "erlang", "abs")
fn float_abs(x: Float) -> Float

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "atan2")
fn float_atan2(y: Float, x: Float) -> Float
