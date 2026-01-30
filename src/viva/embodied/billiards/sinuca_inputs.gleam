//// VIVA Sinuca - Advanced Input Encoding for NEAT
////
//// Enhanced state representation for neural network:
//// - All ball positions (not just target)
//// - Angles to all 6 pockets
//// - Obstruction detection
//// - Strategic context

import gleam/float
import gleam/int
import gleam/list
import gleam/option
import viva/embodied/billiards/sinuca.{type Table, type BallColor, White, Red, Yellow, Green, Brown, Blue, Pink, Black}
import viva/lifecycle/jolt.{type Vec3, Vec3}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Input vector size for different configurations
pub const basic_inputs: Int = 8
pub const full_inputs: Int = 42
pub const spatial_inputs: Int = 64

// =============================================================================
// BASIC INPUTS (8 floats) - Legacy compatibility
// =============================================================================

/// Encode basic inputs (cue, target, best pocket)
pub fn encode_basic(table: Table) -> List(Float) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0

  // Cue ball position
  let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(table) {
    option.Some(Vec3(x, _y, z)) -> #(x, z)
    option.None -> #(0.0, 0.0)
  }

  // Target ball position
  let target = table.target_ball
  let #(target_x, target_z) = case sinuca.get_ball_position(table, target) {
    option.Some(Vec3(x, _y, z)) -> #(x, z)
    option.None -> #(0.0, 0.0)
  }

  // Best pocket angle and distance
  let #(pocket_angle, pocket_dist) = best_pocket_for_ball(table, target)

  // Target ball value (normalized 0-1)
  let target_value = int.to_float(sinuca.point_value(target)) /. 7.0

  // Game state (how many balls left, normalized)
  let balls_left = int.to_float(sinuca.balls_on_table(table)) /. 8.0

  // Normalize all inputs to [-1, 1]
  [
    cue_x /. half_l,
    cue_z /. half_w,
    target_x /. half_l,
    target_z /. half_w,
    pocket_angle /. 3.14159,
    float_clamp(pocket_dist /. 3.0, 0.0, 1.0) *. 2.0 -. 1.0,
    target_value *. 2.0 -. 1.0,
    balls_left *. 2.0 -. 1.0,
  ]
}

// =============================================================================
// FULL INPUTS (42 floats) - Complete table awareness
// =============================================================================

/// Encode full table state
/// Structure:
/// - [0-1] Cue ball (x, z)
/// - [2-15] 7 colored balls (x, z) - White excluded
/// - [16-27] 6 pocket angles from cue
/// - [28-33] 6 pocket distances from target
/// - [34] Target ball index (0-6)
/// - [35] Balls remaining
/// - [36-41] Obstruction scores for each pocket
pub fn encode_full(table: Table) -> List(Float) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0

  // Cue ball position
  let cue_pos = case sinuca.get_cue_ball_position(table) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }
  let Vec3(cue_x, _cue_y, cue_z) = cue_pos

  // All ball positions (7 colored balls, 2 floats each = 14)
  let ball_colors = [Red, Yellow, Green, Brown, Blue, Pink, Black]
  let ball_positions = list.flat_map(ball_colors, fn(color) {
    case sinuca.get_ball_position(table, color) {
      option.Some(Vec3(x, _y, z)) -> [x /. half_l, z /. half_w]
      option.None -> [2.0, 2.0]  // Off table marker
    }
  })

  // Pocket positions
  let pockets = get_pocket_positions()

  // Angles from cue to each pocket (6 floats)
  let pocket_angles = list.map(pockets, fn(pocket) {
    let #(px, pz) = pocket
    float_atan2(pz -. cue_z, px -. cue_x) /. 3.14159
  })

  // Target ball position
  let target = table.target_ball
  let target_pos = case sinuca.get_ball_position(table, target) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }
  let Vec3(target_x, _target_y, target_z) = target_pos

  // Distances from target to each pocket (6 floats)
  let pocket_distances = list.map(pockets, fn(pocket) {
    let #(px, pz) = pocket
    let dx = px -. target_x
    let dz = pz -. target_z
    let dist = float_sqrt(dx *. dx +. dz *. dz)
    float_clamp(dist /. 3.0, 0.0, 1.0) *. 2.0 -. 1.0
  })

  // Target ball index (0-6)
  let target_idx = case target {
    Red -> 0.0
    Yellow -> 1.0 /. 6.0
    Green -> 2.0 /. 6.0
    Brown -> 3.0 /. 6.0
    Blue -> 4.0 /. 6.0
    Pink -> 5.0 /. 6.0
    Black -> 1.0
    White -> 0.0
  }

  // Balls remaining
  let balls_left = int.to_float(sinuca.balls_on_table(table)) /. 8.0

  // Obstruction scores for each pocket
  let obstructions = list.map(pockets, fn(pocket) {
    let #(px, pz) = pocket
    obstruction_score(table, cue_pos, target_pos, Vec3(px, 0.0, pz))
  })

  // Combine all inputs
  list.flatten([
    [cue_x /. half_l, cue_z /. half_w],
    ball_positions,
    pocket_angles,
    pocket_distances,
    [target_idx, balls_left *. 2.0 -. 1.0],
    obstructions,
  ])
}

// =============================================================================
// SPATIAL INPUTS (64 floats) - Grid representation
// =============================================================================

/// Encode as 8x8 spatial grid
/// Each cell: occupancy (ball present), ball value
pub fn encode_spatial(table: Table) -> List(Float) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0
  let cell_l = sinuca.table_length /. 8.0
  let cell_w = sinuca.table_width /. 8.0

  // Get all ball positions
  let positions = sinuca.get_all_positions(table)

  // Create 8x8 grid (64 cells)
  list.flat_map(list.range(0, 7), fn(row) {
    list.map(list.range(0, 7), fn(col) {
      let cell_x = neg(half_l) +. int.to_float(col) *. cell_l +. cell_l /. 2.0
      let cell_z = neg(half_w) +. int.to_float(row) *. cell_w +. cell_w /. 2.0

      // Check if any ball in this cell
      let ball_in_cell = list.find(positions, fn(p) {
        let #(_color, Vec3(x, _y, z)) = p
        let in_x = float_abs(x -. cell_x) <. cell_l /. 2.0
        let in_z = float_abs(z -. cell_z) <. cell_w /. 2.0
        in_x && in_z
      })

      case ball_in_cell {
        Ok(#(color, _pos)) -> {
          // Encode: cue ball = -1, colored = value/7
          case color {
            White -> neg(1.0)
            _ -> int.to_float(sinuca.point_value(color)) /. 7.0
          }
        }
        Error(_) -> 0.0
      }
    })
  })
}

// =============================================================================
// POCKET UTILITIES
// =============================================================================

/// Get pocket positions as (x, z) tuples
fn get_pocket_positions() -> List(#(Float, Float)) {
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0
  let neg_half_l = neg(half_l)
  let neg_half_w = neg(half_w)

  [
    #(neg_half_l, half_w),      // Top-left
    #(half_l, half_w),          // Top-right
    #(neg_half_l, neg_half_w),  // Bottom-left
    #(half_l, neg_half_w),      // Bottom-right
    #(0.0, half_w),             // Middle-top
    #(0.0, neg_half_w),         // Middle-bottom
  ]
}

/// Find best pocket for a specific ball
pub fn best_pocket_for_ball(table: Table, color: BallColor) -> #(Float, Float) {
  let cue_pos = case sinuca.get_cue_ball_position(table) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }

  let ball_pos = case sinuca.get_ball_position(table, color) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }

  find_best_pocket(cue_pos, ball_pos)
}

fn find_best_pocket(cue_pos: Vec3, target_pos: Vec3) -> #(Float, Float) {
  let pockets = get_pocket_positions()

  let Vec3(cx, _cy, cz) = cue_pos
  let Vec3(tx, _ty, tz) = target_pos

  let result = list.fold(pockets, #(999.0, 0.0, 999.0), fn(acc, pocket) {
    let #(best_diff, _best_angle, _best_dist) = acc
    let #(px, pz) = pocket

    // Angle from cue to target
    let cue_to_target = float_atan2(tz -. cz, tx -. cx)

    // Angle from target to pocket
    let target_to_pocket = float_atan2(pz -. tz, px -. tx)

    // Difference (smaller = more direct shot)
    let diff = float_abs(cue_to_target -. target_to_pocket)

    // Distance to pocket
    let dx = px -. tx
    let dz = pz -. tz
    let dist = float_sqrt(dx *. dx +. dz *. dz)

    case diff <. best_diff {
      True -> #(diff, cue_to_target, dist)
      False -> acc
    }
  })

  let #(_diff, angle, dist) = result
  #(angle, dist)
}

// =============================================================================
// OBSTRUCTION DETECTION
// =============================================================================

/// Calculate obstruction score for a shot line (0 = clear, 1 = blocked)
fn obstruction_score(table: Table, cue: Vec3, target: Vec3, pocket: Vec3) -> Float {
  let positions = sinuca.get_all_positions(table)
  let target_color = table.target_ball

  let Vec3(cx, _cy, cz) = cue
  let Vec3(tx, _ty, tz) = target
  let Vec3(px, _py, pz) = pocket

  // Check cue -> target line
  let cue_to_target_blocked = list.any(positions, fn(p) {
    let #(color, Vec3(bx, _by, bz)) = p
    case color == White || color == target_color {
      True -> False
      False -> point_near_line(cx, cz, tx, tz, bx, bz, sinuca.ball_radius *. 2.0)
    }
  })

  // Check target -> pocket line
  let target_to_pocket_blocked = list.any(positions, fn(p) {
    let #(color, Vec3(bx, _by, bz)) = p
    case color == White || color == target_color {
      True -> False
      False -> point_near_line(tx, tz, px, pz, bx, bz, sinuca.ball_radius *. 2.0)
    }
  })

  case cue_to_target_blocked, target_to_pocket_blocked {
    True, True -> 1.0
    True, False -> 0.7
    False, True -> 0.5
    False, False -> 0.0
  }
}

/// Check if point is near a line segment
fn point_near_line(
  x1: Float, z1: Float,
  x2: Float, z2: Float,
  px: Float, pz: Float,
  threshold: Float,
) -> Bool {
  let dx = x2 -. x1
  let dz = z2 -. z1
  let len_sq = dx *. dx +. dz *. dz

  case len_sq <. 0.0001 {
    True -> False
    False -> {
      // Project point onto line
      let t = float_clamp(
        { { px -. x1 } *. dx +. { pz -. z1 } *. dz } /. len_sq,
        0.0,
        1.0,
      )

      let proj_x = x1 +. t *. dx
      let proj_z = z1 +. t *. dz

      let dist_sq = { px -. proj_x } *. { px -. proj_x } +. { pz -. proj_z } *. { pz -. proj_z }
      dist_sq <. threshold *. threshold
    }
  }
}

// =============================================================================
// SHOT DIFFICULTY
// =============================================================================

/// Calculate shot difficulty (0 = easy, 1 = very hard)
pub fn shot_difficulty(table: Table, target_color: BallColor) -> Float {
  let cue_pos = case sinuca.get_cue_ball_position(table) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }

  let target_pos = case sinuca.get_ball_position(table, target_color) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }

  let Vec3(cx, _cy, cz) = cue_pos
  let Vec3(tx, _ty, tz) = target_pos

  // Distance factor
  let dx = tx -. cx
  let dz = tz -. cz
  let distance = float_sqrt(dx *. dx +. dz *. dz)
  let distance_factor = float_clamp(distance /. 2.5, 0.0, 1.0)

  // Best pocket angle (cut angle)
  let #(best_angle, pocket_dist) = best_pocket_for_ball(table, target_color)
  let cue_to_target = float_atan2(tz -. cz, tx -. cx)
  let cut_angle = float_abs(best_angle -. cue_to_target)
  let cut_factor = float_clamp(cut_angle /. 1.57, 0.0, 1.0)  // 90 degrees max

  // Pocket distance factor
  let pocket_factor = float_clamp(pocket_dist /. 2.0, 0.0, 1.0)

  // Combine factors
  { distance_factor *. 0.3 +. cut_factor *. 0.5 +. pocket_factor *. 0.2 }
}

// =============================================================================
// STRATEGIC ANALYSIS
// =============================================================================

/// Analyze table for strategic opportunities
pub fn strategic_analysis(table: Table) -> StrategicInfo {
  let target = table.target_ball
  let positions = sinuca.get_all_positions(table)

  // Count clustered balls
  let clusters = count_clusters(positions)

  // Count clear shots
  let clear_shots = count_clear_shots(table, positions)

  // Safety opportunity (can hide cue ball)
  let safety_available = has_safety_opportunity(table)

  StrategicInfo(
    target_difficulty: shot_difficulty(table, target),
    cluster_count: clusters,
    clear_shot_count: clear_shots,
    safety_available: safety_available,
    balls_remaining: sinuca.balls_on_table(table),
  )
}

pub type StrategicInfo {
  StrategicInfo(
    target_difficulty: Float,
    cluster_count: Int,
    clear_shot_count: Int,
    safety_available: Bool,
    balls_remaining: Int,
  )
}

fn count_clusters(positions: List(#(BallColor, Vec3))) -> Int {
  let cluster_threshold = sinuca.ball_radius *. 4.0

  list.fold(positions, 0, fn(acc, p1) {
    let #(_c1, Vec3(x1, _y1, z1)) = p1
    let nearby = list.count(positions, fn(p2) {
      let #(_c2, Vec3(x2, _y2, z2)) = p2
      let dx = x2 -. x1
      let dz = z2 -. z1
      let dist = float_sqrt(dx *. dx +. dz *. dz)
      dist <. cluster_threshold && dist >. 0.001
    })
    case nearby >= 2 {
      True -> acc + 1
      False -> acc
    }
  })
}

fn count_clear_shots(table: Table, positions: List(#(BallColor, Vec3))) -> Int {
  let cue_pos = case sinuca.get_cue_ball_position(table) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }

  let pockets = get_pocket_positions()

  list.fold(positions, 0, fn(acc, p) {
    let #(color, ball_pos) = p
    case color == White {
      True -> acc
      False -> {
        // Check if any pocket is clear
        let has_clear = list.any(pockets, fn(pocket) {
          let #(px, pz) = pocket
          let pocket_vec = Vec3(px, 0.0, pz)
          let obs = obstruction_score(table, cue_pos, ball_pos, pocket_vec)
          obs <. 0.3
        })
        case has_clear {
          True -> acc + 1
          False -> acc
        }
      }
    }
  })
}

fn has_safety_opportunity(table: Table) -> Bool {
  let cue_pos = case sinuca.get_cue_ball_position(table) {
    option.Some(pos) -> pos
    option.None -> Vec3(0.0, 0.0, 0.0)
  }

  let Vec3(cx, _cy, cz) = cue_pos

  // Check if cue ball is near a cushion (good for safety)
  let half_l = sinuca.table_length /. 2.0
  let half_w = sinuca.table_width /. 2.0
  let margin = 0.3

  float_abs(cx) >. half_l -. margin || float_abs(cz) >. half_w -. margin
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
