//// VIVA Sinuca - Brazilian Bar Pool with Jolt Physics
////
//// Realistic Brazilian sinuca simulation using JoltPhysics.
//// Specifications based on scientific research (Mathavan 2014, pooltool).
////
//// Bar table: 2.80m x 1.52m (playing area: 2.54m x 1.27m)
//// Balls: 7 colored (52mm) + cue ball (56mm)

import gleam/list
import gleam/option.{type Option, None, Some}
import viva/lifecycle/jolt.{
  type BodyId, type Vec3, type World, BodyId, Dynamic, Static, Vec3,
}

// =============================================================================
// CONSTANTS - Brazilian Bar Sinuca (Scientific Research)
// =============================================================================

/// Table dimensions (meters)
pub const table_length: Float = 2.54

pub const table_width: Float = 1.27

pub const table_height: Float = 0.05

/// Cushion (tabela)
pub const cushion_height: Float = 0.04

pub const cushion_thickness: Float = 0.05

/// Colored balls: 52mm diameter
pub const ball_radius: Float = 0.026

/// Cue ball (tacadeira): 56mm diameter
pub const cue_ball_radius: Float = 0.028

/// Pocket: ~10cm opening
pub const pocket_radius: Float = 0.05

/// Ball mass (kg)
pub const ball_mass: Float = 0.16

/// Physics based on Mathavan 2014 + pooltool + Dr. Dave
/// ============================================================

/// Sliding friction (ball-cloth) - when ball slides before rolling
pub const sliding_friction: Float = 0.20

/// Rolling friction (ball-cloth) - very low, ball rolls for long time
pub const rolling_friction: Float = 0.01

/// Spinning friction - ball spinning in place decays
pub const spinning_friction: Float = 0.044

/// Ball-ball friction (causes "throw" effect)
pub const ball_ball_friction: Float = 0.05

/// Ball-cushion friction
pub const cushion_friction: Float = 0.14

/// Coefficient of restitution (ball-ball)
/// Mathavan 2014: 0.89-0.95, using 0.93 for realistic energy loss
pub const ball_restitution: Float = 0.93

/// Cushion restitution (K-66 rubber profile)
/// Real cushions absorb more energy than balls
pub const cushion_restitution: Float = 0.75

/// Gravity (m/s²)
pub const gravity: Float = 9.81

// =============================================================================
// TYPES - Brazilian Sinuca
// =============================================================================

/// Ball colors with their point values
pub type BallColor {
  White       // Cue ball (0 points)
  Red         // 1 point
  Yellow      // 2 points
  Green       // 3 points
  Brown       // 4 points
  Blue        // 5 points
  Pink        // 6 points
  Black       // 7 points
}

/// Ball state
pub type Ball {
  Ball(
    body: BodyId,
    color: BallColor,
    pocketed: Bool,
  )
}

/// Pocket positions
pub type PocketPosition {
  TopLeft
  TopRight
  BottomLeft
  BottomRight
  MiddleTop
  MiddleBottom
}

/// Pocket
pub type Pocket {
  Pocket(position: PocketPosition, center: Vec3)
}

/// Game state
pub type GameState {
  Breaking          // First shot - must hit red
  TargetBall        // Playing target ball (free shot)
  NumberedBall      // Playing numbered (penalty if miss)
  Finished
}

/// Player
pub type Player {
  Player1
  Player2
}

/// Complete sinuca table
pub type Table {
  Table(
    world: World,
    surface: BodyId,
    cushions: List(BodyId),
    balls: List(Ball),
    pockets: List(Pocket),
    cue_ball: BodyId,
    // Game state
    current_player: Player,
    score_p1: Int,
    score_p2: Int,
    state: GameState,
    target_ball: BallColor,
  )
}

/// Shot parameters
pub type Shot {
  Shot(
    /// Angle in radians (0 = positive X axis)
    angle: Float,
    /// Power (0.0 to 1.0)
    power: Float,
    /// English/side spin: -1.0 to 1.0
    english: Float,
    /// Elevation for massé (0.0 to 0.5)
    elevation: Float,
  )
}

/// Shot result
pub type ShotResult {
  ShotResult(
    balls_pocketed: List(BallColor),
    hit_target_ball: Bool,
    foul: Bool,
    points_gained: Int,
    points_lost: Int,
    turn_over: Bool,
    new_target_ball: BallColor,
  )
}

// =============================================================================
// TABLE CREATION
// =============================================================================

/// Create new Brazilian sinuca table
pub fn new() -> Table {
  let world = jolt.world_new()

  // Playing surface (slate)
  let surface = jolt.create_box(
    world,
    Vec3(0.0, neg(table_height /. 2.0), 0.0),
    Vec3(table_length /. 2.0, table_height /. 2.0, table_width /. 2.0),
    Static,
  )
  let _ = jolt.set_friction(world, surface, sliding_friction)

  // Cushions
  let cushions = create_cushions(world)

  // Pockets
  let pockets = create_pockets()

  // Balls in initial position
  let balls = create_balls(world)

  // Cue ball reference
  let cue_ball_body = case list.first(balls) {
    Ok(Ball(body, White, _)) -> body
    _ -> BodyId(0)
  }

  // Optimize broad phase
  jolt.optimize(world)

  Table(
    world: world,
    surface: surface,
    cushions: cushions,
    balls: balls,
    pockets: pockets,
    cue_ball: cue_ball_body,
    current_player: Player1,
    score_p1: 0,
    score_p2: 0,
    state: Breaking,
    target_ball: Red,
  )
}

/// Create the 4 cushions
fn create_cushions(world: World) -> List(BodyId) {
  let half_l = table_length /. 2.0
  let half_w = table_width /. 2.0
  let cush_h = cushion_height /. 2.0
  let cush_t = cushion_thickness /. 2.0

  // Top (Z+)
  let top = jolt.create_box(
    world,
    Vec3(0.0, cush_h, half_w +. cush_t),
    Vec3(half_l +. cushion_thickness, cush_h, cush_t),
    Static,
  )

  // Bottom (Z-)
  let bottom = jolt.create_box(
    world,
    Vec3(0.0, cush_h, neg(half_w +. cush_t)),
    Vec3(half_l +. cushion_thickness, cush_h, cush_t),
    Static,
  )

  // Left (X-)
  let left = jolt.create_box(
    world,
    Vec3(neg(half_l +. cush_t), cush_h, 0.0),
    Vec3(cush_t, cush_h, half_w),
    Static,
  )

  // Right (X+)
  let right = jolt.create_box(
    world,
    Vec3(half_l +. cush_t, cush_h, 0.0),
    Vec3(cush_t, cush_h, half_w),
    Static,
  )

  // Set cushion physics
  [top, bottom, left, right]
  |> list.each(fn(cushion) {
    let _ = jolt.set_friction(world, cushion, cushion_friction)
    let _ = jolt.set_restitution(world, cushion, cushion_restitution)
    Nil
  })

  [top, bottom, left, right]
}

/// Create the 6 pockets
fn create_pockets() -> List(Pocket) {
  let half_l = table_length /. 2.0
  let half_w = table_width /. 2.0

  [
    Pocket(TopLeft, Vec3(neg(half_l), 0.0, half_w)),
    Pocket(TopRight, Vec3(half_l, 0.0, half_w)),
    Pocket(BottomLeft, Vec3(neg(half_l), 0.0, neg(half_w))),
    Pocket(BottomRight, Vec3(half_l, 0.0, neg(half_w))),
    Pocket(MiddleTop, Vec3(0.0, 0.0, half_w)),
    Pocket(MiddleBottom, Vec3(0.0, 0.0, neg(half_w))),
  ]
}

/// Create the 8 balls in initial position
fn create_balls(world: World) -> List(Ball) {
  // Cue ball position (break area)
  let cue_pos = Vec3(neg(table_length /. 4.0), ball_radius, 0.0)

  // Colored balls positions (inverted triangle)
  // Target ball (red) at apex, black at back
  let spacing = ball_radius *. 2.1
  let base_x = table_length /. 4.0

  let positions = [
    #(White, cue_pos, cue_ball_radius),
    // Row 1 (apex) - Red
    #(Red, Vec3(base_x, ball_radius, 0.0), ball_radius),
    // Row 2
    #(Yellow, Vec3(base_x +. spacing *. 0.866, ball_radius, spacing *. 0.5), ball_radius),
    #(Green, Vec3(base_x +. spacing *. 0.866, ball_radius, neg(spacing *. 0.5)), ball_radius),
    // Row 3
    #(Brown, Vec3(base_x +. spacing *. 1.732, ball_radius, spacing), ball_radius),
    #(Blue, Vec3(base_x +. spacing *. 1.732, ball_radius, 0.0), ball_radius),
    #(Pink, Vec3(base_x +. spacing *. 1.732, ball_radius, neg(spacing)), ball_radius),
    // Row 4 (base) - Black in center
    #(Black, Vec3(base_x +. spacing *. 2.598, ball_radius, 0.0), ball_radius),
  ]

  list.map(positions, fn(p) {
    let #(color, pos, radius) = p
    create_ball(world, pos, color, radius)
  })
}

/// Create individual ball with full 3D physics
fn create_ball(world: World, position: Vec3, color: BallColor, radius: Float) -> Ball {
  let body = jolt.create_sphere(world, position, radius, Dynamic)
  let _ = jolt.set_friction(world, body, ball_ball_friction)
  let _ = jolt.set_restitution(world, body, ball_restitution)
  // Gravity enabled for realistic 3D physics - balls can jump off table!
  let _ = jolt.set_gravity_factor(world, body, 1.0)
  Ball(body: body, color: color, pocketed: False)
}

// =============================================================================
// PHYSICS SIMULATION
// =============================================================================

/// Execute a shot with full 3D physics
/// Elevation allows massé shots where the ball jumps
pub fn shoot(table: Table, shot: Shot) -> Table {
  let Shot(angle, power, english, elevation) = shot

  // Impulse direction in 3D (elevation affects Y component)
  let horizontal_factor = float_cos(elevation)
  let vertical_factor = float_sin(elevation)

  let dir_x = float_cos(angle) *. horizontal_factor
  let dir_z = float_sin(angle) *. horizontal_factor
  let dir_y = vertical_factor  // Upward component from elevated cue

  // Max realistic impulse (~8 N*s for hard shot, +2 for jump shots)
  let max_impulse = 8.0 +. { elevation *. 4.0 }  // More power for jump shots
  let magnitude = power *. max_impulse

  // Apply 3D linear impulse
  let impulse = Vec3(dir_x *. magnitude, dir_y *. magnitude *. 0.5, dir_z *. magnitude)
  let _ = jolt.add_impulse(table.world, table.cue_ball, impulse)

  // Apply english (side spin) + topspin/backspin from elevation
  let side_spin = english *. 4.0
  // Elevation creates backspin (negative X rotation for massé)
  let back_spin = elevation *. 8.0 *. float_cos(angle)
  let front_spin = elevation *. 8.0 *. float_sin(angle)
  let spin = Vec3(back_spin, side_spin, front_spin)
  let _ = jolt.add_angular_impulse(table.world, table.cue_ball, spin)

  // Activate cue ball
  let _ = jolt.activate(table.world, table.cue_ball)

  table
}

/// Advance simulation with realistic friction
pub fn step(table: Table, dt: Float) -> Table {
  // Apply realistic friction before physics step
  let table2 = apply_realistic_friction(table, dt)
  let _ = jolt.step(table2.world, dt)
  table2
}

/// Apply correct friction based on ball state (rolling vs sliding)
/// Based on Mathavan 2014 and pooltool physics
fn apply_realistic_friction(table: Table, dt: Float) -> Table {
  list.each(table.balls, fn(ball) {
    case ball.pocketed {
      True -> Nil
      False -> {
        case jolt.get_velocity(table.world, ball.body) {
          Ok(v) -> {
            case jolt.get_angular_velocity(table.world, ball.body) {
              Ok(w) -> {
                let r = case ball.color {
                  White -> cue_ball_radius
                  _ -> ball_radius
                }

                // Speed in XZ plane (table surface)
                let speed = float_sqrt(v.x *. v.x +. v.z *. v.z)

                // Surface speed from rotation (omega * R)
                let surface_speed = float_sqrt(w.x *. w.x +. w.z *. w.z) *. r

                // Slip = difference between linear and rotational surface speed
                let slip = float_abs(speed -. surface_speed)

                // Determine friction regime
                let friction = case slip >. 0.01 {
                  // SLIDING: high friction until pure rolling achieved
                  True -> sliding_friction
                  // ROLLING: low friction, ball rolls for long time
                  False -> rolling_friction
                }

                // Apply linear deceleration
                let decel = friction *. gravity
                case speed >. 0.001 {
                  True -> {
                    let dv = float_min(decel *. dt, speed)
                    let factor = { speed -. dv } /. speed
                    let new_v = Vec3(v.x *. factor, v.y, v.z *. factor)
                    let _ = jolt.set_velocity(table.world, ball.body, new_v)
                    Nil
                  }
                  False -> {
                    let _ = jolt.set_velocity(table.world, ball.body, Vec3(0.0, v.y, 0.0))
                    Nil
                  }
                }

                // SPINNING friction: ball spinning in place after stopping
                case speed <. 0.01 && float_abs(w.y) >. 0.1 {
                  True -> {
                    // Spin decays due to contact patch friction
                    // alpha = 5/(2*R) * mu_spin * g
                    let spin_decel = { 5.0 /. { 2.0 *. r } } *. spinning_friction *. gravity
                    let new_wy = sign_decay(w.y, spin_decel *. dt)
                    let _ = jolt.set_angular_velocity(table.world, ball.body,
                      Vec3(w.x, new_wy, w.z))
                    Nil
                  }
                  False -> Nil
                }

                // Accelerate rotation towards rolling condition when sliding
                case slip >. 0.01 && speed >. 0.01 {
                  True -> {
                    // Torque from friction: alpha = 5/(2*R) * mu * g
                    let alpha = { 5.0 /. { 2.0 *. r } } *. sliding_friction *. gravity

                    // Target angular velocity for pure rolling
                    // omega_x = -v_z / R, omega_z = v_x / R
                    let target_wx = neg(v.z) /. r
                    let target_wz = v.x /. r

                    let new_wx = approach(w.x, target_wx, alpha *. dt)
                    let new_wz = approach(w.z, target_wz, alpha *. dt)

                    let _ = jolt.set_angular_velocity(table.world, ball.body,
                      Vec3(new_wx, w.y, new_wz))
                    Nil
                  }
                  False -> Nil
                }

                Nil
              }
              Error(_) -> Nil
            }
          }
          Error(_) -> Nil
        }
      }
    }
  })

  table
}

/// Decay value towards zero
fn sign_decay(value: Float, amount: Float) -> Float {
  case value >. 0.0 {
    True -> float_max(0.0, value -. amount)
    False -> float_min(0.0, value +. amount)
  }
}

/// Move value towards target by amount
fn approach(current: Float, target: Float, amount: Float) -> Float {
  let diff = target -. current
  case float_abs(diff) <. amount {
    True -> target
    False -> case diff >. 0.0 {
      True -> current +. amount
      False -> current -. amount
    }
  }
}

/// Advance N steps
pub fn step_n(table: Table, n: Int, dt: Float) -> Table {
  let _ = jolt.step_n(table.world, n, dt)
  table
}

/// Check if all balls stopped
pub fn is_settled(table: Table) -> Bool {
  let threshold = 0.001

  list.all(table.balls, fn(ball) {
    case ball.pocketed {
      True -> True
      False -> {
        case jolt.get_velocity(table.world, ball.body) {
          Ok(Vec3(vx, vy, vz)) -> {
            let speed_sq = vx *. vx +. vy *. vy +. vz *. vz
            speed_sq <. threshold
          }
          Error(_) -> True
        }
      }
    }
  })
}

/// Simulate until settled
pub fn simulate_until_settled(table: Table, max_steps: Int) -> Table {
  simulate_loop(table, max_steps, 0)
}

fn simulate_loop(table: Table, max: Int, current: Int) -> Table {
  case current >= max {
    True -> table
    False -> {
      let table2 = step(table, 1.0 /. 60.0)
      case is_settled(table2) {
        True -> table2
        False -> simulate_loop(table2, max, current + 1)
      }
    }
  }
}

// =============================================================================
// POCKET DETECTION
// =============================================================================

/// Check if ball is in pocket
pub fn is_in_pocket(table: Table, ball: Ball) -> Bool {
  case jolt.get_position(table.world, ball.body) {
    Ok(pos) -> {
      list.any(table.pockets, fn(pocket) {
        let Vec3(bx, _by, bz) = pos
        let Vec3(cx, _cy, cz) = pocket.center
        let dx = bx -. cx
        let dz = bz -. cz
        let distance = float_sqrt(dx *. dx +. dz *. dz)
        distance <. pocket_radius
      })
    }
    Error(_) -> False
  }
}

/// Update pocketed status
pub fn update_pocketed(table: Table) -> Table {
  let updated_balls = list.map(table.balls, fn(ball) {
    case ball.pocketed {
      True -> ball
      False -> Ball(..ball, pocketed: is_in_pocket(table, ball))
    }
  })
  Table(..table, balls: updated_balls)
}

/// Get pocketed balls
pub fn get_pocketed_balls(table: Table) -> List(BallColor) {
  table.balls
  |> list.filter(fn(b) { b.pocketed })
  |> list.map(fn(b) { b.color })
}

/// Count balls on table
pub fn balls_on_table(table: Table) -> Int {
  list.count(table.balls, fn(b) { !b.pocketed })
}

// =============================================================================
// GAME RULES
// =============================================================================

/// Point value of a color
pub fn point_value(color: BallColor) -> Int {
  case color {
    White -> 0
    Red -> 1
    Yellow -> 2
    Green -> 3
    Brown -> 4
    Blue -> 5
    Pink -> 6
    Black -> 7
  }
}

/// Determine new target ball (lowest value still in play)
pub fn next_target_ball(table: Table) -> BallColor {
  let in_play = table.balls
    |> list.filter(fn(b) { !b.pocketed && b.color != White })
    |> list.map(fn(b) { b.color })

  // Find lowest value
  let ordered_colors = [Red, Yellow, Green, Brown, Blue, Pink, Black]

  list.find(ordered_colors, fn(color) {
    list.contains(in_play, color)
  })
  |> option_unwrap(Black)
}

/// Check if cue ball was pocketed (foul)
pub fn is_scratch(table: Table) -> Bool {
  case jolt.get_position(table.world, table.cue_ball) {
    Ok(pos) -> {
      list.any(table.pockets, fn(pocket) {
        let Vec3(bx, _by, bz) = pos
        let Vec3(cx, _cy, cz) = pocket.center
        let dx = bx -. cx
        let dz = bz -. cz
        let distance = float_sqrt(dx *. dx +. dz *. dz)
        distance <. pocket_radius
      })
    }
    Error(_) -> False
  }
}

/// Process shot result
pub fn process_shot(
  table_before: Table,
  table_after: Table,
) -> #(Table, ShotResult) {
  let pocketed_before = get_pocketed_balls(table_before)
  let pocketed_after = get_pocketed_balls(table_after)

  // Newly pocketed balls
  let newly_pocketed = list.filter(pocketed_after, fn(color) {
    !list.contains(pocketed_before, color)
  })

  // Check fouls
  let scratch = is_scratch(table_after)

  // Calculate points
  let points_gained = list.fold(newly_pocketed, 0, fn(acc, color) {
    acc + point_value(color)
  })

  // Penalty for scratch
  let points_lost = case scratch {
    True -> 7  // Foul = 7 points to opponent
    False -> 0
  }

  // Check if hit target ball
  let hit_target = list.contains(newly_pocketed, table_before.target_ball)

  // Turn over?
  let turn_over = case list.is_empty(newly_pocketed) {
    True -> True
    False -> scratch
  }

  // New target ball
  let new_target = next_target_ball(table_after)

  // Update table
  let updated_table = case scratch {
    True -> reset_cue_ball(table_after)
    False -> table_after
  }

  let final_table = Table(
    ..updated_table,
    target_ball: new_target,
    state: case balls_on_table(updated_table) <= 1 {
      True -> Finished
      False -> TargetBall
    },
  )

  let result = ShotResult(
    balls_pocketed: newly_pocketed,
    hit_target_ball: hit_target,
    foul: scratch,
    points_gained: points_gained,
    points_lost: points_lost,
    turn_over: turn_over,
    new_target_ball: new_target,
  )

  #(final_table, result)
}

/// Reset cue ball after foul
pub fn reset_cue_ball(table: Table) -> Table {
  let initial_pos = Vec3(neg(table_length /. 4.0), ball_radius, 0.0)
  let _ = jolt.set_position(table.world, table.cue_ball, initial_pos)
  let _ = jolt.set_velocity(table.world, table.cue_ball, jolt.vec3_zero())
  let _ = jolt.set_angular_velocity(table.world, table.cue_ball, jolt.vec3_zero())

  // Update ball state
  let updated_balls = list.map(table.balls, fn(ball) {
    case ball.color {
      White -> Ball(..ball, pocketed: False)
      _ -> ball
    }
  })

  Table(..table, balls: updated_balls)
}

/// Reset entire table
pub fn reset(_table: Table) -> Table {
  new()
}

// =============================================================================
// QUERIES
// =============================================================================

/// Get ball position
pub fn get_ball_position(table: Table, color: BallColor) -> Option(Vec3) {
  case list.find(table.balls, fn(b) { b.color == color }) {
    Ok(ball) -> {
      case jolt.get_position(table.world, ball.body) {
        Ok(pos) -> Some(pos)
        Error(_) -> None
      }
    }
    Error(_) -> None
  }
}

/// Get cue ball position
pub fn get_cue_ball_position(table: Table) -> Option(Vec3) {
  case jolt.get_position(table.world, table.cue_ball) {
    Ok(pos) -> Some(pos)
    Error(_) -> None
  }
}

/// Get all positions
pub fn get_all_positions(table: Table) -> List(#(BallColor, Vec3)) {
  list.filter_map(table.balls, fn(ball) {
    case ball.pocketed {
      True -> Error(Nil)
      False -> {
        case jolt.get_position(table.world, ball.body) {
          Ok(pos) -> Ok(#(ball.color, pos))
          Error(_) -> Error(Nil)
        }
      }
    }
  })
}

// =============================================================================
// DISPLAY
// =============================================================================

/// Color name
pub fn color_name(color: BallColor) -> String {
  case color {
    White -> "White"
    Red -> "Red"
    Yellow -> "Yellow"
    Green -> "Green"
    Brown -> "Brown"
    Blue -> "Blue"
    Pink -> "Pink"
    Black -> "Black"
  }
}

/// Player name
pub fn player_name(player: Player) -> String {
  case player {
    Player1 -> "Player 1"
    Player2 -> "Player 2"
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn neg(x: Float) -> Float {
  0.0 -. x
}

fn option_unwrap(opt: Result(a, Nil), default: a) -> a {
  case opt {
    Ok(v) -> v
    Error(_) -> default
  }
}

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "erlang", "max")
fn float_max(a: Float, b: Float) -> Float

@external(erlang, "erlang", "min")
fn float_min(a: Float, b: Float) -> Float

@external(erlang, "erlang", "abs")
fn float_abs(x: Float) -> Float
