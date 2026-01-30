//// VIVA Billiards - 3D Pool Table with Jolt Physics
////
//// Realistic 8-ball pool simulation using JoltPhysics engine.
//// Standard 9ft table (2.54m x 1.27m), regulation ball size (57.15mm).

import gleam/list
import gleam/int
import viva/jolt.{
  type BodyId, type Vec3, type World, BodyId, Dynamic, Static, Vec3,
}

// =============================================================================
// CONSTANTS - Regulation Pool Table (9ft)
// =============================================================================

/// Table dimensions in meters (9ft table)
pub const table_length: Float = 2.54

pub const table_width: Float = 1.27

pub const table_height: Float = 0.05

/// Rail height above playing surface
pub const rail_height: Float = 0.04

/// Rail thickness
pub const rail_thickness: Float = 0.05

/// Ball radius (regulation 57.15mm diameter)
pub const ball_radius: Float = 0.028575

/// Pocket radius (regulation ~11.5cm opening)
pub const pocket_radius: Float = 0.0575

/// Ball mass in kg
pub const ball_mass: Float = 0.17

/// Friction coefficient for cloth
pub const cloth_friction: Float = 0.2

/// Ball-to-ball friction
pub const ball_friction: Float = 0.05

/// Ball restitution (bounciness)
pub const ball_restitution: Float = 0.92

/// Rail restitution
pub const rail_restitution: Float = 0.75

// =============================================================================
// TYPES
// =============================================================================

/// Ball type for 8-ball pool
pub type BallType {
  CueBall
  Solid(number: Int)
  Stripe(number: Int)
  EightBall
}

/// A pool ball with physics body
pub type Ball {
  Ball(body: BodyId, ball_type: BallType, pocketed: Bool)
}

/// Pocket position on table
pub type PocketPosition {
  TopLeft
  TopMiddle
  TopRight
  BottomLeft
  BottomMiddle
  BottomRight
}

/// A pocket with trigger zone
pub type Pocket {
  Pocket(position: PocketPosition, center: Vec3, body: BodyId)
}

/// Complete pool table state
pub type Table {
  Table(
    world: World,
    surface: BodyId,
    rails: List(BodyId),
    balls: List(Ball),
    pockets: List(Pocket),
    cue_ball: BodyId,
  )
}

/// Shot parameters
pub type Shot {
  Shot(
    /// Direction angle in radians (0 = positive X)
    angle: Float,
    /// Force magnitude (0.0 to 1.0, scaled to max impulse)
    power: Float,
    /// English/spin (-1.0 to 1.0 for left/right)
    english: Float,
    /// Elevation for jump/massé shots (0.0 to 1.0)
    elevation: Float,
  )
}

/// Shot result after simulation
pub type ShotResult {
  ShotResult(
    balls_pocketed: List(BallType),
    cue_ball_pocketed: Bool,
    balls_remaining: Int,
    final_positions: List(#(BallType, Vec3)),
  )
}

// =============================================================================
// TABLE CREATION
// =============================================================================

/// Create a new pool table with all balls racked
pub fn new() -> Table {
  let world = jolt.world_new()

  // Create playing surface (static box)
  let surface =
    jolt.create_box(
      world,
      Vec3(0.0, neg(table_height /. 2.0), 0.0),
      Vec3(table_length /. 2.0, table_height /. 2.0, table_width /. 2.0),
      Static,
    )
  let _ = jolt.set_friction(world, surface, cloth_friction)

  // Create rails
  let rails = create_rails(world)

  // Create pockets (sensor zones)
  let pockets = create_pockets(world)

  // Create balls in rack formation
  let balls = create_rack(world)

  // Get cue ball reference
  let cue_ball = case list.first(balls) {
    Ok(Ball(body, CueBall, _)) -> body
    _ -> BodyId(0)
  }

  // Optimize broad phase after static setup
  jolt.optimize(world)

  Table(
    world: world,
    surface: surface,
    rails: rails,
    balls: balls,
    pockets: pockets,
    cue_ball: cue_ball,
  )
}

/// Create the 6 rails around the table
fn create_rails(world: World) -> List(BodyId) {
  let half_length = table_length /. 2.0
  let half_width = table_width /. 2.0
  let rail_h = rail_height /. 2.0
  let rail_t = rail_thickness /. 2.0

  // Top rail (positive Z)
  let top =
    jolt.create_box(
      world,
      Vec3(0.0, rail_h, half_width +. rail_t),
      Vec3(half_length +. rail_thickness, rail_h, rail_t),
      Static,
    )

  // Bottom rail (negative Z)
  let bottom =
    jolt.create_box(
      world,
      Vec3(0.0, rail_h, neg(half_width +. rail_t)),
      Vec3(half_length +. rail_thickness, rail_h, rail_t),
      Static,
    )

  // Left rail (negative X)
  let left =
    jolt.create_box(
      world,
      Vec3(neg(half_length +. rail_t), rail_h, 0.0),
      Vec3(rail_t, rail_h, half_width),
      Static,
    )

  // Right rail (positive X)
  let right =
    jolt.create_box(
      world,
      Vec3(half_length +. rail_t, rail_h, 0.0),
      Vec3(rail_t, rail_h, half_width),
      Static,
    )

  // Set rail properties
  [top, bottom, left, right]
  |> list.each(fn(rail) {
    let _ = jolt.set_friction(world, rail, 0.1)
    let _ = jolt.set_restitution(world, rail, rail_restitution)
    Nil
  })

  [top, bottom, left, right]
}

/// Create the 6 pockets
fn create_pockets(world: World) -> List(Pocket) {
  let half_length = table_length /. 2.0
  let half_width = table_width /. 2.0
  let pocket_y = neg(table_height)  // Below surface
  let neg_half_length = neg(half_length)
  let neg_half_width = neg(half_width)

  let positions = [
    #(TopLeft, Vec3(neg_half_length, pocket_y, half_width)),
    #(TopMiddle, Vec3(0.0, pocket_y, half_width)),
    #(TopRight, Vec3(half_length, pocket_y, half_width)),
    #(BottomLeft, Vec3(neg_half_length, pocket_y, neg_half_width)),
    #(BottomMiddle, Vec3(0.0, pocket_y, neg_half_width)),
    #(BottomRight, Vec3(half_length, pocket_y, neg_half_width)),
  ]

  list.map(positions, fn(pos) {
    let #(position, center) = pos
    // Create pocket as sphere sensor (for detection)
    let body =
      jolt.create_sphere(world, center, pocket_radius, Static)
    Pocket(position: position, center: center, body: body)
  })
}

/// Create balls in standard 8-ball rack formation
fn create_rack(world: World) -> List(Ball) {
  // Cue ball position (head spot)
  let cue_pos = Vec3(neg(table_length /. 4.0), ball_radius, 0.0)
  let cue = create_ball(world, cue_pos, CueBall)

  // Rack position (foot spot)
  let rack_x = table_length /. 4.0
  let ball_d = ball_radius *. 2.0 *. 1.02  // Slight gap

  // Row positions for triangle rack (5 rows)
  let rack_balls = [
    // Row 1 (apex)
    #(Solid(1), Vec3(rack_x, ball_radius, 0.0)),
    // Row 2
    #(Solid(2), Vec3(rack_x +. ball_d *. 0.866, ball_radius, ball_d *. 0.5)),
    #(Stripe(9), Vec3(rack_x +. ball_d *. 0.866, ball_radius, neg(ball_d *. 0.5))),
    // Row 3 (8-ball in center)
    #(Solid(3), Vec3(rack_x +. ball_d *. 1.732, ball_radius, ball_d)),
    #(EightBall, Vec3(rack_x +. ball_d *. 1.732, ball_radius, 0.0)),
    #(Stripe(10), Vec3(rack_x +. ball_d *. 1.732, ball_radius, neg(ball_d))),
    // Row 4
    #(Stripe(11), Vec3(rack_x +. ball_d *. 2.598, ball_radius, ball_d *. 1.5)),
    #(Solid(4), Vec3(rack_x +. ball_d *. 2.598, ball_radius, ball_d *. 0.5)),
    #(Solid(5), Vec3(rack_x +. ball_d *. 2.598, ball_radius, neg(ball_d *. 0.5))),
    #(Stripe(12), Vec3(rack_x +. ball_d *. 2.598, ball_radius, neg(ball_d *. 1.5))),
    // Row 5
    #(Solid(6), Vec3(rack_x +. ball_d *. 3.464, ball_radius, ball_d *. 2.0)),
    #(Stripe(13), Vec3(rack_x +. ball_d *. 3.464, ball_radius, ball_d)),
    #(Solid(7), Vec3(rack_x +. ball_d *. 3.464, ball_radius, 0.0)),
    #(Stripe(14), Vec3(rack_x +. ball_d *. 3.464, ball_radius, neg(ball_d))),
    #(Stripe(15), Vec3(rack_x +. ball_d *. 3.464, ball_radius, neg(ball_d *. 2.0))),
  ]

  let balls =
    list.map(rack_balls, fn(b) {
      let #(ball_type, pos) = b
      create_ball(world, pos, ball_type)
    })

  [cue, ..balls]
}

/// Create a single ball
fn create_ball(world: World, position: Vec3, ball_type: BallType) -> Ball {
  let body = jolt.create_sphere(world, position, ball_radius, Dynamic)
  let _ = jolt.set_friction(world, body, ball_friction)
  let _ = jolt.set_restitution(world, body, ball_restitution)
  // No gravity on table surface (balls stay on cloth)
  let _ = jolt.set_gravity_factor(world, body, 0.0)
  Ball(body: body, ball_type: ball_type, pocketed: False)
}

// =============================================================================
// PHYSICS SIMULATION
// =============================================================================

/// Execute a shot on the cue ball
pub fn shoot(table: Table, shot: Shot) -> Table {
  let Shot(angle, power, english, _elevation) = shot

  // Calculate impulse direction
  let dir_x = float_cos(angle)
  let dir_z = float_sin(angle)

  // Scale power to realistic impulse (max ~10 N*s for a hard shot)
  let max_impulse = 10.0
  let impulse_magnitude = power *. max_impulse

  // Apply linear impulse
  let impulse = Vec3(dir_x *. impulse_magnitude, 0.0, dir_z *. impulse_magnitude)
  let _ = jolt.add_impulse(table.world, table.cue_ball, impulse)

  // Apply english (side spin) as angular impulse
  let spin_magnitude = english *. 5.0
  let spin = Vec3(0.0, spin_magnitude, 0.0)
  let _ = jolt.add_angular_impulse(table.world, table.cue_ball, spin)

  // Activate cue ball
  let _ = jolt.activate(table.world, table.cue_ball)

  table
}

/// Step the simulation forward
pub fn step(table: Table, dt: Float) -> Table {
  let _ = jolt.step(table.world, dt)
  table
}

/// Step simulation N times (batch)
pub fn step_n(table: Table, n: Int, dt: Float) -> Table {
  let _ = jolt.step_n(table.world, n, dt)
  table
}

/// Check if all balls have stopped moving
pub fn is_settled(table: Table) -> Bool {
  let velocity_threshold = 0.001

  list.all(table.balls, fn(ball) {
    case ball.pocketed {
      True -> True
      False -> {
        case jolt.get_velocity(table.world, ball.body) {
          Ok(Vec3(vx, vy, vz)) -> {
            let speed_sq = vx *. vx +. vy *. vy +. vz *. vz
            speed_sq <. velocity_threshold
          }
          Error(_) -> True
        }
      }
    }
  })
}

/// Simulate until balls settle (max iterations)
pub fn simulate_until_settled(table: Table, max_steps: Int) -> Table {
  simulate_loop(table, max_steps, 0)
}

fn simulate_loop(table: Table, max_steps: Int, current: Int) -> Table {
  case current >= max_steps {
    True -> table
    False -> {
      let table = step(table, 1.0 /. 60.0)
      case is_settled(table) {
        True -> table
        False -> simulate_loop(table, max_steps, current + 1)
      }
    }
  }
}

// =============================================================================
// BALL STATE QUERIES
// =============================================================================

/// Get position of a ball
pub fn get_ball_position(table: Table, ball: Ball) -> Result(Vec3, Nil) {
  jolt.get_position(table.world, ball.body)
}

/// Get cue ball position
pub fn get_cue_ball_position(table: Table) -> Result(Vec3, Nil) {
  jolt.get_position(table.world, table.cue_ball)
}

/// Get velocity of a ball
pub fn get_ball_velocity(table: Table, ball: Ball) -> Result(Vec3, Nil) {
  jolt.get_velocity(table.world, ball.body)
}

/// Get all ball positions
pub fn get_all_positions(table: Table) -> List(#(BallType, Vec3)) {
  list.filter_map(table.balls, fn(ball) {
    case ball.pocketed {
      True -> Error(Nil)
      False -> {
        case jolt.get_position(table.world, ball.body) {
          Ok(pos) -> Ok(#(ball.ball_type, pos))
          Error(_) -> Error(Nil)
        }
      }
    }
  })
}

/// Check if a ball is in a pocket
pub fn is_in_pocket(table: Table, ball: Ball) -> Bool {
  case jolt.get_position(table.world, ball.body) {
    Ok(Vec3(_x, y, _z)) -> y <. 0.0  // Below table surface
    Error(_) -> False
  }
}

/// Update pocketed status for all balls
pub fn update_pocketed(table: Table) -> Table {
  let updated_balls =
    list.map(table.balls, fn(ball) {
      case ball.pocketed {
        True -> ball
        False -> Ball(..ball, pocketed: is_in_pocket(table, ball))
      }
    })
  Table(..table, balls: updated_balls)
}

/// Count balls still on table
pub fn balls_on_table(table: Table) -> Int {
  list.count(table.balls, fn(ball) { !ball.pocketed })
}

/// Get list of pocketed balls
pub fn get_pocketed_balls(table: Table) -> List(BallType) {
  table.balls
  |> list.filter(fn(ball) { ball.pocketed })
  |> list.map(fn(ball) { ball.ball_type })
}

/// Check if cue ball is pocketed (scratch)
pub fn is_scratch(table: Table) -> Bool {
  case jolt.get_position(table.world, table.cue_ball) {
    Ok(Vec3(_x, y, _z)) -> y <. 0.0
    Error(_) -> False
  }
}

// =============================================================================
// RESET FUNCTIONS
// =============================================================================

/// Reset cue ball to head spot (after scratch)
pub fn reset_cue_ball(table: Table) -> Table {
  let head_spot = Vec3(neg(table_length /. 4.0), ball_radius, 0.0)
  let _ = jolt.set_position(table.world, table.cue_ball, head_spot)
  let _ = jolt.set_velocity(table.world, table.cue_ball, jolt.vec3_zero())
  let _ =
    jolt.set_angular_velocity(table.world, table.cue_ball, jolt.vec3_zero())

  // Update ball state
  let updated_balls =
    list.map(table.balls, fn(ball) {
      case ball.ball_type {
        CueBall -> Ball(..ball, pocketed: False)
        _ -> ball
      }
    })
  Table(..table, balls: updated_balls)
}

/// Reset entire table to initial state
pub fn reset(_table: Table) -> Table {
  // For now, create a new table (could optimize by just repositioning)
  new()
}

// =============================================================================
// HELPERS
// =============================================================================

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

/// Negate a float (Gleam has no unary minus)
fn neg(x: Float) -> Float {
  0.0 -. x
}

/// Ball type to string for display
pub fn ball_type_to_string(ball_type: BallType) -> String {
  case ball_type {
    CueBall -> "Cue"
    Solid(n) -> "Solid " <> int.to_string(n)
    Stripe(n) -> "Stripe " <> int.to_string(n)
    EightBall -> "8-Ball"
  }
}
