//// VIVA Sinuca - Advanced Physics Module
////
//// Realistic ball physics based on scientific research:
//// - Mathavan et al. (2014) - Frictional collisions
//// - Ekiefl (2020) - pooltool theory
//// - Han (2005) - Cushion model
////
//// Implements: rolling/sliding states, spin decay, massé curves

import gleam/float
import gleam/option.{type Option, None, Some}
import viva/lifecycle/jolt.{type BodyId, type Vec3, type World, Vec3}

// =============================================================================
// PHYSICS CONSTANTS (Scientific Literature)
// =============================================================================

/// Ball-cloth sliding friction (Mathavan 2014)
pub const mu_slide: Float = 0.2

/// Ball-cloth rolling friction
pub const mu_roll: Float = 0.01

/// Ball-cloth spin friction (for stationary spinning)
pub const mu_spin: Float = 0.044

/// Ball-ball friction (Mathavan 2014)
pub const mu_ball: Float = 0.05

/// Ball-cushion friction
pub const mu_cushion: Float = 0.21

/// Coefficient of restitution ball-ball
pub const e_ball: Float = 0.89

/// Coefficient of restitution ball-cushion
pub const e_cushion: Float = 0.75

/// Ball mass (kg)
pub const ball_mass: Float = 0.16

/// Ball radius (m) - 52mm diameter
pub const ball_radius: Float = 0.026

/// Cue ball radius (m) - 56mm diameter
pub const cue_radius: Float = 0.028

/// Gravity (m/s²)
pub const gravity: Float = 9.81

/// Moment of inertia coefficient (2/5 for solid sphere)
pub const inertia_coeff: Float = 0.4

// =============================================================================
// BALL STATE
// =============================================================================

/// Motion state of a ball
pub type MotionState {
  /// Ball at rest
  Stationary
  /// Ball spinning in place (no translation)
  Spinning
  /// Ball sliding (slip between ball and cloth)
  Sliding
  /// Ball rolling (no slip, v = ω × r)
  Rolling
}

/// Complete ball state with physics
pub type BallState {
  BallState(
    /// Position (x, y, z)
    position: Vec3,
    /// Linear velocity (vx, vy, vz)
    velocity: Vec3,
    /// Angular velocity (ωx, ωy, ωz) - spin
    angular_velocity: Vec3,
    /// Current motion state
    state: MotionState,
  )
}

/// Cue strike parameters
pub type CueStrike {
  CueStrike(
    /// Strike velocity (m/s)
    speed: Float,
    /// Horizontal angle (radians, 0 = +X)
    phi: Float,
    /// Elevation angle (radians, 0 = horizontal, π/2 = straight down)
    theta: Float,
    /// Horizontal offset from center (-1 to 1, for english)
    offset_x: Float,
    /// Vertical offset from center (-1 to 1, for draw/follow)
    offset_y: Float,
  )
}

// =============================================================================
// STATE DETECTION
// =============================================================================

/// Threshold for considering velocity as zero
const velocity_threshold: Float = 0.001

/// Threshold for considering angular velocity as zero
const angular_threshold: Float = 0.01

/// Determine motion state from velocities
pub fn detect_state(velocity: Vec3, angular_velocity: Vec3) -> MotionState {
  let Vec3(vx, vy, vz) = velocity
  let Vec3(wx, wy, wz) = angular_velocity

  let speed_sq = vx *. vx +. vy *. vy +. vz *. vz
  let spin_sq = wx *. wx +. wy *. wy +. wz *. wz

  let is_moving = speed_sq >. velocity_threshold *. velocity_threshold
  let is_spinning = spin_sq >. angular_threshold *. angular_threshold

  case is_moving, is_spinning {
    False, False -> Stationary
    False, True -> Spinning
    True, _ -> {
      // Check if rolling (v = ω × r at contact point)
      let slip = calculate_slip(velocity, angular_velocity)
      let slip_sq = slip.0 *. slip.0 +. slip.1 *. slip.1
      case slip_sq <. velocity_threshold *. velocity_threshold {
        True -> Rolling
        False -> Sliding
      }
    }
  }
}

/// Calculate slip velocity at contact point
/// Slip = v - (ω × r) where r points down to contact
fn calculate_slip(velocity: Vec3, angular_velocity: Vec3) -> #(Float, Float) {
  let Vec3(vx, _vy, vz) = velocity
  let Vec3(wx, _wy, wz) = angular_velocity

  // Contact point is at -R in Y direction
  // ω × (-R ŷ) = (-R) * (ωz x̂ - ωx ẑ) = R * (-ωz, 0, ωx)
  let contact_vx = ball_radius *. wz *. -1.0
  let contact_vz = ball_radius *. wx

  let slip_x = vx -. contact_vx
  let slip_z = vz -. contact_vz

  #(slip_x, slip_z)
}

// =============================================================================
// CUE STRIKE PHYSICS
// =============================================================================

/// Calculate initial ball state from cue strike
pub fn strike_to_state(strike: CueStrike) -> BallState {
  let CueStrike(speed, phi, theta, offset_x, offset_y) = strike

  // Linear velocity from cue direction
  let cos_phi = float_cos(phi)
  let sin_phi = float_sin(phi)
  let cos_theta = float_cos(theta)

  let vx = speed *. cos_phi *. cos_theta
  let vy = 0.0  // Assume flat strike initially
  let vz = speed *. sin_phi *. cos_theta

  // Angular velocity from offset (english and draw/follow)
  // Side spin (english) from horizontal offset
  let side_spin = offset_x *. speed *. 10.0  // ωy

  // Top/back spin from vertical offset
  // offset_y > 0 = follow (topspin), offset_y < 0 = draw (backspin)
  let top_spin = offset_y *. speed *. 10.0

  // Transform spin to ball frame
  let wx = top_spin *. sin_phi *. -1.0  // Backspin around perpendicular axis
  let wy = side_spin
  let wz = top_spin *. cos_phi

  let velocity = Vec3(vx, vy, vz)
  let angular_velocity = Vec3(wx, wy, wz)
  let state = detect_state(velocity, angular_velocity)

  BallState(
    position: Vec3(0.0, ball_radius, 0.0),
    velocity: velocity,
    angular_velocity: angular_velocity,
    state: state,
  )
}

// =============================================================================
// FRICTION FORCES
// =============================================================================

/// Calculate friction deceleration for sliding ball
pub fn sliding_friction(velocity: Vec3, angular_velocity: Vec3) -> #(Vec3, Vec3) {
  let #(slip_x, slip_z) = calculate_slip(velocity, angular_velocity)
  let slip_mag = float_sqrt(slip_x *. slip_x +. slip_z *. slip_z)

  case slip_mag >. velocity_threshold {
    False -> #(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    True -> {
      // Friction force opposes slip
      let friction = mu_slide *. ball_mass *. gravity
      let fx = friction *. slip_x /. slip_mag *. -1.0
      let fz = friction *. slip_z /. slip_mag *. -1.0

      // Linear deceleration
      let ax = fx /. ball_mass
      let az = fz /. ball_mass

      // Angular acceleration (torque = r × F)
      let torque_x = ball_radius *. fz *. -1.0
      let torque_z = ball_radius *. fx
      let inertia = inertia_coeff *. ball_mass *. ball_radius *. ball_radius
      let alpha_x = torque_x /. inertia
      let alpha_z = torque_z /. inertia

      #(Vec3(ax, 0.0, az), Vec3(alpha_x, 0.0, alpha_z))
    }
  }
}

/// Calculate friction deceleration for rolling ball
pub fn rolling_friction(velocity: Vec3) -> Vec3 {
  let Vec3(vx, _vy, vz) = velocity
  let speed = float_sqrt(vx *. vx +. vz *. vz)

  case speed >. velocity_threshold {
    False -> Vec3(0.0, 0.0, 0.0)
    True -> {
      let friction = mu_roll *. gravity
      let ax = friction *. vx /. speed *. -1.0
      let az = friction *. vz /. speed *. -1.0
      Vec3(ax, 0.0, az)
    }
  }
}

/// Calculate spin decay for stationary spinning ball
pub fn spin_friction(angular_velocity: Vec3) -> Vec3 {
  let Vec3(wx, wy, wz) = angular_velocity
  let spin = float_sqrt(wx *. wx +. wy *. wy +. wz *. wz)

  case spin >. angular_threshold {
    False -> Vec3(0.0, 0.0, 0.0)
    True -> {
      // Spin decays due to friction
      let decay = mu_spin *. gravity /. ball_radius
      let alpha_x = decay *. wx /. spin *. -1.0
      let alpha_y = decay *. wy /. spin *. -1.0
      let alpha_z = decay *. wz /. spin *. -1.0
      Vec3(alpha_x, alpha_y, alpha_z)
    }
  }
}

// =============================================================================
// MASSÉ PHYSICS
// =============================================================================

/// Calculate curve trajectory for massé shot
/// Returns lateral acceleration based on sidespin and forward velocity
pub fn masse_curve(velocity: Vec3, angular_velocity: Vec3) -> Vec3 {
  let Vec3(vx, _vy, vz) = velocity
  let Vec3(_wx, wy, _wz) = angular_velocity

  let speed = float_sqrt(vx *. vx +. vz *. vz)

  case speed >. velocity_threshold {
    False -> Vec3(0.0, 0.0, 0.0)
    True -> {
      // Side spin causes lateral force (Magnus-like effect on cloth)
      // Perpendicular to velocity direction
      let curve_strength = wy *. mu_slide *. 0.1  // Empirical factor

      // Perpendicular direction to velocity
      let perp_x = vz /. speed *. -1.0
      let perp_z = vx /. speed

      Vec3(curve_strength *. perp_x, 0.0, curve_strength *. perp_z)
    }
  }
}

// =============================================================================
// INTEGRATION STEP
// =============================================================================

/// Integrate ball state forward by dt
pub fn integrate(state: BallState, dt: Float) -> BallState {
  let BallState(position, velocity, angular_velocity, motion_state) = state

  // Calculate accelerations based on current state
  let #(linear_accel, angular_accel) = case motion_state {
    Stationary -> #(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    Spinning -> #(Vec3(0.0, 0.0, 0.0), spin_friction(angular_velocity))
    Sliding -> {
      let #(la, aa) = sliding_friction(velocity, angular_velocity)
      let curve = masse_curve(velocity, angular_velocity)
      #(vec3_add(la, curve), aa)
    }
    Rolling -> #(rolling_friction(velocity), Vec3(0.0, 0.0, 0.0))
  }

  // Update velocity
  let new_velocity = vec3_add(velocity, vec3_scale(linear_accel, dt))

  // Update angular velocity
  let new_angular = vec3_add(angular_velocity, vec3_scale(angular_accel, dt))

  // Update position
  let new_position = vec3_add(position, vec3_scale(velocity, dt))

  // Detect new state
  let new_state = detect_state(new_velocity, new_angular)

  // If rolling, enforce rolling constraint (v = ω × r)
  let #(final_velocity, final_angular) = case new_state {
    Rolling -> enforce_rolling(new_velocity, new_angular)
    _ -> #(new_velocity, new_angular)
  }

  BallState(
    position: new_position,
    velocity: final_velocity,
    angular_velocity: final_angular,
    state: new_state,
  )
}

/// Enforce rolling constraint: no slip at contact point
fn enforce_rolling(velocity: Vec3, angular_velocity: Vec3) -> #(Vec3, Vec3) {
  let Vec3(vx, vy, vz) = velocity
  let speed = float_sqrt(vx *. vx +. vz *. vz)

  case speed >. velocity_threshold {
    False -> #(velocity, angular_velocity)
    True -> {
      // For rolling: v = ω × r at contact
      // This means: vx = R * ωz, vz = -R * ωx
      let wx = vz /. ball_radius *. -1.0
      let wz = vx /. ball_radius
      let Vec3(_owx, wy, _owz) = angular_velocity

      #(velocity, Vec3(wx, wy, wz))
    }
  }
}

// =============================================================================
// APPLY TO JOLT
// =============================================================================

/// Apply advanced physics to a Jolt body
pub fn apply_to_body(world: World, body: BodyId, dt: Float) -> Option(MotionState) {
  case jolt.get_velocity(world, body) {
    Ok(velocity) -> {
      case jolt.get_angular_velocity(world, body) {
        Ok(angular) -> {
          let state = detect_state(velocity, angular)

          // Calculate and apply forces based on state
          let #(linear_accel, angular_accel) = case state {
            Stationary -> #(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
            Spinning -> #(Vec3(0.0, 0.0, 0.0), spin_friction(angular))
            Sliding -> sliding_friction(velocity, angular)
            Rolling -> #(rolling_friction(velocity), Vec3(0.0, 0.0, 0.0))
          }

          // Apply as forces
          let force = vec3_scale(linear_accel, ball_mass)
          let _ = jolt.add_force(world, body, force)

          // Apply angular (as torque would need inertia tensor)
          let new_angular = vec3_add(angular, vec3_scale(angular_accel, dt))
          let _ = jolt.set_angular_velocity(world, body, new_angular)

          Some(state)
        }
        Error(_) -> None
      }
    }
    Error(_) -> None
  }
}

// =============================================================================
// VECTOR HELPERS
// =============================================================================

fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
  let Vec3(ax, ay, az) = a
  let Vec3(bx, by, bz) = b
  Vec3(ax +. bx, ay +. by, az +. bz)
}

fn vec3_scale(v: Vec3, s: Float) -> Vec3 {
  let Vec3(x, y, z) = v
  Vec3(x *. s, y *. s, z *. s)
}

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
