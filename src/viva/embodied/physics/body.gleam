//// MemoryBody - A body in 4D semantic space

import viva/embodied/physics/vec4.{type Vec4}
import viva_glyph/glyph.{type Glyph}

/// Motion type
pub type MotionType {
  /// Long-term memory, immovable (archetypes)
  Static
  /// Active thought, moves toward attractors
  Dynamic
  /// Externally controlled (current context)
  Kinematic
}

/// Sleep state
pub type SleepState {
  Awake
  Sleeping
}

/// A memory body in semantic space
pub type MemoryBody {
  MemoryBody(
    /// Unique ID
    id: Int,
    /// Position in 4D (PAD + Intensity)
    position: Vec4,
    /// Velocity (change per step)
    velocity: Vec4,
    /// Content as Glyph
    glyph: Glyph,
    /// Motion type
    motion_type: MotionType,
    /// Sleep state
    sleep_state: SleepState,
    /// Island (cluster) ID
    island_id: Int,
    /// Inactivity counter
    inactive_ticks: Int,
    /// Weight/importance (karma)
    mass: Float,
    /// Creation tick
    created_at: Int,
    /// Last access tick
    last_accessed: Int,
  )
}

/// Create dynamic body
pub fn new_dynamic(
  id: Int,
  position: Vec4,
  glyph: Glyph,
  mass: Float,
  tick: Int,
) -> MemoryBody {
  MemoryBody(
    id: id,
    position: position,
    velocity: vec4.zero,
    glyph: glyph,
    motion_type: Dynamic,
    sleep_state: Awake,
    island_id: -1,
    inactive_ticks: 0,
    mass: mass,
    created_at: tick,
    last_accessed: tick,
  )
}

/// Create static body (archetype)
pub fn new_static(
  id: Int,
  position: Vec4,
  glyph: Glyph,
  tick: Int,
) -> MemoryBody {
  MemoryBody(
    id: id,
    position: position,
    velocity: vec4.zero,
    glyph: glyph,
    motion_type: Static,
    sleep_state: Awake,
    island_id: -1,
    inactive_ticks: 0,
    mass: 1000.0,
    created_at: tick,
    last_accessed: tick,
  )
}

/// Check if sleeping
pub fn is_sleeping(body: MemoryBody) -> Bool {
  body.sleep_state == Sleeping
}

/// Check if can move
pub fn is_dynamic(body: MemoryBody) -> Bool {
  body.motion_type == Dynamic
}

/// Wake up body
pub fn wake_up(body: MemoryBody, tick: Int) -> MemoryBody {
  MemoryBody(..body, sleep_state: Awake, inactive_ticks: 0, last_accessed: tick)
}

/// Put to sleep
pub fn put_to_sleep(body: MemoryBody) -> MemoryBody {
  MemoryBody(..body, sleep_state: Sleeping, velocity: vec4.zero)
}

/// Apply velocity and update position
pub fn integrate(body: MemoryBody, dt: Float) -> MemoryBody {
  case body.motion_type {
    Static -> body
    _ -> {
      let new_pos = vec4.add(body.position, vec4.scale(body.velocity, dt))
      let clamped = vec4.clamp_pad(new_pos)
      MemoryBody(..body, position: clamped)
    }
  }
}

/// Apply force (acceleration based on mass)
pub fn apply_force(body: MemoryBody, force: Vec4, dt: Float) -> MemoryBody {
  case body.motion_type {
    Static -> body
    _ -> {
      let acceleration = vec4.scale(force, 1.0 /. body.mass)
      let new_velocity = vec4.add(body.velocity, vec4.scale(acceleration, dt))
      MemoryBody(..body, velocity: new_velocity)
    }
  }
}

/// Apply damping to velocity
pub fn apply_damping(body: MemoryBody, damping: Float) -> MemoryBody {
  MemoryBody(..body, velocity: vec4.scale(body.velocity, damping))
}

/// Increment inactive counter
pub fn tick_inactive(body: MemoryBody) -> MemoryBody {
  let is_moving = vec4.length_sq(body.velocity) >. 0.0001
  case is_moving {
    True -> MemoryBody(..body, inactive_ticks: 0)
    False -> MemoryBody(..body, inactive_ticks: body.inactive_ticks + 1)
  }
}
