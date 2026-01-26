//// Body - Memory bodies in semantic space
////
//// Each memory is a "rigid body" with position, velocity, and shape (HRR).
//// Dynamic bodies move and interact. Static bodies are immutable landmarks.
////
//// Inspired by Jolt Physics concepts, but for semantic space.

import gleam/list
import viva/memory/hrr.{type HRR}
import viva/neural/tensor.{type Tensor}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// Motion type determines how a body behaves in simulation
pub type MotionType {
  /// Dynamic bodies move, interact, and can sleep
  Dynamic
  /// Static bodies are immutable (long-term memories, archetypes)
  Static
  /// Kinematic bodies move but aren't affected by forces
  Kinematic
}

/// Memory body in semantic space
pub type Body {
  Body(
    /// Unique identifier
    id: Int,
    /// Position in semantic space (PAD+Intensity or custom dims)
    position: Tensor,
    /// Velocity (rate of change in semantic space)
    velocity: Tensor,
    /// The HRR shape (holographic content)
    shape: HRR,
    /// Motion type
    motion_type: MotionType,
    /// Is this body currently sleeping (inactive)?
    sleeping: Bool,
    /// Island ID (-1 if not assigned)
    island_id: Int,
    /// Activation energy (decays over time)
    energy: Float,
    /// Creation timestamp
    created_at: Int,
    /// Last access timestamp
    last_accessed: Int,
    /// Optional label/tag
    label: String,
  )
}

/// Body configuration for creation
pub type BodyConfig {
  BodyConfig(
    position: Tensor,
    shape: HRR,
    motion_type: MotionType,
    label: String,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create a new dynamic memory body
pub fn new(id: Int, config: BodyConfig) -> Body {
  let dim = tensor.size(config.position)
  let now = erlang_system_time()

  Body(
    id: id,
    position: config.position,
    velocity: tensor.zeros([dim]),
    shape: config.shape,
    motion_type: config.motion_type,
    sleeping: False,
    island_id: -1,
    energy: 1.0,
    created_at: now,
    last_accessed: now,
    label: config.label,
  )
}

/// Create a dynamic body (default)
pub fn dynamic(id: Int, position: Tensor, shape: HRR, label: String) -> Body {
  new(
    id,
    BodyConfig(
      position: position,
      shape: shape,
      motion_type: Dynamic,
      label: label,
    ),
  )
}

/// Create a static body (archetype/landmark)
pub fn static_body(id: Int, position: Tensor, shape: HRR, label: String) -> Body {
  new(
    id,
    BodyConfig(
      position: position,
      shape: shape,
      motion_type: Static,
      label: label,
    ),
  )
}

/// Create body at origin with random HRR
pub fn at_origin(id: Int, dim: Int, hrr_dim: Int, label: String) -> Body {
  dynamic(id, tensor.zeros([dim]), hrr.random(hrr_dim), label)
}

// =============================================================================
// ACCESSORS
// =============================================================================

/// Get body ID
pub fn id(body: Body) -> Int {
  body.id
}

/// Get body position
pub fn position(body: Body) -> Tensor {
  body.position
}

/// Get body shape (HRR)
pub fn shape(body: Body) -> HRR {
  body.shape
}

/// Check if body is sleeping
pub fn is_sleeping(body: Body) -> Bool {
  body.sleeping
}

/// Check if body is dynamic
pub fn is_dynamic(body: Body) -> Bool {
  case body.motion_type {
    Dynamic -> True
    _ -> False
  }
}

/// Check if body is static
pub fn is_static(body: Body) -> Bool {
  case body.motion_type {
    Static -> True
    _ -> False
  }
}

// =============================================================================
// MUTATIONS
// =============================================================================

/// Update body position
pub fn set_position(body: Body, pos: Tensor) -> Body {
  Body(..body, position: pos, last_accessed: erlang_system_time())
}

/// Update body velocity
pub fn set_velocity(body: Body, vel: Tensor) -> Body {
  Body(..body, velocity: vel)
}

/// Apply force to body (adds to velocity)
pub fn apply_force(body: Body, force: Tensor, dt: Float) -> Body {
  case body.motion_type {
    Static -> body
    _ -> {
      let scaled_force = tensor.scale(force, dt)
      case tensor.add(body.velocity, scaled_force) {
        Ok(new_vel) -> Body(..body, velocity: new_vel)
        Error(_) -> body
      }
    }
  }
}

/// Integrate position based on velocity
pub fn integrate(body: Body, dt: Float) -> Body {
  case body.motion_type {
    Static -> body
    _ -> {
      let displacement = tensor.scale(body.velocity, dt)
      case tensor.add(body.position, displacement) {
        Ok(new_pos) -> Body(..body, position: new_pos)
        Error(_) -> body
      }
    }
  }
}

/// Put body to sleep
pub fn sleep(body: Body) -> Body {
  Body(..body, sleeping: True, velocity: tensor.zeros(body.position.shape))
}

/// Wake body up
pub fn wake(body: Body) -> Body {
  Body(
    ..body,
    sleeping: False,
    energy: 1.0,
    last_accessed: erlang_system_time(),
  )
}

/// Decay energy over time
pub fn decay_energy(body: Body, decay_rate: Float) -> Body {
  let new_energy = body.energy *. decay_rate
  case new_energy <. 0.01 {
    True -> sleep(body)
    False -> Body(..body, energy: new_energy)
  }
}

/// Touch body (reset last_accessed, boost energy)
pub fn touch(body: Body) -> Body {
  Body(
    ..body,
    last_accessed: erlang_system_time(),
    energy: float_min(body.energy +. 0.5, 1.0),
    sleeping: False,
  )
}

/// Assign to island
pub fn assign_island(body: Body, island_id: Int) -> Body {
  Body(..body, island_id: island_id)
}

// =============================================================================
// PHYSICS HELPERS
// =============================================================================

/// Calculate distance between two bodies (Euclidean in position space)
pub fn distance(a: Body, b: Body) -> Float {
  case tensor.sub(a.position, b.position) {
    Ok(diff) -> {
      let squared = list.map(td(diff), fn(x) { x *. x })
      let sum = list.fold(squared, 0.0, fn(acc, x) { acc +. x })
      float_sqrt(sum)
    }
    Error(_) -> 9999.0
  }
}

/// Calculate semantic similarity between two bodies (via HRR)
pub fn semantic_similarity(a: Body, b: Body) -> Float {
  hrr.similarity(a.shape, b.shape)
}

/// Calculate attraction force between body and attractor point
/// Force = direction * (similarity / distance²)
pub fn attraction_to(body: Body, attractor: Tensor, strength: Float) -> Tensor {
  case tensor.sub(attractor, body.position) {
    Ok(direction) -> {
      let dist_sq = list.fold(td(direction), 0.0, fn(acc, x) { acc +. x *. x })
      case dist_sq >. 0.0001 {
        True -> {
          let dist = float_sqrt(dist_sq)
          let force_mag = strength /. dist_sq
          tensor.scale(direction, force_mag /. dist)
        }
        False -> tensor.zeros(direction.shape)
      }
    }
    Error(_) -> tensor.zeros(body.position.shape)
  }
}

/// Calculate repulsion force from another body
/// Prevents collapse of similar memories
pub fn repulsion_from(body: Body, other: Body, strength: Float) -> Tensor {
  case tensor.sub(body.position, other.position) {
    Ok(direction) -> {
      let dist_sq = list.fold(td(direction), 0.0, fn(acc, x) { acc +. x *. x })
      case dist_sq >. 0.0001 && dist_sq <. 100.0 {
        True -> {
          let dist = float_sqrt(dist_sq)
          let force_mag = strength /. dist_sq
          tensor.scale(direction, force_mag /. dist)
        }
        False -> tensor.zeros(direction.shape)
      }
    }
    Error(_) -> tensor.zeros(body.position.shape)
  }
}

// =============================================================================
// SERIALIZATION
// =============================================================================

/// Convert body to simple representation for storage
pub fn to_tuple(body: Body) -> #(Int, List(Float), List(Float), String) {
  #(body.id, td(body.position), td(body.shape.vector), body.label)
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "erlang", "system_time")
fn erlang_system_time() -> Int

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

fn float_min(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}
