//// PhysicsWorld - Semantic simulation space
////
//// Inspired by Jolt Physics concepts:
//// - Bodies with motion types (Static/Dynamic/Kinematic)
//// - Islands for sleep/wake clustering
//// - Broadphase queries (simplified for < 1000 bodies)
//// - Attractors instead of gravity

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/physics/body.{type MemoryBody}
import viva/physics/vec4.{type Vec4}
import viva_glyph
import viva_glyph/glyph.{type Glyph}

/// Physics configuration
pub type PhysicsConfig {
  PhysicsConfig(
    /// Emotional bias direction
    gravity: Vec4,
    /// Attraction strength (0-1)
    attraction_strength: Float,
    /// Velocity damping per step (0-1)
    damping: Float,
    /// Similarity threshold for resonance
    resonance_threshold: Float,
    /// Ticks before auto-sleep
    sleep_threshold: Int,
  )
}

/// Default config
pub fn default_config() -> PhysicsConfig {
  PhysicsConfig(
    gravity: vec4.zero,
    attraction_strength: 0.1,
    damping: 0.95,
    resonance_threshold: 0.7,
    sleep_threshold: 100,
  )
}

/// The physics world
pub type PhysicsWorld {
  PhysicsWorld(
    /// All bodies by ID
    bodies: Dict(Int, MemoryBody),
    /// Configuration
    config: PhysicsConfig,
    /// Current tick
    tick: Int,
    /// Next body ID
    next_id: Int,
    /// Current attractor (context)
    attractor: Option(Vec4),
  )
}

/// Create new world
pub fn new() -> PhysicsWorld {
  PhysicsWorld(
    bodies: dict.new(),
    config: default_config(),
    tick: 0,
    next_id: 1,
    attractor: None,
  )
}

/// Create world with config
pub fn with_config(config: PhysicsConfig) -> PhysicsWorld {
  PhysicsWorld(..new(), config: config)
}

/// Add dynamic body
pub fn add_dynamic(
  world: PhysicsWorld,
  position: Vec4,
  g: Glyph,
  mass: Float,
) -> #(PhysicsWorld, Int) {
  let id = world.next_id
  let b = body.new_dynamic(id, position, g, mass, world.tick)

  let new_world =
    PhysicsWorld(
      ..world,
      bodies: dict.insert(world.bodies, id, b),
      next_id: id + 1,
    )

  #(new_world, id)
}

/// Add static body (archetype)
pub fn add_static(
  world: PhysicsWorld,
  position: Vec4,
  g: Glyph,
) -> #(PhysicsWorld, Int) {
  let id = world.next_id
  let b = body.new_static(id, position, g, world.tick)

  let new_world =
    PhysicsWorld(
      ..world,
      bodies: dict.insert(world.bodies, id, b),
      next_id: id + 1,
    )

  #(new_world, id)
}

/// Remove body
pub fn remove(world: PhysicsWorld, id: Int) -> PhysicsWorld {
  PhysicsWorld(..world, bodies: dict.delete(world.bodies, id))
}

/// Get body by ID
pub fn get(world: PhysicsWorld, id: Int) -> Option(MemoryBody) {
  dict.get(world.bodies, id) |> option.from_result
}

/// Set attractor (context)
pub fn set_attractor(world: PhysicsWorld, position: Vec4) -> PhysicsWorld {
  PhysicsWorld(..world, attractor: Some(position))
}

/// Clear attractor
pub fn clear_attractor(world: PhysicsWorld) -> PhysicsWorld {
  PhysicsWorld(..world, attractor: None)
}

/// Step simulation
pub fn step(world: PhysicsWorld, dt: Float) -> PhysicsWorld {
  let new_tick = world.tick + 1

  // Update each body
  let new_bodies =
    world.bodies
    |> dict.map_values(fn(_id, b) { step_body(b, world, dt) })

  // Auto-sleep check
  let threshold = world.config.sleep_threshold
  let final_bodies =
    new_bodies
    |> dict.map_values(fn(_id, b) {
      case b.inactive_ticks > threshold && body.is_dynamic(b) {
        True -> body.put_to_sleep(b)
        False -> b
      }
    })

  PhysicsWorld(..world, bodies: final_bodies, tick: new_tick)
}

/// Step single body
fn step_body(b: MemoryBody, world: PhysicsWorld, dt: Float) -> MemoryBody {
  // Skip sleeping or static
  case body.is_sleeping(b) || !body.is_dynamic(b) {
    True -> b
    False -> {
      let b1 = case world.attractor {
        Some(attractor) -> {
          // Direction to attractor
          let to_attractor = vec4.sub(attractor, b.position)
          let distance = vec4.length(to_attractor)

          case distance >. 0.001 {
            True -> {
              // Attraction force (inverse distance, capped)
              let dir = vec4.normalize(to_attractor)
              let strength =
                world.config.attraction_strength /. { 1.0 +. distance }
              let force = vec4.scale(dir, strength)
              body.apply_force(b, force, dt)
            }
            False -> b
          }
        }
        None -> b
      }

      // Apply gravity (emotional bias)
      let b2 = body.apply_force(b1, world.config.gravity, dt)

      // Apply damping
      let b3 = body.apply_damping(b2, world.config.damping)

      // Integrate position
      let b4 = body.integrate(b3, dt)

      // Update inactive counter
      body.tick_inactive(b4)
    }
  }
}

/// Wake body and connected island
pub fn wake(world: PhysicsWorld, id: Int) -> PhysicsWorld {
  case dict.get(world.bodies, id) {
    Ok(b) -> {
      let woken = body.wake_up(b, world.tick)
      PhysicsWorld(..world, bodies: dict.insert(world.bodies, id, woken))
    }
    Error(_) -> world
  }
}

// =============================================================================
// QUERIES
// =============================================================================

/// Query bodies within radius (brute force, fine for < 1000)
pub fn query_radius(world: PhysicsWorld, center: Vec4, radius: Float) -> List(Int) {
  let radius_sq = radius *. radius

  world.bodies
  |> dict.to_list
  |> list.filter_map(fn(pair) {
    let #(id, b) = pair
    let dist_sq = vec4.distance_sq(center, b.position)
    case dist_sq <=. radius_sq {
      True -> Ok(id)
      False -> Error(Nil)
    }
  })
}

/// Query k nearest bodies
pub fn query_nearest(world: PhysicsWorld, center: Vec4, k: Int) -> List(#(Int, Float)) {
  world.bodies
  |> dict.to_list
  |> list.map(fn(pair) {
    let #(id, b) = pair
    let dist = vec4.distance(center, b.position)
    #(id, dist)
  })
  |> list.sort(fn(a, b) { float.compare(a.1, b.1) })
  |> list.take(k)
}

/// Query by glyph resonance (similarity)
pub fn query_resonance(
  world: PhysicsWorld,
  g: Glyph,
  min_similarity: Float,
) -> List(#(Int, Float)) {
  world.bodies
  |> dict.to_list
  |> list.filter_map(fn(pair) {
    let #(id, b) = pair
    let sim = viva_glyph.similarity(g, b.glyph)
    case sim >=. min_similarity {
      True -> Ok(#(id, sim))
      False -> Error(Nil)
    }
  })
  |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
}

/// Combined query: near in space AND resonating
pub fn query_near_and_resonant(
  world: PhysicsWorld,
  center: Vec4,
  g: Glyph,
  radius: Float,
  min_similarity: Float,
) -> List(#(Int, Float, Float)) {
  let radius_sq = radius *. radius

  world.bodies
  |> dict.to_list
  |> list.filter_map(fn(pair) {
    let #(id, b) = pair
    let dist_sq = vec4.distance_sq(center, b.position)
    let sim = viva_glyph.similarity(g, b.glyph)

    case dist_sq <=. radius_sq && sim >=. min_similarity {
      True -> Ok(#(id, float_sqrt(dist_sq), sim))
      False -> Error(Nil)
    }
  })
}

// =============================================================================
// STATS
// =============================================================================

pub type WorldStats {
  WorldStats(
    total: Int,
    awake: Int,
    sleeping: Int,
    dynamic: Int,
    static_count: Int,
    tick: Int,
  )
}

pub fn stats(world: PhysicsWorld) -> WorldStats {
  let bodies = dict.values(world.bodies)
  let total = list.length(bodies)
  let awake = list.count(bodies, fn(b) { !body.is_sleeping(b) })
  let dynamic = list.count(bodies, fn(b) { body.is_dynamic(b) })

  WorldStats(
    total: total,
    awake: awake,
    sleeping: total - awake,
    dynamic: dynamic,
    static_count: total - dynamic,
    tick: world.tick,
  )
}

pub fn stats_string(s: WorldStats) -> String {
  "PhysicsWorld: "
  <> int.to_string(s.total)
  <> " bodies ("
  <> int.to_string(s.awake)
  <> " awake, "
  <> int.to_string(s.sleeping)
  <> " sleeping), tick="
  <> int.to_string(s.tick)
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
