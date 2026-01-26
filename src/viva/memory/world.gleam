//// World - Semantic space simulation
////
//// A force-directed graph where memories interact.
//// - Attractors pull similar memories together
//// - Repulsion prevents collapse
//// - Islands group connected memories
//// - Sleep mechanism for inactive memories

import gleam/dict.{type Dict}
import gleam/float
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/memory/body.{type Body}
import viva/memory/hrr.{type HRR}
import viva_tensor/tensor.{type Tensor}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// The semantic world containing all memory bodies
pub type World {
  World(
    /// All bodies indexed by ID
    bodies: Dict(Int, Body),
    /// Next available ID
    next_id: Int,
    /// Islands (groups of connected bodies)
    islands: List(Island),
    /// Current context attractor
    attractor: Option(Tensor),
    /// World configuration
    config: WorldConfig,
    /// Simulation tick count
    tick: Int,
  )
}

/// Island: group of connected, active memories
pub type Island {
  Island(id: Int, body_ids: List(Int), sleeping: Bool)
}

/// World configuration
pub type WorldConfig {
  WorldConfig(
    /// Spatial dimensions (e.g., 4 for PAD+I)
    spatial_dims: Int,
    /// HRR dimensions
    hrr_dims: Int,
    /// Attraction strength to context
    attraction_strength: Float,
    /// Repulsion strength between bodies
    repulsion_strength: Float,
    /// Energy decay per tick (0.95 = 5% decay)
    energy_decay: Float,
    /// Distance threshold for island formation
    island_threshold: Float,
    /// Velocity damping (0.9 = 10% damping)
    damping: Float,
    /// Maximum velocity
    max_velocity: Float,
  )
}

/// Query result
pub type QueryResult {
  QueryResult(body: Body, distance: Float, similarity: Float, score: Float)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create a new world with default config
pub fn new() -> World {
  new_with_config(default_config())
}

/// Create world with custom config
pub fn new_with_config(config: WorldConfig) -> World {
  World(
    bodies: dict.new(),
    next_id: 1,
    islands: [],
    attractor: None,
    config: config,
    tick: 0,
  )
}

/// Default world configuration
pub fn default_config() -> WorldConfig {
  WorldConfig(
    spatial_dims: 4,
    hrr_dims: 256,
    attraction_strength: 0.5,
    repulsion_strength: 0.1,
    energy_decay: 0.98,
    island_threshold: 5.0,
    damping: 0.95,
    max_velocity: 10.0,
  )
}

/// VIVA-optimized config (emotional space)
pub fn viva_config() -> WorldConfig {
  WorldConfig(
    spatial_dims: 4,
    // Pleasure, Arousal, Dominance, Intensity
    hrr_dims: 512,
    // Rich semantic content
    attraction_strength: 0.3,
    repulsion_strength: 0.05,
    energy_decay: 0.99,
    // Slower decay for longer retention
    island_threshold: 3.0,
    damping: 0.9,
    max_velocity: 5.0,
  )
}

// =============================================================================
// BODY MANAGEMENT
// =============================================================================

/// Add a memory to the world
pub fn add_memory(
  world: World,
  shape: HRR,
  position: Tensor,
  label: String,
) -> #(World, Int) {
  let id = world.next_id
  let new_body = body.dynamic(id, position, shape, label)

  let new_world =
    World(
      ..world,
      bodies: dict.insert(world.bodies, id, new_body),
      next_id: id + 1,
    )

  #(new_world, id)
}

/// Add memory at random position
pub fn add_memory_random(
  world: World,
  shape: HRR,
  label: String,
) -> #(World, Int) {
  let position = random_position(world.config.spatial_dims)
  add_memory(world, shape, position, label)
}

/// Add a static archetype
pub fn add_archetype(
  world: World,
  shape: HRR,
  position: Tensor,
  label: String,
) -> #(World, Int) {
  let id = world.next_id
  let new_body = body.static_body(id, position, shape, label)

  let new_world =
    World(
      ..world,
      bodies: dict.insert(world.bodies, id, new_body),
      next_id: id + 1,
    )

  #(new_world, id)
}

/// Remove a memory
pub fn remove_memory(world: World, id: Int) -> World {
  World(..world, bodies: dict.delete(world.bodies, id))
}

/// Get a body by ID
pub fn get_body(world: World, id: Int) -> Option(Body) {
  dict.get(world.bodies, id)
  |> option.from_result
}

/// Touch a body (access it, boost energy)
pub fn touch_body(world: World, id: Int) -> World {
  case dict.get(world.bodies, id) {
    Ok(b) ->
      World(..world, bodies: dict.insert(world.bodies, id, body.touch(b)))
    Error(_) -> world
  }
}

// =============================================================================
// CONTEXT / ATTRACTOR
// =============================================================================

/// Set the current context attractor
/// Memories similar to this will be "pulled" toward it
pub fn set_context(world: World, context: Tensor) -> World {
  World(..world, attractor: Some(context))
}

/// Clear context attractor
pub fn clear_context(world: World) -> World {
  World(..world, attractor: None)
}

/// Set context from HRR (uses its position if encoded)
pub fn set_context_hrr(world: World, context_hrr: HRR) -> World {
  // Project HRR to spatial dims (simple: take first N dims)
  let spatial = project_to_spatial(context_hrr, world.config.spatial_dims)
  set_context(world, spatial)
}

// =============================================================================
// SIMULATION STEP
// =============================================================================

/// Advance simulation by one tick
pub fn step(world: World) -> World {
  let dt = 0.016
  // ~60fps timestep

  world
  |> step_forces(dt)
  |> step_integration(dt)
  |> step_energy_decay()
  |> step_update_islands()
  |> increment_tick()
}

/// Step with custom delta time
pub fn step_dt(world: World, dt: Float) -> World {
  world
  |> step_forces(dt)
  |> step_integration(dt)
  |> step_energy_decay()
  |> step_update_islands()
  |> increment_tick()
}

/// Apply forces to all dynamic bodies
fn step_forces(world: World, dt: Float) -> World {
  let bodies_list = dict.to_list(world.bodies)
  let active_bodies =
    list.filter(bodies_list, fn(pair) {
      let #(_, b) = pair
      body.is_dynamic(b) && !body.is_sleeping(b)
    })

  let updated_bodies =
    list.fold(active_bodies, world.bodies, fn(acc, pair) {
      let #(id, b) = pair

      // Attraction to context
      let b_with_attraction = case world.attractor {
        Some(attractor) -> {
          let force =
            body.attraction_to(b, attractor, world.config.attraction_strength)
          body.apply_force(b, force, dt)
        }
        None -> b
      }

      // Repulsion from other bodies
      let b_with_repulsion =
        list.fold(active_bodies, b_with_attraction, fn(current, other_pair) {
          let #(other_id, other) = other_pair
          case id == other_id {
            True -> current
            False -> {
              let force =
                body.repulsion_from(
                  current,
                  other,
                  world.config.repulsion_strength,
                )
              body.apply_force(current, force, dt)
            }
          }
        })

      dict.insert(acc, id, b_with_repulsion)
    })

  World(..world, bodies: updated_bodies)
}

/// Integrate velocities and apply damping
fn step_integration(world: World, dt: Float) -> World {
  let updated =
    dict.map_values(world.bodies, fn(_, b) {
      case body.is_dynamic(b) && !body.is_sleeping(b) {
        True -> {
          let integrated = body.integrate(b, dt)
          // Apply damping
          let damped_vel =
            tensor.scale(integrated.velocity, world.config.damping)
          // Clamp velocity
          let clamped_vel =
            clamp_velocity(damped_vel, world.config.max_velocity)
          body.set_velocity(integrated, clamped_vel)
        }
        False -> b
      }
    })

  World(..world, bodies: updated)
}

/// Decay energy for all bodies
fn step_energy_decay(world: World) -> World {
  let updated =
    dict.map_values(world.bodies, fn(_, b) {
      case body.is_dynamic(b) {
        True -> body.decay_energy(b, world.config.energy_decay)
        False -> b
      }
    })

  World(..world, bodies: updated)
}

/// Update islands based on proximity
fn step_update_islands(world: World) -> World {
  // Simple island detection: group bodies within threshold distance
  let active =
    dict.to_list(world.bodies)
    |> list.filter(fn(pair) {
      let #(_, b) = pair
      !body.is_sleeping(b)
    })

  let islands = build_islands(active, world.config.island_threshold)

  // Assign island IDs to bodies
  let updated_bodies =
    list.fold(islands, world.bodies, fn(acc, island) {
      list.fold(island.body_ids, acc, fn(inner_acc, body_id) {
        case dict.get(inner_acc, body_id) {
          Ok(b) ->
            dict.insert(inner_acc, body_id, body.assign_island(b, island.id))
          Error(_) -> inner_acc
        }
      })
    })

  World(..world, bodies: updated_bodies, islands: islands)
}

fn increment_tick(world: World) -> World {
  World(..world, tick: world.tick + 1)
}

// =============================================================================
// QUERYING
// =============================================================================

/// Query memories by HRR similarity
pub fn query(world: World, cue: HRR, limit: Int) -> List(QueryResult) {
  dict.to_list(world.bodies)
  |> list.map(fn(pair) {
    let #(_, b) = pair
    let sim = hrr.similarity(cue, b.shape)
    let dist = case world.attractor {
      Some(att) -> body_distance_to_point(b, att)
      None -> 0.0
    }
    // Score combines similarity and proximity
    let score = sim -. { dist *. 0.1 }
    QueryResult(body: b, distance: dist, similarity: sim, score: score)
  })
  |> list.sort(fn(a, b) { float.compare(b.score, a.score) })
  |> list.take(limit)
}

/// Query memories near a position
pub fn query_near(world: World, position: Tensor, radius: Float) -> List(Body) {
  dict.to_list(world.bodies)
  |> list.filter_map(fn(pair) {
    let #(_, b) = pair
    let dist = body_distance_to_point(b, position)
    case dist <. radius {
      True -> Ok(b)
      False -> Error(Nil)
    }
  })
}

/// Query awake memories only
pub fn query_awake(world: World, cue: HRR, limit: Int) -> List(QueryResult) {
  dict.to_list(world.bodies)
  |> list.filter(fn(pair) {
    let #(_, b) = pair
    !body.is_sleeping(b)
  })
  |> list.map(fn(pair) {
    let #(_, b) = pair
    let sim = hrr.similarity(cue, b.shape)
    QueryResult(body: b, distance: 0.0, similarity: sim, score: sim)
  })
  |> list.sort(fn(a, b) { float.compare(b.score, a.score) })
  |> list.take(limit)
}

/// Wake all memories similar to cue
pub fn wake_similar(world: World, cue: HRR, threshold: Float) -> World {
  let updated =
    dict.map_values(world.bodies, fn(_, b) {
      case hrr.similarity(cue, b.shape) >. threshold {
        True -> body.wake(b)
        False -> b
      }
    })

  World(..world, bodies: updated)
}

// =============================================================================
// STATISTICS
// =============================================================================

/// Count total bodies
pub fn body_count(world: World) -> Int {
  dict.size(world.bodies)
}

/// Count awake bodies
pub fn awake_count(world: World) -> Int {
  dict.to_list(world.bodies)
  |> list.filter(fn(pair) {
    let #(_, b) = pair
    !body.is_sleeping(b)
  })
  |> list.length
}

/// Count sleeping bodies
pub fn sleeping_count(world: World) -> Int {
  body_count(world) - awake_count(world)
}

/// Count islands
pub fn island_count(world: World) -> Int {
  list.length(world.islands)
}

/// Get simulation tick
pub fn current_tick(world: World) -> Int {
  world.tick
}

// =============================================================================
// HELPERS
// =============================================================================

fn random_position(dims: Int) -> Tensor {
  let data =
    list.range(1, dims)
    |> list.map(fn(_) { { random_float() -. 0.5 } *. 10.0 })
  tensor.Tensor(data: data, shape: [dims])
}

fn project_to_spatial(h: HRR, dims: Int) -> Tensor {
  let data = list.take(td(h.vector), dims)
  let padded = case list.length(data) < dims {
    True -> list.append(data, list.repeat(0.0, dims - list.length(data)))
    False -> data
  }
  tensor.Tensor(data: padded, shape: [dims])
}

fn body_distance_to_point(b: Body, point: Tensor) -> Float {
  case tensor.sub(b.position, point) {
    Ok(diff) -> {
      let sq_sum = list.fold(td(diff), 0.0, fn(acc, x) { acc +. x *. x })
      float_sqrt(sq_sum)
    }
    Error(_) -> 9999.0
  }
}

fn clamp_velocity(vel: Tensor, max: Float) -> Tensor {
  let mag_sq = list.fold(td(vel), 0.0, fn(acc, x) { acc +. x *. x })
  let mag = float_sqrt(mag_sq)
  case mag >. max {
    True -> tensor.scale(vel, max /. mag)
    False -> vel
  }
}

fn build_islands(bodies: List(#(Int, Body)), threshold: Float) -> List(Island) {
  // Simple O(n²) island builder - could optimize with spatial hash
  case bodies {
    [] -> []
    _ -> {
      let #(islands, _) =
        list.fold(bodies, #([], 0), fn(state, pair) {
          let #(islands, next_island_id) = state
          let #(id, _b) = pair

          // Check if already in an island
          let already_assigned =
            list.any(islands, fn(island: Island) {
              list.contains(island.body_ids, id)
            })

          case already_assigned {
            True -> state
            False -> {
              // Find all bodies within threshold
              let nearby_ids =
                list.filter_map(bodies, fn(other_pair) {
                  let #(other_id, other_body) = other_pair
                  let #(_, this_body) = pair
                  case body.distance(this_body, other_body) <. threshold {
                    True -> Ok(other_id)
                    False -> Error(Nil)
                  }
                })

              let new_island =
                Island(
                  id: next_island_id,
                  body_ids: nearby_ids,
                  sleeping: False,
                )

              #([new_island, ..islands], next_island_id + 1)
            }
          }
        })

      islands
    }
  }
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "rand", "uniform")
fn random_float() -> Float

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
