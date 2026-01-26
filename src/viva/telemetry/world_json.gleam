//// World JSON Serialization
////
//// Converts World state to JSON for real-time telemetry.
//// Optimized for WebSocket streaming to visualization clients.

import gleam/dict
import gleam/json.{type Json}
import gleam/list
import gleam/option.{None, Some}
import viva/memory/body.{type Body}
import viva/memory/world.{type Island, type World, type WorldConfig}
import viva/neural/tensor.{type Tensor}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// FULL WORLD SERIALIZATION
// =============================================================================

/// Serialize entire World state to JSON
pub fn world_to_json(w: World) -> Json {
  json.object([
    #("tick", json.int(w.tick)),
    #("body_count", json.int(dict.size(w.bodies))),
    #("island_count", json.int(list.length(w.islands))),
    #("bodies", bodies_to_json(w)),
    #("islands", islands_to_json(w.islands)),
    #("attractor", attractor_to_json(w)),
    #("config", config_to_json(w.config)),
  ])
}

/// Serialize World to JSON string
pub fn world_to_string(w: World) -> String {
  world_to_json(w) |> json.to_string
}

// =============================================================================
// LIGHTWEIGHT FRAME (for 60fps streaming)
// =============================================================================

/// Lightweight frame with only positions and energies
/// Ideal for high-frequency updates
pub fn frame_to_json(w: World) -> Json {
  let bodies =
    dict.to_list(w.bodies)
    |> list.map(fn(pair) {
      let #(id, b) = pair
      json.object([
        #("id", json.int(id)),
        #("pos", floats_to_json(td(b.position))),
        #("e", json.float(b.energy)),
        #("s", json.bool(b.sleeping)),
      ])
    })

  json.object([
    #("t", json.int(w.tick)),
    #("b", json.preprocessed_array(bodies)),
  ])
}

/// Frame to JSON string
pub fn frame_to_string(w: World) -> String {
  frame_to_json(w) |> json.to_string
}

// =============================================================================
// DELTA UPDATES (only changed bodies)
// =============================================================================

/// Delta update with specific body IDs
pub fn delta_to_json(w: World, changed_ids: List(Int)) -> Json {
  let changed_bodies =
    list.filter_map(changed_ids, fn(id) {
      case dict.get(w.bodies, id) {
        Ok(b) -> Ok(body_to_json(b))
        Error(_) -> Error(Nil)
      }
    })

  json.object([
    #("t", json.int(w.tick)),
    #("delta", json.preprocessed_array(changed_bodies)),
  ])
}

// =============================================================================
// BODY SERIALIZATION
// =============================================================================

/// Full body serialization
pub fn body_to_json(b: Body) -> Json {
  json.object([
    #("id", json.int(b.id)),
    #("label", json.string(b.label)),
    #("position", floats_to_json(td(b.position))),
    #("velocity", floats_to_json(td(b.velocity))),
    #("energy", json.float(b.energy)),
    #("sleeping", json.bool(b.sleeping)),
    #("island_id", json.int(b.island_id)),
    #("motion_type", motion_type_to_json(b.motion_type)),
    #("hrr_norm", json.float(hrr_norm(td(b.shape.vector)))),
  ])
}

fn motion_type_to_json(mt: body.MotionType) -> Json {
  case mt {
    body.Dynamic -> json.string("dynamic")
    body.Static -> json.string("static")
    body.Kinematic -> json.string("kinematic")
  }
}

fn bodies_to_json(w: World) -> Json {
  dict.to_list(w.bodies)
  |> list.map(fn(pair) {
    let #(_, b) = pair
    body_to_json(b)
  })
  |> json.preprocessed_array
}

// =============================================================================
// ISLAND SERIALIZATION
// =============================================================================

fn island_to_json(island: Island) -> Json {
  json.object([
    #("id", json.int(island.id)),
    #("body_ids", json.array(island.body_ids, json.int)),
    #("sleeping", json.bool(island.sleeping)),
    #("size", json.int(list.length(island.body_ids))),
  ])
}

fn islands_to_json(islands: List(Island)) -> Json {
  islands
  |> list.map(island_to_json)
  |> json.preprocessed_array
}

// =============================================================================
// ATTRACTOR / CONFIG
// =============================================================================

fn attractor_to_json(w: World) -> Json {
  case w.attractor {
    Some(att) -> floats_to_json(td(att))
    None -> json.null()
  }
}

fn config_to_json(c: WorldConfig) -> Json {
  json.object([
    #("spatial_dims", json.int(c.spatial_dims)),
    #("hrr_dims", json.int(c.hrr_dims)),
    #("attraction_strength", json.float(c.attraction_strength)),
    #("repulsion_strength", json.float(c.repulsion_strength)),
    #("energy_decay", json.float(c.energy_decay)),
    #("island_threshold", json.float(c.island_threshold)),
    #("damping", json.float(c.damping)),
    #("max_velocity", json.float(c.max_velocity)),
  ])
}

// =============================================================================
// HELPERS
// =============================================================================

fn floats_to_json(data: List(Float)) -> Json {
  json.array(data, json.float)
}

fn hrr_norm(data: List(Float)) -> Float {
  let sum = list.fold(data, 0.0, fn(acc, x) { acc +. x *. x })
  float_sqrt(sum)
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
