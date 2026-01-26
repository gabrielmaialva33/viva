//// Telemetry Server
////
//// Real-time HTTP/JSON server for World visualization.
//// Uses polling (REST API) for reliability across mist versions.
////
//// Endpoints:
////   /           - Embedded frontend (Three.js + D3.js)
////   /api/health - Health check
////   /api/world  - Current world state JSON
////   /api/metrics - Aggregated metrics JSON
////   /api/graph  - D3.js force graph JSON

import gleam/bool
import gleam/bytes_tree
import gleam/dict
import gleam/erlang/process.{type Subject}
import gleam/float
import gleam/http/request.{type Request}
import gleam/http/response.{type Response}
import gleam/int
import gleam/json
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import gleam/set.{type Set}
import mist.{type Connection, type ResponseData}
import viva/memory/world.{type World}
import viva/neural/tensor.{type Tensor}
import viva/telemetry/frontend
import viva/telemetry/metrics
import viva/telemetry/perf
import viva/telemetry/system
import viva/telemetry/world_json

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

pub type TelemetryMessage {
  Subscribe(Subject(String))
  Unsubscribe(Subject(String))
  Broadcast(String)
  UpdateWorld(World)
  GetWorld(Subject(Option(World)))
}

pub type TelemetryState {
  TelemetryState(
    subscribers: Set(Subject(String)),
    current_world: Option(World),
  )
}

pub type Broadcaster =
  Subject(TelemetryMessage)

// =============================================================================
// BROADCASTER ACTOR
// =============================================================================

fn telemetry_loop(
  state: TelemetryState,
  message: TelemetryMessage,
) -> actor.Next(TelemetryState, TelemetryMessage) {
  case message {
    Subscribe(client) -> {
      actor.continue(
        TelemetryState(
          ..state,
          subscribers: set.insert(state.subscribers, client),
        ),
      )
    }

    Unsubscribe(client) -> {
      actor.continue(
        TelemetryState(
          ..state,
          subscribers: set.delete(state.subscribers, client),
        ),
      )
    }

    Broadcast(data) -> {
      let _ =
        set.fold(state.subscribers, Nil, fn(_, client) {
          process.send(client, data)
          Nil
        })
      actor.continue(state)
    }

    UpdateWorld(world) -> {
      let data = world_json.world_to_string(world)
      let _ =
        set.fold(state.subscribers, Nil, fn(_, client) {
          process.send(client, data)
          Nil
        })
      actor.continue(TelemetryState(..state, current_world: Some(world)))
    }

    GetWorld(reply_to) -> {
      process.send(reply_to, state.current_world)
      actor.continue(state)
    }
  }
}

pub fn start_broadcaster() -> Result(Broadcaster, actor.StartError) {
  let initial_state =
    TelemetryState(subscribers: set.new(), current_world: None)

  let builder =
    actor.new(initial_state)
    |> actor.on_message(telemetry_loop)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Broadcast world state update
pub fn broadcast_world(broadcaster: Broadcaster, world: World) {
  process.send(broadcaster, UpdateWorld(world))
}

/// Legacy broadcast function
pub fn broadcast(broadcaster: Broadcaster, data: String) -> Nil {
  process.send(broadcaster, Broadcast(data))
}

// =============================================================================
// HTTP HANDLERS
// =============================================================================

fn json_response(body: String) -> Response(ResponseData) {
  response.new(200)
  |> response.set_header("content-type", "application/json")
  |> response.set_header("access-control-allow-origin", "*")
  |> response.set_body(mist.Bytes(bytes_tree.from_string(body)))
}

fn html_response(body: String) -> Response(ResponseData) {
  response.new(200)
  |> response.set_header("content-type", "text/html; charset=utf-8")
  |> response.set_body(mist.Bytes(bytes_tree.from_string(body)))
}

fn not_found() -> Response(ResponseData) {
  response.new(404)
  |> response.set_body(mist.Bytes(bytes_tree.from_string("Not Found")))
}

fn csv_response(body: String) -> Response(ResponseData) {
  response.new(200)
  |> response.set_header("content-type", "text/csv; charset=utf-8")
  |> response.set_header("access-control-allow-origin", "*")
  |> response.set_header(
    "content-disposition",
    "attachment; filename=\"viva_export.csv\"",
  )
  |> response.set_body(mist.Bytes(bytes_tree.from_string(body)))
}

/// Convert World state to CSV format for R/Python analysis
fn world_to_csv(w: World) -> String {
  let header = "id,label,x,y,z,w,energy,sleeping,island_id\n"
  let rows =
    dict.fold(w.bodies, "", fn(acc, id, body) {
      let pos = td(body.position)
      let x = float.to_string(list_get_float(pos, 0))
      let y = float.to_string(list_get_float(pos, 1))
      let z = float.to_string(list_get_float(pos, 2))
      let w_coord = float.to_string(list_get_float(pos, 3))
      let energy = float.to_string(body.energy)
      let sleeping = bool.to_string(body.sleeping)
      let island = int.to_string(body.island_id)

      acc
      <> int.to_string(id)
      <> ","
      <> body.label
      <> ","
      <> x
      <> ","
      <> y
      <> ","
      <> z
      <> ","
      <> w_coord
      <> ","
      <> energy
      <> ","
      <> sleeping
      <> ","
      <> island
      <> "\n"
    })
  header <> rows
}

fn list_get_float(l: List(Float), idx: Int) -> Float {
  case list.drop(l, idx) {
    [x, ..] -> x
    [] -> 0.0
  }
}

fn get_world_sync(broadcaster: Broadcaster) -> Option(World) {
  let reply_subject = process.new_subject()
  process.send(broadcaster, GetWorld(reply_subject))
  case process.receive(reply_subject, 1000) {
    Ok(world_opt) -> world_opt
    Error(_) -> None
  }
}

// =============================================================================
// MAIN SERVER
// =============================================================================

pub fn start(port: Int, broadcaster: Broadcaster) {
  let handler = fn(req: Request(Connection)) {
    let path = request.path_segments(req)

    case path {
      // Health check
      ["api", "health"] ->
        json_response("{\"status\":\"ok\",\"service\":\"viva-telemetry\"}")

      // World state
      ["api", "world"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> json_response(world_json.world_to_string(world))
          None ->
            json_response(
              "{\"error\":\"no_world_state\",\"bodies\":[],\"islands\":[],\"tick\":0}",
            )
        }
      }

      // Metrics
      ["api", "metrics"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> {
            let m = metrics.collect(world)
            json_response(metrics.metrics_to_string(m))
          }
          None -> json_response("{\"error\":\"no_world_state\"}")
        }
      }

      // Force graph for D3.js
      ["api", "graph"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> {
            let graph = metrics.to_force_graph(world, 0.3)
            json_response(json.to_string(graph))
          }
          None -> json_response("{\"nodes\":[],\"links\":[]}")
        }
      }

      // CSV export for R/Python analysis
      ["api", "export", "csv"] -> {
        case get_world_sync(broadcaster) {
          Some(world) -> csv_response(world_to_csv(world))
          None -> csv_response("id,label,x,y,z,w,energy,sleeping,island_id\n")
        }
      }

      // System metrics (CPU, Memory, GPU)
      ["api", "system"] -> {
        let sys = system.collect()
        json_response(system.to_string(sys))
      }

      // Performance metrics (tick time, GC, memory, reductions)
      ["api", "perf"] -> {
        let p = perf.collect(0, 0.0)
        json_response(perf.to_string(p))
      }

      // Embedded frontend
      [] -> html_response(frontend.html(port))

      _ -> not_found()
    }
  }

  mist.new(handler)
  |> mist.port(port)
  |> mist.start
}
