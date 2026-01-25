//// Telemetry Demo
////
//// Example usage of the telemetry system.
//// Run with: gleam run -m viva/telemetry/demo

import gleam/dict
import gleam/erlang/process
import gleam/int
import gleam/io
import gleam/list
import viva/memory/hrr
import viva/memory/world
import viva/neural/tensor
import viva/telemetry/server

/// Demo entry point
pub fn main() {
  io.println("=== VIVA Memory World Telemetry Demo ===")
  io.println("")

  // Start broadcaster
  let assert Ok(broadcaster) = server.start_broadcaster()
  io.println("[OK] Broadcaster started")

  // Start HTTP server
  let port = 8888
  let assert Ok(_) = server.start(port, broadcaster)
  io.println(
    "[OK] Telemetry server running on http://localhost:" <> int.to_string(port),
  )
  io.println("")
  io.println(
    "Open http://localhost:" <> int.to_string(port) <> " in your browser",
  )
  io.println("to see the Three.js + D3.js visualization.")
  io.println("")

  // Create a demo world
  let w = create_demo_world()
  io.println(
    "[OK] Created demo world with "
    <> int.to_string(world.body_count(w))
    <> " bodies",
  )

  // Broadcast initial state
  server.broadcast_world(broadcaster, w)
  io.println("[OK] Initial state broadcasted")
  io.println("")

  // Run simulation loop
  io.println("Starting simulation loop (Ctrl+C to stop)...")
  simulation_loop(w, broadcaster, 0)
}

/// Create a demo world with some bodies
fn create_demo_world() -> world.World {
  let config = world.viva_config()
  let w = world.new_with_config(config)

  // Add some dynamic memories
  let positions = [
    #(tensor.Tensor(data: [2.0, 3.0, 1.0, 0.5], shape: [4]), "joy"),
    #(tensor.Tensor(data: [-2.0, 2.0, -1.0, 0.3], shape: [4]), "sadness"),
    #(tensor.Tensor(data: [1.0, -2.0, 2.0, 0.8], shape: [4]), "excitement"),
    #(tensor.Tensor(data: [-1.0, -1.0, -2.0, 0.4], shape: [4]), "calm"),
    #(tensor.Tensor(data: [3.0, 0.0, 0.0, 0.6], shape: [4]), "surprise"),
    #(tensor.Tensor(data: [0.0, 4.0, 1.0, 0.7], shape: [4]), "anticipation"),
    #(tensor.Tensor(data: [-3.0, 1.0, 2.0, 0.2], shape: [4]), "trust"),
    #(tensor.Tensor(data: [1.5, 1.5, -1.5, 0.5], shape: [4]), "curiosity"),
  ]

  list.fold(positions, w, fn(world_acc, pair) {
    let #(pos, label) = pair
    let shape = hrr.random(config.hrr_dims)
    let #(new_world, _id) = world.add_memory(world_acc, shape, pos, label)
    new_world
  })
}

/// Simulation loop - steps the world and broadcasts state
fn simulation_loop(w: world.World, broadcaster: server.Broadcaster, step: Int) {
  // Step the simulation
  let stepped = world.step(w)

  // Inject chaos every 10 steps - keep things alive!
  let new_world = case step % 10 {
    0 -> inject_chaos(stepped, step)
    _ -> stepped
  }

  // Broadcast every tick
  server.broadcast_world(broadcaster, new_world)

  // Log progress every 100 steps
  case step % 100 {
    0 -> {
      let tick = world.current_tick(new_world)
      let awake = world.awake_count(new_world)
      let islands = world.island_count(new_world)
      io.println(
        "Tick: "
        <> int.to_string(tick)
        <> " | Awake: "
        <> int.to_string(awake)
        <> " | Islands: "
        <> int.to_string(islands),
      )
    }
    _ -> Nil
  }

  // Sleep ~16ms (60fps)
  process.sleep(16)

  // Continue loop
  simulation_loop(new_world, broadcaster, step + 1)
}

/// Inject chaos into the world - stimulate random bodies
fn inject_chaos(w: world.World, step: Int) -> world.World {
  // Pick a "random" body based on step
  let body_ids = dict.keys(w.bodies) |> list.sort(int.compare)
  let n = list.length(body_ids)
  case n > 0 {
    False -> w
    True -> {
      let target_idx = step % n
      case list.drop(body_ids, target_idx) {
        [id, ..] -> {
          // Touch the body to boost its energy
          let touched = world.touch_body(w, id)

          // Also wake similar bodies for chain reaction
          case dict.get(touched.bodies, id) {
            Ok(b) -> world.wake_similar(touched, b.shape, 0.3)
            Error(_) -> touched
          }
        }
        [] -> w
      }
    }
  }
}
