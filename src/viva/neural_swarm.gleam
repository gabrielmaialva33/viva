//// Neural GPU Swarm - Massive Agency
////
//// "The Hive Mind is just a matrix multiplication."
////
//// Architecture: SoA (Structure of Arrays)
//// - pos: Tensor [N, 2]
//// - vel: Tensor [N, 2]
//// - brain: Simple MLP
////
//// Logic:
//// 1. Input = Self Vel + Random Noise
//// 2. Action = Acceleration (X, Y)
//// 3. Physics = Euler Integration (Pos += Vel * dt)

import gleam/erlang/process.{type Subject}
import gleam/int
import gleam/io
import gleam/otp/actor
import viva/gpu.{type Backend}
import viva/neural/activation
import viva/neural/tensor.{type Tensor}

// =============================================================================
// CONFIG
// =============================================================================

const input_size = 2

// vx, vy

const hidden_size = 16

const output_size = 2

// ax, ay

// =============================================================================
// TYPES
// =============================================================================

pub type SwarmState {
  SwarmState(
    count: Int,
    // Physics State (SoA)
    pos: Tensor,
    // [N, 2]
    vel: Tensor,
    // [N, 2]
    // Brain (Shared Weights)
    w1: Tensor,
    // [Input, Hidden]
    b1: Tensor,
    // [Hidden]
    w2: Tensor,
    // [Hidden, Output]
    b2: Tensor,
    // [Output]
    // Meta
    tick: Int,
    backend: Backend,
  )
}

pub type Message {
  Tick(dt: Float)
  GetCount(reply: Subject(Int))
  GetState(reply: Subject(#(Tensor, Tensor)))
  // Debug
}

// =============================================================================
// API
// =============================================================================

pub fn start(count: Int) -> Result(Subject(Message), actor.StartError) {
  let builder =
    actor.new(init(count))
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

pub fn tick(swarm: Subject(Message), dt: Float) {
  process.send(swarm, Tick(dt))
}

pub fn get_count(swarm: Subject(Message)) -> Int {
  process.call(swarm, 1000, GetCount)
}

// =============================================================================
// LOGIC
// =============================================================================

fn init(count: Int) -> SwarmState {
  let backend = gpu.detect()
  io.println(
    "Initializing Neural Swarm ("
    <> int.to_string(count)
    <> " boids) on "
    <> debug_backend(backend),
  )

  // 1. Init Physics
  let pos = tensor.random_uniform([count, 2])
  // 0-1 range
  let vel = tensor.random_uniform([count, 2])
  // 0-1 range (drift)
  // Center velocities to -0.5 to 0.5
  let vel = tensor.add_scalar(vel, -0.5)

  // 2. Init Brain (Random Weights)
  let w1 = tensor.xavier_init(input_size, hidden_size)
  let b1 = tensor.zeros([hidden_size])
  let w2 = tensor.xavier_init(hidden_size, output_size)
  let b2 = tensor.zeros([output_size])

  SwarmState(
    count: count,
    pos: pos,
    vel: vel,
    w1: w1,
    b1: b1,
    w2: w2,
    b2: b2,
    tick: 0,
    backend: backend,
  )
}

fn handle_message(
  state: SwarmState,
  msg: Message,
) -> actor.Next(SwarmState, Message) {
  case msg {
    Tick(dt) -> {
      // --- STEP 1: SENSE ---
      // Input = Velocity (Simple Navigation)
      let inputs = state.vel

      // --- STEP 2: THINK ---
      // Layer 1
      let assert Ok(h1) = tensor.matmul(inputs, state.w1)
      let assert Ok(h1) = tensor.add_broadcast(h1, state.b1)
      let h1 = activation.forward(h1, activation.ReLU)

      // Layer 2
      let assert Ok(outputs) = tensor.matmul(h1, state.w2)
      let assert Ok(accel) = tensor.add_broadcast(outputs, state.b2)
      let accel = activation.forward(accel, activation.Tanh)
      // Limit acceleration to -1..1

      // --- STEP 3: ACT (PHYSICS) ---
      // Vel += Accel * dt
      let delta_v = tensor.scale(accel, dt)
      let assert Ok(new_vel) = tensor.add(state.vel, delta_v)

      // Drag (Friction) e.g. 0.99
      let new_vel = tensor.scale(new_vel, 0.99)

      // Pos += Vel * dt
      let delta_p = tensor.scale(new_vel, dt)
      let assert Ok(new_pos) = tensor.add(state.pos, delta_p)

      // TODO: Boundary Wrap (Requires Modulo/Where op not yet in tensor.gleam)
      actor.continue(
        SwarmState(..state, pos: new_pos, vel: new_vel, tick: state.tick + 1),
      )
    }

    GetCount(reply) -> {
      process.send(reply, state.count)
      actor.continue(state)
    }

    GetState(reply) -> {
      process.send(reply, #(state.pos, state.vel))
      actor.continue(state)
    }
  }
}

fn debug_backend(b: Backend) -> String {
  case b {
    gpu.GPU -> "GPU (CUDA)"
    gpu.ExlaCpu -> "CPU (EXLA Accelerated)"
    gpu.CPU -> "CPU (Pure Gleam)"
  }
}
