//// Pendulum Environment - Continuous Control Benchmark
////
//// Classic inverted pendulum swingup task. The goal is to swing the pendulum
//// up so it stays upright. This is a continuous control task.
////
//// Reference:
////   - OpenAI Gym Pendulum-v1
////   - Brax Pendulum environment
////
//// Observation Space (3 dimensions):
////   0: cos(theta)  [-1, 1]
////   1: sin(theta)  [-1, 1]
////   2: theta_dot   [-8, 8]
////
//// Action Space (Continuous, 1 dimension):
////   0: Torque applied to pendulum [-2, 2]
////
//// Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*action^2)
//// Episode length: 200 steps (no early termination)

import gleam/float
import gleam/list
import viva/infra/environments/environment.{
  type Action, type EnvInfo, type Environment, type EnvOps, type EnvSpec,
  type Observation, type StepResult, Continuous, Discrete, EnvInfo, EnvOps,
  EnvSpec, Environment, Observation, StepResult,
}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Maximum angular velocity
const max_speed: Float = 8.0

/// Maximum torque
const max_torque: Float = 2.0

/// Physics timestep
const dt: Float = 0.05

/// Gravity
const gravity: Float = 10.0

/// Pendulum mass
const mass: Float = 1.0

/// Pendulum length
const length: Float = 1.0

/// Maximum episode steps
const max_steps: Int = 200

// =============================================================================
// PENDULUM STATE
// =============================================================================

/// Internal state of Pendulum environment
pub type PendulumState {
  PendulumState(
    /// Angle (radians, 0 = upright)
    theta: Float,
    /// Angular velocity
    theta_dot: Float,
  )
}

// =============================================================================
// ENVIRONMENT OPERATIONS
// =============================================================================

/// Get Pendulum environment specification
pub fn spec() -> EnvSpec {
  EnvSpec(
    name: "Pendulum-v1",
    observation_dim: 3,
    action_dim: 1,
    discrete_actions: False,
    max_episode_steps: max_steps,
    reward_range: #(-16.27, 0.0),
  )
}

/// Create Pendulum environment operations
pub fn ops() -> EnvOps(PendulumState) {
  EnvOps(
    reset: reset,
    step: step,
    spec: spec,
    clone: clone,
  )
}

/// Reset environment
pub fn reset(seed: Int) -> Environment(PendulumState) {
  // Initialize at random angle
  let r1 = pseudo_random(seed)
  let r2 = pseudo_random(seed + 1)

  let state = PendulumState(
    theta: { r1 -. 0.5 } *. 2.0 *. 3.14159,  // Random in [-pi, pi]
    theta_dot: { r2 -. 0.5 } *. 2.0,          // Random in [-1, 1]
  )

  Environment(
    state: state,
    observation: state_to_obs(state),
    timestep: 0,
    episode_return: 0.0,
    done: False,
  )
}

/// Take a step
pub fn step(env: Environment(PendulumState), action: Action) -> StepResult {
  let state = env.state

  // Get torque from action
  let torque = case action {
    Continuous([t, ..]) -> clamp(t, neg(max_torque), max_torque)
    Continuous([]) -> 0.0
    Discrete(a) -> case a {
      0 -> neg(max_torque)
      1 -> 0.0
      _ -> max_torque
    }
  }

  // Normalize angle to [-pi, pi]
  let theta = normalize_angle(state.theta)

  // Calculate reward (before update)
  // reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
  let angle_cost = theta *. theta
  let velocity_cost = 0.1 *. state.theta_dot *. state.theta_dot
  let action_cost = 0.001 *. torque *. torque
  let reward = neg(angle_cost +. velocity_cost +. action_cost)

  // Physics update (Euler integration)
  // theta_ddot = (3*g)/(2*l) * sin(theta) + (3/(m*l^2)) * torque
  let theta_ddot = {
    3.0 *. gravity /. { 2.0 *. length } *. float_sin(theta)
    +. 3.0 /. { mass *. length *. length } *. torque
  }

  let new_theta_dot = state.theta_dot +. theta_ddot *. dt
  let new_theta_dot = clamp(new_theta_dot, neg(max_speed), max_speed)
  let new_theta = state.theta +. new_theta_dot *. dt

  let new_state = PendulumState(
    theta: new_theta,
    theta_dot: new_theta_dot,
  )

  let new_timestep = env.timestep + 1
  let truncated = new_timestep >= max_steps
  let done = truncated  // Pendulum never terminates early

  let new_return = env.episode_return +. reward

  let info = EnvInfo(
    timestep: new_timestep,
    episode_return: new_return,
    metrics: [
      #("theta", new_theta),
      #("theta_dot", new_theta_dot),
      #("torque", torque),
      #("cos_theta", float_cos(new_theta)),
    ],
  )

  StepResult(
    observation: state_to_obs(new_state),
    reward: reward,
    done: done,
    truncated: truncated,
    info: info,
  )
}

/// Clone environment
pub fn clone(env: Environment(PendulumState)) -> Environment(PendulumState) {
  Environment(
    state: PendulumState(
      theta: env.state.theta,
      theta_dot: env.state.theta_dot,
    ),
    observation: env.observation,
    timestep: env.timestep,
    episode_return: env.episode_return,
    done: env.done,
  )
}

// =============================================================================
// BATCH OPERATIONS
// =============================================================================

/// Batch of Pendulum states
pub type PendulumBatch {
  PendulumBatch(
    theta: List(Float),
    theta_dot: List(Float),
    done: List(Bool),
    timesteps: List(Int),
    returns: List(Float),
  )
}

/// Create batch of N environments
pub fn create_batch(n: Int, seed: Int) -> PendulumBatch {
  let indices = list.range(0, n - 1)

  let states = list.map(indices, fn(i) {
    let s = seed + i * 2
    let r1 = pseudo_random(s)
    let r2 = pseudo_random(s + 1)
    #(
      { r1 -. 0.5 } *. 2.0 *. 3.14159,
      { r2 -. 0.5 } *. 2.0,
    )
  })

  PendulumBatch(
    theta: list.map(states, fn(s) { s.0 }),
    theta_dot: list.map(states, fn(s) { s.1 }),
    done: list.map(indices, fn(_) { False }),
    timesteps: list.map(indices, fn(_) { 0 }),
    returns: list.map(indices, fn(_) { 0.0 }),
  )
}

/// Batch step (vectorized)
pub fn batch_step(
  batch: PendulumBatch,
  actions: List(Float),
) -> #(PendulumBatch, List(Float), List(Bool)) {
  let n = list.length(batch.theta)
  let indices = list.range(0, n - 1)

  let results = list.map(indices, fn(i) {
    let theta = list_at_float(batch.theta, i)
    let theta_dot = list_at_float(batch.theta_dot, i)
    let was_done = list_at_bool(batch.done, i)
    let timestep = list_at_int(batch.timesteps, i)
    let ret = list_at_float(batch.returns, i)
    let torque = clamp(list_at_float(actions, i), neg(max_torque), max_torque)

    case was_done {
      True -> #(theta, theta_dot, True, timestep, ret, 0.0)
      False -> {
        let norm_theta = normalize_angle(theta)
        let angle_cost = norm_theta *. norm_theta
        let velocity_cost = 0.1 *. theta_dot *. theta_dot
        let action_cost = 0.001 *. torque *. torque
        let reward = neg(angle_cost +. velocity_cost +. action_cost)

        let theta_ddot = {
          3.0 *. gravity /. { 2.0 *. length } *. float_sin(norm_theta)
          +. 3.0 /. { mass *. length *. length } *. torque
        }

        let new_theta_dot = clamp(theta_dot +. theta_ddot *. dt, neg(max_speed), max_speed)
        let new_theta = theta +. new_theta_dot *. dt

        let new_timestep = timestep + 1
        let done = new_timestep >= max_steps
        let new_return = ret +. reward

        #(new_theta, new_theta_dot, done, new_timestep, new_return, reward)
      }
    }
  })

  let new_batch = PendulumBatch(
    theta: list.map(results, fn(r) { r.0 }),
    theta_dot: list.map(results, fn(r) { r.1 }),
    done: list.map(results, fn(r) { r.2 }),
    timesteps: list.map(results, fn(r) { r.3 }),
    returns: list.map(results, fn(r) { r.4 }),
  )

  let rewards = list.map(results, fn(r) { r.5 })
  let dones = list.map(results, fn(r) { r.2 })

  #(new_batch, rewards, dones)
}

/// Reset finished episodes
pub fn batch_reset(batch: PendulumBatch, seed: Int) -> PendulumBatch {
  let n = list.length(batch.theta)
  let indices = list.range(0, n - 1)

  let results = list.map(indices, fn(i) {
    case list_at_bool(batch.done, i) {
      True -> {
        let s = seed + i * 2
        let r1 = pseudo_random(s)
        let r2 = pseudo_random(s + 1)
        #(
          { r1 -. 0.5 } *. 2.0 *. 3.14159,
          { r2 -. 0.5 } *. 2.0,
          False,
          0,
          0.0,
        )
      }
      False -> #(
        list_at_float(batch.theta, i),
        list_at_float(batch.theta_dot, i),
        False,
        list_at_int(batch.timesteps, i),
        list_at_float(batch.returns, i),
      )
    }
  })

  PendulumBatch(
    theta: list.map(results, fn(r) { r.0 }),
    theta_dot: list.map(results, fn(r) { r.1 }),
    done: list.map(results, fn(r) { r.2 }),
    timesteps: list.map(results, fn(r) { r.3 }),
    returns: list.map(results, fn(r) { r.4 }),
  )
}

/// Get batch observations (cos, sin, theta_dot)
pub fn batch_observations(batch: PendulumBatch) -> List(List(Float)) {
  let n = list.length(batch.theta)
  let indices = list.range(0, n - 1)

  list.map(indices, fn(i) {
    let theta = list_at_float(batch.theta, i)
    [float_cos(theta), float_sin(theta), list_at_float(batch.theta_dot, i)]
  })
}

// =============================================================================
// HELPERS
// =============================================================================

fn state_to_obs(state: PendulumState) -> Observation {
  Observation(
    values: [
      float_cos(state.theta),
      float_sin(state.theta),
      state.theta_dot,
    ],
    dim: 3,
  )
}

fn normalize_angle(angle: Float) -> Float {
  // Normalize to [-pi, pi]
  let pi = 3.14159265359
  let two_pi = 2.0 *. pi
  let a = float_mod(angle +. pi, two_pi)
  case a <. 0.0 {
    True -> a +. two_pi -. pi
    False -> a -. pi
  }
}

fn clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False -> case x >. max {
      True -> max
      False -> x
    }
  }
}

fn neg(x: Float) -> Float {
  0.0 -. x
}

fn pseudo_random(seed: Int) -> Float {
  let s = int_abs(seed) + 1
  let x = int_xor(s, int_shl(s, 13))
  let x = int_xor(x, int_shr(x, 17))
  let x = int_xor(x, int_shl(x, 5))
  let x = int_abs(x) % 1_000_000
  int_to_float(x) /. 1_000_000.0
}

fn list_at_float(lst: List(Float), idx: Int) -> Float {
  lst |> list.drop(idx) |> list.first |> result_unwrap(0.0)
}

fn list_at_int(lst: List(Int), idx: Int) -> Int {
  lst |> list.drop(idx) |> list.first |> result_unwrap(0)
}

fn list_at_bool(lst: List(Bool), idx: Int) -> Bool {
  lst |> list.drop(idx) |> list.first |> result_unwrap(False)
}

fn result_unwrap(r: Result(a, e), default: a) -> a {
  case r { Ok(v) -> v Error(_) -> default }
}

@external(erlang, "erlang", "float")
fn int_to_float(x: Int) -> Float

fn int_abs(x: Int) -> Int {
  case x < 0 { True -> 0 - x False -> x }
}

fn int_xor(a: Int, b: Int) -> Int {
  erlang_bxor(a, b)
}

fn int_shl(a: Int, b: Int) -> Int {
  erlang_bsl(a, b)
}

fn int_shr(a: Int, b: Int) -> Int {
  erlang_bsr(a, b)
}

@external(erlang, "erlang", "bxor")
fn erlang_bxor(a: Int, b: Int) -> Int

@external(erlang, "erlang", "bsl")
fn erlang_bsl(a: Int, b: Int) -> Int

@external(erlang, "erlang", "bsr")
fn erlang_bsr(a: Int, b: Int) -> Int

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

@external(erlang, "math", "fmod")
fn float_mod(a: Float, b: Float) -> Float
