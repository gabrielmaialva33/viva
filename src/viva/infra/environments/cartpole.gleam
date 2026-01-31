//// CartPole Environment - Classic Control Benchmark
////
//// A pole is attached by an un-actuated joint to a cart, which moves along a
//// frictionless track. The pendulum starts upright, and the goal is to prevent
//// it from falling over by increasing or reducing the cart's velocity.
////
//// Reference:
////   - Barto et al. (1983) - Neuronlike adaptive elements
////   - OpenAI Gym CartPole-v1 specification
////
//// Observation Space (4 dimensions):
////   0: Cart position     (-4.8, 4.8)
////   1: Cart velocity     (-Inf, Inf)
////   2: Pole angle        (-0.418 rad, 0.418 rad) ~ 24 degrees
////   3: Pole angular vel  (-Inf, Inf)
////
//// Action Space (Discrete, 2 actions):
////   0: Push cart left
////   1: Push cart right
////
//// Reward: +1 for every step the pole remains upright
//// Episode ends: pole angle > 12 degrees OR cart position > 2.4

import gleam/float
import gleam/list
import viva/infra/environments/environment.{
  type Action, type EnvInfo, type Environment, type EnvOps, type EnvSpec,
  type Observation, type StepResult, Continuous, Discrete, EnvInfo, EnvOps,
  EnvSpec, Environment, Observation, StepResult,
}

// =============================================================================
// CONSTANTS (from OpenAI Gym)
// =============================================================================

/// Gravity (m/s^2)
const gravity: Float = 9.8

/// Cart mass (kg)
const cart_mass: Float = 1.0

/// Pole mass (kg)
const pole_mass: Float = 0.1

/// Total mass
const total_mass: Float = 1.1

/// Pole half-length (m)
const pole_length: Float = 0.5

/// Pole mass times length
const pole_mass_length: Float = 0.05

/// Force magnitude applied to cart
const force_mag: Float = 10.0

/// Physics timestep (s)
const tau: Float = 0.02

/// Theta threshold for failure (radians) ~ 12 degrees
const theta_threshold: Float = 0.2095

/// X position threshold for failure
const x_threshold: Float = 2.4

/// Maximum episode steps
const max_steps: Int = 500

// =============================================================================
// CARTPOLE STATE
// =============================================================================

/// Internal state of CartPole environment
pub type CartPoleState {
  CartPoleState(
    /// Cart position (x)
    x: Float,
    /// Cart velocity (x_dot)
    x_dot: Float,
    /// Pole angle (theta, 0 = upright)
    theta: Float,
    /// Pole angular velocity (theta_dot)
    theta_dot: Float,
  )
}

// =============================================================================
// ENVIRONMENT OPERATIONS
// =============================================================================

/// Get CartPole environment specification
pub fn spec() -> EnvSpec {
  EnvSpec(
    name: "CartPole-v1",
    observation_dim: 4,
    action_dim: 2,
    discrete_actions: True,
    max_episode_steps: max_steps,
    reward_range: #(0.0, 500.0),
  )
}

/// Create new CartPole environment operations
pub fn ops() -> EnvOps(CartPoleState) {
  EnvOps(
    reset: reset,
    step: step,
    spec: spec,
    clone: clone,
  )
}

/// Reset environment to initial state
pub fn reset(seed: Int) -> Environment(CartPoleState) {
  // Initialize state with small random values
  let r1 = pseudo_random(seed)
  let r2 = pseudo_random(seed + 1)
  let r3 = pseudo_random(seed + 2)
  let r4 = pseudo_random(seed + 3)

  let state = CartPoleState(
    x: { r1 -. 0.5 } *. 0.1,
    x_dot: { r2 -. 0.5 } *. 0.1,
    theta: { r3 -. 0.5 } *. 0.1,
    theta_dot: { r4 -. 0.5 } *. 0.1,
  )

  Environment(
    state: state,
    observation: state_to_obs(state),
    timestep: 0,
    episode_return: 0.0,
    done: False,
  )
}

/// Take a step in the environment
pub fn step(env: Environment(CartPoleState), action: Action) -> StepResult {
  let state = env.state

  // Decode action (0 = left, 1 = right)
  let force = case action {
    Discrete(0) -> 0.0 -. force_mag
    Discrete(1) -> force_mag
    Discrete(_) -> 0.0
    Continuous([f, ..]) -> f *. force_mag
    Continuous([]) -> 0.0
  }

  // Physics calculations (Euler integration)
  let cos_theta = float_cos(state.theta)
  let sin_theta = float_sin(state.theta)

  // Intermediate calculation
  let temp = { force +. pole_mass_length *. state.theta_dot *. state.theta_dot *. sin_theta } /. total_mass

  // Angular acceleration
  let theta_acc = {
    gravity *. sin_theta -. cos_theta *. temp
  } /. {
    pole_length *. { 4.0 /. 3.0 -. pole_mass *. cos_theta *. cos_theta /. total_mass }
  }

  // Linear acceleration
  let x_acc = temp -. pole_mass_length *. theta_acc *. cos_theta /. total_mass

  // Euler integration
  let new_x = state.x +. tau *. state.x_dot
  let new_x_dot = state.x_dot +. tau *. x_acc
  let new_theta = state.theta +. tau *. state.theta_dot
  let new_theta_dot = state.theta_dot +. tau *. theta_acc

  let new_state = CartPoleState(
    x: new_x,
    x_dot: new_x_dot,
    theta: new_theta,
    theta_dot: new_theta_dot,
  )

  // Check termination
  let x_out = float.absolute_value(new_x) >. x_threshold
  let theta_out = float.absolute_value(new_theta) >. theta_threshold
  let terminated = x_out || theta_out

  let new_timestep = env.timestep + 1
  let truncated = new_timestep >= max_steps

  let done = terminated || truncated

  // Reward: +1 for each step survived
  let reward = case terminated {
    True -> 0.0
    False -> 1.0
  }

  let new_return = env.episode_return +. reward

  let info = EnvInfo(
    timestep: new_timestep,
    episode_return: new_return,
    metrics: [
      #("x", new_x),
      #("theta", new_theta),
      #("x_dot", new_x_dot),
      #("theta_dot", new_theta_dot),
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

/// Clone environment state
pub fn clone(env: Environment(CartPoleState)) -> Environment(CartPoleState) {
  let state = env.state
  let new_state = CartPoleState(
    x: state.x,
    x_dot: state.x_dot,
    theta: state.theta,
    theta_dot: state.theta_dot,
  )
  Environment(
    state: new_state,
    observation: env.observation,
    timestep: env.timestep,
    episode_return: env.episode_return,
    done: env.done,
  )
}

// =============================================================================
// BATCH OPERATIONS (for GPU/parallel evaluation)
// =============================================================================

/// Batch of CartPole states (for vectorized simulation)
pub type CartPoleBatch {
  CartPoleBatch(
    x: List(Float),
    x_dot: List(Float),
    theta: List(Float),
    theta_dot: List(Float),
    done: List(Bool),
    timesteps: List(Int),
    returns: List(Float),
  )
}

/// Create batch of N environments
pub fn create_batch(n: Int, seed: Int) -> CartPoleBatch {
  let indices = list.range(0, n - 1)

  let states = list.map(indices, fn(i) {
    let s = seed + i * 4
    let r1 = pseudo_random(s)
    let r2 = pseudo_random(s + 1)
    let r3 = pseudo_random(s + 2)
    let r4 = pseudo_random(s + 3)
    #(
      { r1 -. 0.5 } *. 0.1,
      { r2 -. 0.5 } *. 0.1,
      { r3 -. 0.5 } *. 0.1,
      { r4 -. 0.5 } *. 0.1,
    )
  })

  CartPoleBatch(
    x: list.map(states, fn(s) { s.0 }),
    x_dot: list.map(states, fn(s) { s.1 }),
    theta: list.map(states, fn(s) { s.2 }),
    theta_dot: list.map(states, fn(s) { s.3 }),
    done: list.map(indices, fn(_) { False }),
    timesteps: list.map(indices, fn(_) { 0 }),
    returns: list.map(indices, fn(_) { 0.0 }),
  )
}

/// Batch step (vectorized)
pub fn batch_step(
  batch: CartPoleBatch,
  actions: List(Int),
) -> #(CartPoleBatch, List(Float), List(Bool)) {
  let n = list.length(batch.x)
  let indices = list.range(0, n - 1)

  let results = list.map(indices, fn(i) {
    let x = list_at_float(batch.x, i)
    let x_dot = list_at_float(batch.x_dot, i)
    let theta = list_at_float(batch.theta, i)
    let theta_dot = list_at_float(batch.theta_dot, i)
    let was_done = list_at_bool(batch.done, i)
    let timestep = list_at_int(batch.timesteps, i)
    let ret = list_at_float(batch.returns, i)
    let action = list_at_int(actions, i)

    case was_done {
      True -> #(x, x_dot, theta, theta_dot, True, timestep, ret, 0.0)
      False -> {
        let force = case action {
          0 -> 0.0 -. force_mag
          _ -> force_mag
        }

        let cos_theta = float_cos(theta)
        let sin_theta = float_sin(theta)
        let temp = { force +. pole_mass_length *. theta_dot *. theta_dot *. sin_theta } /. total_mass
        let theta_acc = {
          gravity *. sin_theta -. cos_theta *. temp
        } /. {
          pole_length *. { 4.0 /. 3.0 -. pole_mass *. cos_theta *. cos_theta /. total_mass }
        }
        let x_acc = temp -. pole_mass_length *. theta_acc *. cos_theta /. total_mass

        let new_x = x +. tau *. x_dot
        let new_x_dot = x_dot +. tau *. x_acc
        let new_theta = theta +. tau *. theta_dot
        let new_theta_dot = theta_dot +. tau *. theta_acc

        let x_out = float.absolute_value(new_x) >. x_threshold
        let theta_out = float.absolute_value(new_theta) >. theta_threshold
        let terminated = x_out || theta_out
        let new_timestep = timestep + 1
        let truncated = new_timestep >= max_steps
        let done = terminated || truncated

        let reward = case terminated { True -> 0.0 False -> 1.0 }
        let new_return = ret +. reward

        #(new_x, new_x_dot, new_theta, new_theta_dot, done, new_timestep, new_return, reward)
      }
    }
  })

  let new_batch = CartPoleBatch(
    x: list.map(results, fn(r) { r.0 }),
    x_dot: list.map(results, fn(r) { r.1 }),
    theta: list.map(results, fn(r) { r.2 }),
    theta_dot: list.map(results, fn(r) { r.3 }),
    done: list.map(results, fn(r) { r.4 }),
    timesteps: list.map(results, fn(r) { r.5 }),
    returns: list.map(results, fn(r) { r.6 }),
  )

  let rewards = list.map(results, fn(r) { r.7 })
  let dones = list.map(results, fn(r) { r.4 })

  #(new_batch, rewards, dones)
}

/// Reset finished episodes in batch
pub fn batch_reset(batch: CartPoleBatch, seed: Int) -> CartPoleBatch {
  let n = list.length(batch.x)
  let indices = list.range(0, n - 1)

  let results = list.map(indices, fn(i) {
    case list_at_bool(batch.done, i) {
      True -> {
        let s = seed + i * 4
        let r1 = pseudo_random(s)
        let r2 = pseudo_random(s + 1)
        let r3 = pseudo_random(s + 2)
        let r4 = pseudo_random(s + 3)
        #(
          { r1 -. 0.5 } *. 0.1,
          { r2 -. 0.5 } *. 0.1,
          { r3 -. 0.5 } *. 0.1,
          { r4 -. 0.5 } *. 0.1,
          False,
          0,
          0.0,
        )
      }
      False -> #(
        list_at_float(batch.x, i),
        list_at_float(batch.x_dot, i),
        list_at_float(batch.theta, i),
        list_at_float(batch.theta_dot, i),
        False,
        list_at_int(batch.timesteps, i),
        list_at_float(batch.returns, i),
      )
    }
  })

  CartPoleBatch(
    x: list.map(results, fn(r) { r.0 }),
    x_dot: list.map(results, fn(r) { r.1 }),
    theta: list.map(results, fn(r) { r.2 }),
    theta_dot: list.map(results, fn(r) { r.3 }),
    done: list.map(results, fn(r) { r.4 }),
    timesteps: list.map(results, fn(r) { r.5 }),
    returns: list.map(results, fn(r) { r.6 }),
  )
}

// =============================================================================
// HELPERS
// =============================================================================

fn state_to_obs(state: CartPoleState) -> Observation {
  Observation(
    values: [state.x, state.x_dot, state.theta, state.theta_dot],
    dim: 4,
  )
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
