//// Billiards Environment - VIVA Showcase Benchmark
////
//// Brazilian sinuca (bar pool) with realistic physics.
//// This is VIVA's flagship environment demonstrating GPU-accelerated
//// physics simulation for evolutionary optimization.
////
//// Reference:
////   - Mathavan et al. (2014) - Ball physics research
////   - Ekiefl (2020) - pooltool
////
//// Observation Space (16 dimensions):
////   0-1:   Cue ball position (x, z) normalized
////   2-3:   Target ball position (x, z) normalized
////   4-5:   Nearest pocket direction (dx, dz)
////   6:     Distance to target ball (normalized)
////   7:     Distance to nearest pocket (normalized)
////   8-9:   Best pocket angle (sin, cos)
////   10:    Target ball value (normalized 0-1)
////   11:    Balls remaining (normalized 0-1)
////   12-15: Cue ball velocity (if moving)
////
//// Action Space (Continuous, 4 dimensions):
////   0: Shot angle (0 to 2*pi)
////   1: Shot power (0 to 1)
////   2: English/side spin (-1 to 1)
////   3: Elevation for masse (0 to 0.5)
////
//// Reward:
////   +10 per ball pocketed
////   +5 for hitting target ball
////   -7 for scratch (cue ball pocketed)
////   +0.5 for good position (cue near target)
////   -0.1 per step (encourage fast play)

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{None, Some}
import viva/embodied/billiards/sinuca.{type Shot, type Table, Shot}
import viva/infra/environments/environment.{
  type Action, type EnvInfo, type Environment, type EnvOps, type EnvSpec,
  type Observation, type StepResult, Continuous, Discrete, EnvInfo, EnvOps,
  EnvSpec, Environment, Observation, StepResult,
}
import viva/lifecycle/jolt.{Vec3}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Maximum episode steps
const max_steps: Int = 50

/// Maximum physics steps per action
const physics_steps_per_action: Int = 300

/// Table half dimensions for normalization
const half_length: Float = 1.27

const half_width: Float = 0.635

// =============================================================================
// BILLIARDS STATE
// =============================================================================

/// Billiards environment state (wraps sinuca Table)
pub type BilliardsState {
  BilliardsState(
    table: Table,
    shot_count: Int,
    total_pocketed: Int,
    total_scratches: Int,
    last_hit_target: Bool,
  )
}

// =============================================================================
// ENVIRONMENT OPERATIONS
// =============================================================================

/// Get Billiards environment specification
pub fn spec() -> EnvSpec {
  EnvSpec(
    name: "Billiards-v1 (VIVA Sinuca)",
    observation_dim: 16,
    action_dim: 4,
    discrete_actions: False,
    max_episode_steps: max_steps,
    reward_range: #(-50.0, 100.0),
  )
}

/// Create Billiards environment operations
pub fn ops() -> EnvOps(BilliardsState) {
  EnvOps(
    reset: reset,
    step: step,
    spec: spec,
    clone: clone,
  )
}

/// Reset environment
pub fn reset(_seed: Int) -> Environment(BilliardsState) {
  let table = sinuca.new()

  let state = BilliardsState(
    table: table,
    shot_count: 0,
    total_pocketed: 0,
    total_scratches: 0,
    last_hit_target: False,
  )

  Environment(
    state: state,
    observation: state_to_obs(state),
    timestep: 0,
    episode_return: 0.0,
    done: False,
  )
}

/// Take a step (execute shot)
pub fn step(env: Environment(BilliardsState), action: Action) -> StepResult {
  let state = env.state
  let table = state.table

  // Decode action to shot parameters
  let shot = action_to_shot(action)

  // Count balls before shot
  let balls_before = sinuca.balls_on_table(table)

  // Execute shot
  let table2 = sinuca.shoot(table, shot)

  // Simulate physics until settled
  let table3 = sinuca.simulate_until_settled(table2, physics_steps_per_action)

  // Update pocketed status
  let table4 = sinuca.update_pocketed(table3)

  // Check results
  let balls_after = sinuca.balls_on_table(table4)
  let pocketed = balls_before - balls_after
  let scratch = sinuca.is_scratch(table4)

  // Calculate reward
  let pocketed_reward = int.to_float(pocketed) *. 10.0
  let scratch_penalty = case scratch { True -> -7.0 False -> 0.0 }

  // Position bonus (cue ball near target ball after shot)
  let position_bonus = calculate_position_bonus(table4)

  // Step penalty (encourage efficiency)
  let step_penalty = -0.1

  let reward = pocketed_reward +. scratch_penalty +. position_bonus +. step_penalty

  // Handle scratch
  let table5 = case scratch {
    True -> sinuca.reset_cue_ball(table4)
    False -> table4
  }

  // Update state
  let new_state = BilliardsState(
    table: table5,
    shot_count: state.shot_count + 1,
    total_pocketed: state.total_pocketed + pocketed,
    total_scratches: state.total_scratches + case scratch { True -> 1 False -> 0 },
    last_hit_target: pocketed > 0,
  )

  let new_timestep = env.timestep + 1
  let truncated = new_timestep >= max_steps

  // Episode ends when all balls pocketed or truncated
  let finished = balls_after <= 1  // Only cue ball left
  let done = finished || truncated

  let new_return = env.episode_return +. reward

  let info = EnvInfo(
    timestep: new_timestep,
    episode_return: new_return,
    metrics: [
      #("balls_pocketed", int.to_float(pocketed)),
      #("total_pocketed", int.to_float(new_state.total_pocketed)),
      #("scratches", int.to_float(new_state.total_scratches)),
      #("balls_remaining", int.to_float(balls_after)),
      #("shot_angle", shot.angle),
      #("shot_power", shot.power),
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
pub fn clone(env: Environment(BilliardsState)) -> Environment(BilliardsState) {
  // Note: sinuca.Table contains Jolt world which is mutable
  // For true cloning, would need deep copy of physics world
  // For now, create fresh table (approximate clone)
  let new_table = sinuca.new()

  Environment(
    state: BilliardsState(
      table: new_table,
      shot_count: env.state.shot_count,
      total_pocketed: env.state.total_pocketed,
      total_scratches: env.state.total_scratches,
      last_hit_target: env.state.last_hit_target,
    ),
    observation: env.observation,
    timestep: env.timestep,
    episode_return: env.episode_return,
    done: env.done,
  )
}

// =============================================================================
// BATCH OPERATIONS (GPU-accelerated)
// =============================================================================

/// Batch state for vectorized simulation
pub type BilliardsBatch {
  BilliardsBatch(
    tables: List(Table),
    shot_counts: List(Int),
    total_pocketed: List(Int),
    total_scratches: List(Int),
    done: List(Bool),
    returns: List(Float),
  )
}

/// Create batch of N environments
pub fn create_batch(n: Int, _seed: Int) -> BilliardsBatch {
  let tables = list.map(list.range(0, n - 1), fn(_) { sinuca.new() })

  BilliardsBatch(
    tables: tables,
    shot_counts: list.map(list.range(0, n - 1), fn(_) { 0 }),
    total_pocketed: list.map(list.range(0, n - 1), fn(_) { 0 }),
    total_scratches: list.map(list.range(0, n - 1), fn(_) { 0 }),
    done: list.map(list.range(0, n - 1), fn(_) { False }),
    returns: list.map(list.range(0, n - 1), fn(_) { 0.0 }),
  )
}

/// Batch step (parallel evaluation)
pub fn batch_step(
  batch: BilliardsBatch,
  actions: List(Shot),
) -> #(BilliardsBatch, List(Float), List(Bool)) {
  let n = list.length(batch.tables)
  let indices = list.range(0, n - 1)

  let results = list.map(indices, fn(i) {
    let table = list_at_table(batch.tables, i)
    let shot = list_at_shot(actions, i)
    let was_done = list_at_bool(batch.done, i)
    let shot_count = list_at_int(batch.shot_counts, i)
    let pocketed_total = list_at_int(batch.total_pocketed, i)
    let scratches_total = list_at_int(batch.total_scratches, i)
    let ret = list_at_float(batch.returns, i)

    case was_done {
      True -> #(table, shot_count, pocketed_total, scratches_total, True, ret, 0.0)
      False -> {
        let balls_before = sinuca.balls_on_table(table)
        let t2 = sinuca.shoot(table, shot)
        let t3 = sinuca.simulate_until_settled(t2, physics_steps_per_action)
        let t4 = sinuca.update_pocketed(t3)

        let balls_after = sinuca.balls_on_table(t4)
        let pocketed = balls_before - balls_after
        let scratch = sinuca.is_scratch(t4)

        let pocketed_reward = int.to_float(pocketed) *. 10.0
        let scratch_penalty = case scratch { True -> -7.0 False -> 0.0 }
        let reward = pocketed_reward +. scratch_penalty -. 0.1

        let t5 = case scratch {
          True -> sinuca.reset_cue_ball(t4)
          False -> t4
        }

        let new_count = shot_count + 1
        let new_pocketed = pocketed_total + pocketed
        let new_scratches = scratches_total + case scratch { True -> 1 False -> 0 }
        let done = balls_after <= 1 || new_count >= max_steps
        let new_return = ret +. reward

        #(t5, new_count, new_pocketed, new_scratches, done, new_return, reward)
      }
    }
  })

  let new_batch = BilliardsBatch(
    tables: list.map(results, fn(r) { r.0 }),
    shot_counts: list.map(results, fn(r) { r.1 }),
    total_pocketed: list.map(results, fn(r) { r.2 }),
    total_scratches: list.map(results, fn(r) { r.3 }),
    done: list.map(results, fn(r) { r.4 }),
    returns: list.map(results, fn(r) { r.5 }),
  )

  let rewards = list.map(results, fn(r) { r.6 })
  let dones = list.map(results, fn(r) { r.4 })

  #(new_batch, rewards, dones)
}

/// Reset finished episodes
pub fn batch_reset(batch: BilliardsBatch, _seed: Int) -> BilliardsBatch {
  let n = list.length(batch.tables)
  let indices = list.range(0, n - 1)

  let results = list.map(indices, fn(i) {
    case list_at_bool(batch.done, i) {
      True -> #(sinuca.new(), 0, 0, 0, False, 0.0)
      False -> #(
        list_at_table(batch.tables, i),
        list_at_int(batch.shot_counts, i),
        list_at_int(batch.total_pocketed, i),
        list_at_int(batch.total_scratches, i),
        False,
        list_at_float(batch.returns, i),
      )
    }
  })

  BilliardsBatch(
    tables: list.map(results, fn(r) { r.0 }),
    shot_counts: list.map(results, fn(r) { r.1 }),
    total_pocketed: list.map(results, fn(r) { r.2 }),
    total_scratches: list.map(results, fn(r) { r.3 }),
    done: list.map(results, fn(r) { r.4 }),
    returns: list.map(results, fn(r) { r.5 }),
  )
}

/// Get batch observations
pub fn batch_observations(batch: BilliardsBatch) -> List(List(Float)) {
  list.map(batch.tables, fn(table) {
    let state = BilliardsState(
      table: table,
      shot_count: 0,
      total_pocketed: 0,
      total_scratches: 0,
      last_hit_target: False,
    )
    let obs = state_to_obs(state)
    obs.values
  })
}

// =============================================================================
// OBSERVATION ENCODING
// =============================================================================

fn state_to_obs(state: BilliardsState) -> Observation {
  let table = state.table

  // Cue ball position
  let #(cue_x, cue_z) = case sinuca.get_cue_ball_position(table) {
    Some(Vec3(x, _y, z)) -> #(x /. half_length, z /. half_width)
    None -> #(0.0, 0.0)
  }

  // Target ball position
  let target = table.target_ball
  let #(target_x, target_z) = case sinuca.get_ball_position(table, target) {
    Some(Vec3(x, _y, z)) -> #(x /. half_length, z /. half_width)
    None -> #(0.0, 0.0)
  }

  // Distance to target
  let dx = target_x -. cue_x
  let dz = target_z -. cue_z
  let dist_to_target = float_sqrt(dx *. dx +. dz *. dz)

  // Nearest pocket direction
  let #(pocket_dx, pocket_dz, dist_to_pocket) = nearest_pocket_direction(cue_x, cue_z)

  // Best pocket angle through target
  let #(angle_sin, angle_cos) = best_pocket_angle(cue_x, cue_z, target_x, target_z)

  // Target value (normalized 0-1)
  let target_value = int.to_float(sinuca.point_value(target)) /. 7.0

  // Balls remaining
  let balls_remaining = int.to_float(sinuca.balls_on_table(table)) /. 8.0

  Observation(
    values: [
      cue_x, cue_z,           // 0-1: Cue ball position
      target_x, target_z,     // 2-3: Target ball position
      pocket_dx, pocket_dz,   // 4-5: Nearest pocket direction
      dist_to_target,         // 6: Distance to target
      dist_to_pocket,         // 7: Distance to nearest pocket
      angle_sin, angle_cos,   // 8-9: Best pocket angle
      target_value,           // 10: Target ball value
      balls_remaining,        // 11: Balls remaining
      0.0, 0.0, 0.0, 0.0,     // 12-15: Reserved for velocity
    ],
    dim: 16,
  )
}

fn action_to_shot(action: Action) -> Shot {
  case action {
    Continuous([angle, power, english, elevation, ..]) -> {
      Shot(
        angle: angle *. 2.0 *. 3.14159,
        power: clamp(power, 0.1, 1.0),
        english: clamp(english *. 2.0 -. 1.0, -1.0, 1.0),
        elevation: clamp(elevation, 0.0, 0.5),
      )
    }
    Continuous([angle, power, english]) -> {
      Shot(
        angle: angle *. 2.0 *. 3.14159,
        power: clamp(power, 0.1, 1.0),
        english: clamp(english *. 2.0 -. 1.0, -1.0, 1.0),
        elevation: 0.0,
      )
    }
    Continuous([angle, power]) -> {
      Shot(
        angle: angle *. 2.0 *. 3.14159,
        power: clamp(power, 0.1, 1.0),
        english: 0.0,
        elevation: 0.0,
      )
    }
    _ -> Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)
  }
}

fn nearest_pocket_direction(cue_x: Float, cue_z: Float) -> #(Float, Float, Float) {
  // Corner pockets (normalized coordinates)
  let pockets = [
    #(-1.0, 1.0), #(1.0, 1.0),
    #(-1.0, -1.0), #(1.0, -1.0),
    #(0.0, 1.0), #(0.0, -1.0),
  ]

  let #(best_dx, best_dz, best_dist) = list.fold(pockets, #(0.0, 0.0, 999.0), fn(acc, pocket) {
    let #(px, pz) = pocket
    let dx = px -. cue_x
    let dz = pz -. cue_z
    let dist = float_sqrt(dx *. dx +. dz *. dz)
    case dist <. acc.2 {
      True -> #(dx /. dist, dz /. dist, dist)
      False -> acc
    }
  })

  #(best_dx, best_dz, clamp(best_dist /. 2.0, 0.0, 1.0))
}

fn best_pocket_angle(cue_x: Float, cue_z: Float, target_x: Float, target_z: Float) -> #(Float, Float) {
  // Angle from cue to target
  let dx = target_x -. cue_x
  let dz = target_z -. cue_z
  let angle = float_atan2(dz, dx)
  #(float_sin(angle), float_cos(angle))
}

fn calculate_position_bonus(table: Table) -> Float {
  // Bonus for cue ball being near target ball
  let target = table.target_ball
  case sinuca.get_cue_ball_position(table), sinuca.get_ball_position(table, target) {
    Some(Vec3(cx, _, cz)), Some(Vec3(tx, _, tz)) -> {
      let dx = tx -. cx
      let dz = tz -. cz
      let dist = float_sqrt(dx *. dx +. dz *. dz)
      case dist <. 0.5 {
        True -> 0.5 *. { 1.0 -. dist /. 0.5 }
        False -> 0.0
      }
    }
    _, _ -> 0.0
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False -> case x >. max {
      True -> max
      False -> x
    }
  }
}

fn list_at_table(lst: List(Table), idx: Int) -> Table {
  lst |> list.drop(idx) |> list.first |> result_unwrap_table
}

fn list_at_shot(lst: List(Shot), idx: Int) -> Shot {
  lst |> list.drop(idx) |> list.first |> result_unwrap_shot
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

fn result_unwrap_table(r: Result(Table, e)) -> Table {
  case r { Ok(t) -> t Error(_) -> sinuca.new() }
}

fn result_unwrap_shot(r: Result(Shot, e)) -> Shot {
  case r { Ok(s) -> s Error(_) -> Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0) }
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "atan2")
fn float_atan2(y: Float, x: Float) -> Float
