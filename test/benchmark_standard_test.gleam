//// Tests for VIVA Standard Benchmark Suite

import gleeunit/should
import viva/infra/benchmark_standard
import viva/infra/environments/cartpole
import viva/infra/environments/environment
import viva/infra/environments/pendulum

// =============================================================================
// ENVIRONMENT TESTS
// =============================================================================

pub fn cartpole_spec_test() {
  let spec = cartpole.spec()
  spec.name |> should.equal("CartPole-v1")
  spec.observation_dim |> should.equal(4)
  spec.action_dim |> should.equal(2)
  spec.discrete_actions |> should.equal(True)
  spec.max_episode_steps |> should.equal(500)
}

pub fn cartpole_reset_test() {
  let env = cartpole.reset(42)
  env.timestep |> should.equal(0)
  env.done |> should.equal(False)
  env.observation.dim |> should.equal(4)
}

pub fn cartpole_step_test() {
  let env = cartpole.reset(42)
  let result = cartpole.step(env, environment.Discrete(1))
  result.done |> should.equal(False)
  result.truncated |> should.equal(False)
  result.observation.dim |> should.equal(4)
}

pub fn cartpole_batch_test() {
  let batch = cartpole.create_batch(10, 42)
  batch.x |> list_length |> should.equal(10)
  batch.done |> list_length |> should.equal(10)
}

pub fn cartpole_batch_step_test() {
  let batch = cartpole.create_batch(5, 42)
  let actions = [0, 1, 0, 1, 0]
  let #(new_batch, rewards, dones) = cartpole.batch_step(batch, actions)

  new_batch.x |> list_length |> should.equal(5)
  rewards |> list_length |> should.equal(5)
  dones |> list_length |> should.equal(5)
}

pub fn pendulum_spec_test() {
  let spec = pendulum.spec()
  spec.name |> should.equal("Pendulum-v1")
  spec.observation_dim |> should.equal(3)
  spec.action_dim |> should.equal(1)
  spec.discrete_actions |> should.equal(False)
  spec.max_episode_steps |> should.equal(200)
}

pub fn pendulum_reset_test() {
  let env = pendulum.reset(42)
  env.timestep |> should.equal(0)
  env.done |> should.equal(False)
  env.observation.dim |> should.equal(3)
}

pub fn pendulum_step_test() {
  let env = pendulum.reset(42)
  let result = pendulum.step(env, environment.Continuous([0.5]))
  result.done |> should.equal(False)
  result.observation.dim |> should.equal(3)
}

pub fn pendulum_batch_test() {
  let batch = pendulum.create_batch(10, 42)
  batch.theta |> list_length |> should.equal(10)
}

// =============================================================================
// BENCHMARK CONFIG TESTS
// =============================================================================

pub fn default_config_test() {
  let config = benchmark_standard.default_config()
  config.num_envs |> should.equal(1000)
  config.population_size |> should.equal(100)
}

pub fn fast_config_test() {
  let config = benchmark_standard.fast_config()
  config.num_envs |> should.equal(100)
  config.population_size |> should.equal(50)
}

pub fn literature_baselines_test() {
  let baselines = benchmark_standard.literature_baselines()
  baselines |> list_length |> should.equal(4)
}

// =============================================================================
// HELPERS
// =============================================================================

fn list_length(lst: List(a)) -> Int {
  list_length_loop(lst, 0)
}

fn list_length_loop(lst: List(a), acc: Int) -> Int {
  case lst {
    [] -> acc
    [_, ..rest] -> list_length_loop(rest, acc + 1)
  }
}
