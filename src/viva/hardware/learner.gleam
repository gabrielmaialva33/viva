//// VIVA Body Learner - Body learning via HRR
////
//// Like a baby discovering its body:
//// - Explores random actions (LED, sound)
//// - Observes effects on sensors
//// - Creates action -> effect associations via HRR bind
//// - Uses knowledge to predict/plan

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/hardware/body.{
  type BodyCommand, type Sensation, PlayTone, SetLed, StopAll,
}
import viva/memory/hrr.{type HRR}
import viva/memory/world.{type World}
import viva/neural/tensor.{type Tensor}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// Observed effect of an action
pub type Effect {
  Effect(
    delta_light: Int,
    // Light change (-1023 to 1023)
    delta_noise: Int,
    // Noise change
    got_touch: Bool,
    // Received touch during action?
  )
}

/// Available body action
pub type Action {
  Action(
    name: String,
    // "led_red", "tone_low", etc.
    command: BodyCommand,
    hrr: HRR,
    // Unique HRR representation
  )
}

/// Memory of an experience
pub type Experience {
  Experience(action_name: String, effect: Effect, timestamp: Int)
}

/// Learner state
pub type Learner {
  Learner(
    /// Available actions bank
    actions: Dict(String, Action),
    /// Spatial memory (World) with associations
    world: World,
    /// Curiosity (1.0 = max, 0.1 = min)
    curiosity: Float,
    /// Exploration count per action
    exploration_count: Dict(String, Int),
    /// Current tick
    tick: Int,
    /// HRR dimension
    hrr_dim: Int,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new learner with default actions
pub fn new() -> Learner {
  let hrr_dim = 256

  // Create unique HRRs for each action
  let actions =
    [
      #("led_off", SetLed(0, 0)),
      #("led_red", SetLed(255, 0)),
      #("led_green", SetLed(0, 255)),
      #("led_yellow", SetLed(255, 255)),
      #("tone_low", PlayTone(9, 200, 100)),
      #("tone_mid", PlayTone(9, 500, 100)),
      #("tone_high", PlayTone(9, 1000, 100)),
      #("tone_right", PlayTone(10, 800, 100)),
      #("stop", StopAll),
    ]
    |> list.fold(dict.new(), fn(acc, pair) {
      let #(name, cmd) = pair
      let action = Action(name: name, command: cmd, hrr: hrr.random(hrr_dim))
      dict.insert(acc, name, action)
    })

  let exploration_count =
    dict.keys(actions)
    |> list.fold(dict.new(), fn(acc, name) { dict.insert(acc, name, 0) })

  Learner(
    actions: actions,
    world: world.new_with_config(learner_world_config()),
    curiosity: 1.0,
    exploration_count: exploration_count,
    tick: 0,
    hrr_dim: hrr_dim,
  )
}

/// World configuration optimized for learner
fn learner_world_config() -> world.WorldConfig {
  world.WorldConfig(
    spatial_dims: 4,
    // P, A, D, Intensity
    hrr_dims: 256,
    attraction_strength: 0.2,
    repulsion_strength: 0.03,
    energy_decay: 0.995,
    // Memórias duram mais
    island_threshold: 4.0,
    damping: 0.9,
    max_velocity: 3.0,
  )
}

// =============================================================================
// EXPLORATION
// =============================================================================

/// Choose next action to explore
/// Returns (action name, command for body)
pub fn explore(learner: Learner) -> #(String, BodyCommand) {
  let action_names = dict.keys(learner.actions)

  let chosen_name = case random_float() <. learner.curiosity {
    // High curiosity: explore random
    True -> {
      let idx =
        float.truncate(
          random_float() *. int_to_float(list.length(action_names)),
        )
      list_get(action_names, idx, "stop")
    }
    // Low curiosity: choose least explored
    False -> {
      let least_explored =
        dict.to_list(learner.exploration_count)
        |> list.sort(fn(a, b) { int.compare(a.1, b.1) })
        |> list.first

      case least_explored {
        Ok(#(name, _)) -> name
        Error(_) -> "stop"
      }
    }
  }

  case dict.get(learner.actions, chosen_name) {
    Ok(action) -> #(chosen_name, action.command)
    Error(_) -> #("stop", StopAll)
  }
}

// =============================================================================
// LEARNING
// =============================================================================

/// Learn from an experience (action + observed effect)
pub fn learn(
  learner: Learner,
  action_name: String,
  before: Sensation,
  after: Sensation,
) -> Learner {
  let effect =
    Effect(
      delta_light: after.light - before.light,
      delta_noise: after.noise - before.noise,
      got_touch: after.touch && !before.touch,
    )

  case dict.get(learner.actions, action_name) {
    Error(_) -> learner
    Ok(action) -> {
      // Create HRR of effect
      let effect_hrr = encode_effect(effect, learner.hrr_dim)

      // Bind action with effect
      case hrr.bind(action.hrr, effect_hrr) {
        Error(_) -> learner
        Ok(association) -> {
          // Position in emotional space (based on effect)
          let position = effect_to_position(effect)

          // Add to world
          let label = action_name <> " -> " <> effect_label(effect)
          let #(new_world, _id) =
            world.add_memory(learner.world, association, position, label)

          // Update exploration count
          let current_count = case
            dict.get(learner.exploration_count, action_name)
          {
            Ok(n) -> n
            Error(_) -> 0
          }
          let new_count =
            dict.insert(
              learner.exploration_count,
              action_name,
              current_count + 1,
            )

          // Decay curiosity
          let new_curiosity = float.max(0.1, learner.curiosity *. 0.998)

          Learner(
            ..learner,
            world: new_world,
            exploration_count: new_count,
            curiosity: new_curiosity,
            tick: learner.tick + 1,
          )
        }
      }
    }
  }
}

// =============================================================================
// PREDICTION
// =============================================================================

/// Predict effect of an action based on memory
pub fn predict(learner: Learner, action_name: String) -> Option(Effect) {
  case dict.get(learner.actions, action_name) {
    Error(_) -> None
    Ok(action) -> {
      // Search memories similar to action
      let results = world.query(learner.world, action.hrr, 5)

      case list.first(results) {
        Error(_) -> None
        Ok(result) -> {
          // Try to recover effect via unbind
          case hrr.unbind(result.body.shape, action.hrr) {
            Error(_) -> None
            Ok(effect_hrr) -> Some(decode_effect(effect_hrr))
          }
        }
      }
    }
  }
}

/// Find best action for a goal
pub fn find_action_for(learner: Learner, goal: Effect) -> Option(String) {
  let goal_hrr = encode_effect(goal, learner.hrr_dim)

  // For each action, check if any memory predicts desired effect
  let candidates =
    dict.to_list(learner.actions)
    |> list.filter_map(fn(pair) {
      let #(name, action) = pair

      // Query memories related to this action
      let results = world.query(learner.world, action.hrr, 3)

      case list.first(results) {
        Error(_) -> Error(Nil)
        Ok(result) -> {
          case hrr.unbind(result.body.shape, action.hrr) {
            Error(_) -> Error(Nil)
            Ok(predicted_effect) -> {
              let sim = hrr.similarity(predicted_effect, goal_hrr)
              Ok(#(name, sim))
            }
          }
        }
      }
    })
    |> list.sort(fn(a, b) { float.compare(b.1, a.1) })

  case list.first(candidates) {
    Ok(#(name, sim)) if sim >. 0.3 -> Some(name)
    _ -> None
  }
}

// =============================================================================
// INTROSPECTION
// =============================================================================

/// Describe what learner knows about an action
pub fn describe_action(learner: Learner, action_name: String) -> String {
  case dict.get(learner.exploration_count, action_name) {
    Error(_) -> "unknown action"
    Ok(count) if count == 0 -> "never tried"
    Ok(count) -> {
      case predict(learner, action_name) {
        None -> "tried " <> int.to_string(count) <> "x, no clear pattern"
        Some(effect) ->
          effect_label(effect) <> " (" <> int.to_string(count) <> "x)"
      }
    }
  }
}

/// List all actions and what was learned
pub fn knowledge_summary(learner: Learner) -> List(#(String, String)) {
  dict.keys(learner.actions)
  |> list.map(fn(name) { #(name, describe_action(learner, name)) })
}

/// Learner statistics
pub fn stats(learner: Learner) -> Dict(String, String) {
  dict.new()
  |> dict.insert("tick", int.to_string(learner.tick))
  |> dict.insert("curiosity", float.to_string(learner.curiosity))
  |> dict.insert("memories", int.to_string(world.body_count(learner.world)))
  |> dict.insert("awake", int.to_string(world.awake_count(learner.world)))
  |> dict.insert("islands", int.to_string(world.island_count(learner.world)))
}

// =============================================================================
// HRR ENCODING
// =============================================================================

/// Encode effect as HRR
fn encode_effect(effect: Effect, dim: Int) -> HRR {
  // Normalize values to [-1, 1]
  let light_norm = int_to_float(effect.delta_light) /. 512.0
  let noise_norm = int_to_float(effect.delta_noise) /. 512.0
  let touch_val = case effect.got_touch {
    True -> 1.0
    False -> 0.0
  }

  // Create vector with pattern based on values
  // (technique: modulate each dimension by normalized values)
  let data =
    list.range(0, dim - 1)
    |> list.map(fn(i) {
      let phase = int_to_float(i) /. int_to_float(dim) *. 6.28318
      light_norm
      *. float_cos(phase)
      +. noise_norm
      *. float_sin(phase)
      +. touch_val
      *. float_cos(phase *. 2.0)
    })

  hrr.from_list(data)
  |> hrr.normalize
}

/// Decode HRR back to effect
fn decode_effect(h: HRR) -> Effect {
  // Extract approximate values from HRR
  let data = td(h.vector)

  // Estimate delta_light from first third of vector
  let light_sum =
    list.take(data, h.dim / 3)
    |> list.fold(0.0, fn(acc, x) { acc +. x })
  let light_avg = light_sum /. int_to_float(h.dim / 3)

  // Estimate delta_noise from second third
  let noise_slice =
    list.drop(data, h.dim / 3)
    |> list.take(h.dim / 3)
  let noise_sum = list.fold(noise_slice, 0.0, fn(acc, x) { acc +. x })
  let noise_avg = noise_sum /. int_to_float(h.dim / 3)

  // Estimate touch from overall magnitude
  let total_mag =
    list.fold(data, 0.0, fn(acc, x) { acc +. float.absolute_value(x) })
    /. int_to_float(h.dim)

  Effect(
    delta_light: float.truncate(light_avg *. 512.0),
    delta_noise: float.truncate(noise_avg *. 512.0),
    got_touch: total_mag >. 0.5,
  )
}

// =============================================================================
// HELPERS
// =============================================================================

/// Convert effect to position in emotional space (4D: P, A, D, I)
fn effect_to_position(effect: Effect) -> Tensor {
  // Pleasure: increases with positive light, touch
  let pleasure =
    int_to_float(effect.delta_light)
    /. 200.0
    +. case effect.got_touch {
      True -> 0.5
      False -> 0.0
    }

  // Arousal: increases with large changes
  let arousal =
    float.absolute_value(int_to_float(effect.delta_light))
    /. 300.0
    +. float.absolute_value(int_to_float(effect.delta_noise))
    /. 300.0

  // Dominance: fixed for now (could vary with prediction success)
  let dominance = 0.5

  // Intensity: overall magnitude of effect
  let intensity =
    float.absolute_value(int_to_float(effect.delta_light))
    /. 500.0
    +. float.absolute_value(int_to_float(effect.delta_noise))
    /. 500.0
    +. case effect.got_touch {
      True -> 0.3
      False -> 0.0
    }

  tensor.Tensor(data: [pleasure, arousal, dominance, intensity], shape: [4])
}

/// Generate readable label for an effect
fn effect_label(effect: Effect) -> String {
  let parts =
    []
    |> fn(acc) {
      case effect.delta_light {
        d if d > 10 -> ["light+" <> int.to_string(d), ..acc]
        d if d < -10 -> ["light" <> int.to_string(d), ..acc]
        _ -> acc
      }
    }
    |> fn(acc) {
      case effect.delta_noise {
        d if d > 10 -> ["noise+" <> int.to_string(d), ..acc]
        d if d < -10 -> ["noise" <> int.to_string(d), ..acc]
        _ -> acc
      }
    }
    |> fn(acc) {
      case effect.got_touch {
        True -> ["TOUCH", ..acc]
        False -> acc
      }
    }

  case list.is_empty(parts) {
    True -> "no effect"
    False -> string_join(list.reverse(parts), ", ")
  }
}

fn list_get(l: List(a), idx: Int, default: a) -> a {
  case list.drop(l, idx) {
    [x, ..] -> x
    [] -> default
  }
}

fn string_join(parts: List(String), sep: String) -> String {
  case parts {
    [] -> ""
    [single] -> single
    [first, ..rest] -> list.fold(rest, first, fn(acc, s) { acc <> sep <> s })
  }
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "rand", "uniform")
fn random_float() -> Float

@external(erlang, "erlang", "float")
fn int_to_float(i: Int) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float
