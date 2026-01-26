//// VIVA Body Learner Demo
////
//// Demonstrates VIVA learning to use the Arduino body in real-time.
//// Run with: gleam run -m viva/hardware/learner_demo

import gleam/erlang/process.{type Subject}
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{None, Some}
import viva/hardware/body.{type Message, type Sensation}
import viva/hardware/learner.{type Learner}

// =============================================================================
// DEMO RUNNER
// =============================================================================

pub fn main() {
  io.println("╔════════════════════════════════════════════╗")
  io.println("║   VIVA - Learning to use the body (Gleam)  ║")
  io.println("╚════════════════════════════════════════════╝")
  io.println("")

  // Connect to the body
  let device = "/dev/ttyUSB0"
  io.println("Connecting to Arduino at " <> device <> "...")

  case body.start(device) {
    Error(e) -> {
      io.println("Error connecting: " <> debug_error(e))
      io.println("")
      io.println("Tip: Check if Arduino is connected.")
      io.println("     Use 'ls /dev/ttyUSB*' to find the port.")
    }
    Ok(body_actor) -> {
      io.println("Connected!")
      io.println("")
      io.println("VIVA will explore the body and learn...")
      io.println("Interact! Touch pins, change ambient light.")
      io.println("")

      // Create learner
      let l = learner.new()

      // Wait for serial to open and Arduino to stabilize
      process.sleep(3000)

      // Run learning loop
      run_learning_loop(body_actor, l, 0)
    }
  }
}

fn run_learning_loop(body_actor: Subject(Message), l: Learner, iteration: Int) {
  // Iteration limit (for demo)
  case iteration >= 1000 {
    True -> {
      io.println("")
      io.println("")
      show_learning_summary(l)
      body.stop(body_actor)
      io.println("")
      io.println("VIVA went to sleep, but will remember!")
    }
    False -> {
      // Get state before
      let before = case body.get_sensation(body_actor) {
        Some(s) -> s
        None -> body.Sensation(light: 500, noise: 500, touch: False)
      }

      // Choose and execute action
      let #(action_name, command) = learner.explore(l)
      body.send(body_actor, command)

      // Wait for effect
      process.sleep(150)

      // Get state after
      let after = case body.get_sensation(body_actor) {
        Some(s) -> s
        None -> body.Sensation(light: 500, noise: 500, touch: False)
      }

      // Learn
      let new_learner = learner.learn(l, action_name, before, after)

      // Show progress
      let delta_light = after.light - before.light
      show_progress(iteration, action_name, after, delta_light, new_learner.curiosity)

      // Small pause
      process.sleep(100)

      // Continue
      run_learning_loop(body_actor, new_learner, iteration + 1)
    }
  }
}

fn show_progress(
  iteration: Int,
  action_name: String,
  sensation: Sensation,
  delta_light: Int,
  curiosity: Float,
) {
  let iter_str = int.to_string(iteration) |> pad_left(4, " ")
  let action_str = pad_right(action_name, 12, " ")
  let light_str = int.to_string(sensation.light) |> pad_left(4, " ")
  let delta_str = format_delta(delta_light)
  let touch_str = case sensation.touch {
    True -> "1"
    False -> "0"
  }
  let curiosity_bar = curiosity_to_bar(curiosity)

  io.print("\r[" <> iter_str <> "] " <> action_str <> " | ")
  io.print("Light: " <> light_str <> " (" <> delta_str <> ") | ")
  io.print("Touch: " <> touch_str <> " | ")
  io.print("Curiosity: " <> curiosity_bar)
}

fn show_learning_summary(l: Learner) {
  io.println("What VIVA learned about the body:")
  io.println("")

  learner.knowledge_summary(l)
  |> list.each(fn(pair) {
    let #(name, desc) = pair
    let name_str = pad_right(name, 12, " ")
    io.println("  " <> name_str <> " -> " <> desc)
  })

  io.println("")

  // Stats
  let stats = learner.stats(l)
  io.println("Stats:")
  io.println("  Tick: " <> dict_get(stats, "tick", "?"))
  io.println("  Curiosity: " <> dict_get(stats, "curiosity", "?"))
  io.println("  Memories: " <> dict_get(stats, "memories", "?"))
  io.println("  Awake: " <> dict_get(stats, "awake", "?"))
  io.println("  Islands: " <> dict_get(stats, "islands", "?"))
}

// =============================================================================
// HELPERS
// =============================================================================

fn pad_left(s: String, len: Int, char: String) -> String {
  let s_len = string_length(s)
  case s_len >= len {
    True -> s
    False -> string_repeat(char, len - s_len) <> s
  }
}

fn pad_right(s: String, len: Int, char: String) -> String {
  let s_len = string_length(s)
  case s_len >= len {
    True -> s
    False -> s <> string_repeat(char, len - s_len)
  }
}

fn format_delta(d: Int) -> String {
  let s = int.to_string(int.absolute_value(d))
  let sign = case d >= 0 {
    True -> "+"
    False -> "-"
  }
  sign <> pad_left(s, 3, " ")
}

fn curiosity_to_bar(c: Float) -> String {
  let n = float_truncate(c *. 10.0)
  string_repeat("█", n) <> string_repeat(" ", 10 - n)
}

fn dict_get(d: dict.Dict(String, String), key: String, default: String) -> String {
  case dict.get(d, key) {
    Ok(v) -> v
    Error(_) -> default
  }
}

fn string_repeat(s: String, n: Int) -> String {
  case n <= 0 {
    True -> ""
    False -> s <> string_repeat(s, n - 1)
  }
}

fn debug_error(_e: actor.StartError) -> String {
  "StartError"
}

// External imports
import gleam/dict
import gleam/otp/actor

@external(erlang, "string", "length")
fn string_length(s: String) -> Int

@external(erlang, "erlang", "trunc")
fn float_truncate(f: Float) -> Int
