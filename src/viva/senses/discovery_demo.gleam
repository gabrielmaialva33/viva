//// VIVA Sense Discovery Demo
////
//// VIVA receives raw signals from Arduino and Windows
//// without knowing what they are. She discovers her senses!
////
//// Run: gleam run -m viva/senses/discovery_demo

import gleam/dict
import gleam/erlang/process
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{None, Some}
import gleam/string
import viva/hardware/body
import viva/senses/sense_learner.{type ChannelKnowledge}

// ============================================================================
// Main
// ============================================================================

pub fn main() {
  io.println("╔════════════════════════════════════════════════════╗")
  io.println("║   VIVA SENSE DISCOVERY                             ║")
  io.println("║   She doesn't know what senses are... yet          ║")
  io.println("╚════════════════════════════════════════════════════╝")
  io.println("")

  // Start sense learner
  case sense_learner.start() {
    Error(_) -> {
      io.println("Failed to start sense learner")
    }
    Ok(learner) -> {
      io.println("Learner: I'm awake... what is this world?")
      io.println("")

      // Try to connect Arduino
      let arduino = case body.start("/dev/ttyUSB0") {
        Ok(b) -> {
          io.println("Connected to... something (Arduino)")
          Some(b)
        }
        Error(_) -> {
          io.println("No Arduino - using simulated signals")
          None
        }
      }

      // Wait for systems to initialize
      process.sleep(2000)

      // Run discovery loop
      io.println("")
      io.println("Starting discovery... VIVA will learn what her senses are")
      io.println("")

      discovery_loop(learner, arduino, 0)
    }
  }
}

// ============================================================================
// Discovery Loop
// ============================================================================

fn discovery_loop(
  learner: process.Subject(sense_learner.Message),
  arduino: option.Option(process.Subject(body.Message)),
  tick: Int,
) -> Nil {
  // Stop after 200 ticks
  case tick >= 200 {
    True -> {
      io.println("")
      io.println("")
      show_discoveries(learner)
    }
    False -> {
      // Get sensor data
      let sensor_data = case arduino {
        Some(body_actor) -> get_arduino_data(body_actor)
        None -> get_simulated_data(tick)
      }

      // Get Windows data (webcam activity simulation for now)
      let windows_data = get_windows_data(tick)

      // Combine all data
      let all_data = dict.merge(sensor_data, windows_data)

      // Feed to learner - VIVA doesn't know what any of this is!
      sense_learner.feed_batch(learner, all_data)

      // Show progress
      show_tick(tick, all_data)

      // Small pause
      process.sleep(200)

      // Continue
      discovery_loop(learner, arduino, tick + 1)
    }
  }
}

// ============================================================================
// Data Sources
// ============================================================================

fn get_arduino_data(
  body_actor: process.Subject(body.Message),
) -> dict.Dict(String, Float) {
  case body.get_sensation(body_actor) {
    Some(sensation) -> {
      dict.new()
      |> dict.insert("chan_0", int.to_float(sensation.light) /. 1023.0)
      |> dict.insert("chan_1", int.to_float(sensation.noise) /. 1023.0)
      |> dict.insert("chan_2", case sensation.touch {
        True -> 1.0
        False -> 0.0
      })
    }
    None -> {
      // No data yet
      dict.new()
    }
  }
}

fn get_simulated_data(tick: Int) -> dict.Dict(String, Float) {
  // Simulate Arduino-like sensors
  let t = int.to_float(tick)

  // chan_0: "light" - varies with sine wave (like day/night)
  let light = 0.5 +. 0.3 *. float_sin(t /. 20.0)

  // chan_1: "noise" - random with occasional spikes
  let noise = 0.3 +. 0.2 *. random_float()

  // chan_2: "touch" - occasional binary events
  let touch = case random_float() <. 0.05 {
    True -> 1.0
    False -> 0.0
  }

  dict.new()
  |> dict.insert("chan_0", light)
  |> dict.insert("chan_1", noise)
  |> dict.insert("chan_2", touch)
}

fn get_windows_data(tick: Int) -> dict.Dict(String, Float) {
  // Simulate webcam/mic activity
  // In real version, this would capture actual data

  let t = int.to_float(tick)

  // chan_3: "visual_activity" - how much the image changes
  let visual = 0.2 +. 0.1 *. random_float()

  // chan_4: "audio_level" - mic volume
  let audio = 0.1 +. 0.3 *. random_float()

  dict.new()
  |> dict.insert("chan_3", visual)
  |> dict.insert("chan_4", audio)
}

// ============================================================================
// Display
// ============================================================================

fn show_tick(tick: Int, data: dict.Dict(String, Float)) -> Nil {
  let tick_str = int.to_string(tick) |> pad_left(3, " ")

  // Build channel display
  let channel_strs =
    dict.to_list(data)
    |> list.sort(fn(a, b) { string.compare(a.0, b.0) })
    |> list.map(fn(pair) {
      let #(id, value) = pair
      let bar = value_to_bar(value, 5)
      id <> ":" <> bar
    })

  let channels_display = string.join(channel_strs, " ")

  io.print("\r[" <> tick_str <> "] " <> channels_display <> "  ")
}

fn show_discoveries(learner: process.Subject(sense_learner.Message)) -> Nil {
  io.println("╔════════════════════════════════════════════════════╗")
  io.println("║   WHAT VIVA DISCOVERED ABOUT HER SENSES            ║")
  io.println("╚════════════════════════════════════════════════════╝")
  io.println("")

  let knowledge = sense_learner.what_do_you_know(learner)

  list.each(knowledge, fn(k: ChannelKnowledge) {
    let name = case k.learned_name {
      Some(n) -> "\"" <> n <> "\""
      None -> "(unnamed)"
    }

    let status = case k.is_active {
      True -> "ACTIVE"
      False -> "quiet"
    }

    io.println("  " <> k.id <> " -> " <> name <> " [" <> status <> "]")
  })

  io.println("")
  io.println(
    "VIVA discovered " <> int.to_string(list.length(knowledge)) <> " senses!",
  )
}

fn value_to_bar(value: Float, width: Int) -> String {
  let filled = float.truncate(value *. int.to_float(width))
  let empty = width - filled
  string.repeat("█", filled) <> string.repeat("░", empty)
}

fn pad_left(s: String, len: Int, char: String) -> String {
  let s_len = string.length(s)
  case s_len >= len {
    True -> s
    False -> string.repeat(char, len - s_len) <> s
  }
}

// ============================================================================
// FFI
// ============================================================================

@external(erlang, "rand", "uniform")
fn random_float() -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float
