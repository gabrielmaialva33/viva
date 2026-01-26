//// VIVA Perceiver - Autonomous sensory perception
////
//// VIVA decides WHEN to look, WHAT to ask, and HOW to feel about it.
//// This is her eyes and ears - she controls them, not us.

import gleam/erlang/process.{type Subject}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import gleam/string

// ============================================================================
// Types
// ============================================================================

/// What VIVA perceives
pub type Percept {
  Percept(
    visual: Option(String),
    // What she saw (description)
    auditory: Option(String),
    // What she heard (transcription)
    timestamp: Int,
    salience: Float,
    // How important/interesting (0-1)
  )
}

/// VIVA's internal drives that trigger perception
pub type Drive {
  Curiosity(Float)
  // Want to explore (0-1)
  Vigilance(Float)
  // Watching for changes (0-1)
  Boredom(Float)
  // Nothing happening, look around (0-1)
  Social(Float)
  // Looking for interaction (0-1)
}

/// Perceiver state
pub type State {
  State(
    // Hardware config
    camera: String,
    microphone: String,
    temp_dir: String,
    // Internal state
    last_percept: Option(Percept),
    curiosity: Float,
    vigilance: Float,
    tick: Int,
    // Self-reference for timers
    self_subject: Option(Subject(Message)),
  )
}

/// Actor messages
pub type Message {
  // External triggers
  Look
  // Force a look
  Listen(Int)
  // Force listen (duration ms)
  Ask(String)

  // Ask about what she sees
  // Internal drives
  Tick
  // Periodic check - should I look?
  PerceptionComplete(Percept)

  // Got a percept back
  // Queries
  GetLastPercept(Subject(Option(Percept)))
  GetCuriosity(Subject(Float))

  // Control
  SetSelf(Subject(Message))
  SetCuriosity(Float)
  Shutdown
}

// ============================================================================
// Public API
// ============================================================================

/// Start the perceiver - VIVA's autonomous sensory system
pub fn start() -> Result(Subject(Message), actor.StartError) {
  let state =
    State(
      camera: "Logi C270 HD WebCam",
      microphone: "Microfone (Realtek USB Audio)",
      temp_dir: "/mnt/h/Temp",
      last_percept: None,
      curiosity: 0.8,
      // Start curious!
      vigilance: 0.5,
      tick: 0,
      self_subject: None,
    )

  let builder =
    actor.new(state)
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> {
      let subject = started.data
      process.send(subject, SetSelf(subject))
      // Start the autonomous perception loop
      process.send(subject, Tick)
      io.println("Perceiver: VIVA's senses are awake")
      Ok(subject)
    }
    Error(e) -> Error(e)
  }
}

/// Poke VIVA's curiosity
pub fn poke_curiosity(perceiver: Subject(Message), amount: Float) -> Nil {
  let new_curiosity = float.min(1.0, amount)
  process.send(perceiver, SetCuriosity(new_curiosity))
}

/// Ask VIVA to look now
pub fn look_now(perceiver: Subject(Message)) -> Nil {
  process.send(perceiver, Look)
}

/// Ask VIVA a question about what she sees
pub fn ask(perceiver: Subject(Message), question: String) -> Nil {
  process.send(perceiver, Ask(question))
}

/// Get what VIVA last perceived
pub fn last_percept(perceiver: Subject(Message)) -> Option(Percept) {
  process.call(perceiver, 5000, GetLastPercept)
}

// ============================================================================
// Actor Logic - VIVA's autonomous perception
// ============================================================================

fn handle_message(state: State, msg: Message) -> actor.Next(State, Message) {
  case msg {
    Tick -> {
      // VIVA decides: should I look?
      let should_look = decide_to_look(state)

      case should_look, state.self_subject {
        True, Some(self) -> {
          // Yes! Capture and analyze
          io.println("Perceiver: *curious* looking around...")
          let percept = do_perception(state, None)
          process.send(self, PerceptionComplete(percept))
          Nil
        }
        _, _ -> Nil
      }

      // Schedule next tick (every 5 seconds)
      case state.self_subject {
        Some(self) -> {
          let _ = process.send_after(self, 5000, Tick)
          Nil
        }
        None -> Nil
      }

      // Curiosity slowly decays
      let new_curiosity = float.max(0.1, state.curiosity *. 0.99)
      actor.continue(
        State(..state, tick: state.tick + 1, curiosity: new_curiosity),
      )
    }

    Look -> {
      io.println("Perceiver: *looking*")
      let percept = do_perception(state, None)

      case state.self_subject {
        Some(self) -> process.send(self, PerceptionComplete(percept))
        None -> Nil
      }

      actor.continue(state)
    }

    Ask(question) -> {
      io.println("Perceiver: *looking to answer* " <> question)
      let percept = do_perception(state, Some(question))

      case state.self_subject {
        Some(self) -> process.send(self, PerceptionComplete(percept))
        None -> Nil
      }

      actor.continue(state)
    }

    Listen(duration_ms) -> {
      io.println(
        "Perceiver: *listening for " <> int.to_string(duration_ms) <> "ms*",
      )
      // TODO: Implement audio capture + Whisper
      actor.continue(state)
    }

    PerceptionComplete(percept) -> {
      // VIVA received a percept - react!
      case percept.visual {
        Some(desc) -> {
          io.println("Perceiver: I see: " <> truncate(desc, 100))

          // Interesting things boost curiosity
          let curiosity_boost = percept.salience *. 0.3
          let new_curiosity = float.min(1.0, state.curiosity +. curiosity_boost)

          actor.continue(
            State(
              ..state,
              last_percept: Some(percept),
              curiosity: new_curiosity,
            ),
          )
        }
        None -> actor.continue(State(..state, last_percept: Some(percept)))
      }
    }

    GetLastPercept(reply) -> {
      process.send(reply, state.last_percept)
      actor.continue(state)
    }

    GetCuriosity(reply) -> {
      process.send(reply, state.curiosity)
      actor.continue(state)
    }

    SetSelf(subject) -> {
      actor.continue(State(..state, self_subject: Some(subject)))
    }

    SetCuriosity(c) -> {
      io.println("Perceiver: curiosity now " <> float.to_string(c))
      actor.continue(State(..state, curiosity: c))
    }

    Shutdown -> {
      io.println("Perceiver: closing eyes...")
      actor.stop()
    }
  }
}

// ============================================================================
// Decision Logic - VIVA decides when to look
// ============================================================================

fn decide_to_look(state: State) -> Bool {
  // Random factor
  let random = random_float()

  // Look if:
  // 1. High curiosity (> 0.7) and random chance
  // 2. Haven't looked in a while (tick > 10 since last)
  // 3. Random exploration (5% chance always)

  let curious_look = state.curiosity >. 0.7 && random <. state.curiosity
  let bored_look = state.tick > 10 && random <. 0.3
  let random_look = random <. 0.05

  curious_look || bored_look || random_look
}

// ============================================================================
// Perception Execution
// ============================================================================

fn do_perception(state: State, question: Option(String)) -> Percept {
  let timestamp = erlang_now_ms()

  // Build capture command
  let capture_cmd =
    string.concat([
      "powershell.exe -Command \"cd $env:TEMP; ", "ffmpeg -f dshow -i video='",
      state.camera, "' ", "-frames:v 1 -update 1 -y viva_percept.jpg\" 2>&1",
    ])

  // Capture image
  let _ = run_shell(capture_cmd)

  // Copy to Linux
  let copy_cmd =
    "cp "
    <> state.temp_dir
    <> "/viva_percept.jpg /tmp/viva_percept.jpg 2>/dev/null"
  let _ = run_shell(copy_cmd)

  // Analyze with NVIDIA
  let prompt = case question {
    Some(q) -> q
    None ->
      "Describe what you see briefly. Focus on interesting or unusual things."
  }

  let analyze_cmd =
    string.concat([
      "source ~/.zshrc && python3 /home/mrootx/viva_gleam/scripts/viva_see.py ",
      "/tmp/viva_percept.jpg '",
      prompt,
      "' 2>/dev/null",
    ])

  let result = run_shell(analyze_cmd)

  // Parse result
  let description = case string.split(result, "\n") {
    [_, ..rest] -> string.trim(string.join(rest, " "))
    _ -> string.trim(result)
  }

  // Calculate salience (how interesting)
  let salience = calculate_salience(description)

  Percept(
    visual: Some(description),
    auditory: None,
    timestamp: timestamp,
    salience: salience,
  )
}

fn calculate_salience(description: String) -> Float {
  let lower = string.lowercase(description)

  // Interesting things
  let person_bonus = case
    string.contains(lower, "person") || string.contains(lower, "face")
  {
    True -> 0.4
    False -> 0.0
  }

  let movement_bonus = case
    string.contains(lower, "moving") || string.contains(lower, "motion")
  {
    True -> 0.3
    False -> 0.0
  }

  let unusual_bonus = case
    string.contains(lower, "unusual") || string.contains(lower, "strange")
  {
    True -> 0.3
    False -> 0.0
  }

  // Base salience
  let base = 0.3

  float.min(1.0, base +. person_bonus +. movement_bonus +. unusual_bonus)
}

// ============================================================================
// Helpers
// ============================================================================

fn truncate(s: String, max: Int) -> String {
  case string.length(s) > max {
    True -> string.slice(s, 0, max) <> "..."
    False -> s
  }
}

// ============================================================================
// FFI
// ============================================================================

@external(erlang, "viva_senses_ffi", "run_shell")
fn run_shell(cmd: String) -> String

@external(erlang, "rand", "uniform")
fn random_float() -> Float

@external(erlang, "erlang", "system_time")
fn erlang_system_time(unit: SystemTimeUnit) -> Int

type SystemTimeUnit {
  Millisecond
}

fn erlang_now_ms() -> Int {
  erlang_system_time(Millisecond)
}
