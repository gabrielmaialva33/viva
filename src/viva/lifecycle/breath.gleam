//// Breath - The Autonomic Loop ("The Pulse")
////
//// This module is the "Heart" of VIVA. It drives the time (Aion),
//// triggers the Soul (Consciousness), and manages the lifecycle.

import gleam/erlang/process.{type Subject, type Timer}
import gleam/int
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import viva/soul/soul
import viva_telemetry/log

// =============================================================================
// CONSTANTS
// =============================================================================

/// Heartbeat interval in milliseconds (10Hz = 100ms)
pub const heartbeat_interval = 100

// =============================================================================
// TYPES
// =============================================================================

/// Breath State
pub type BreathState {
  BreathState(
    /// Self reference (to schedule ticks)
    self: Option(Subject(BreathMsg)),
    /// Reference to the Soul actor
    soul: Subject(soul.Message),
    /// Current tick count (local to Breath)
    tick_count: Int,
    /// Timer reference for next tick
    timer: Option(Timer),
  )
}

/// Messages for the Breath actor
pub type BreathMsg {
  /// Initialize self reference
  SetSelf(Subject(BreathMsg))
  /// The Heartbeat - drives time forward
  Tick
  /// Teach the VIVA something (Console/User Input)
  Teach(instruction: String)
  /// Stop breathing (shutdown)
  Stop
}

// =============================================================================
// API
// =============================================================================

/// Start the Breath (and the Soul)
pub fn start() -> Result(Subject(BreathMsg), actor.StartError) {
  // 1. Ignite the Soul
  case soul.start(1) {
    Ok(soul_subject) -> {
      // 2. Prepare Breath State
      let state =
        BreathState(self: None, soul: soul_subject, tick_count: 0, timer: None)

      // 3. Start Actor using Builder Pattern
      let builder =
        actor.new(state)
        |> actor.on_message(handle_message)

      case actor.start(builder) {
        Ok(started) -> {
          let breath_subject = started.data
          // 4. Inject Self and Start Heartbeat
          process.send(breath_subject, SetSelf(breath_subject))
          process.send(breath_subject, Tick)
          Ok(breath_subject)
        }
        Error(e) -> Error(e)
      }
    }
    Error(e) -> Error(e)
  }
}

/// Send a teaching instruction
pub fn teach(breath: Subject(BreathMsg), instruction: String) {
  process.send(breath, Teach(instruction))
}

/// Stop the Breath
pub fn stop(breath: Subject(BreathMsg)) {
  process.send(breath, Stop)
}

// =============================================================================
// INTERNAL
// =============================================================================

fn handle_message(
  state: BreathState,
  msg: BreathMsg,
) -> actor.Next(BreathState, BreathMsg) {
  case msg {
    SetSelf(subject) -> {
      actor.continue(BreathState(..state, self: Some(subject)))
    }

    Tick -> {
      // 1. Tick the Soul (0.1s delta)
      soul.tick(state.soul, int.to_float(heartbeat_interval) /. 1000.0)

      // 2. Schedule next heartbeat
      let timer = case state.self {
        Some(self) -> {
          Some(process.send_after(self, heartbeat_interval, Tick))
        }
        None -> None
      }

      // Log occasionally (every 5 seconds = 50 ticks)
      case state.tick_count % 50 {
        0 -> log.info("Pulse: " <> int.to_string(state.tick_count), [])
        _ -> Nil
      }

      actor.continue(
        BreathState(..state, tick_count: state.tick_count + 1, timer: timer),
      )
    }

    Teach(instruction) -> {
      log.info("Teaching: " <> instruction, [])

      soul.receive_sensation(
        state.soul,
        1000,
        // light
        500,
        // sound
        False,
        // touch
        Some("Teacher: " <> instruction),
        // entity
      )

      actor.continue(state)
    }

    Stop -> {
      log.info("Breath stopped.", [])
      soul.kill(state.soul)
      actor.stop()
    }
  }
}
