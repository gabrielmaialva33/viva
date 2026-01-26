//// VIVA Body - Pure Gleam interface to the physical body (Arduino)
////
//// The Soul senses through the Body and expresses through it.
//// No Python - pure Gleam/Erlang serial communication.

import gleam/erlang/process.{type Subject}
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import gleam/string
import viva/serial.{type Serial}

// ============================================================================
// Types
// ============================================================================

/// Sensations from the body
pub type Sensation {
  Sensation(
    light: Int,
    // 0-1023: ambient light
    noise: Int,
    // 0-1023: electrical noise (primitive touch)
    touch: Bool,
    // Button/touch pressed
  )
}

/// Commands to the body
pub type BodyCommand {
  SetLed(red: Int, green: Int)
  PlayTone(pin: Int, freq: Int, duration: Int)
  StopAll
}

/// Actor messages
pub type Message {
  // External
  SendCommand(BodyCommand)
  GetSensation(reply: Subject(Option(Sensation)))
  Subscribe(Subject(Sensation))
  Shutdown

  // Internal
  InitSerial
  SerialData(String)
  PollSerial
  SetSelf(Subject(Message))
}

/// Body state
pub type State {
  State(
    serial: Option(Serial),
    device: String,
    last_sensation: Option(Sensation),
    subscribers: List(Subject(Sensation)),
    self_subject: Option(Subject(Message)),
  )
}

// ============================================================================
// Public API
// ============================================================================

/// Start connection to the body
pub fn start(device: String) -> Result(Subject(Message), actor.StartError) {
  // Don't open serial here - it must be opened inside the actor process
  let state =
    State(
      serial: None,
      device: device,
      last_sensation: None,
      subscribers: [],
      self_subject: None,
    )

  let builder =
    actor.new(state)
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> {
      let subject = started.data
      // Send init message to open serial inside actor
      process.send(subject, SetSelf(subject))
      process.send(subject, InitSerial)
      Ok(subject)
    }
    Error(e) -> Error(e)
  }
}

/// Send command to the body
pub fn send(body: Subject(Message), cmd: BodyCommand) -> Nil {
  process.send(body, SendCommand(cmd))
}

/// Set LED color (0-255)
pub fn set_led(body: Subject(Message), red: Int, green: Int) -> Nil {
  send(body, SetLed(red, green))
}

/// Play tone on speaker (pin 9 or 10)
pub fn play_tone(
  body: Subject(Message),
  pin: Int,
  freq: Int,
  duration: Int,
) -> Nil {
  send(body, PlayTone(pin, freq, duration))
}

/// Stop all outputs
pub fn stop(body: Subject(Message)) -> Nil {
  send(body, StopAll)
}

/// Get last sensation
pub fn get_sensation(body: Subject(Message)) -> Option(Sensation) {
  process.call(body, 1000, fn(reply) { GetSensation(reply) })
}

/// Subscribe to receive sensations
pub fn subscribe(body: Subject(Message), sub: Subject(Sensation)) -> Nil {
  process.send(body, Subscribe(sub))
}

// ============================================================================
// Actor
// ============================================================================

fn handle_message(state: State, msg: Message) -> actor.Next(State, Message) {
  case msg {
    SendCommand(cmd) -> {
      case state.serial {
        Some(ser) -> {
          let data = encode_command(cmd)
          let _ = serial.write(ser, data)
          Nil
        }
        None -> Nil
      }
      actor.continue(state)
    }

    GetSensation(reply) -> {
      process.send(reply, state.last_sensation)
      actor.continue(state)
    }

    Subscribe(sub) -> {
      actor.continue(State(..state, subscribers: [sub, ..state.subscribers]))
    }

    SerialData(line) -> {
      case parse_sensation(line) {
        Some(sensation) -> {
          // Notify subscribers
          list.each(state.subscribers, fn(sub) { process.send(sub, sensation) })
          actor.continue(State(..state, last_sensation: Some(sensation)))
        }
        None -> actor.continue(state)
      }
    }

    PollSerial -> {
      // Read line from serial (non-blocking with short timeout)
      case state.serial, state.self_subject {
        Some(ser), Some(self) -> {
          case serial.read_line(ser, 100) {
            Some(line) -> {
              // Got data, process it
              case parse_sensation(line) {
                Some(sensation) -> {
                  // Notify subscribers
                  list.each(state.subscribers, fn(sub) {
                    process.send(sub, sensation)
                  })
                  // Schedule next poll
                  let _ = process.send_after(self, 50, PollSerial)
                  actor.continue(
                    State(..state, last_sensation: Some(sensation)),
                  )
                }
                None -> {
                  // Invalid data, keep polling
                  let _ = process.send_after(self, 50, PollSerial)
                  actor.continue(state)
                }
              }
            }
            None -> {
              // No data, keep polling
              let _ = process.send_after(self, 50, PollSerial)
              actor.continue(state)
            }
          }
        }
        _, _ -> actor.continue(state)
      }
    }

    SetSelf(subject) -> {
      actor.continue(State(..state, self_subject: Some(subject)))
    }

    InitSerial -> {
      // Open serial inside this actor's process
      case serial.open_arduino(state.device) {
        Error(_e) -> {
          io.println("Body: Failed to open " <> state.device)
          actor.continue(state)
        }
        Ok(ser) -> {
          io.println("Body: Connected to " <> state.device)
          // Start polling for data
          case state.self_subject {
            Some(self) -> {
              let _ = process.send_after(self, 100, PollSerial)
              Nil
            }
            None -> Nil
          }
          actor.continue(State(..state, serial: Some(ser)))
        }
      }
    }

    Shutdown -> {
      case state.serial {
        Some(ser) -> {
          serial.close(ser)
          Nil
        }
        None -> Nil
      }
      actor.stop()
    }
  }
}

// ============================================================================
// Protocol
// ============================================================================

fn encode_command(cmd: BodyCommand) -> BitArray {
  case cmd {
    SetLed(r, g) -> {
      <<0x4C, clamp(r, 0, 255), clamp(g, 0, 255)>>
      // 'L' r g
    }
    PlayTone(pin, freq, dur) -> {
      let fh = freq / 256
      let fl = freq % 256
      let dh = dur / 256
      let dl = dur % 256
      <<0x53, pin, fh, fl, dh, dl>>
      // 'S' pin fH fL dH dL
    }
    StopAll -> {
      <<0x58>>
      // 'X'
    }
  }
}

fn parse_sensation(line: String) -> Option(Sensation) {
  // Format: "S,light,noise,touch"
  case string.starts_with(line, "S,") {
    False -> None
    True -> {
      let parts = string.split(string.drop_start(line, 2), ",")
      case parts {
        [light_s, noise_s, touch_s] -> {
          case int.parse(light_s), int.parse(noise_s), int.parse(touch_s) {
            Ok(light), Ok(noise), Ok(touch) -> {
              Some(Sensation(light: light, noise: noise, touch: touch == 1))
            }
            _, _, _ -> None
          }
        }
        _ -> None
      }
    }
  }
}

fn clamp(value: Int, min: Int, max: Int) -> Int {
  case value < min {
    True -> min
    False ->
      case value > max {
        True -> max
        False -> value
      }
  }
}
// No more FFI - using pure Gleam viva/serial library!
