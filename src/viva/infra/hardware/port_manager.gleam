//// VIVA-Link Port Manager - OTP Actor for serial communication
////
//// Manages the Erlang Port that bridges to the Arduino body.
//// Uses Subject/Selector pattern from gleam_otp 1.0+

import gleam/bit_array
import gleam/erlang/process.{type Subject}
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import viva_telemetry/log
import viva/infra/hardware/cobs
import viva/infra/hardware/packet.{type Packet}

// ============================================================================
// Types
// ============================================================================

/// Messages the Port Manager can receive
pub type Message {
  // External API
  Send(packet: Packet)
  Subscribe(subject: Subject(Packet))
  Unsubscribe(subject: Subject(Packet))
  GetStats(reply: Subject(Stats))
  Shutdown

  // Internal (from Erlang Port)
  PortData(data: BitArray)
  PortClosed
  HeartbeatTick
}

/// Port Manager state
pub type State {
  State(
    port: Option(Port),
    device: String,
    baud: Int,
    buffer: BitArray,
    seq: Int,
    subscribers: List(Subject(Packet)),
    stats: Stats,
  )
}

/// Statistics
pub type Stats {
  Stats(
    packets_sent: Int,
    packets_received: Int,
    crc_errors: Int,
    reconnects: Int,
  )
}

/// Opaque Erlang port type
pub type Port

// ============================================================================
// Public API
// ============================================================================

/// Start the port manager actor
pub fn start(
  device: String,
  baud: Int,
) -> Result(Subject(Message), actor.StartError) {
  let state =
    State(
      port: None,
      device: device,
      baud: baud,
      buffer: <<>>,
      seq: 0,
      subscribers: [],
      stats: Stats(0, 0, 0, 0),
    )

  let builder =
    actor.new(state)
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

/// Send a packet to the Arduino body
pub fn send(manager: Subject(Message), pkt: Packet) -> Nil {
  process.send(manager, Send(pkt))
}

/// Subscribe to receive packets from Arduino
pub fn subscribe(manager: Subject(Message), subscriber: Subject(Packet)) -> Nil {
  process.send(manager, Subscribe(subscriber))
}

/// Unsubscribe from packets
pub fn unsubscribe(
  manager: Subject(Message),
  subscriber: Subject(Packet),
) -> Nil {
  process.send(manager, Unsubscribe(subscriber))
}

/// Get current statistics (synchronous)
pub fn get_stats(manager: Subject(Message)) -> Stats {
  process.call(manager, 1000, fn(reply) { GetStats(reply) })
}

/// Gracefully shutdown
pub fn shutdown(manager: Subject(Message)) -> Nil {
  process.send(manager, Shutdown)
}

// ============================================================================
// Actor Implementation
// ============================================================================

fn handle_message(state: State, message: Message) -> actor.Next(State, Message) {
  case message {
    // Send packet to Arduino
    Send(pkt) -> {
      case state.port {
        Some(port) -> {
          let encoded = packet.encode(pkt)
          let framed = cobs.encode(encoded)
          let _ = port_command(port, framed)

          let new_stats =
            Stats(..state.stats, packets_sent: state.stats.packets_sent + 1)
          actor.continue(State(..state, stats: new_stats))
        }
        None -> actor.continue(state)
      }
    }

    // Subscribe to packets
    Subscribe(subject) -> {
      let new_subs = [subject, ..state.subscribers]
      actor.continue(State(..state, subscribers: new_subs))
    }

    // Unsubscribe
    Unsubscribe(subject) -> {
      let new_subs = list.filter(state.subscribers, fn(s) { s != subject })
      actor.continue(State(..state, subscribers: new_subs))
    }

    // Get stats
    GetStats(reply) -> {
      process.send(reply, state.stats)
      actor.continue(state)
    }

    // Data from Arduino
    PortData(data) -> {
      let new_buffer = bit_array.append(state.buffer, data)
      let #(remaining, frames) = cobs.extract_frames(new_buffer)

      // Decode and dispatch each frame
      let new_stats =
        list.fold(frames, state.stats, fn(stats, frame) {
          case packet.decode(frame) {
            Ok(pkt) -> {
              // Broadcast to all subscribers
              list.each(state.subscribers, fn(sub) { process.send(sub, pkt) })
              Stats(..stats, packets_received: stats.packets_received + 1)
            }
            Error(_) -> {
              Stats(..stats, crc_errors: stats.crc_errors + 1)
            }
          }
        })

      actor.continue(State(..state, buffer: remaining, stats: new_stats))
    }

    // Port closed unexpectedly
    PortClosed -> {
      log.warning("Port closed, attempting reconnect...", [])
      case reconnect(state) {
        Ok(new_state) -> actor.continue(new_state)
        Error(_) -> actor.continue(state)
      }
    }

    // Heartbeat tick
    HeartbeatTick -> {
      case state.port {
        Some(port) -> {
          let hb = packet.heartbeat(state.seq)
          let encoded = packet.encode(hb)
          let framed = cobs.encode(encoded)
          let _ = port_command(port, framed)

          let new_seq = { state.seq + 1 } % 256
          actor.continue(State(..state, seq: new_seq))
        }
        None -> actor.continue(state)
      }
    }

    // Shutdown
    Shutdown -> {
      case state.port {
        Some(port) -> {
          let _ = close_port(port)
          Nil
        }
        None -> Nil
      }
      actor.stop()
    }
  }
}

fn reconnect(state: State) -> Result(State, String) {
  case open_serial_port(state.device, state.baud) {
    Ok(port) -> {
      log.info("Reconnected", [#("device", state.device)])
      let new_stats =
        Stats(..state.stats, reconnects: state.stats.reconnects + 1)
      Ok(State(..state, port: Some(port), stats: new_stats))
    }
    Error(e) -> Error(e)
  }
}

// ============================================================================
// Erlang FFI
// ============================================================================

/// Open serial port via Erlang port
fn open_serial_port(device: String, baud: Int) -> Result(Port, String) {
  let port = open_port_spawn(device, baud)
  Ok(port)
}

@external(erlang, "viva_hardware_ffi", "open_serial_port")
fn open_port_spawn(device: String, baud: Int) -> Port

@external(erlang, "erlang", "port_command")
fn port_command(port: Port, data: BitArray) -> Bool

@external(erlang, "erlang", "port_close")
fn close_port(port: Port) -> Bool
