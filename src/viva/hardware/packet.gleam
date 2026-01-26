//// VIVA-Link Packet definitions
////
//// All packets exchanged between Soul (Gleam) and Body (Arduino).
//// Binary format: [TYPE:1] [SEQ:1] [PAYLOAD:N] [CRC8:1]

import gleam/bit_array
import gleam/float
import gleam/int
import gleam/option.{type Option, None, Some}
import viva/hardware/crc8

// ============================================================================
// Packet Type IDs
// ============================================================================

pub const type_heartbeat = 0x00

pub const type_sensor_data = 0x01

pub const type_command = 0x02

pub const type_pad_state = 0x10

pub const type_audio_cmd = 0x11

pub const type_ack = 0xFE

pub const type_error = 0xFF

// ============================================================================
// Packet Types
// ============================================================================

/// Heartbeat - bidirectional keep-alive
pub type Heartbeat {
  Heartbeat(seq: Int)
}

/// Sensor data from Arduino body
pub type SensorData {
  SensorData(
    seq: Int,
    temperature: Float,
    light: Int,
    touch: Bool,
    audio_level: Int,
  )
}

/// Command from Soul to Body
pub type Command {
  Command(seq: Int, servo_angle: Int, led_state: Bool, vibration: Int)
}

/// PAD emotional state broadcast
pub type PadState {
  PadState(seq: Int, pleasure: Float, arousal: Float, dominance: Float)
}

/// Audio command for speakers
pub type AudioCommand {
  AudioCommand(
    seq: Int,
    freq_left: Int,
    freq_right: Int,
    duration_ms: Int,
    waveform: Int,
    // 0=sine, 1=square, 2=saw, 3=noise
  )
}

/// Acknowledgment
pub type Ack {
  Ack(seq: Int, acked_seq: Int)
}

/// Error report
pub type ErrorPacket {
  ErrorPacket(seq: Int, error_code: Int)
}

/// Union of all packet types
pub type Packet {
  PacketHeartbeat(Heartbeat)
  PacketSensorData(SensorData)
  PacketCommand(Command)
  PacketPadState(PadState)
  PacketAudioCommand(AudioCommand)
  PacketAck(Ack)
  PacketError(ErrorPacket)
}

// ============================================================================
// Encoding (Gleam -> Wire)
// ============================================================================

/// Encode any packet to binary (with CRC, without COBS)
pub fn encode(packet: Packet) -> BitArray {
  let raw = case packet {
    PacketHeartbeat(p) -> encode_heartbeat(p)
    PacketSensorData(p) -> encode_sensor_data(p)
    PacketCommand(p) -> encode_command(p)
    PacketPadState(p) -> encode_pad_state(p)
    PacketAudioCommand(p) -> encode_audio_command(p)
    PacketAck(p) -> encode_ack(p)
    PacketError(p) -> encode_error(p)
  }
  crc8.append(raw)
}

fn encode_heartbeat(p: Heartbeat) -> BitArray {
  <<type_heartbeat:8, p.seq:8>>
}

fn encode_sensor_data(p: SensorData) -> BitArray {
  let touch_byte = case p.touch {
    True -> 1
    False -> 0
  }
  <<
    type_sensor_data:8,
    p.seq:8,
    p.temperature:float-32-little,
    p.light:16-little,
    touch_byte:8,
    p.audio_level:16-little,
  >>
}

fn encode_command(p: Command) -> BitArray {
  let led = case p.led_state {
    True -> 1
    False -> 0
  }
  <<
    type_command:8,
    p.seq:8,
    p.servo_angle:16-little,
    led:8,
    p.vibration:8,
  >>
}

fn encode_pad_state(p: PadState) -> BitArray {
  <<
    type_pad_state:8,
    p.seq:8,
    p.pleasure:float-32-little,
    p.arousal:float-32-little,
    p.dominance:float-32-little,
  >>
}

fn encode_audio_command(p: AudioCommand) -> BitArray {
  <<
    type_audio_cmd:8,
    p.seq:8,
    p.freq_left:16-little,
    p.freq_right:16-little,
    p.duration_ms:16-little,
    p.waveform:8,
  >>
}

fn encode_ack(p: Ack) -> BitArray {
  <<type_ack:8, p.seq:8, p.acked_seq:8>>
}

fn encode_error(p: ErrorPacket) -> BitArray {
  <<type_error:8, p.seq:8, p.error_code:8>>
}

// ============================================================================
// Decoding (Wire -> Gleam)
// ============================================================================

/// Decode binary to packet (expects CRC already verified/stripped)
pub fn decode(data: BitArray) -> Result(Packet, String) {
  // Verify CRC first
  case crc8.verify(data) {
    False -> Error("CRC mismatch")
    True -> {
      // Strip CRC byte for parsing
      let size = bit_array.byte_size(data)
      case bit_array.slice(data, 0, size - 1) {
        Ok(payload) -> decode_payload(payload)
        _ -> Error("Invalid packet size")
      }
    }
  }
}

fn decode_payload(data: BitArray) -> Result(Packet, String) {
  case data {
    // Heartbeat
    <<t:8, seq:8>> if t == type_heartbeat -> {
      Ok(PacketHeartbeat(Heartbeat(seq)))
    }

    // Sensor Data
    <<
      t:8,
      seq:8,
      temp:float-32-little,
      light:16-little,
      touch:8,
      audio:16-little,
    >>
      if t == type_sensor_data
    -> {
      Ok(
        PacketSensorData(SensorData(
          seq: seq,
          temperature: temp,
          light: light,
          touch: touch == 1,
          audio_level: audio,
        )),
      )
    }

    // Command
    <<t:8, seq:8, angle:16-little-signed, led:8, vib:8>> if t == type_command -> {
      Ok(
        PacketCommand(Command(
          seq: seq,
          servo_angle: angle,
          led_state: led == 1,
          vibration: vib,
        )),
      )
    }

    // PAD State
    <<t:8, seq:8, p:float-32-little, a:float-32-little, d:float-32-little>>
      if t == type_pad_state
    -> {
      Ok(
        PacketPadState(PadState(seq: seq, pleasure: p, arousal: a, dominance: d)),
      )
    }

    // Audio Command
    <<t:8, seq:8, fl:16-little, fr:16-little, dur:16-little, wave:8>>
      if t == type_audio_cmd
    -> {
      Ok(
        PacketAudioCommand(AudioCommand(
          seq: seq,
          freq_left: fl,
          freq_right: fr,
          duration_ms: dur,
          waveform: wave,
        )),
      )
    }

    // Ack
    <<t:8, seq:8, acked:8>> if t == type_ack -> {
      Ok(PacketAck(Ack(seq, acked)))
    }

    // Error
    <<t:8, seq:8, code:8>> if t == type_error -> {
      Ok(PacketError(ErrorPacket(seq, code)))
    }

    _ -> Error("Unknown packet type")
  }
}

// ============================================================================
// Helpers
// ============================================================================

/// Get sequence number from any packet
pub fn get_seq(packet: Packet) -> Int {
  case packet {
    PacketHeartbeat(p) -> p.seq
    PacketSensorData(p) -> p.seq
    PacketCommand(p) -> p.seq
    PacketPadState(p) -> p.seq
    PacketAudioCommand(p) -> p.seq
    PacketAck(p) -> p.seq
    PacketError(p) -> p.seq
  }
}

/// Create a new heartbeat with given sequence
pub fn heartbeat(seq: Int) -> Packet {
  PacketHeartbeat(Heartbeat(seq))
}

/// Create audio command for binaural beat
pub fn binaural_beat(
  seq: Int,
  base_freq: Int,
  beat_freq: Int,
  duration: Int,
) -> Packet {
  PacketAudioCommand(AudioCommand(
    seq: seq,
    freq_left: base_freq,
    freq_right: base_freq + beat_freq,
    duration_ms: duration,
    waveform: 0,
    // sine
  ))
}

/// Create PAD state packet
pub fn pad_state(
  seq: Int,
  pleasure: Float,
  arousal: Float,
  dominance: Float,
) -> Packet {
  PacketPadState(PadState(seq:, pleasure:, arousal:, dominance:))
}

/// Describe a packet in human-readable form
pub fn describe(packet: Packet) -> String {
  case packet {
    PacketHeartbeat(p) -> "Heartbeat(seq=" <> int.to_string(p.seq) <> ")"
    PacketSensorData(p) -> {
      "SensorData(seq="
      <> int.to_string(p.seq)
      <> ", temp="
      <> float.to_string(p.temperature)
      <> ", light="
      <> int.to_string(p.light)
      <> ", touch="
      <> bool_to_string(p.touch)
      <> ", audio="
      <> int.to_string(p.audio_level)
      <> ")"
    }
    PacketCommand(p) -> {
      "Command(seq="
      <> int.to_string(p.seq)
      <> ", servo="
      <> int.to_string(p.servo_angle)
      <> ", led="
      <> bool_to_string(p.led_state)
      <> ", vib="
      <> int.to_string(p.vibration)
      <> ")"
    }
    PacketPadState(p) -> {
      "PadState(seq="
      <> int.to_string(p.seq)
      <> ", P="
      <> float.to_string(p.pleasure)
      <> ", A="
      <> float.to_string(p.arousal)
      <> ", D="
      <> float.to_string(p.dominance)
      <> ")"
    }
    PacketAudioCommand(p) -> {
      "AudioCmd(seq="
      <> int.to_string(p.seq)
      <> ", L="
      <> int.to_string(p.freq_left)
      <> "Hz"
      <> ", R="
      <> int.to_string(p.freq_right)
      <> "Hz"
      <> ", dur="
      <> int.to_string(p.duration_ms)
      <> "ms)"
    }
    PacketAck(p) -> {
      "Ack(seq="
      <> int.to_string(p.seq)
      <> ", acked="
      <> int.to_string(p.acked_seq)
      <> ")"
    }
    PacketError(p) -> {
      "Error(seq="
      <> int.to_string(p.seq)
      <> ", code="
      <> int.to_string(p.error_code)
      <> ")"
    }
  }
}

fn bool_to_string(b: Bool) -> String {
  case b {
    True -> "true"
    False -> "false"
  }
}
