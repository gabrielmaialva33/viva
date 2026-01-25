//// Tests for VIVA-Link hardware modules

import gleam/bit_array
import gleam/list
import gleeunit/should
import viva/hardware/cobs
import viva/hardware/crc8
import viva/hardware/packet

// Erlang FFI for binary to list
@external(erlang, "erlang", "binary_to_list")
fn binary_to_list(data: BitArray) -> List(Int)

// ============================================================================
// COBS Tests
// ============================================================================

pub fn cobs_encode_simple_test() {
  // Data without zeros
  let data = <<1, 2, 3, 4, 5>>
  let encoded = cobs.encode(data)

  // Should have overhead byte + data + delimiter
  bit_array.byte_size(encoded) |> should.equal(7)

  // Last byte should be delimiter
  let size = bit_array.byte_size(encoded)
  case bit_array.slice(encoded, size - 1, 1) {
    Ok(<<0>>) -> should.be_true(True)
    _ -> should.fail()
  }
}

pub fn cobs_encode_with_zeros_test() {
  // Data with zero bytes
  let data = <<1, 0, 2, 0, 3>>
  let encoded = cobs.encode(data)

  // Encoded data should not contain zeros except delimiter
  let without_delimiter = case bit_array.slice(encoded, 0, bit_array.byte_size(encoded) - 1) {
    Ok(d) -> d
    _ -> <<>>
  }

  // Check no internal zeros
  without_delimiter
  |> binary_to_list
  |> list.any(fn(b) { b == 0 })
  |> should.be_false
}

pub fn cobs_roundtrip_test() {
  // Test encode -> decode roundtrip
  let original = <<0, 1, 2, 0, 0, 3, 4, 5, 0, 6>>
  let encoded = cobs.encode(original)

  // Strip delimiter for decoding
  let size = bit_array.byte_size(encoded)
  case bit_array.slice(encoded, 0, size - 1) {
    Ok(without_delim) -> {
      case cobs.decode(without_delim) {
        Ok(decoded) -> decoded |> should.equal(original)
        Error(e) -> should.fail()
      }
    }
    _ -> should.fail()
  }
}

pub fn cobs_extract_frames_test() {
  // Multiple frames in buffer
  let frame1 = cobs.encode(<<1, 2, 3>>)
  let frame2 = cobs.encode(<<4, 5, 6>>)
  let buffer = bit_array.append(frame1, frame2)

  let #(remaining, frames) = cobs.extract_frames(buffer)

  // Should have extracted both frames
  list.length(frames) |> should.equal(2)

  // Buffer should be empty
  bit_array.byte_size(remaining) |> should.equal(0)
}

pub fn cobs_partial_frame_test() {
  // Incomplete frame (no delimiter)
  let complete = cobs.encode(<<1, 2, 3>>)
  let partial = <<7, 8, 9>>  // No delimiter
  let buffer = bit_array.append(complete, partial)

  let #(remaining, frames) = cobs.extract_frames(buffer)

  // Should have extracted one frame
  list.length(frames) |> should.equal(1)

  // Remaining should be the partial data
  bit_array.byte_size(remaining) |> should.equal(3)
}

// ============================================================================
// CRC-8 Tests
// ============================================================================

pub fn crc8_compute_test() {
  // Known test vectors for CRC-8-CCITT
  let data = <<"123456789">>
  let crc = crc8.compute(data)

  // Standard CRC-8 of "123456789" should be 0xF4
  crc |> should.equal(0xF4)
}

pub fn crc8_append_verify_test() {
  let data = <<1, 2, 3, 4, 5>>
  let with_crc = crc8.append(data)

  // Should be one byte longer
  bit_array.byte_size(with_crc) |> should.equal(6)

  // Verification should pass
  crc8.verify(with_crc) |> should.be_true
}

pub fn crc8_verify_corrupted_test() {
  let data = <<1, 2, 3, 4, 5>>
  let with_crc = crc8.append(data)

  // Corrupt one byte
  let corrupted = <<1, 2, 99, 4, 5, 0>>

  // Verification should fail
  crc8.verify(corrupted) |> should.be_false
}

// ============================================================================
// Packet Tests
// ============================================================================

pub fn packet_heartbeat_roundtrip_test() {
  let original = packet.PacketHeartbeat(packet.Heartbeat(42))
  let encoded = packet.encode(original)

  case packet.decode(encoded) {
    Ok(packet.PacketHeartbeat(hb)) -> hb.seq |> should.equal(42)
    _ -> should.fail()
  }
}

pub fn packet_sensor_data_roundtrip_test() {
  let original = packet.PacketSensorData(packet.SensorData(
    seq: 1,
    temperature: 36.5,
    light: 512,
    touch: True,
    audio_level: 100,
  ))

  let encoded = packet.encode(original)

  case packet.decode(encoded) {
    Ok(packet.PacketSensorData(sd)) -> {
      sd.seq |> should.equal(1)
      sd.light |> should.equal(512)
      sd.touch |> should.be_true
      sd.audio_level |> should.equal(100)
      // Float comparison with tolerance
      let temp_ok = sd.temperature >. 36.0 && sd.temperature <. 37.0
      temp_ok |> should.be_true
    }
    _ -> should.fail()
  }
}

pub fn packet_pad_state_roundtrip_test() {
  let original = packet.PacketPadState(packet.PadState(
    seq: 5,
    pleasure: 0.8,
    arousal: -0.3,
    dominance: 0.5,
  ))

  let encoded = packet.encode(original)

  case packet.decode(encoded) {
    Ok(packet.PacketPadState(ps)) -> {
      ps.seq |> should.equal(5)
      // Approximate float comparison
      let p_ok = ps.pleasure >. 0.7 && ps.pleasure <. 0.9
      let a_ok = ps.arousal >. -0.4 && ps.arousal <. -0.2
      let d_ok = ps.dominance >. 0.4 && ps.dominance <. 0.6
      p_ok |> should.be_true
      a_ok |> should.be_true
      d_ok |> should.be_true
    }
    _ -> should.fail()
  }
}

pub fn packet_audio_command_roundtrip_test() {
  let original = packet.PacketAudioCommand(packet.AudioCommand(
    seq: 10,
    freq_left: 440,
    freq_right: 450,
    duration_ms: 1000,
    waveform: 0,
  ))

  let encoded = packet.encode(original)

  case packet.decode(encoded) {
    Ok(packet.PacketAudioCommand(ac)) -> {
      ac.seq |> should.equal(10)
      ac.freq_left |> should.equal(440)
      ac.freq_right |> should.equal(450)
      ac.duration_ms |> should.equal(1000)
      ac.waveform |> should.equal(0)
    }
    _ -> should.fail()
  }
}

pub fn packet_binaural_beat_helper_test() {
  // Test the binaural beat helper
  let pkt = packet.binaural_beat(1, 400, 10, 5000)

  case pkt {
    packet.PacketAudioCommand(ac) -> {
      ac.freq_left |> should.equal(400)
      ac.freq_right |> should.equal(410)  // 400 + 10
      ac.duration_ms |> should.equal(5000)
    }
    _ -> should.fail()
  }
}

pub fn packet_invalid_crc_test() {
  // Create a valid packet then corrupt it
  let original = packet.PacketHeartbeat(packet.Heartbeat(1))
  let encoded = packet.encode(original)

  // Corrupt the CRC byte (last byte)
  let size = bit_array.byte_size(encoded)
  case bit_array.slice(encoded, 0, size - 1) {
    Ok(without_crc) -> {
      let corrupted = bit_array.append(without_crc, <<0xFF>>)
      case packet.decode(corrupted) {
        Error("CRC mismatch") -> should.be_true(True)
        _ -> should.fail()
      }
    }
    _ -> should.fail()
  }
}

// ============================================================================
// Integration Tests
// ============================================================================

pub fn full_pipeline_test() {
  // Test complete flow: packet -> encode -> COBS -> decode -> packet

  let original = packet.PacketPadState(packet.PadState(
    seq: 99,
    pleasure: 1.0,
    arousal: 0.0,
    dominance: -1.0,
  ))

  // Encode packet (adds CRC)
  let packet_bytes = packet.encode(original)

  // COBS encode (for wire transmission)
  let wire_bytes = cobs.encode(packet_bytes)

  // Simulate receiving and extracting frame
  let #(_, frames) = cobs.extract_frames(wire_bytes)

  case frames {
    [frame] -> {
      // Decode packet (verifies CRC)
      case packet.decode(frame) {
        Ok(packet.PacketPadState(ps)) -> {
          ps.seq |> should.equal(99)
        }
        _ -> should.fail()
      }
    }
    _ -> should.fail()
  }
}
