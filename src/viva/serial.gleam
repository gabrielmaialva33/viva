//// VIVA Serial - Pure Gleam serial communication
////
//// No Python, no external deps - just Erlang ports

import gleam/erlang/process.{type Subject}
import gleam/bit_array
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import gleam/result

// ============================================================================
// TYPES
// ============================================================================

/// Serial port handle
pub opaque type Serial {
  Serial(port: Port, device: String, baud: Int)
}

/// Raw Erlang port
pub type Port

/// Serial config
pub type Config {
  Config(
    device: String,
    baud: Int,
    data_bits: Int,
    stop_bits: Int,
    parity: Parity,
  )
}

pub type Parity {
  NoParity
  Even
  Odd
}

/// Serial errors
pub type SerialError {
  OpenFailed(reason: String)
  WriteFailed
  ReadFailed
  ConfigFailed
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Default config for Arduino
pub fn arduino_config(device: String) -> Config {
  Config(
    device: device,
    baud: 115_200,
    data_bits: 8,
    stop_bits: 1,
    parity: NoParity,
  )
}

/// Open serial port
pub fn open(config: Config) -> Result(Serial, SerialError) {
  // Configure with stty
  case configure_port(config) {
    False -> Error(ConfigFailed)
    True -> {
      // Wait for Arduino reset
      process.sleep(2000)

      // Open as Erlang port
      let port = open_device(config.device)
      Ok(Serial(port: port, device: config.device, baud: config.baud))
    }
  }
}

/// Open with defaults (Arduino)
pub fn open_arduino(device: String) -> Result(Serial, SerialError) {
  open(arduino_config(device))
}

/// Write bytes to serial
pub fn write(serial: Serial, data: BitArray) -> Result(Nil, SerialError) {
  case write_port(serial.port, data) {
    True -> Ok(Nil)
    False -> Error(WriteFailed)
  }
}

/// Write string to serial
pub fn write_string(serial: Serial, s: String) -> Result(Nil, SerialError) {
  write(serial, bit_array.from_string(s))
}

/// Read available data (non-blocking)
pub fn read(serial: Serial) -> Option(BitArray) {
  read_port(serial.port)
}

/// Read line (blocking with timeout)
pub fn read_line(serial: Serial, timeout_ms: Int) -> Option(String) {
  read_line_port(serial.port, timeout_ms)
}

/// Close serial port
pub fn close(serial: Serial) -> Nil {
  close_port(serial.port)
}

// ============================================================================
// FFI - Erlang implementation
// ============================================================================

@external(erlang, "viva_serial_ffi", "configure_port")
fn configure_port(config: Config) -> Bool

@external(erlang, "viva_serial_ffi", "open_device")
fn open_device(device: String) -> Port

@external(erlang, "viva_serial_ffi", "write_port")
fn write_port(port: Port, data: BitArray) -> Bool

@external(erlang, "viva_serial_ffi", "read_port")
fn read_port(port: Port) -> Option(BitArray)

@external(erlang, "viva_serial_ffi", "read_line_port")
fn read_line_port(port: Port, timeout: Int) -> Option(String)

@external(erlang, "viva_serial_ffi", "close_port")
fn close_port(port: Port) -> Nil
