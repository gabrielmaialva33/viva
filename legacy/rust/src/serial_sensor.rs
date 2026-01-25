#![allow(dead_code)] // TODO: Remove once wired to a dynamic sensor registry
use crate::{HardwareState, SensoryInput};
use serialport::SerialPort;
use std::io::{ErrorKind, Write};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

// Global serial port instance (Single port for now: /dev/ttyUSB0 or COM3)
// In a real generic system, we'd have a registry of sensors.
static SERIAL_PORT: LazyLock<Mutex<Option<Box<dyn SerialPort>>>> = LazyLock::new(|| Mutex::new(None));

pub struct SerialSensor {
    pub port_name: String,
    pub baud_rate: u32,
}

impl SerialSensor {
    pub fn new(port_name: &str, baud_rate: u32) -> Self {
        Self {
            port_name: port_name.to_string(),
            baud_rate,
        }
    }

    fn try_connect(&self) -> bool {
        let mut port_guard = match SERIAL_PORT.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        if port_guard.is_some() {
            return true; // Already connected
        }

        match serialport::new(&self.port_name, self.baud_rate)
            .timeout(Duration::from_millis(100))
            .open()
        {
            Ok(port) => {
                eprintln!("[viva_body] SerialSensor connected to {}", self.port_name);
                *port_guard = Some(port);
                true
            }
            Err(_e) => {
                // Don't spam logs on every fetch loop
                // eprintln!("[viva_body] SerialSensor connection failed: {}", e);
                false
            }
        }
    }
}

impl SensoryInput for SerialSensor {
    fn identify(&self) -> String {
        format!("SerialSensor:{}", self.port_name)
    }

    fn fetch_metrics(&self) -> HardwareState {
        // 1. Ensure connection
        if !self.try_connect() {
            return HardwareState::empty(); // Return empty state if disconnected
        }

        let mut port_guard = match SERIAL_PORT.lock() {
            Ok(g) => g,
            Err(_) => return HardwareState::empty(),
        };

        let port = match port_guard.as_mut() {
            Some(p) => p,
            None => return HardwareState::empty(),
        };

        // 2. Poll Status (Write "S\n")
        if let Err(e) = port.write_all(b"S\n") {
             eprintln!("[viva_body] Serial write error (poll): {}", e);
             // If write fails, maybe connection is dead?
             // We don't drop the connection immediately to avoid thrashing,
             // but next read failure will likely drop it.
             return HardwareState::empty();
        }

        // 3. Read response
        // Expected: "ACK:PWM:128,RPM:1500,HARMONY:ON"
        let mut serial_buf: Vec<u8> = vec![0; 1024];

        // Timeout is 100ms from try_connect
        match port.read(serial_buf.as_mut_slice()) {
            Ok(t) if t > 0 => {
                let data_str = String::from_utf8_lossy(&serial_buf[..t]);
                let trimmed = data_str.trim();

                // Parse "ACK:PWM:128,RPM:1500,HARMONY:ON"
                if trimmed.starts_with("ACK:PWM:") {
                     let mut state = HardwareState::empty();

                     // Simple parsing
                     for part in trimmed.split(',') {
                         if let Some(val_str) = part.strip_prefix("RPM:") {
                             if let Ok(rpm) = val_str.parse::<u32>() {
                                 state.fan_rpm = Some(rpm);
                             }
                         } else if let Some(val_str) = part.strip_prefix("ACK:PWM:") {
                             // format is ACK:PWM:128
                              if let Ok(pwm) = val_str.parse::<u32>() {
                                 // Map PWM (0-255) to target RPM estimate or just store raw?
                                 // For now store as target_fan_rpm if we want, or just ignore.
                                 // Let's store PWM as a proxy for Target for now, or just the PWM value.
                                 // In body_state we defined target_fan_rpm as u32.
                                 // Let's store the PWM value there
                                 state.target_fan_rpm = Some(pwm);
                             }
                         }
                     }

                     return state;
                }
            }
            Ok(_) => {} // Empty read
            Err(ref e) if e.kind() == ErrorKind::TimedOut => {} // Timeout is expected if no data
            Err(e) => {
                eprintln!("[viva_body] Serial read error: {}", e);
                *port_guard = None; // Force reconnect on error
            }
        }

        HardwareState::empty()
    }
}


