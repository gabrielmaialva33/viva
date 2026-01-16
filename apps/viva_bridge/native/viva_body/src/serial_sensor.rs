#![allow(dead_code)] // TODO: Remove once wired to a dynamic sensor registry
use crate::{HardwareState, SensoryInput};
use serialport::SerialPort;
use std::io::ErrorKind;
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

        // 2. Read data (Naive implementation: Read line)
        // Ideally: Use a buffer, look for newline, parse JSON.
        // For NIF safety, we just try to peek/read available bytes.

        let mut serial_buf: Vec<u8> = vec![0; 1024];
        match port.read(serial_buf.as_mut_slice()) {
            Ok(t) if t > 0 => {
                let data_str = String::from_utf8_lossy(&serial_buf[..t]);
                // Try to find JSON object structure
                if let Some(start) = data_str.find('{') {
                    if let Some(end) = data_str[start..].find('}') {
                        let json_str = &data_str[start..start + end + 1];
                        if let Ok(partial_state) = serde_json::from_str::<HardwareState>(json_str) {
                            return partial_state;
                        }
                    }
                }
            }
            Ok(_) => {}
            Err(ref e) if e.kind() == ErrorKind::TimedOut => {}
            Err(e) => {
                eprintln!("[viva_body] Serial read error: {}", e);
                // Maybe disconnect?
                *port_guard = None;
            }
        }

        HardwareState::empty()
    }
}


