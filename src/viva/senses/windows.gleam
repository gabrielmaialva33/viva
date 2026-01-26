//// VIVA Windows Senses - Vision and Hearing via FFmpeg
////
//// Captures webcam images and microphone audio from Windows host.
//// Uses FFmpeg DirectShow (dshow) through PowerShell.

import gleam/erlang/process
import gleam/int
import gleam/option.{type Option, None, Some}
import gleam/string

// ============================================================================
// Types
// ============================================================================

/// Visual input from webcam
pub type Vision {
  Vision(
    width: Int,
    height: Int,
    path: String,      // Path to captured image
    timestamp: Int,
  )
}

/// Audio input from microphone
pub type Hearing {
  Hearing(
    duration_ms: Int,
    sample_rate: Int,
    channels: Int,
    path: String,      // Path to captured audio
    timestamp: Int,
  )
}

/// Available devices
pub type Devices {
  Devices(
    cameras: List(String),
    microphones: List(String),
  )
}

/// Sense configuration
pub type SenseConfig {
  SenseConfig(
    camera: String,
    microphone: String,
    temp_dir: String,
  )
}

// ============================================================================
// Default Configuration
// ============================================================================

/// Default config for Gabriel's setup
pub fn default_config() -> SenseConfig {
  SenseConfig(
    camera: "Logi C270 HD WebCam",
    microphone: "Microfone (Realtek USB Audio)",
    temp_dir: "/mnt/h/Temp",
  )
}

// ============================================================================
// Vision - See
// ============================================================================

/// Capture a single frame from webcam
pub fn see(config: SenseConfig) -> Result(Vision, String) {
  let timestamp = erlang_now_ms()
  let filename = "viva_eye_" <> int.to_string(timestamp) <> ".jpg"
  let win_path = "viva_eye.jpg"  // Temp file in Windows
  let linux_path = config.temp_dir <> "/" <> win_path

  // Build FFmpeg command for Windows
  let cmd = string.concat([
    "powershell.exe -Command \"cd $env:TEMP; ",
    "ffmpeg -f dshow -i video='", config.camera, "' ",
    "-frames:v 1 -update 1 -y ", win_path, " 2>&1\"",
  ])

  case run_command(cmd) {
    Ok(_output) -> {
      // Copy from Windows temp to Linux
      let copy_cmd = "cp " <> linux_path <> " /tmp/" <> filename
      case run_command(copy_cmd) {
        Ok(_) -> {
          Ok(Vision(
            width: 640,
            height: 480,
            path: "/tmp/" <> filename,
            timestamp: timestamp,
          ))
        }
        Error(e) -> Error("Failed to copy image: " <> e)
      }
    }
    Error(e) -> Error("FFmpeg capture failed: " <> e)
  }
}

/// Continuous vision - captures frames at interval
pub fn see_continuous(
  config: SenseConfig,
  interval_ms: Int,
  callback: fn(Vision) -> Nil,
) -> Nil {
  case see(config) {
    Ok(vision) -> callback(vision)
    Error(_) -> Nil
  }
  process.sleep(interval_ms)
  see_continuous(config, interval_ms, callback)
}

// ============================================================================
// Hearing - Listen
// ============================================================================

/// Record audio for specified duration
pub fn listen(config: SenseConfig, duration_ms: Int) -> Result(Hearing, String) {
  let timestamp = erlang_now_ms()
  let filename = "viva_ear_" <> int.to_string(timestamp) <> ".wav"
  let win_path = "viva_ear.wav"
  let linux_path = config.temp_dir <> "/" <> win_path
  let duration_sec = int.to_string(duration_ms / 1000)

  // Build FFmpeg command
  let cmd = string.concat([
    "powershell.exe -Command \"cd $env:TEMP; ",
    "ffmpeg -f dshow -i audio='", config.microphone, "' ",
    "-t ", duration_sec, " -y ", win_path, " 2>&1\"",
  ])

  case run_command(cmd) {
    Ok(_output) -> {
      // Copy from Windows temp to Linux
      let copy_cmd = "cp " <> linux_path <> " /tmp/" <> filename
      case run_command(copy_cmd) {
        Ok(_) -> {
          Ok(Hearing(
            duration_ms: duration_ms,
            sample_rate: 44100,
            channels: 2,
            path: "/tmp/" <> filename,
            timestamp: timestamp,
          ))
        }
        Error(e) -> Error("Failed to copy audio: " <> e)
      }
    }
    Error(e) -> Error("FFmpeg record failed: " <> e)
  }
}

/// Listen for a short burst (default 1 second)
pub fn listen_burst(config: SenseConfig) -> Result(Hearing, String) {
  listen(config, 1000)
}

// ============================================================================
// Device Discovery
// ============================================================================

/// List available audio/video devices
pub fn list_devices() -> Result(Devices, String) {
  let cmd = "powershell.exe -Command \"ffmpeg -f dshow -list_devices true -i dummy 2>&1\""

  case run_command(cmd) {
    Ok(output) -> {
      let cameras = parse_devices(output, "video")
      let microphones = parse_devices(output, "audio")
      Ok(Devices(cameras: cameras, microphones: microphones))
    }
    Error(e) -> Error("Failed to list devices: " <> e)
  }
}

fn parse_devices(output: String, device_type: String) -> List(String) {
  // Parse FFmpeg device list output
  // Format: [dshow @ ...] "Device Name" (video) or (audio)
  output
  |> string.split("\n")
  |> filter_device_lines(device_type, [])
}

fn filter_device_lines(
  lines: List(String),
  device_type: String,
  acc: List(String),
) -> List(String) {
  case lines {
    [] -> acc
    [line, ..rest] -> {
      case string.contains(line, "(" <> device_type <> ")") {
        True -> {
          // Extract device name between quotes
          case extract_quoted(line) {
            Some(name) -> filter_device_lines(rest, device_type, [name, ..acc])
            None -> filter_device_lines(rest, device_type, acc)
          }
        }
        False -> filter_device_lines(rest, device_type, acc)
      }
    }
  }
}

fn extract_quoted(s: String) -> Option(String) {
  case string.split(s, "\"") {
    [_, name, ..] -> Some(name)
    _ -> None
  }
}

// ============================================================================
// Combined Perception
// ============================================================================

/// Capture both vision and hearing simultaneously
pub type Perception {
  Perception(
    vision: Option(Vision),
    hearing: Option(Hearing),
    timestamp: Int,
  )
}

/// Perceive - capture both senses
pub fn perceive(config: SenseConfig, audio_ms: Int) -> Perception {
  let timestamp = erlang_now_ms()

  // Capture vision
  let vision = case see(config) {
    Ok(v) -> Some(v)
    Error(_) -> None
  }

  // Capture hearing
  let hearing = case listen(config, audio_ms) {
    Ok(h) -> Some(h)
    Error(_) -> None
  }

  Perception(
    vision: vision,
    hearing: hearing,
    timestamp: timestamp,
  )
}

// ============================================================================
// Helpers
// ============================================================================

fn run_command(cmd: String) -> Result(String, String) {
  let result = run_shell(cmd)
  // FFmpeg outputs to stderr, so "error" in output doesn't mean failure
  // Check for specific failure patterns instead
  case string.contains(result, "Conversion failed") {
    True -> Error(result)
    False -> Ok(result)
  }
}

@external(erlang, "viva_senses_ffi", "run_shell")
fn run_shell(cmd: String) -> String

@external(erlang, "erlang", "system_time")
fn erlang_system_time(unit: SystemTimeUnit) -> Int

type SystemTimeUnit {
  Millisecond
}

fn erlang_now_ms() -> Int {
  erlang_system_time(Millisecond)
}
