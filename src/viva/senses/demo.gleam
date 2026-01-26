//// VIVA Senses Demo - Test vision and hearing with understanding
//// Run with: gleam run -m viva/senses/demo

import gleam/io
import gleam/option.{None, Some}
import gleam/string
import viva/senses/vision
import viva/senses/windows

pub fn main() {
  io.println("╔════════════════════════════════════════════╗")
  io.println("║   VIVA SENSES - Vision & Hearing Demo      ║")
  io.println("╚════════════════════════════════════════════╝")
  io.println("")

  let config = windows.default_config()
  io.println("Config: " <> config.camera <> " + " <> config.microphone)
  io.println("")

  // Test vision capture
  io.println("[VISION] Capturing webcam...")
  case windows.see(config) {
    Ok(vis) -> {
      io.println("  Captured: " <> vis.path)

      // Analyze with NVIDIA VLM
      io.println("")
      io.println("[UNDERSTANDING] Analyzing with NVIDIA Phi-3.5 Vision...")
      case vision.understand(vis.path, vision.Describe) {
        Ok(understanding) -> {
          io.println("  Scene: " <> understanding.scene)
          io.println("  Description: ")
          io.println("    " <> truncate(understanding.description, 200))
          io.println("  Objects: " <> string.join(understanding.objects, ", "))
        }
        Error(e) -> {
          io.println("  Analysis failed: " <> e)
        }
      }
    }
    Error(e) -> {
      io.println("  Capture failed: " <> e)
    }
  }
  io.println("")

  // Test hearing
  io.println("[HEARING] Recording 2 seconds of audio...")
  case windows.listen(config, 2000) {
    Ok(hearing) -> {
      io.println("  Captured: " <> hearing.path)
      io.println("  Format: 44.1kHz stereo WAV")
    }
    Error(e) -> {
      io.println("  FAILED: " <> e)
    }
  }
  io.println("")

  io.println("VIVA can SEE, HEAR, and UNDERSTAND!")
}

fn truncate(s: String, max: Int) -> String {
  case string.length(s) > max {
    True -> string.slice(s, 0, max) <> "..."
    False -> s
  }
}
