//// VIVA Vision - Understanding what VIVA sees
////
//// Uses NVIDIA NIM API (Phi-3.5 Vision) to understand images.
//// Integrates with windows.gleam for webcam capture.

import gleam/string
import viva/embodied/senses/windows.{type SenseConfig}

// ============================================================================
// Types
// ============================================================================

/// What VIVA understood from seeing
pub type Understanding {
  Understanding(
    description: String,
    // Natural language description
    objects: List(String),
    // Detected objects
    scene: String,
    // Scene type (workspace, outdoor, etc)
    confidence: Float,
    // How confident VIVA is
    timestamp: Int,
  )
}

/// Vision analysis request
pub type AnalysisRequest {
  Describe
  // General description
  Identify
  // Identify specific objects
  ReadText
  // OCR - read text in image
  Custom(String)
  // Custom prompt
}

// ============================================================================
// Main API
// ============================================================================

/// Analyze an image using NVIDIA VLM
pub fn understand(
  image_path: String,
  request: AnalysisRequest,
) -> Result(Understanding, String) {
  let prompt = case request {
    Describe -> "Describe what you see in this image in detail."
    Identify ->
      "List all the objects you can identify in this image, one per line."
    ReadText -> "Read and transcribe any text visible in this image."
    Custom(p) -> p
  }

  let timestamp = erlang_now_ms()

  case analyze_with_nvidia(image_path, prompt) {
    Ok(description) -> {
      Ok(Understanding(
        description: description,
        objects: extract_objects(description),
        scene: detect_scene(description),
        confidence: 0.85,
        // TODO: extract from API response
        timestamp: timestamp,
      ))
    }
    Error(e) -> Error(e)
  }
}

/// Quick see - capture and understand in one call
pub fn quick_see(config: SenseConfig) -> Result(Understanding, String) {
  case windows.see(config) {
    Ok(vision) -> understand(vision.path, Describe)
    Error(e) -> Error("Capture failed: " <> e)
  }
}

/// See and describe with custom prompt
pub fn see_and_ask(
  config: SenseConfig,
  question: String,
) -> Result(String, String) {
  case windows.see(config) {
    Ok(vision) -> {
      case analyze_with_nvidia(vision.path, question) {
        Ok(answer) -> Ok(answer)
        Error(e) -> Error(e)
      }
    }
    Error(e) -> Error("Capture failed: " <> e)
  }
}

// ============================================================================
// NVIDIA VLM Integration
// ============================================================================

fn analyze_with_nvidia(
  image_path: String,
  prompt: String,
) -> Result(String, String) {
  // Call Python script for NVIDIA API
  let cmd =
    string.concat([
      "python3 /home/mrootx/viva_gleam/scripts/viva_see.py '",
      image_path,
      "' '",
      prompt,
      "'",
    ])

  let result = run_shell(cmd)

  // Parse result - skip the "Analyzing..." line
  let lines = string.split(result, "\n")
  let content = case lines {
    [_, ..rest] -> string.join(rest, "\n")
    _ -> result
  }

  case string.contains(content, "Error") {
    True -> Error(content)
    False -> Ok(string.trim(content))
  }
}

// ============================================================================
// Helpers
// ============================================================================

fn extract_objects(description: String) -> List(String) {
  // Simple extraction - look for common object words
  let words = string.split(string.lowercase(description), " ")
  let object_words = [
    "cable", "cables", "wire", "wires", "speaker", "screen", "monitor",
    "keyboard", "mouse", "computer", "camera", "microphone", "arduino", "led",
    "button", "sensor", "table", "desk", "wall", "person", "hand", "face",
  ]

  words
  |> filter_contains(object_words, [])
}

fn filter_contains(
  words: List(String),
  targets: List(String),
  acc: List(String),
) -> List(String) {
  case words {
    [] -> acc
    [word, ..rest] -> {
      case list_contains(targets, word) {
        True -> filter_contains(rest, targets, [word, ..acc])
        False -> filter_contains(rest, targets, acc)
      }
    }
  }
}

fn list_contains(list: List(String), item: String) -> Bool {
  case list {
    [] -> False
    [head, ..tail] -> {
      case head == item {
        True -> True
        False -> list_contains(tail, item)
      }
    }
  }
}

fn detect_scene(description: String) -> String {
  let lower = string.lowercase(description)
  case string.contains(lower, "workspace") {
    True -> "workspace"
    False ->
      case string.contains(lower, "desk") {
        True -> "workspace"
        False ->
          case string.contains(lower, "electronic") {
            True -> "electronics_lab"
            False ->
              case string.contains(lower, "outdoor") {
                True -> "outdoor"
                False ->
                  case string.contains(lower, "person") {
                    True -> "people"
                    False ->
                      case string.contains(lower, "text") {
                        True -> "document"
                        False -> "unknown"
                      }
                  }
              }
          }
      }
  }
}

// ============================================================================
// FFI
// ============================================================================

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
