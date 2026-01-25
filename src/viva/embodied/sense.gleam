//// Sense - VIVA's sensory input types and FFI
////
//// Low-level interface to NIMs (NV-CLIP, PaddleOCR, DeepSeek)
//// Pure Gleam types with Erlang/Elixir FFI for actual NIM calls.
////
//// Architecture:
////   Gleam (types, logic) → FFI → Elixir (HTTP) → NIM (GPU)

import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import gleam/string

// =============================================================================
// SENSE TYPES
// =============================================================================

/// What VIVA can see
pub type Vision {
  Vision(
    labels: List(String),
    confidence: List(Float),
    dominant: String,
    dominant_confidence: Float,
    scene: SceneType,
  )
}

/// Scene classification
pub type SceneType {
  Workspace
  Communication
  Browsing
  Entertainment
  ReadingScene
  Viewing
  Unknown
}

/// What VIVA can read (OCR)
pub type Reading {
  Reading(
    text: String,
    blocks: List(TextBlock),
    language: String,
    has_code: Bool,
    word_count: Int,
    line_count: Int,
  )
}

/// OCR text block with position
pub type TextBlock {
  TextBlock(
    text: String,
    x: Int,
    y: Int,
    width: Int,
    height: Int,
    confidence: Float,
  )
}

/// What VIVA can hear
pub type Hearing {
  Hearing(
    text: String,
    language: String,
    confidence: Float,
  )
}

/// VIVA's thought about a perception
pub type Thought {
  Thought(
    content: String,
    emotion: Emotion,
    action: SuggestedAction,
  )
}

/// PAD emotional state
pub type Emotion {
  Emotion(
    valence: Float,     // -1 (negative) to 1 (positive)
    arousal: Float,     // 0 (calm) to 1 (excited)
    dominance: Float,   // 0 (submissive) to 1 (dominant)
  )
}

/// What action VIVA might take
pub type SuggestedAction {
  Observe
  OfferHelp
  Alert
  Celebrate
  Empathize
  Rest
}

/// Sense errors
pub type SenseError {
  ConnectionFailed(service: String)
  ProcessingFailed(reason: String)
  Timeout
  InvalidInput(reason: String)
}

// =============================================================================
// SENSE CONSTRUCTORS
// =============================================================================

/// Create neutral emotion
pub fn neutral_emotion() -> Emotion {
  Emotion(valence: 0.0, arousal: 0.3, dominance: 0.5)
}

/// Create positive emotion
pub fn positive_emotion(intensity: Float) -> Emotion {
  Emotion(
    valence: intensity,
    arousal: 0.5 +. intensity *. 0.3,
    dominance: 0.5 +. intensity *. 0.2,
  )
}

/// Create negative emotion
pub fn negative_emotion(intensity: Float) -> Emotion {
  Emotion(
    valence: 0.0 -. intensity,
    arousal: 0.5 +. intensity *. 0.4,
    dominance: 0.5 -. intensity *. 0.2,
  )
}

/// Empty vision (nothing seen)
pub fn empty_vision() -> Vision {
  Vision(
    labels: [],
    confidence: [],
    dominant: "nothing",
    dominant_confidence: 0.0,
    scene: Unknown,
  )
}

/// Empty reading (nothing read)
pub fn empty_reading() -> Reading {
  Reading(
    text: "",
    blocks: [],
    language: "unknown",
    has_code: False,
    word_count: 0,
    line_count: 0,
  )
}

// =============================================================================
// SCENE CLASSIFICATION
// =============================================================================

/// Classify scene from labels
pub fn classify_scene(labels: List(String)) -> SceneType {
  case has_label(labels, "code"), has_label(labels, "terminal") {
    True, _ -> Workspace
    _, True -> Workspace
    _, _ ->
      case has_label(labels, "chat"), has_label(labels, "message") {
        True, _ -> Communication
        _, True -> Communication
        _, _ ->
          case has_label(labels, "browser"), has_label(labels, "web") {
            True, _ -> Browsing
            _, True -> Browsing
            _, _ ->
              case has_label(labels, "game"), has_label(labels, "video") {
                True, _ -> Entertainment
                _, True -> Entertainment
                _, _ ->
                  case has_label(labels, "document"), has_label(labels, "text") {
                    True, _ -> ReadingScene
                    _, True -> ReadingScene
                    _, _ -> Unknown
                  }
              }
          }
      }
  }
}

fn has_label(labels: List(String), keyword: String) -> Bool {
  list.any(labels, fn(label) { string_contains(label, keyword) })
}

// =============================================================================
// EMOTION OPERATIONS
// =============================================================================

/// Blend two emotions
pub fn blend_emotions(a: Emotion, b: Emotion, weight: Float) -> Emotion {
  let w = clamp(weight, 0.0, 1.0)
  let inv_w = 1.0 -. w

  Emotion(
    valence: a.valence *. inv_w +. b.valence *. w,
    arousal: a.arousal *. inv_w +. b.arousal *. w,
    dominance: a.dominance *. inv_w +. b.dominance *. w,
  )
}

/// Calculate emotional intensity
pub fn emotion_intensity(e: Emotion) -> Float {
  let v_sq = e.valence *. e.valence
  let a_sq = { e.arousal -. 0.5 } *. { e.arousal -. 0.5 }
  let d_sq = { e.dominance -. 0.5 } *. { e.dominance -. 0.5 }
  float_sqrt(v_sq +. a_sq +. d_sq)
}

/// Emotion delta (difference)
pub fn emotion_delta(current: Emotion, baseline: Emotion) -> Emotion {
  Emotion(
    valence: current.valence -. baseline.valence,
    arousal: current.arousal -. baseline.arousal,
    dominance: current.dominance -. baseline.dominance,
  )
}

// =============================================================================
// SENSE FFI - NIM Calls via Elixir
// =============================================================================

/// See an image (NV-CLIP)
pub fn see(image_path: String) -> Result(Vision, SenseError) {
  case nim_see(image_path) {
    Ok(raw) -> Ok(parse_vision_result(raw))
    Error(reason) -> Error(ProcessingFailed(reason))
  }
}

/// See raw bytes
pub fn see_bytes(image_bytes: BitArray) -> Result(Vision, SenseError) {
  case nim_see_bytes(image_bytes) {
    Ok(raw) -> Ok(parse_vision_result(raw))
    Error(reason) -> Error(ProcessingFailed(reason))
  }
}

/// Read text from image (PaddleOCR)
pub fn read(image_path: String) -> Result(Reading, SenseError) {
  case nim_read(image_path) {
    Ok(raw) -> Ok(parse_reading_result(raw))
    Error(reason) -> Error(ProcessingFailed(reason))
  }
}

/// Read from bytes
pub fn read_bytes(image_bytes: BitArray) -> Result(Reading, SenseError) {
  case nim_read_bytes(image_bytes) {
    Ok(raw) -> Ok(parse_reading_result(raw))
    Error(reason) -> Error(ProcessingFailed(reason))
  }
}

/// Hear audio (Whisper)
pub fn hear(audio_path: String) -> Result(Hearing, SenseError) {
  case nim_hear(audio_path) {
    Ok(raw) -> Ok(parse_hearing_result(raw))
    Error(reason) -> Error(ProcessingFailed(reason))
  }
}

/// Think about something (DeepSeek)
pub fn think(perception: String) -> Result(Thought, SenseError) {
  case nim_think(perception) {
    Ok(raw) -> Ok(parse_thought_result(raw))
    Error(reason) -> Error(ProcessingFailed(reason))
  }
}

// =============================================================================
// RESULT PARSERS
// =============================================================================

fn parse_vision_result(raw: SenseResult) -> Vision {
  let labels = raw.labels
  let confidence = raw.confidence
  let #(dominant, dom_conf) = case labels, confidence {
    [l, ..], [c, ..] -> #(l, c)
    _, _ -> #("unknown", 0.0)
  }

  Vision(
    labels: labels,
    confidence: confidence,
    dominant: dominant,
    dominant_confidence: dom_conf,
    scene: classify_scene(labels),
  )
}

fn parse_reading_result(raw: SenseResult) -> Reading {
  Reading(
    text: raw.text,
    blocks: [],  // TODO: parse blocks
    language: raw.language,
    has_code: detect_code(raw.text),
    word_count: count_words(raw.text),
    line_count: count_lines(raw.text),
  )
}

fn parse_hearing_result(raw: SenseResult) -> Hearing {
  Hearing(
    text: raw.text,
    language: raw.language,
    confidence: list.first(raw.confidence) |> result.unwrap(0.9),
  )
}

fn parse_thought_result(raw: SenseResult) -> Thought {
  Thought(
    content: raw.text,
    emotion: Emotion(
      valence: raw.valence,
      arousal: raw.arousal,
      dominance: raw.dominance,
    ),
    action: parse_action(raw.action),
  )
}

fn parse_action(action: String) -> SuggestedAction {
  case action {
    "offer_help" -> OfferHelp
    "alert" -> Alert
    "celebrate" -> Celebrate
    "empathize" -> Empathize
    "rest" -> Rest
    _ -> Observe
  }
}

// =============================================================================
// TEXT ANALYSIS HELPERS
// =============================================================================

fn detect_code(text: String) -> Bool {
  let patterns = [
    "def ", "fn ", "pub fn", "function ", "class ",
    "import ", "module ", "defmodule", "=>", "|>",
    "let ", "const ", "var ", "if ", "case ",
  ]
  list.any(patterns, fn(p) { string_contains(text, p) })
}

fn count_words(text: String) -> Int {
  text
  |> string_split(" ")
  |> list.filter(fn(w) { string_length(w) > 0 })
  |> list.length()
}

fn count_lines(text: String) -> Int {
  text
  |> string_split("\n")
  |> list.length()
}

// =============================================================================
// RAW SENSE RESULT (from FFI)
// =============================================================================

/// Raw result from NIM FFI
pub type SenseResult {
  SenseResult(
    labels: List(String),
    confidence: List(Float),
    text: String,
    language: String,
    valence: Float,
    arousal: Float,
    dominance: Float,
    action: String,
  )
}

// =============================================================================
// FFI DECLARATIONS
// =============================================================================

@external(erlang, "Elixir.Viva.Embodied.Senses", "see")
fn nim_see(path: String) -> Result(SenseResult, String)

@external(erlang, "Elixir.Viva.Embodied.Senses", "see_bytes")
fn nim_see_bytes(bytes: BitArray) -> Result(SenseResult, String)

@external(erlang, "Elixir.Viva.Embodied.Senses", "read")
fn nim_read(path: String) -> Result(SenseResult, String)

@external(erlang, "Elixir.Viva.Embodied.Senses", "read_bytes")
fn nim_read_bytes(bytes: BitArray) -> Result(SenseResult, String)

@external(erlang, "Elixir.Viva.Embodied.Senses", "hear")
fn nim_hear(path: String) -> Result(SenseResult, String)

@external(erlang, "Elixir.Viva.Embodied.Senses", "think")
fn nim_think(perception: String) -> Result(SenseResult, String)

// =============================================================================
// MATH HELPERS (FFI)
// =============================================================================

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

fn clamp(value: Float, min: Float, max: Float) -> Float {
  case value <. min {
    True -> min
    False ->
      case value >. max {
        True -> max
        False -> value
      }
  }
}

fn string_contains(haystack: String, needle: String) -> Bool {
  string.contains(haystack, needle)
}

fn string_split(text: String, delimiter: String) -> List(String) {
  string.split(text, delimiter)
}

fn string_length(text: String) -> Int {
  string.length(text)
}
