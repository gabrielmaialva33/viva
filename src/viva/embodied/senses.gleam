//// Senses - High-level sensory API for VIVA
////
//// This module provides the main interface for VIVA to sense the world:
////   - see()   → NV-CLIP for visual classification
////   - read()  → PaddleOCR for text extraction
////   - hear()  → Whisper for audio transcription
////   - think() → DeepSeek for reasoning
////
//// Returns unified Percept for integration with Soul.

import gleam/option.{type Option, None, Some}
import gleam/result
import gleam/string
import viva/embodied/awareness.{type AwarenessResult, type AwarenessState}
import viva/embodied/percept.{type Percept}
import viva/embodied/sense.{
  type Emotion, type Hearing, type Reading, type SenseError, type Thought,
  type Vision,
}

// =============================================================================
// HIGH-LEVEL API
// =============================================================================

/// Look at an image and understand it
/// Returns unified percept with vision, reading, and thought
pub fn perceive_image(path: String) -> Result(Percept, SenseError) {
  // See (NV-CLIP)
  use vision <- result.try(sense.see(path))

  // Read (PaddleOCR)
  let reading = case sense.read(path) {
    Ok(r) -> r
    Error(_) -> sense.empty_reading()
  }

  // Think about what was seen
  let perception_text = format_perception(vision, reading)
  use thought <- result.try(sense.think(perception_text))

  // Create unified percept
  Ok(percept.visual_percept(vision, reading, thought, path))
}

/// Listen to audio and understand it
pub fn perceive_audio(path: String) -> Result(Percept, SenseError) {
  // Hear (Whisper)
  use hearing <- result.try(sense.hear(path))

  // Think about what was heard
  let perception_text = "I heard: " <> hearing.text
  use thought <- result.try(sense.think(perception_text))

  // Create unified percept
  Ok(percept.auditory_percept(hearing, thought, path))
}

/// Internal thought (no external stimulus)
pub fn contemplate(content: String) -> Result(Percept, SenseError) {
  use thought <- result.try(sense.think(content))
  Ok(percept.thought_percept(thought))
}

/// Perceive image and process through awareness
pub fn perceive_and_process(
  path: String,
  state: AwarenessState,
) -> Result(#(AwarenessState, AwarenessResult), SenseError) {
  use percept <- result.try(perceive_image(path))
  Ok(awareness.process_with_state(state, percept))
}

// =============================================================================
// QUICK SENSING (single modality)
// =============================================================================

/// Quick vision check (no OCR, no thinking)
pub fn quick_see(path: String) -> Result(Vision, SenseError) {
  sense.see(path)
}

/// Quick read (OCR only)
pub fn quick_read(path: String) -> Result(Reading, SenseError) {
  sense.read(path)
}

/// Quick listen (transcription only)
pub fn quick_hear(path: String) -> Result(Hearing, SenseError) {
  sense.hear(path)
}

/// Quick think (reasoning only)
pub fn quick_think(prompt: String) -> Result(Thought, SenseError) {
  sense.think(prompt)
}

// =============================================================================
// BATCH SENSING
// =============================================================================

/// Perceive multiple images
pub fn perceive_images(paths: List(String)) -> List(Result(Percept, SenseError)) {
  // Process sequentially (NIMs may not handle parallel well)
  list_map(paths, perceive_image)
}

/// Get the most salient percept from images
pub fn perceive_most_salient(paths: List(String)) -> Option(Percept) {
  let percepts =
    paths
    |> list_map(perceive_image)
    |> list_filter_ok()

  case percepts {
    [] -> None
    _ -> awareness.process_many(percepts) |> option_map(fn(r) { r.percept })
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn format_perception(vision: Vision, reading: Reading) -> String {
  let visual_desc = "I see: " <> vision.dominant

  let text_desc = case reading.text {
    "" -> ""
    text -> ". Text visible: " <> truncate(text, 200)
  }

  let code_desc = case reading.has_code {
    True -> " [contains code]"
    False -> ""
  }

  visual_desc <> text_desc <> code_desc
}

fn truncate(text: String, max_len: Int) -> String {
  case string_length(text) > max_len {
    True -> string_slice(text, 0, max_len) <> "..."
    False -> text
  }
}

// =============================================================================
// LIST HELPERS (avoid stdlib dependency issues)
// =============================================================================

fn list_map(lst: List(a), f: fn(a) -> b) -> List(b) {
  case lst {
    [] -> []
    [head, ..tail] -> [f(head), ..list_map(tail, f)]
  }
}

fn list_filter_ok(lst: List(Result(a, e))) -> List(a) {
  case lst {
    [] -> []
    [Ok(v), ..tail] -> [v, ..list_filter_ok(tail)]
    [Error(_), ..tail] -> list_filter_ok(tail)
  }
}

fn option_map(opt: Option(a), f: fn(a) -> b) -> Option(b) {
  case opt {
    Some(v) -> Some(f(v))
    None -> None
  }
}

// =============================================================================
// STRING HELPERS
// =============================================================================

fn string_length(s: String) -> Int {
  string.length(s)
}

fn string_slice(s: String, start: Int, len: Int) -> String {
  string.slice(s, start, len)
}
