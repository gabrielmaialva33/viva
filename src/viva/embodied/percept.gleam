//// Percept - Unified perception structure
////
//// A Percept is VIVA's unit of conscious experience.
//// It binds together what was seen, read, thought, and felt
//// into a single coherent moment of awareness.
////
//// Inspired by phenomenology: the "what it is like" of experience.

import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/string
import viva/embodied/sense.{
  type Emotion, type Hearing, type Reading, type SceneType,
  type Thought, type Vision, Thought,
}

// =============================================================================
// PERCEPT TYPE
// =============================================================================

/// A single moment of unified perception
pub type Percept {
  Percept(
    /// What was seen (visual classification)
    vision: Option(Vision),
    /// What was read (text extraction)
    reading: Option(Reading),
    /// What was heard (audio transcription)
    hearing: Option(Hearing),
    /// What VIVA thought about it
    thought: Thought,
    /// When this was perceived
    timestamp: Int,
    /// Where attention was focused
    attention: AttentionFocus,
    /// How novel this perception is (0-1)
    novelty: Float,
    /// How salient/important (0-1)
    salience: Float,
    /// Source of the perception
    source: PerceptSource,
  )
}

/// What VIVA is paying attention to
pub type AttentionFocus {
  /// Analyzing code
  CodeAnalysis
  /// Detected an error
  ErrorDetection
  /// Observing work environment
  WorkObservation
  /// Social/communication context
  SocialInteraction
  /// Passive watching
  PassiveObservation
  /// Someone addressed VIVA directly
  DirectAddress
  /// Request for help
  HelpRequest
  /// Just listening
  Listening
  /// General awareness
  GeneralAwareness
  /// No focus (idle)
  Idle
}

/// Where the perception came from
pub type PerceptSource {
  Screenshot(path: String)
  ImageFile(path: String)
  AudioFile(path: String)
  DirectInput(content: String)
  Stream(stream_id: String)
}

// =============================================================================
// PERCEPT CONSTRUCTORS
// =============================================================================

/// Create a visual percept (from image)
pub fn visual_percept(
  vision: Vision,
  reading: Reading,
  thought: Thought,
  source: String,
) -> Percept {
  Percept(
    vision: Some(vision),
    reading: Some(reading),
    hearing: None,
    thought: thought,
    timestamp: now_timestamp(),
    attention: determine_visual_attention(vision, reading),
    novelty: 1.0,
    // Will be calculated when added to memory
    salience: calculate_salience(thought),
    source: ImageFile(source),
  )
}

/// Create an auditory percept (from audio)
pub fn auditory_percept(
  hearing: Hearing,
  thought: Thought,
  source: String,
) -> Percept {
  Percept(
    vision: None,
    reading: None,
    hearing: Some(hearing),
    thought: thought,
    timestamp: now_timestamp(),
    attention: determine_auditory_attention(hearing),
    novelty: 1.0,
    salience: calculate_salience(thought),
    source: AudioFile(source),
  )
}

/// Create a thought-only percept (internal)
pub fn thought_percept(thought: Thought) -> Percept {
  Percept(
    vision: None,
    reading: None,
    hearing: None,
    thought: thought,
    timestamp: now_timestamp(),
    attention: GeneralAwareness,
    novelty: 0.5,
    salience: calculate_salience(thought),
    source: DirectInput("internal"),
  )
}

/// Empty/null percept
pub fn empty_percept() -> Percept {
  Percept(
    vision: None,
    reading: None,
    hearing: None,
    thought: Thought(
      content: "",
      emotion: sense.neutral_emotion(),
      action: sense.Observe,
    ),
    timestamp: now_timestamp(),
    attention: Idle,
    novelty: 0.0,
    salience: 0.0,
    source: DirectInput("none"),
  )
}

// =============================================================================
// ATTENTION DETERMINATION
// =============================================================================

fn determine_visual_attention(
  vision: Vision,
  reading: Reading,
) -> AttentionFocus {
  // Priority: error detection > code > scene type
  case has_error_text(reading.text) {
    True -> ErrorDetection
    False ->
      case reading.has_code {
        True -> CodeAnalysis
        False ->
          case vision.scene {
            sense.Workspace -> WorkObservation
            sense.Communication -> SocialInteraction
            sense.Entertainment -> PassiveObservation
            sense.Browsing -> PassiveObservation
            sense.ReadingScene -> GeneralAwareness
            sense.Viewing -> PassiveObservation
            sense.Unknown -> GeneralAwareness
          }
      }
  }
}

fn determine_auditory_attention(hearing: Hearing) -> AttentionFocus {
  let text = hearing.text
  case
    string_contains_any(text, ["VIVA", "viva", "hey", "Hey"]),
    string_contains_any(text, ["help", "?", "how", "what", "why"])
  {
    True, _ -> DirectAddress
    _, True -> HelpRequest
    _, _ -> Listening
  }
}

fn has_error_text(text: String) -> Bool {
  string_contains_any(text, [
    "error", "Error", "ERROR", "fail", "Fail", "FAIL", "exception", "Exception",
    "crash", "panic",
  ])
}

fn string_contains_any(text: String, patterns: List(String)) -> Bool {
  list.any(patterns, fn(p) { string_contains(text, p) })
}

// =============================================================================
// SALIENCE CALCULATION
// =============================================================================

fn calculate_salience(thought: Thought) -> Float {
  // Emotional intensity contributes to salience
  let emotional_salience = sense.emotion_intensity(thought.emotion)

  // Action urgency contributes
  let action_salience = case thought.action {
    sense.Alert -> 0.9
    sense.OfferHelp -> 0.7
    sense.Celebrate -> 0.5
    sense.Empathize -> 0.5
    sense.Rest -> 0.1
    sense.Observe -> 0.2
  }

  // Combine (max 1.0)
  let combined = emotional_salience *. 0.6 +. action_salience *. 0.4
  float_min(combined, 1.0)
}

// =============================================================================
// NOVELTY CALCULATION (requires history)
// =============================================================================

/// Calculate novelty compared to recent percepts
pub fn calculate_novelty(
  percept: Percept,
  recent_history: List(Percept),
) -> Float {
  case recent_history {
    [] -> 1.0
    // First percept is fully novel
    history -> {
      // Compare with recent percepts
      let similarities =
        list.map(history, fn(past) { percept_similarity(percept, past) })

      let avg_similarity = case list.length(similarities) {
        0 -> 0.0
        n -> list.fold(similarities, 0.0, float_add) /. int.to_float(n)
      }

      // Novelty is inverse of similarity
      1.0 -. avg_similarity
    }
  }
}

/// Calculate similarity between two percepts
pub fn percept_similarity(a: Percept, b: Percept) -> Float {
  let attention_sim = case a.attention == b.attention {
    True -> 0.3
    False -> 0.0
  }

  let scene_sim = case a.vision, b.vision {
    Some(va), Some(vb) ->
      case va.scene == vb.scene {
        True -> 0.3
        False -> 0.0
      }
    _, _ -> 0.0
  }

  let emotion_sim = {
    let e1 = a.thought.emotion
    let e2 = b.thought.emotion
    let diff = sense.emotion_delta(e1, e2)
    let dist = sense.emotion_intensity(diff)
    // Convert distance to similarity (closer = more similar)
    float_max(0.0, 0.4 -. dist *. 0.4)
  }

  attention_sim +. scene_sim +. emotion_sim
}

/// Update percept with calculated novelty
pub fn with_novelty(percept: Percept, history: List(Percept)) -> Percept {
  let novelty = calculate_novelty(percept, history)
  Percept(..percept, novelty: novelty)
}

// =============================================================================
// PERCEPT ANALYSIS
// =============================================================================

/// Is this percept emotionally significant?
pub fn is_emotionally_significant(percept: Percept) -> Bool {
  sense.emotion_intensity(percept.thought.emotion) >. 0.5
}

/// Does this percept require action?
pub fn requires_action(percept: Percept) -> Bool {
  case percept.thought.action {
    sense.Alert -> True
    sense.OfferHelp -> True
    _ -> False
  }
}

/// Get the dominant content type
pub fn content_type(percept: Percept) -> String {
  case percept.vision, percept.hearing {
    Some(_), _ -> "visual"
    _, Some(_) -> "auditory"
    _, _ -> "internal"
  }
}

/// Extract text content (from reading or hearing)
pub fn text_content(percept: Percept) -> String {
  case percept.reading, percept.hearing {
    Some(r), _ -> r.text
    _, Some(h) -> h.text
    _, _ -> ""
  }
}

/// Get scene description
pub fn scene_description(percept: Percept) -> String {
  case percept.vision {
    Some(v) -> v.dominant <> " (" <> scene_to_string(v.scene) <> ")"
    None -> "no visual"
  }
}

fn scene_to_string(scene: SceneType) -> String {
  case scene {
    sense.Workspace -> "workspace"
    sense.Communication -> "communication"
    sense.Browsing -> "browsing"
    sense.Entertainment -> "entertainment"
    sense.ReadingScene -> "reading"
    sense.Viewing -> "viewing"
    sense.Unknown -> "unknown"
  }
}

// =============================================================================
// PERCEPT TO HRR VECTOR
// =============================================================================

/// Convert percept to HRR memory vector (512 dimensions)
/// This creates a holographic representation for storage/retrieval
pub fn to_memory_vector(percept: Percept) -> List(Float) {
  let dim = 512

  // Component vectors
  let attention_vec = attention_to_vector(percept.attention, dim)
  let emotion_vec = emotion_to_vector(percept.thought.emotion, dim)
  let content_vec = content_to_vector(percept, dim)
  let time_vec = time_to_vector(percept.timestamp, dim)

  // Bind components (simplified: weighted sum, normalized)
  list.range(0, dim - 1)
  |> list.map(fn(i) {
    let a = list_at(attention_vec, i, 0.0)
    let e = list_at(emotion_vec, i, 0.0)
    let c = list_at(content_vec, i, 0.0)
    let t = list_at(time_vec, i, 0.0)
    a *. 0.3 +. e *. 0.3 +. c *. 0.3 +. t *. 0.1
  })
  |> normalize_vector()
}

fn attention_to_vector(attention: AttentionFocus, dim: Int) -> List(Float) {
  let seed = case attention {
    CodeAnalysis -> 1
    ErrorDetection -> 2
    WorkObservation -> 3
    SocialInteraction -> 4
    PassiveObservation -> 5
    DirectAddress -> 6
    HelpRequest -> 7
    Listening -> 8
    GeneralAwareness -> 9
    Idle -> 0
  }
  pseudo_random_vector(seed, dim)
}

fn emotion_to_vector(emotion: Emotion, dim: Int) -> List(Float) {
  // Encode PAD as periodic functions
  list.range(0, dim - 1)
  |> list.map(fn(i) {
    let phase = 2.0 *. pi() *. int.to_float(i) /. int.to_float(dim)
    emotion.valence
    *. float_sin(phase)
    +. emotion.arousal
    *. float_cos(phase *. 2.0)
    +. emotion.dominance
    *. float_sin(phase *. 3.0)
  })
  |> normalize_vector()
}

fn content_to_vector(percept: Percept, dim: Int) -> List(Float) {
  let text = text_content(percept)
  case text {
    "" -> list.repeat(0.0, dim)
    _ -> {
      // Hash-based vector from text
      let seed = string_hash(text)
      pseudo_random_vector(seed, dim)
    }
  }
}

fn time_to_vector(timestamp: Int, dim: Int) -> List(Float) {
  // Cyclic encoding of time
  let hour_phase = int.to_float(timestamp / 3600 % 24) /. 24.0 *. 2.0 *. pi()

  list.range(0, dim - 1)
  |> list.map(fn(i) {
    let phase = 2.0 *. pi() *. int.to_float(i) /. int.to_float(dim)
    float_sin(hour_phase +. phase)
  })
  |> normalize_vector()
}

fn pseudo_random_vector(seed: Int, dim: Int) -> List(Float) {
  // Simple LCG-based pseudo-random
  list.range(0, dim - 1)
  |> list.map(fn(i) {
    let x = { seed * 1_103_515_245 + i * 12_345 } % 2_147_483_648
    int.to_float(x) /. 2_147_483_648.0 *. 2.0 -. 1.0
  })
  |> normalize_vector()
}

fn normalize_vector(vec: List(Float)) -> List(Float) {
  let sum_sq = list.fold(vec, 0.0, fn(acc, x) { acc +. x *. x })
  let norm = float_sqrt(sum_sq)
  case norm >. 0.0 {
    True -> list.map(vec, fn(x) { x /. norm })
    False -> vec
  }
}

fn list_at(lst: List(Float), index: Int, default: Float) -> Float {
  list_at_impl(lst, index, default, 0)
}

fn list_at_impl(
  lst: List(Float),
  target: Int,
  default: Float,
  current: Int,
) -> Float {
  case lst {
    [] -> default
    [head, ..tail] -> {
      case current == target {
        True -> head
        False -> list_at_impl(tail, target, default, current + 1)
      }
    }
  }
}

// =============================================================================
// FFI HELPERS
// =============================================================================

@external(erlang, "os", "system_time")
fn now_timestamp() -> Int

@external(erlang, "erlang", "phash2")
fn string_hash(s: String) -> Int

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "pi")
fn pi() -> Float

fn float_add(a: Float, b: Float) -> Float {
  a +. b
}

fn float_min(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}

fn float_max(a: Float, b: Float) -> Float {
  case a >. b {
    True -> a
    False -> b
  }
}

fn string_contains(haystack: String, needle: String) -> Bool {
  string.contains(haystack, needle)
}
