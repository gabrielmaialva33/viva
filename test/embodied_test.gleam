//// Tests for VIVA's Embodied Perception System
////
//// Tests sense types, percepts, and awareness integration

import gleam/option.{None, Some}
import gleeunit/should
import viva/embodied/awareness.{
  AlertUser, DoNothing, Express, ObserveOnly, OfferAssistance,
}
import viva/embodied/percept.{
  CodeAnalysis, DirectAddress, ErrorDetection, GeneralAwareness, HelpRequest,
  Idle, Listening, PassiveObservation, SocialInteraction, WorkObservation,
}
import viva/embodied/sense.{
  Alert, Celebrate, Emotion, Empathize, Observe, OfferHelp, Reading, Rest,
  Thought, Vision,
}

// =============================================================================
// SENSE TYPES TESTS
// =============================================================================

pub fn neutral_emotion_test() {
  let emotion = sense.neutral_emotion()

  // Neutral: valence near 0, moderate arousal, moderate dominance
  should.be_true(emotion.valence >. -0.1 && emotion.valence <. 0.1)
  should.be_true(emotion.arousal >. 0.2 && emotion.arousal <. 0.5)
  should.be_true(emotion.dominance >. 0.3 && emotion.dominance <. 0.7)
}

pub fn positive_emotion_test() {
  let emotion = sense.positive_emotion(0.8)

  should.be_true(emotion.valence >. 0.5)
  should.be_true(emotion.arousal >. 0.5)
  should.be_true(emotion.dominance >. 0.5)
}

pub fn negative_emotion_test() {
  let emotion = sense.negative_emotion(0.8)

  should.be_true(emotion.valence <. -0.5)
  should.be_true(emotion.arousal >. 0.5)
  should.be_true(emotion.dominance <. 0.5)
}

pub fn blend_emotions_test() {
  let positive = sense.positive_emotion(1.0)
  let negative = sense.negative_emotion(1.0)

  // 50% blend should be roughly neutral
  let blended = sense.blend_emotions(positive, negative, 0.5)

  // Should be closer to neutral
  should.be_true(blended.valence >. -0.5 && blended.valence <. 0.5)
}

pub fn emotion_intensity_neutral_low_test() {
  let neutral = sense.neutral_emotion()
  let intensity = sense.emotion_intensity(neutral)

  // Neutral should have low intensity
  should.be_true(intensity <. 0.5)
}

pub fn emotion_intensity_extreme_high_test() {
  let extreme = sense.positive_emotion(1.0)
  let intensity = sense.emotion_intensity(extreme)

  // Extreme emotion should have high intensity
  should.be_true(intensity >. 0.5)
}

pub fn emotion_delta_test() {
  let baseline = sense.neutral_emotion()
  let current = sense.positive_emotion(0.8)

  let delta = sense.emotion_delta(current, baseline)

  // Delta should show increase in valence
  should.be_true(delta.valence >. 0.5)
}

// =============================================================================
// SCENE CLASSIFICATION TESTS
// =============================================================================

pub fn classify_code_scene_test() {
  let labels = ["code", "terminal", "text"]
  let scene = sense.classify_scene(labels)
  should.equal(scene, sense.Workspace)
}

pub fn classify_communication_scene_test() {
  let labels = ["chat", "window"]
  let scene = sense.classify_scene(labels)
  should.equal(scene, sense.Communication)
}

pub fn classify_browsing_scene_test() {
  let labels = ["browser", "search"]
  let scene = sense.classify_scene(labels)
  should.equal(scene, sense.Browsing)
}

pub fn classify_entertainment_scene_test() {
  let labels = ["video", "player"]
  let scene = sense.classify_scene(labels)
  should.equal(scene, sense.Entertainment)
}

pub fn classify_reading_scene_test() {
  let labels = ["document", "pdf"]
  let scene = sense.classify_scene(labels)
  should.equal(scene, sense.ReadingScene)
}

pub fn classify_unknown_scene_test() {
  let labels = ["random", "stuff"]
  let scene = sense.classify_scene(labels)
  should.equal(scene, sense.Unknown)
}

// =============================================================================
// PERCEPT TESTS
// =============================================================================

pub fn empty_percept_test() {
  let p = percept.empty_percept()

  should.equal(p.vision, None)
  should.equal(p.reading, None)
  should.equal(p.hearing, None)
  should.equal(p.attention, Idle)
  should.equal(p.novelty, 0.0)
  should.equal(p.salience, 0.0)
}

pub fn thought_percept_test() {
  let thought =
    Thought(
      content: "Interesting observation",
      emotion: sense.positive_emotion(0.5),
      action: Observe,
    )

  let p = percept.thought_percept(thought)

  should.equal(p.vision, None)
  should.equal(p.hearing, None)
  should.equal(p.attention, GeneralAwareness)
  should.be_true(p.novelty >. 0.0)
}

pub fn visual_percept_creates_with_attention_test() {
  let vision =
    Vision(
      labels: ["code", "terminal"],
      confidence: [0.9, 0.8],
      dominant: "code editor",
      dominant_confidence: 0.9,
      scene: sense.Workspace,
    )

  let reading =
    Reading(
      text: "fn main() { println!(\"Hello\"); }",
      blocks: [],
      language: "rust",
      has_code: True,
      word_count: 5,
      line_count: 1,
    )

  let thought =
    Thought(
      content: "I see code",
      emotion: sense.positive_emotion(0.3),
      action: Observe,
    )

  let p = percept.visual_percept(vision, reading, thought, "/tmp/screenshot.png")

  should.equal(p.vision, Some(vision))
  should.equal(p.reading, Some(reading))
  // Code with has_code=True should give CodeAnalysis attention
  should.equal(p.attention, CodeAnalysis)
}

pub fn visual_percept_error_detection_test() {
  let vision =
    Vision(
      labels: ["terminal"],
      confidence: [0.9],
      dominant: "terminal",
      dominant_confidence: 0.9,
      scene: sense.Workspace,
    )

  let reading =
    Reading(
      text: "error: cannot find module\nError: compilation failed",
      blocks: [],
      language: "en",
      has_code: False,
      word_count: 6,
      line_count: 2,
    )

  let thought =
    Thought(
      content: "I see an error",
      emotion: sense.negative_emotion(0.5),
      action: Alert,
    )

  let p = percept.visual_percept(vision, reading, thought, "/tmp/error.png")

  // Error text should trigger ErrorDetection
  should.equal(p.attention, ErrorDetection)
}

// =============================================================================
// PERCEPT ANALYSIS TESTS
// =============================================================================

pub fn is_emotionally_significant_high_test() {
  let thought =
    Thought(
      content: "Very exciting!",
      emotion: sense.positive_emotion(0.9),
      action: Celebrate,
    )

  let p = percept.thought_percept(thought)

  should.be_true(percept.is_emotionally_significant(p))
}

pub fn is_emotionally_significant_low_test() {
  let thought =
    Thought(content: "Meh", emotion: sense.neutral_emotion(), action: Observe)

  let p = percept.thought_percept(thought)

  should.be_false(percept.is_emotionally_significant(p))
}

pub fn requires_action_alert_test() {
  let thought =
    Thought(
      content: "Danger!",
      emotion: sense.negative_emotion(0.9),
      action: Alert,
    )

  let p = percept.thought_percept(thought)

  should.be_true(percept.requires_action(p))
}

pub fn requires_action_observe_test() {
  let thought =
    Thought(content: "...", emotion: sense.neutral_emotion(), action: Observe)

  let p = percept.thought_percept(thought)

  should.be_false(percept.requires_action(p))
}

pub fn content_type_internal_test() {
  let p = percept.empty_percept()
  should.equal(percept.content_type(p), "internal")
}

// =============================================================================
// NOVELTY CALCULATION TESTS
// =============================================================================

pub fn novelty_first_percept_is_one_test() {
  let p = percept.empty_percept()
  let novelty = percept.calculate_novelty(p, [])

  should.equal(novelty, 1.0)
}

pub fn novelty_decreases_with_repetition_test() {
  let p = percept.empty_percept()

  // First percept has full novelty
  let novelty_first = percept.calculate_novelty(p, [])
  should.equal(novelty_first, 1.0)

  // With history of similar percepts, novelty should be less than first
  let history = [p]
  let novelty_with_history = percept.calculate_novelty(p, history)

  // Novelty with history should be <= novelty of first
  should.be_true(novelty_with_history <=. novelty_first)
}

pub fn percept_similarity_same_attention_test() {
  let p1 = percept.empty_percept()
  let p2 = percept.empty_percept()

  let sim = percept.percept_similarity(p1, p2)

  // Same percepts should have similarity > 0
  should.be_true(sim >. 0.0)
}

// =============================================================================
// AWARENESS TESTS
// =============================================================================

pub fn awareness_process_creates_result_test() {
  let thought =
    Thought(
      content: "Test observation",
      emotion: sense.neutral_emotion(),
      action: Observe,
    )

  let p = percept.thought_percept(thought)
  let result = awareness.process(p)

  // Should have a result with memory vector
  should.be_true(list_length(result.memory_vector) == 512)
}

pub fn awareness_alert_is_urgent_test() {
  let thought =
    Thought(content: "Error!", emotion: sense.negative_emotion(0.9), action: Alert)

  let p = percept.thought_percept(thought)
  let result = awareness.process(p)

  should.be_true(result.urgent)
}

pub fn awareness_observe_not_urgent_test() {
  let thought =
    Thought(content: "...", emotion: sense.neutral_emotion(), action: Observe)

  let p = percept.thought_percept(thought)
  let result = awareness.process(p)

  should.be_false(result.urgent)
}

pub fn awareness_response_alert_test() {
  let thought =
    Thought(content: "Danger!", emotion: sense.negative_emotion(0.9), action: Alert)

  let p = percept.thought_percept(thought)
  let result = awareness.process(p)

  case result.response {
    AlertUser(_) -> should.be_true(True)
    _ -> should.fail()
  }
}

pub fn awareness_response_offer_help_test() {
  let thought =
    Thought(
      content: "Need help?",
      emotion: sense.positive_emotion(0.3),
      action: OfferHelp,
    )

  let p = percept.thought_percept(thought)
  let result = awareness.process(p)

  case result.response {
    OfferAssistance(_) -> should.be_true(True)
    _ -> should.fail()
  }
}

pub fn awareness_response_celebrate_test() {
  let thought =
    Thought(
      content: "Success!",
      emotion: sense.positive_emotion(0.9),
      action: Celebrate,
    )

  let p = percept.thought_percept(thought)
  let result = awareness.process(p)

  case result.response {
    Express(t) -> should.equal(t, "celebration")
    _ -> should.fail()
  }
}

pub fn awareness_pad_delta_positive_test() {
  let thought =
    Thought(
      content: "Happy",
      emotion: sense.positive_emotion(0.8),
      action: Celebrate,
    )

  let p = percept.thought_percept(thought)
  let #(pleasure, _, _) = awareness.to_pad_delta(p)

  // Positive emotion should give positive pleasure delta
  should.be_true(pleasure >. 0.0)
}

pub fn awareness_emotional_summary_test() {
  let thought =
    Thought(
      content: "Happy",
      emotion: sense.positive_emotion(0.8),
      action: Observe,
    )

  let p = percept.thought_percept(thought)
  let summary = awareness.emotional_summary(p)

  // Should contain "positive"
  should.be_true(string_contains(summary, "positive"))
}

// =============================================================================
// CONTINUOUS AWARENESS TESTS
// =============================================================================

pub fn awareness_state_new_test() {
  let state = awareness.new_state()

  should.equal(state.tick, 0)
  should.equal(state.current_attention, Idle)
}

pub fn awareness_process_with_state_updates_tick_test() {
  let state = awareness.new_state()
  let thought =
    Thought(content: "...", emotion: sense.neutral_emotion(), action: Observe)

  let p = percept.thought_percept(thought)
  let #(new_state, _result) = awareness.process_with_state(state, p)

  should.equal(new_state.tick, 1)
}

pub fn awareness_momentum_blends_test() {
  let state = awareness.new_state()

  let thought1 =
    Thought(
      content: "Happy!",
      emotion: sense.positive_emotion(1.0),
      action: Observe,
    )

  let p1 = percept.thought_percept(thought1)
  let #(state2, _) = awareness.process_with_state(state, p1)

  // Momentum should be influenced by positive emotion
  let momentum = awareness.get_momentum(state2)
  should.be_true(momentum.valence >. 0.0)
}

pub fn awareness_worth_remembering_high_salience_test() {
  let thought =
    Thought(content: "Important!", emotion: sense.positive_emotion(0.9), action: Alert)

  // Create percept with high salience by having Alert action
  let p = percept.thought_percept(thought)

  should.be_true(awareness.worth_remembering(p))
}

// =============================================================================
// HRR VECTOR TESTS
// =============================================================================

pub fn hrr_vector_dimension_test() {
  let p = percept.empty_percept()
  let vec = percept.to_memory_vector(p)

  should.equal(list_length(vec), 512)
}

pub fn hrr_vector_normalized_test() {
  let thought =
    Thought(content: "Test", emotion: sense.positive_emotion(0.5), action: Observe)

  let p = percept.thought_percept(thought)
  let vec = percept.to_memory_vector(p)

  // Vector should be normalized (magnitude close to 1)
  let magnitude = sqrt(list_sum_squared(vec))
  should.be_true(magnitude >. 0.9 && magnitude <. 1.1)
}

// =============================================================================
// HELPERS
// =============================================================================

fn list_length(lst: List(a)) -> Int {
  case lst {
    [] -> 0
    [_, ..rest] -> 1 + list_length(rest)
  }
}

fn list_sum_squared(lst: List(Float)) -> Float {
  case lst {
    [] -> 0.0
    [head, ..tail] -> head *. head +. list_sum_squared(tail)
  }
}

@external(erlang, "math", "sqrt")
fn sqrt(x: Float) -> Float

import gleam/string

fn string_contains(haystack: String, needle: String) -> Bool {
  string.contains(haystack, needle)
}
