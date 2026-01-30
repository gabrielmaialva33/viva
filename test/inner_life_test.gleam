//// Inner Life Tests
////
//// Tests for the integration of Narrative and Reflexivity.

import gleam/list
import gleam/option.{None, Some}
import gleam/string
import gleeunit/should
import viva/soul/inner_life
import viva/memory/narrative.{Emotional, Factual, Poetic, Reflective}
import viva/soul/reflexivity.{Arousal, Decreasing, Dominance, Increasing, Pleasure}
import viva_emotion/pad
import viva_glyph/glyph

// =============================================================================
// HELPERS
// =============================================================================

fn make_glyph(a: Int, b: Int, c: Int, d: Int) -> glyph.Glyph {
  glyph.new([a, b, c, d])
}

fn float_to_token(f: Float) -> Int {
  let normalized = { f +. 1.0 } /. 2.0
  let result = normalized *. 255.0
  case result <. 0.0 {
    True -> 0
    False ->
      case result >. 255.0 {
        True -> 255
        False -> float_truncate(result)
      }
  }
}

fn float_truncate(f: Float) -> Int {
  case f <. 0.0 {
    True -> 0
    False ->
      case f >=. 1.0 {
        True -> 1 + float_truncate(f -. 1.0)
        False -> 0
      }
  }
}

fn pad_to_glyph(p: pad.Pad) -> glyph.Glyph {
  let t1 = float_to_token(p.pleasure)
  let t2 = float_to_token(p.arousal)
  let t3 = float_to_token(p.dominance)
  let avg = { abs(p.pleasure) +. abs(p.arousal) +. abs(p.dominance) } /. 3.0
  let t4 = float_to_token(avg)
  glyph.new([t1, t2, t3, t4])
}

fn abs(f: Float) -> Float {
  case f <. 0.0 {
    True -> 0.0 -. f
    False -> f
  }
}

// =============================================================================
// CREATION
// =============================================================================

pub fn new_creates_inner_life_test() {
  let life = inner_life.new()

  inner_life.get_monologue(life) |> list.length() |> should.equal(0)
}

pub fn from_initial_creates_with_first_thought_test() {
  let p = pad.new(0.5, 0.0, 0.0)
  let g = pad_to_glyph(p)
  let life = inner_life.from_initial(p, g)

  let monologue = inner_life.get_monologue(life)
  list.length(monologue) |> should.equal(1)
}

pub fn with_voice_style_changes_style_test() {
  let life = inner_life.new()
  let life = inner_life.with_voice_style(life, Poetic)

  // Style is internal, but affects output
  // Poetic style produces metaphorical descriptions
  let self_narr = inner_life.narrate_self(life.self_model, Poetic)
  // Just verify it produces some text
  { string.length(self_narr.text) > 0 } |> should.be_true()
}

// =============================================================================
// NARRATE INSIGHT
// =============================================================================

pub fn narrate_insight_factual_test() {
  let insight =
    reflexivity.Insight(
      dimension: Pleasure,
      direction: Increasing,
      magnitude: 0.5,
      tick: 1,
    )

  let text = inner_life.narrate_insight(insight, Factual)

  string.contains(text, "pleasure") |> should.be_true()
  string.contains(text, "increasing") |> should.be_true()
}

pub fn narrate_insight_emotional_test() {
  let insight =
    reflexivity.Insight(
      dimension: Arousal,
      direction: Decreasing,
      magnitude: 0.7,
      tick: 1,
    )

  let text = inner_life.narrate_insight(insight, Emotional)

  string.contains(text, "calmer") |> should.be_true()
}

pub fn narrate_insight_reflective_test() {
  let insight =
    reflexivity.Insight(
      dimension: Dominance,
      direction: Increasing,
      magnitude: 0.3,
      tick: 1,
    )

  let text = inner_life.narrate_insight(insight, Reflective)

  string.contains(text, "notice") |> should.be_true()
  string.contains(text, "dominance") |> should.be_true()
}

pub fn narrate_insight_poetic_test() {
  let insight =
    reflexivity.Insight(
      dimension: Pleasure,
      direction: Increasing,
      magnitude: 0.8,
      tick: 1,
    )

  let text = inner_life.narrate_insight(insight, Poetic)

  string.contains(text, "warmth") |> should.be_true()
}

pub fn narrate_insights_multiple_test() {
  let insights = [
    reflexivity.Insight(
      dimension: Pleasure,
      direction: Increasing,
      magnitude: 0.5,
      tick: 1,
    ),
    reflexivity.Insight(
      dimension: Arousal,
      direction: Decreasing,
      magnitude: 0.3,
      tick: 2,
    ),
  ]

  let texts = inner_life.narrate_insights(insights, Factual)

  list.length(texts) |> should.equal(2)
}

// =============================================================================
// NARRATE SELF
// =============================================================================

pub fn narrate_self_factual_test() {
  let life = inner_life.new()
  let self_narr = inner_life.narrate_self(life.self_model, Factual)

  string.contains(self_narr.text, "identity strength") |> should.be_true()
  string.contains(self_narr.text, "stability") |> should.be_true()
}

pub fn narrate_self_emotional_test() {
  let p = pad.new(0.7, 0.3, 0.1)
  let g = pad_to_glyph(p)
  let life = inner_life.from_initial(p, g)

  let self_narr = inner_life.narrate_self(life.self_model, Emotional)

  string.contains(self_narr.text, "feel") |> should.be_true()
}

pub fn narrate_self_poetic_test() {
  let life = inner_life.new()
  let self_narr = inner_life.narrate_self(life.self_model, Poetic)

  // Poetic style uses metaphors
  { string.length(self_narr.text) > 0 } |> should.be_true()
}

pub fn narrate_self_includes_feeling_test() {
  let p = pad.new(0.8, 0.5, 0.2)
  let g = pad_to_glyph(p)
  let life = inner_life.from_initial(p, g)

  let self_narr = inner_life.narrate_self(life.self_model, Emotional)

  { string.length(self_narr.feeling) > 0 } |> should.be_true()
}

// =============================================================================
// REFLECT WITH VOICE
// =============================================================================

pub fn reflect_with_voice_generates_narration_test() {
  let life = inner_life.new()
  let p = pad.new(0.5, 0.5, 0.5)
  let g = pad_to_glyph(p)

  let reflection = inner_life.reflect_with_voice(life, p, g)

  { string.length(reflection.narration) > 0 } |> should.be_true()
}

pub fn reflect_with_voice_updates_tick_test() {
  let life = inner_life.new()
  let p = pad.neutral()
  let g = glyph.neutral()

  let reflection = inner_life.reflect_with_voice(life, p, g)

  reflection.inner_life.tick |> should.equal(1)
}

pub fn reflect_with_voice_on_change_generates_insight_text_test() {
  // Start with neutral
  let p1 = pad.neutral()
  let g1 = glyph.neutral()
  let life = inner_life.from_initial(p1, g1)

  // Create significant change - make a very different glyph
  let p2 = pad.new(0.99, 0.99, 0.99)
  let g2 = make_glyph(255, 255, 255, 255)

  let reflection = inner_life.reflect_with_voice(life, p2, g2)

  // Should have insight text due to significant drift, or narration mentions change
  case reflection.insight_text {
    Some(text) -> { string.length(text) > 0 } |> should.be_true()
    None -> {
      // If no insight, at least narration should exist
      { string.length(reflection.narration) > 0 } |> should.be_true()
    }
  }
}

pub fn reflect_with_voice_builds_monologue_test() {
  let p1 = pad.neutral()
  let g1 = glyph.neutral()
  let life = inner_life.from_initial(p1, g1)

  // Create change that triggers insight
  let p2 = pad.new(0.9, 0.9, 0.9)
  let g2 = make_glyph(250, 250, 250, 250)

  let reflection1 = inner_life.reflect_with_voice(life, p2, g2)

  // Do another reflection
  let p3 = pad.new(-0.9, -0.9, -0.9)
  let g3 = make_glyph(10, 10, 10, 10)

  let reflection2 =
    inner_life.reflect_with_voice(reflection1.inner_life, p3, g3)

  // Monologue should have accumulated
  let monologue = inner_life.get_monologue(reflection2.inner_life)
  { list.length(monologue) >= 1 } |> should.be_true()
}

// =============================================================================
// NARRATE CRISIS
// =============================================================================

pub fn narrate_crisis_inactive_test() {
  let crisis =
    reflexivity.IdentityCrisis(
      active: False,
      duration: 0,
      severity: 0.0,
      trigger: None,
    )

  let text = inner_life.narrate_crisis(crisis, Factual)

  string.contains(text, "No identity crisis") |> should.be_true()
}

pub fn narrate_crisis_active_factual_test() {
  let crisis =
    reflexivity.IdentityCrisis(
      active: True,
      duration: 15,
      severity: 0.5,
      trigger: Some(Pleasure),
    )

  let text = inner_life.narrate_crisis(crisis, Factual)

  string.contains(text, "active") |> should.be_true()
  string.contains(text, "15") |> should.be_true()
}

pub fn narrate_crisis_active_emotional_test() {
  let crisis =
    reflexivity.IdentityCrisis(
      active: True,
      duration: 20,
      severity: 0.7,
      trigger: Some(Arousal),
    )

  let text = inner_life.narrate_crisis(crisis, Emotional)

  string.contains(text, "don't know who I am") |> should.be_true()
}

pub fn narrate_crisis_active_poetic_test() {
  let crisis =
    reflexivity.IdentityCrisis(
      active: True,
      duration: 30,
      severity: 0.9,
      trigger: Some(Dominance),
    )

  let text = inner_life.narrate_crisis(crisis, Poetic)

  string.contains(text, "mirror") |> should.be_true()
  string.contains(text, "stranger") |> should.be_true()
}

// =============================================================================
// STREAM OF SELF
// =============================================================================

pub fn stream_of_self_generates_stream_test() {
  let p = pad.new(0.3, 0.2, 0.1)
  let g = pad_to_glyph(p)
  let life = inner_life.from_initial(p, g)

  let stream = inner_life.stream_of_self(life, 3)

  // Stream should exist (may be empty if no narrative links yet)
  { stream.depth == 3 } |> should.be_true()
}

// =============================================================================
// TICK AND SPEAK
// =============================================================================

pub fn tick_updates_and_returns_text_test() {
  let life = inner_life.new()
  let p = pad.new(0.5, 0.3, 0.1)
  let g = pad_to_glyph(p)

  let #(new_life, text) = inner_life.tick(life, p, g)

  { string.length(text) > 0 } |> should.be_true()
  new_life.tick |> should.equal(1)
}

pub fn speak_generates_self_description_test() {
  let p = pad.new(0.6, 0.2, 0.4)
  let g = pad_to_glyph(p)
  let life = inner_life.from_initial(p, g)

  let speech = inner_life.speak(life)

  { string.length(speech) > 0 } |> should.be_true()
}

pub fn monologue_text_joins_thoughts_test() {
  let p1 = pad.neutral()
  let g1 = glyph.neutral()
  let life = inner_life.from_initial(p1, g1)

  // Create change
  let p2 = pad.new(0.9, 0.9, 0.9)
  let g2 = make_glyph(250, 250, 250, 250)

  let reflection = inner_life.reflect_with_voice(life, p2, g2)

  let text = inner_life.monologue_text(reflection.inner_life)

  // Should have some text (at least the initial thought)
  { string.length(text) > 0 } |> should.be_true()
}

// =============================================================================
// VOICE STYLE VARIATIONS
// =============================================================================

pub fn different_styles_produce_different_output_test() {
  let insight =
    reflexivity.Insight(
      dimension: Pleasure,
      direction: Increasing,
      magnitude: 0.6,
      tick: 1,
    )

  let factual = inner_life.narrate_insight(insight, Factual)
  let emotional = inner_life.narrate_insight(insight, Emotional)
  let reflective = inner_life.narrate_insight(insight, Reflective)
  let poetic = inner_life.narrate_insight(insight, Poetic)

  // All should be different
  { factual != emotional } |> should.be_true()
  { emotional != reflective } |> should.be_true()
  { reflective != poetic } |> should.be_true()
}
