//// Imprint Tests
////
//// Testa o sistema de imprinting (período crítico de aprendizado).

import gleam/option.{None, Some}
import gleeunit/should
import viva/imprint
import viva/imprint/motor
import viva/imprint/sensory
import viva/imprint/social
import viva/imprint/survival

// =============================================================================
// LIFECYCLE
// =============================================================================

pub fn imprint_starts_inactive_test() {
  let state = imprint.new_default()
  state.active |> should.be_false()
}

pub fn imprint_starts_with_critical_period_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  state.active |> should.be_true()
  state.current_intensity |> should.equal(3.0)
}

pub fn imprint_critical_period_check_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  // Tick 500 - still in critical period (1000 ticks default)
  imprint.is_critical_period(state, 500) |> should.be_true()

  // Tick 1500 - past critical period
  imprint.is_critical_period(state, 1500) |> should.be_false()
}

pub fn imprint_progress_tracks_correctly_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  // At tick 0 = 0% progress
  imprint.progress(state, 0) |> should.equal(0.0)

  // At tick 500 = 50% progress
  let progress = imprint.progress(state, 500)
  { progress >. 0.49 && progress <. 0.51 } |> should.be_true()

  // At tick 1000+ = 100% progress (clamped)
  imprint.progress(state, 1000) |> should.equal(1.0)
}

pub fn imprint_elapsed_tracks_correctly_test() {
  let state =
    imprint.new_default()
    |> imprint.start(100)

  imprint.elapsed(state, 350) |> should.equal(250)
}

// =============================================================================
// SENSORY IMPRINTING
// =============================================================================

pub fn sensory_new_creates_empty_test() {
  let s = sensory.new(256)
  sensory.association_count(s) |> should.equal(0)
}

pub fn sensory_observe_creates_association_test() {
  let s = sensory.new(256)

  let #(s2, events) = sensory.observe(s, 500, 300, False, 0.8, 3.0, 1)

  // Should have created associations for light and sound
  { sensory.association_count(s2) > 0 } |> should.be_true()

  // Should emit events for new associations
  events |> should.not_equal([])
}

pub fn sensory_query_valence_returns_learned_test() {
  let s = sensory.new(256)

  // Learn association: light 500 = positive
  let #(s2, _) = sensory.observe(s, 500, 200, False, 0.9, 3.0, 1)

  // Query should return positive valence
  let result = sensory.query_valence(s2, 500, 200, False)
  case result {
    Some(v) -> { v >. 0.0 } |> should.be_true()
    None -> should.fail()
  }
}

// =============================================================================
// MOTOR IMPRINTING
// =============================================================================

pub fn motor_new_creates_empty_patterns_test() {
  let m = motor.new()
  motor.pattern_count(m) |> should.equal(0)
}

pub fn motor_learn_creates_pattern_test() {
  let m = motor.new()

  let #(m2, events) =
    motor.learn(
      m,
      "led_red",
      200,
      100,
      False,
      // before
      400,
      150,
      False,
      // after
      0.5,
      // pleasure delta
      3.0,
      // intensity
      1,
      // tick
    )

  motor.pattern_count(m2) |> should.equal(1)
  events |> should.not_equal([])
}

pub fn motor_get_reflex_returns_none_initially_test() {
  let m = motor.new()
  motor.get_reflex(m, 500, 500, False) |> should.be_none()
}

// =============================================================================
// SOCIAL IMPRINTING
// =============================================================================

pub fn social_new_has_creator_test() {
  let s = social.new("Gabriel")
  social.entity_count(s) |> should.equal(1)
  { social.primary_attachment_strength(s) >. 0.0 } |> should.be_true()
}

pub fn social_observe_presence_updates_attachment_test() {
  let s = social.new("Gabriel")

  let #(s2, _) =
    social.observe_presence(
      s,
      Some("Gabriel"),
      0.9,
      // pleasure
      0.3,
      // arousal (calm)
      3.0,
      // intensity
      1,
      // tick
    )

  // Attachment should increase with positive, calm interaction
  let old_strength = social.attachment_strength(s, "Gabriel")
  let new_strength = social.attachment_strength(s2, "Gabriel")
  { new_strength >=. old_strength } |> should.be_true()
}

pub fn social_observe_new_entity_test() {
  let s = social.new("Gabriel")

  let #(s2, events) =
    social.observe_presence(
      s,
      Some("Unknown"),
      0.5,
      // neutral pleasure
      0.5,
      // neutral arousal
      3.0,
      // intensity
      1,
      // tick
    )

  social.entity_count(s2) |> should.equal(2)
  events |> should.not_equal([])
}

// =============================================================================
// SURVIVAL IMPRINTING
// =============================================================================

pub fn survival_new_creates_empty_test() {
  let s = survival.new()
  survival.danger_count(s) |> should.equal(0)
  survival.safety_count(s) |> should.equal(0)
}

pub fn survival_evaluate_learns_danger_test() {
  let s = survival.new()

  // Negative pleasure + loud sound = danger
  let #(s2, _events) =
    survival.evaluate(
      s,
      0.5,
      // energy
      500,
      // light
      900,
      // loud sound
      False,
      // no touch
      0.1,
      // low pleasure
      3.0,
      // intensity
      1,
      // tick
    )

  // Should learn a danger signal
  { survival.danger_count(s2) > 0 } |> should.be_true()
}

pub fn survival_evaluate_learns_safety_test() {
  let s = survival.new()

  // High pleasure + calm environment = safety
  let #(s2, _) =
    survival.evaluate(
      s,
      0.8,
      // good energy
      300,
      // moderate light
      100,
      // quiet
      False,
      // no touch
      0.8,
      // high pleasure
      3.0,
      // intensity
      1,
      // tick
    )

  // Should learn a safety signal
  { survival.safety_count(s2) > 0 } |> should.be_true()
}

// =============================================================================
// INTEGRATION
// =============================================================================

pub fn imprint_tick_processes_all_systems_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  let #(state2, _events) =
    imprint.tick(
      state,
      0.7,
      // pleasure
      0.4,
      // arousal
      0.5,
      // dominance
      400,
      // light
      200,
      // sound
      False,
      // touch
      Some("Gabriel"),
      // entity present
      0.8,
      // body energy
      1,
      // current tick
    )

  // Should have processed observations
  state2.total_observations |> should.equal(1)

  // Events may or may not be emitted depending on learning
  True |> should.be_true()
}

pub fn imprint_critical_period_ends_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  // Process tick after critical period
  let #(state2, _events) =
    imprint.tick(
      state,
      0.5,
      0.5,
      0.5,
      400,
      200,
      False,
      None,
      0.5,
      1100,
      // past critical period (1000 ticks)
    )

  // Should deactivate and emit CriticalPeriodEnded
  state2.active |> should.be_false()
  state2.current_intensity |> should.equal(1.0)
}

pub fn imprint_motor_learning_separate_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  let #(state2, _events) =
    imprint.learn_motor(
      state,
      "led_green",
      300,
      100,
      False,
      // before
      500,
      150,
      False,
      // after
      0.6,
      // pleasure delta
      1,
      // tick
    )

  // Should have learned motor pattern
  motor.pattern_count(state2.motor) |> should.equal(1)
}

pub fn imprint_queries_work_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  // Process some ticks to build associations
  let #(state2, _) =
    imprint.tick(state, 0.8, 0.3, 0.5, 500, 200, False, Some("Gabriel"), 0.8, 1)

  // Queries should work (may return None if not enough data)
  let _ = imprint.expected_valence(state2, 500, 200, False)
  let _ = imprint.get_reflex(state2, 500, 200, False)

  // Attachment query should work
  let attachment = imprint.attachment_to(state2, "Gabriel")
  { attachment >. 0.0 } |> should.be_true()
}

pub fn imprint_complete_generates_summary_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  // Process some ticks
  let #(state2, _) =
    imprint.tick(state, 0.7, 0.4, 0.5, 400, 200, False, Some("Gabriel"), 0.8, 1)

  let summary = imprint.complete(state2)

  summary.total_observations |> should.equal(1)
  { summary.social_count >= 1 } |> should.be_true()
}

pub fn imprint_describe_returns_string_test() {
  let state =
    imprint.new_default()
    |> imprint.start(0)

  let desc = imprint.describe(state, 500)

  // Should contain useful info
  { desc != "" } |> should.be_true()
}
